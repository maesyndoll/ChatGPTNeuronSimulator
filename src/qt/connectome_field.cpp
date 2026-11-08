#include "qt/connectome_field.h"

#include <QDateTime>
#include <QSGGeometry>
#include <QSGGeometryNode>
#include <QSGNode>
#include <QSGVertexColorMaterial>
#include <QMutexLocker>
#include <QVariant>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace {
struct FieldNode final : QSGNode {
    QSGGeometryNode* edges = nullptr;
    QSGGeometryNode* neurons = nullptr;\n    QSGGeometryNode* pulses = nullptr;

    FieldNode() {
        edges = new QSGGeometryNode;
        auto* edgeGeometry = new QSGGeometry(
            QSGGeometry::defaultAttributes_ColoredPoint2D(), 0);
        edgeGeometry->setDrawingMode(QSGGeometry::DrawLines);
        edgeGeometry->setLineWidth(1.0f);
        edges->setGeometry(edgeGeometry);
        edges->setFlag(QSGNode::OwnsGeometry);
        auto* edgeMaterial = new QSGVertexColorMaterial;
        edges->setMaterial(edgeMaterial);
        edges->setFlag(QSGNode::OwnsMaterial);
        appendChildNode(edges);

        neurons = new QSGGeometryNode;
        auto* neuronGeometry = new QSGGeometry(
            QSGGeometry::defaultAttributes_ColoredPoint2D(), 0);
        neuronGeometry->setDrawingMode(QSGGeometry::DrawTriangles);
        neurons->setGeometry(neuronGeometry);
        neurons->setFlag(QSGNode::OwnsGeometry);
        auto* neuronMaterial = new QSGVertexColorMaterial;
        neurons->setMaterial(neuronMaterial);
        neurons->setFlag(QSGNode::OwnsMaterial);
        appendChildNode(neurons);

        pulses = new QSGGeometryNode;
        auto* pulseGeometry = new QSGGeometry(
            QSGGeometry::defaultAttributes_ColoredPoint2D(), 0);
        pulseGeometry->setDrawingMode(QSGGeometry::DrawTriangles);
        pulses->setGeometry(pulseGeometry);
        pulses->setFlag(QSGNode::OwnsGeometry);
        auto* pulseMaterial = new QSGVertexColorMaterial;
        pulses->setMaterial(pulseMaterial);
        pulses->setFlag(QSGNode::OwnsMaterial);
        appendChildNode(pulses);
    }
};

using Point = QSGGeometry::ColoredPoint2D;

void put(Point& p, float x, float y, uchar r, uchar g, uchar b, uchar a) {
    p.set(x, y, r, g, b, a);
}
}

ConnectomeField::ConnectomeField(QQuickItem* parent) : QQuickItem(parent) {
    setFlag(ItemHasContents, true);
}

void ConnectomeField::setTopology(const QVariantMap& value) {
    QMutexLocker locker(&mutex_);
    if (topology_ == value) return;
    topology_ = value;
    nodes_.clear();
    edges_.clear();
    const auto nodeValues = value.value(QStringLiteral("nodes")).toList();
    nodes_.reserve(nodeValues.size());
    for (const auto& item : nodeValues)
        nodes_.push_back(item.toUInt());
    const auto edgeValues = value.value(QStringLiteral("edges")).toList();
    edges_.reserve(edgeValues.size());
    for (const auto& item : edgeValues) {
        const auto map = item.toMap();
        Edge edge;
        edge.source = map.value(QStringLiteral("source")).toUInt();
        edge.target = map.value(QStringLiteral("target")).toUInt();
        edge.delay = std::max(1u, map.value(QStringLiteral("delay")).toUInt());
        edge.excitatory = map.value(QStringLiteral("excitatory")).toBool();
        edges_.push_back(edge);
    }
    emit topologyChanged();
    update();
}

void ConnectomeField::setSpikeEvents(const QVariantList& value) {
    QMutexLocker locker(&mutex_);
    if (spike_events_ == value) return;
    spike_events_ = value;
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    const QString generation = topology_.value(
        QStringLiteral("simulation_generation")).toString();
    if (!generation.isEmpty() && generation_ != generation) {
        generation_ = generation;
        spikes_.clear();
        last_sequence_ = 0;
    }
    for (const auto& item : value) {
        const auto map = item.toMap();
        const qint64 sequence = map.value(QStringLiteral("sequence")).toString().toLongLong();
        if (sequence <= last_sequence_) continue;
        last_sequence_ = sequence;
        spikes_.push_back({map.value(QStringLiteral("neuron_id")).toUInt(), now});
    }
    while (!spikes_.isEmpty() && now - spikes_.front().received_ms > 2600)
        spikes_.removeFirst();
    if (spikes_.size() > 4096)
        spikes_.remove(0, spikes_.size() - 4096);
    emit spikeEventsChanged();
    update();
}

void ConnectomeField::setRotationX(qreal value) {
    QMutexLocker locker(&mutex_);
    if (qFuzzyCompare(rotation_x_, value)) return;
    rotation_x_ = value; emit cameraChanged(); update();
}
void ConnectomeField::setRotationY(qreal value) {
    QMutexLocker locker(&mutex_);
    if (qFuzzyCompare(rotation_y_, value)) return;
    rotation_y_ = value; emit cameraChanged(); update();
}
void ConnectomeField::setCameraDistance(qreal value) {
    value = std::clamp(value, qreal(1.45), qreal(7.0));
    QMutexLocker locker(&mutex_);
    if (qFuzzyCompare(camera_distance_, value)) return;
    camera_distance_ = value; emit cameraChanged(); update();
}

void ConnectomeField::advanceFrame() { update(); }

QSGNode* ConnectomeField::updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*) {
    QMutexLocker locker(&mutex_);
    auto* root = oldNode ? static_cast<FieldNode*>(oldNode) : new FieldNode;
    const int edgeCount = edges_.size();
    const int nodeCount = nodes_.size();
    auto* edgeGeometry = root->edges->geometry();
    auto* nodeGeometry = root->neurons->geometry();
    edgeGeometry->allocate(edgeCount * 2);
    nodeGeometry->allocate(nodeCount * 6);

    const float w = float(width()), h = float(height());
    const float extent = std::min(w, h) * 0.43f;
    const float cy = std::cos(float(rotation_y_)), sy = std::sin(float(rotation_y_));
    const float cx = std::cos(float(rotation_x_)), sx = std::sin(float(rotation_x_));
    const float count = float(std::max(1, nodeCount));
    const qint64 now = QDateTime::currentMSecsSinceEpoch();

    std::unordered_map<std::uint32_t, float> activity;
    activity.reserve(static_cast<std::size_t>(spikes_.size()));
    for (const auto& spike : spikes_) {
        const qint64 age = now - spike.received_ms;
        if (age >= 0 && age < 2600)
            activity[spike.neuron] += std::exp(-float(age) / 720.0f);
    }

    struct Position { float x, y, z, scale, intensity; };
    std::unordered_map<std::uint32_t, Position> positions;
    positions.reserve(static_cast<std::size_t>(nodeCount));
    for (int i = 0; i < nodeCount; ++i) {
        const float z0 = 1.0f - 2.0f * (float(i) + 0.5f) / count;
        const float ring = std::sqrt(std::max(0.0f, 1.0f - z0 * z0));
        const float angle = float(i) * 2.399963229728653f;
        const auto id = nodes_[i];
        const float intensity = std::min(2.8f, activity[id]);
        const float wave = float(now) * 0.00125f + angle * 1.7f + z0 * 5.0f;
        const float radial = 1.0f + intensity * 0.075f +
            std::sin(wave) * (0.012f + intensity * 0.025f);
        const float x0 = std::cos(angle) * ring * radial;
        const float y0 = std::sin(angle) * ring * radial;
        const float z = z0 * radial + std::sin(wave * 0.7f) * intensity * 0.045f;
        const float twist = intensity * 0.10f + std::sin(wave * 0.35f) * 0.018f;
        const float tx = x0 * std::cos(twist) - y0 * std::sin(twist);
        const float ty = x0 * std::sin(twist) + y0 * std::cos(twist);
        const float x1 = tx * cy + z * sy;
        const float z1 = -tx * sy + z * cy;
        const float y1 = ty * cx - z1 * sx;
        const float z2 = ty * sx + z1 * cx;
        const float perspective = float(camera_distance_) /
            std::max(0.35f, float(camera_distance_) - z2);
        positions.emplace(id, Position{
            w * 0.5f + x1 * extent * perspective,
            h * 0.5f + y1 * extent * perspective,
            z2, perspective, intensity});
    }

    auto* edgeVertices = static_cast<Point*>(edgeGeometry->vertexData());
    for (int i = 0; i < edgeCount; ++i) {
        const auto& edge = edges_[i];
        const auto fromIt = positions.find(edge.source);
        const auto toIt = positions.find(edge.target);
        const Position from = fromIt == positions.end() ? Position{0,0,0,1,0} : fromIt->second;
        const Position to = toIt == positions.end() ? Position{0,0,0,1,0} : toIt->second;
        const float intensity = std::max(from.intensity, to.intensity);
        const uchar alpha = static_cast<uchar>(std::clamp(8.0f + intensity * 45.0f, 8.0f, 145.0f));
        const uchar r = edge.excitatory ? 55 : 177;
        const uchar g = edge.excitatory ? 154 : 112;
        const uchar b = edge.excitatory ? 153 : 65;
        put(edgeVertices[i*2], from.x, from.y, r, g, b, alpha);
        put(edgeVertices[i*2+1], to.x, to.y, r, g, b, alpha);

        const auto sourcePulses = pulseTimes.find(edge.source);
        if (sourcePulses != pulseTimes.end()) {
            const float duration = 240.0f + 1760.0f *
                (float(edge.delay) / float(maxDelay));
            int emitted = 0;
            for (const qint64 firedAt : sourcePulses->second) {
                const float age = float(now - firedAt);
                if (age < 0.0f || age >= duration) continue;
                const float t = age / duration;
                const float px = from.x + (to.x - from.x) * t;
                const float py = from.y + (to.y - from.y) * t;
                const float radius = 1.5f + (1.0f - t) * 1.2f;
                const uchar pr = edge.excitatory ? 190 : 255;
                const uchar pg = edge.excitatory ? 255 : 198;
                const uchar pb = edge.excitatory ? 232 : 132;
                const int base = pulseVertices.size();
                pulseVertices.resize(base + 6);
                put(pulseVertices[base], px-radius, py-radius, pr, pg, pb, 245);
                put(pulseVertices[base+1], px+radius, py-radius, pr, pg, pb, 245);
                put(pulseVertices[base+2], px+radius, py+radius, pr, pg, pb, 245);
                put(pulseVertices[base+3], px-radius, py-radius, pr, pg, pb, 245);
                put(pulseVertices[base+4], px+radius, py+radius, pr, pg, pb, 245);
                put(pulseVertices[base+5], px-radius, py+radius, pr, pg, pb, 245);
                if (++emitted >= 3) break;
            }
        }
    }
    pulseGeometry->allocate(pulseVertices.size());
    auto* pulseData = static_cast<Point*>(pulseGeometry->vertexData());
    std::copy(pulseVertices.cbegin(), pulseVertices.cend(), pulseData);

    auto* neuronVertices = static_cast<Point*>(nodeGeometry->vertexData());
    const float excitatoryCount = topology_.value(QStringLiteral("excitatory_count")).toFloat();
    for (int i = 0; i < nodeCount; ++i) {
        const auto id = nodes_[i];
        const auto p = positions[id];
        const float radius = (0.9f + (p.z + 1.0f) * 0.7f) * p.scale *
            (1.0f + std::min(2.0f, p.intensity) * 0.55f);
        const bool excitatory = float(id) < excitatoryCount;
        const uchar r = p.intensity > 0.12f ? 245 : (excitatory ? 94 : 237);
        const uchar g = p.intensity > 0.12f ? 255 : (excitatory ? 205 : 166);
        const uchar b = p.intensity > 0.12f ? 255 : (excitatory ? 190 : 103);
        const float x0 = p.x-radius, y0 = p.y-radius, x1 = p.x+radius, y1 = p.y+radius;
        const int k = i * 6;
        put(neuronVertices[k], x0,y0,r,g,b,245);
        put(neuronVertices[k+1], x1,y0,r,g,b,245);
        put(neuronVertices[k+2], x1,y1,r,g,b,245);
        put(neuronVertices[k+3], x0,y0,r,g,b,245);
        put(neuronVertices[k+4], x1,y1,r,g,b,245);
        put(neuronVertices[k+5], x0,y1,r,g,b,245);
    }

    edgeGeometry->markVertexDataDirty();
    nodeGeometry->markVertexDataDirty();
    root->edges->markDirty(QSGNode::DirtyGeometry);
    root->neurons->markDirty(QSGNode::DirtyGeometry);
    return root;
}
