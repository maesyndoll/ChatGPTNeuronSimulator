#include "qt/connectome_field.h"

#include "connectome/temporal_embedding.h"

#include <QDateTime>
#include <QSGGeometry>
#include <QSGGeometryNode>
#include <QSGNode>
#include <QSGVertexColorMaterial>
#include <QMutexLocker>
#include <QVariant>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace {
struct FieldNode final : QSGNode {
    QSGGeometryNode* edges = nullptr;
    QSGGeometryNode* neurons = nullptr;
    QSGGeometryNode* pulses = nullptr;

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
    connect(&embedding_watcher_,
            &QFutureWatcher<connectome::EmbeddingResult>::finished,
            this, [this]() {
        QMutexLocker locker(&mutex_);
        if (embedding_watcher_generation_ != embedding_request_generation_)
            return;

        connectome::EmbeddingResult embedding;
        try {
            embedding = embedding_watcher_.result();
        } catch (const std::exception&) {
            return; // Keep the last valid geometry if an embedding fails.
        }
        if (embedding.positions.size() !=
            static_cast<std::size_t>(nodes_.size()))
            return;

        QVector<RenderPosition> next;
        next.reserve(nodes_.size());
        for (const auto& point : embedding.positions)
            next.push_back({static_cast<float>(point.x),
                            static_cast<float>(point.y),
                            static_cast<float>(point.z)});

        // MDS eigenvectors have arbitrary signs. Align each axis to the
        // previous solution before interpolation to avoid whole-field flips.
        if (next.size() == previous_temporal_positions_.size() &&
            next.size() == temporal_positions_.size()) {
            for (int axis = 0; axis < 3; ++axis) {
                double correlation = 0.0;
                for (int i = 0; i < next.size(); ++i) {
                    const float oldValue = axis == 0
                        ? temporal_positions_[i].x
                        : axis == 1 ? temporal_positions_[i].y
                                    : temporal_positions_[i].z;
                    const float newValue = axis == 0 ? next[i].x
                        : axis == 1 ? next[i].y : next[i].z;
                    correlation += double(oldValue) * double(newValue);
                }
                if (correlation < 0.0) {
                    for (auto& point : next) {
                        if (axis == 0) point.x = -point.x;
                        else if (axis == 1) point.y = -point.y;
                        else point.z = -point.z;
                    }
                }
            }
        }

        const bool sameNeuronOrder = embedded_node_ids_ == nodes_;
        previous_temporal_positions_ = sameNeuronOrder
            ? temporal_positions_ : QVector<RenderPosition>{};
        temporal_positions_ = std::move(next);
        embedded_node_ids_ = nodes_;
        embedding_transition_started_ms_ =
            QDateTime::currentMSecsSinceEpoch();
        update();
    });
}

void ConnectomeField::setTopology(const QVariantMap& value) {
    QMutexLocker locker(&mutex_);
    if (topology_ == value) return;
    if (!topology_.isEmpty() &&
        topology_.value(QStringLiteral("revision")) ==
            value.value(QStringLiteral("revision"))) {
        topology_ = value;
        return;
    }
    const bool hadPreviousTopology = !topology_.isEmpty();
    const auto previousEdges = edges_;
    topology_ = value;
    nodes_.clear();
    edges_.clear();
    const auto nodeValues = value.value(QStringLiteral("nodes")).toList();
    nodes_.reserve(nodeValues.size());
    for (const auto& item : nodeValues)
        nodes_.push_back(item.toUInt());
    const auto edgeValues = value.value(QStringLiteral("edges")).toList();
    edges_.reserve(edgeValues.size());
    std::unordered_map<std::uint32_t, std::uint32_t> denseIndex;
    denseIndex.reserve(static_cast<std::size_t>(nodes_.size()));
    for (int i = 0; i < nodes_.size(); ++i)
        denseIndex.emplace(nodes_[i], static_cast<std::uint32_t>(i));

    std::vector<connectome::TemporalEdge> temporalEdges;
    temporalEdges.reserve(static_cast<std::size_t>(edgeValues.size()));
    for (const auto& item : edgeValues) {
        const auto map = item.toMap();
        Edge edge;
        edge.source = map.value(QStringLiteral("source")).toUInt();
        edge.target = map.value(QStringLiteral("target")).toUInt();
        edge.delay = std::max(1u, map.value(QStringLiteral("delay")).toUInt());
        edge.excitatory = map.value(QStringLiteral("excitatory")).toBool();
        edges_.push_back(edge);
        const auto source = denseIndex.find(edge.source);
        const auto target = denseIndex.find(edge.target);
        if (source != denseIndex.end() && target != denseIndex.end())
            temporalEdges.push_back({source->second, target->second,
                                     static_cast<double>(edge.delay)});
    }
    // Diff by directed endpoint pair. Delay/weight changes remain live
    // edge updates; only appearance/disappearance creates a transition.
    const auto edgeKey = [](const Edge& edge) {
        return (static_cast<std::uint64_t>(edge.source) << 32) | edge.target;
    };
    std::unordered_map<std::uint64_t, Edge> previousByKey;
    std::unordered_map<std::uint64_t, Edge> currentByKey;
    previousByKey.reserve(static_cast<std::size_t>(previousEdges.size()));
    currentByKey.reserve(static_cast<std::size_t>(edges_.size()));
    for (const auto& edge : previousEdges) previousByKey.emplace(edgeKey(edge), edge);
    for (const auto& edge : edges_) currentByKey.emplace(edgeKey(edge), edge);
    const qint64 transitionTime = QDateTime::currentMSecsSinceEpoch();
    QVector<EdgeTransition> nextTransitions;
    nextTransitions.reserve(std::min<qsizetype>(12000,
        edge_transitions_.size() + previousEdges.size() + edges_.size()));
    for (const auto& transition : edge_transitions_) {
        if (transition.created_ms + 1800 > transitionTime)
            nextTransitions.push_back(transition);
    }
    if (hadPreviousTopology) {
        for (const auto& edge : edges_) {
            if (previousByKey.find(edgeKey(edge)) == previousByKey.end())
                nextTransitions.push_back({edge, transitionTime, true});
        }
        for (const auto& edge : previousEdges) {
            if (currentByKey.find(edgeKey(edge)) == currentByKey.end())
                nextTransitions.push_back({edge, transitionTime, false});
        }
    }
    if (nextTransitions.size() > 12000)
        nextTransitions.remove(0, nextTransitions.size() - 12000);
    edge_transitions_ = std::move(nextTransitions);

    // Compute the new geometry off the GUI/render thread. Replacing the
    // watcher's future supersedes stale results when topology updates arrive
    // faster than an embedding can finish.
    if (!nodes_.isEmpty()) {
        const quint64 request = ++embedding_request_generation_;
        embedding_watcher_generation_ = request;
        const std::size_t nodeCount = static_cast<std::size_t>(nodes_.size());
        embedding_watcher_.setFuture(QtConcurrent::run(
            [nodeCount, graphEdges = std::move(temporalEdges)]() {
                return connectome::embed_temporal_graph(nodeCount, graphEdges);
            }));
    } else {
        ++embedding_request_generation_;
        embedding_watcher_generation_ = embedding_request_generation_;
        previous_temporal_positions_.clear();
        temporal_positions_.clear();
        embedded_node_ids_.clear();
    }
    emit topologyChanged();
    update();
}

void ConnectomeField::setSimulationGeneration(const QString& value) {
    QMutexLocker locker(&mutex_);
    if (generation_ == value) return;
    generation_ = value;
    spikes_.clear();
    last_sequence_ = 0;
    update();
}

void ConnectomeField::setSimulationStep(const QString& value) {
    bool ok = false;
    const std::uint64_t nextStep = value.toULongLong(&ok);
    if (!ok) return;
    QMutexLocker locker(&mutex_);
    if (simulation_step_text_ == value) return;
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (step_clock_updated_ms_ > 0 && now > step_clock_updated_ms_ &&
        nextStep >= simulation_step_) {
        const double elapsedSeconds =
            double(now - step_clock_updated_ms_) / 1000.0;
        const double instantaneousRate =
            double(nextStep - simulation_step_) / elapsedSeconds;
        if (instantaneousRate > 0.0 && std::isfinite(instantaneousRate)) {
            simulation_steps_per_second_ =
                simulation_steps_per_second_ > 0.0
                    ? 0.75 * simulation_steps_per_second_ + 0.25 * instantaneousRate
                    : instantaneousRate;
        }
    }
    simulation_step_ = nextStep;
    simulation_step_text_ = value;
    step_clock_updated_ms_ = now;
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
        bool stepOk = false;
        const std::uint64_t eventStep =
            map.value(QStringLiteral("step")).toString().toULongLong(&stepOk);
        if (!stepOk) continue;
        spikes_.push_back({
            map.value(QStringLiteral("neuron_id")).toUInt(), eventStep, now});
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
    const int transitionCount = edge_transitions_.size();
    const int nodeCount = nodes_.size();
    auto* edgeGeometry = root->edges->geometry();
    auto* nodeGeometry = root->neurons->geometry();
    auto* pulseGeometry = root->pulses->geometry();
    edgeGeometry->allocate((edgeCount + transitionCount) * 2);
    nodeGeometry->allocate(nodeCount * 6);

    const float w = float(width()), h = float(height());
    const float extent = std::min(w, h) * 0.43f;
    const float cy = std::cos(float(rotation_y_)), sy = std::sin(float(rotation_y_));
    const float cx = std::cos(float(rotation_x_)), sx = std::sin(float(rotation_x_));
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    const float embeddingBlend = previous_temporal_positions_.isEmpty()
        ? 1.0f
        : std::clamp(float(now - embedding_transition_started_ms_) / 900.0f,
                     0.0f, 1.0f);
    float coordinateScale = 1.0f;
    for (const auto& point : temporal_positions_) {
        coordinateScale = std::max(coordinateScale, std::abs(point.x));
        coordinateScale = std::max(coordinateScale, std::abs(point.y));
        coordinateScale = std::max(coordinateScale, std::abs(point.z));
    }

    std::unordered_map<std::uint32_t, float> activity;
    std::unordered_map<std::uint32_t, QVector<const Spike*>> pulseEvents;
    std::unordered_map<std::uint32_t, QVector<std::uint64_t>> spikeSteps;
    std::unordered_map<std::uint32_t, qint64> lastSpikeTime;
    activity.reserve(static_cast<std::size_t>(spikes_.size()));
    pulseEvents.reserve(static_cast<std::size_t>(spikes_.size()));
    spikeSteps.reserve(static_cast<std::size_t>(spikes_.size()));
    lastSpikeTime.reserve(static_cast<std::size_t>(spikes_.size()));
    for (const auto& spike : spikes_) {
        const qint64 age = now - spike.received_ms;
        if (age >= 0 && age < 2600) {
            activity[spike.neuron] += std::exp(-float(age) / 720.0f);
            auto& events = pulseEvents[spike.neuron];
            if (events.size() < 8) events.push_back(&spike);
            spikeSteps[spike.neuron].push_back(spike.step);
            lastSpikeTime[spike.neuron] = spike.received_ms;
        }
    }
    std::unordered_map<std::uint32_t, float> firingRateHz;
    firingRateHz.reserve(spikeSteps.size());
    for (const auto& entry : spikeSteps) {
        const auto& steps = entry.second;
        if (steps.size() < 2 || simulation_steps_per_second_ <= 0.0)
            continue;
        std::uint64_t intervalSum = 0;
        std::uint64_t intervalCount = 0;
        for (qsizetype i = 1; i < steps.size(); ++i) {
            if (steps[i] > steps[i - 1]) {
                intervalSum += steps[i] - steps[i - 1];
                ++intervalCount;
            }
        }
        if (intervalCount != 0 && intervalSum != 0) {
            const double meanInterval =
                double(intervalSum) / double(intervalCount);
            firingRateHz[entry.first] =
                float(simulation_steps_per_second_ / meanInterval);
        }
    }

    QVector<Point> pulseVertices;
    pulseVertices.reserve(std::min(edgeCount * 6, 12000));

    struct Position { float x, y, z, scale, intensity; };
    std::unordered_map<std::uint32_t, Position> positions;
    positions.reserve(static_cast<std::size_t>(nodeCount));
    for (int i = 0; i < nodeCount; ++i) {
        const auto id = nodes_[i];
        float intensity = std::min(2.8f, activity[id]);
        const auto rateIt = firingRateHz.find(id);
        if (rateIt != firingRateHz.end() && intensity > 0.0f) {
            const float frequency = std::clamp(rateIt->second, 0.1f, 80.0f);
            const qint64 lastSpike = lastSpikeTime[id];
            const float phase = float(now - lastSpike) * 0.001f *
                frequency * 6.28318530718f;
            const float strobe = 0.20f + 0.80f *
                (0.5f + 0.5f * std::sin(phase));
            intensity *= strobe;
        }
        RenderPosition point = i < temporal_positions_.size()
            ? temporal_positions_[i] : RenderPosition{};
        if (i < previous_temporal_positions_.size()) {
            const auto& oldPoint = previous_temporal_positions_[i];
            point.x = oldPoint.x + (point.x - oldPoint.x) * embeddingBlend;
            point.y = oldPoint.y + (point.y - oldPoint.y) * embeddingBlend;
            point.z = oldPoint.z + (point.z - oldPoint.z) * embeddingBlend;
        }
        // Activity changes illumination and node size, never the temporal
        // coordinate itself. Structural revisions are the only source of
        // spatial re-embedding.
        const float x0 = point.x / coordinateScale;
        const float y0 = point.y / coordinateScale;
        const float z0 = point.z / coordinateScale;
        const float x1 = x0 * cy + z0 * sy;
        const float z1 = -x0 * sy + z0 * cy;
        const float y1 = y0 * cx - z1 * sx;
        const float z2 = y0 * sx + z1 * cx;
        const float perspective = float(camera_distance_) /
            std::max(0.35f, float(camera_distance_) - z2);
        positions.emplace(id, Position{
            w * 0.5f + x1 * extent * perspective,
            h * 0.5f + y1 * extent * perspective,
            z2, perspective, intensity});
    }
    if (embeddingBlend >= 1.0f)
        previous_temporal_positions_.clear();

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

        const auto sourcePulses = pulseEvents.find(edge.source);
        if (sourcePulses != pulseEvents.end()) {
            int emitted = 0;
            for (const Spike* spike : sourcePulses->second) {
                if (simulation_step_ < spike->step) continue;
                const std::uint64_t ageSteps = simulation_step_ - spike->step;
                if (ageSteps >= edge.delay) continue;
                const float t = float(ageSteps) / float(edge.delay);
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
    for (int i = 0; i < transitionCount; ++i) {
        const auto& transition = edge_transitions_[i];
        const auto fromIt = positions.find(transition.edge.source);
        const auto toIt = positions.find(transition.edge.target);
        if (fromIt == positions.end() || toIt == positions.end()) continue;
        const float age = float(now - transition.created_ms);
        const float progress = std::clamp(age / 1800.0f, 0.0f, 1.0f);
        const uchar alpha = static_cast<uchar>(
            220.0f * (transition.created ? progress : (1.0f - progress)));
        const uchar r = transition.created ? 72 : 255;
        const uchar g = transition.created ? 255 : 72;
        const uchar b = transition.created ? 210 : 92;
        const int vertex = (edgeCount + i) * 2;
        put(edgeVertices[vertex], fromIt->second.x, fromIt->second.y,
            r, g, b, alpha);
        put(edgeVertices[vertex+1], toIt->second.x, toIt->second.y,
            r, g, b, alpha);
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
    pulseGeometry->markVertexDataDirty();
    root->pulses->markDirty(QSGNode::DirtyGeometry);
    return root;
}
