#pragma once

#include <QFutureWatcher>
#include <QMutex>
#include <QQuickItem>
#include <QVariantList>
#include <QVariantMap>
#include <QVector>

#include "connectome/temporal_embedding.h"

#include <cstdint>

class ConnectomeField : public QQuickItem {
    Q_OBJECT
    Q_PROPERTY(QVariantMap topology READ topology WRITE setTopology NOTIFY topologyChanged)
    Q_PROPERTY(QString simulationGeneration READ simulationGeneration WRITE setSimulationGeneration NOTIFY spikeEventsChanged)
    Q_PROPERTY(QVariantList spikeEvents READ spikeEvents WRITE setSpikeEvents NOTIFY spikeEventsChanged)
    Q_PROPERTY(qreal rotationX READ rotationX WRITE setRotationX NOTIFY cameraChanged)
    Q_PROPERTY(qreal rotationY READ rotationY WRITE setRotationY NOTIFY cameraChanged)
    Q_PROPERTY(qreal cameraDistance READ cameraDistance WRITE setCameraDistance NOTIFY cameraChanged)

public:
    explicit ConnectomeField(QQuickItem* parent = nullptr);

    QVariantMap topology() const { return topology_; }
    void setTopology(const QVariantMap& value);
    QString simulationGeneration() const { return generation_; }
    void setSimulationGeneration(const QString& value);
    QVariantList spikeEvents() const { return spike_events_; }
    void setSpikeEvents(const QVariantList& value);
    qreal rotationX() const { return rotation_x_; }
    void setRotationX(qreal value);
    qreal rotationY() const { return rotation_y_; }
    void setRotationY(qreal value);
    qreal cameraDistance() const { return camera_distance_; }
    void setCameraDistance(qreal value);

    Q_INVOKABLE void advanceFrame();

signals:
    void topologyChanged();
    void spikeEventsChanged();
    void cameraChanged();

protected:
    QSGNode* updatePaintNode(QSGNode* oldNode, UpdatePaintNodeData*) override;

private:
    struct Edge {
        std::uint32_t source = 0;
        std::uint32_t target = 0;
        std::uint32_t delay = 1;
        bool excitatory = true;
    };
    struct RenderPosition { float x = 0.0f, y = 0.0f, z = 0.0f; };
    struct EdgeTransition {
        Edge edge;
        qint64 created_ms = 0;
        bool created = true;
    };
    struct Spike {
        std::uint32_t neuron = 0;
        qint64 received_ms = 0;
    };

    QVariantMap topology_;
    QVariantList spike_events_;
    QVector<std::uint32_t> nodes_;
    QVector<Edge> edges_;
    QVector<EdgeTransition> edge_transitions_;
    QVector<RenderPosition> temporal_positions_;
    QVector<RenderPosition> previous_temporal_positions_;
    QVector<std::uint32_t> embedded_node_ids_;
    QFutureWatcher<connectome::EmbeddingResult> embedding_watcher_;
    quint64 embedding_request_generation_ = 0;
    quint64 embedding_watcher_generation_ = 0;
    qint64 embedding_transition_started_ms_ = 0;
    QVector<Spike> spikes_;
    qreal rotation_x_ = -0.18;
    qreal rotation_y_ = 0.35;
    qreal camera_distance_ = 3.15;
    qint64 last_sequence_ = 0;
    QString generation_;
    mutable QMutex mutex_;
};
