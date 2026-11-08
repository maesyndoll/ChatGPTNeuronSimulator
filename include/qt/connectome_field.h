#pragma once

#include <QMutex>
#include <QQuickItem>
#include <QVariantList>
#include <QVariantMap>
#include <QVector>

#include <cstdint>

class ConnectomeField final : public QQuickItem {
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
    struct Spike {
        std::uint32_t neuron = 0;
        qint64 received_ms = 0;
    };

    QVariantMap topology_;
    QVariantList spike_events_;
    QVector<std::uint32_t> nodes_;
    QVector<Edge> edges_;
    QVector<Spike> spikes_;
    qreal rotation_x_ = -0.18;
    qreal rotation_y_ = 0.35;
    qreal camera_distance_ = 3.15;
    qint64 last_sequence_ = 0;
    QString generation_;
    mutable QMutex mutex_;
};
