#pragma once

#include "qt/simulation_client.h"

#include <QObject>
#include <QJsonObject>
#include <QString>
#include <QVariantMap>

class SimulationBridge : public QObject {
    Q_OBJECT

    Q_PROPERTY(bool connected READ connected NOTIFY connectionChanged)
    Q_PROPERTY(QString connectionStatus READ connectionStatus NOTIFY connectionChanged)
    Q_PROPERTY(QString simulationStatus READ simulationStatus NOTIFY simulationStatusChanged)
    Q_PROPERTY(QString statusMessage READ statusMessage NOTIFY statusMessageChanged)
    Q_PROPERTY(QString errorMessage READ errorMessage NOTIFY errorMessageChanged)

    Q_PROPERTY(QString completedSteps READ completedSteps NOTIFY telemetryChanged)
    Q_PROPERTY(QString excitatorySpikes READ excitatorySpikes NOTIFY telemetryChanged)
    Q_PROPERTY(QString inhibitorySpikes READ inhibitorySpikes NOTIFY telemetryChanged)
    Q_PROPERTY(double excitatoryRateHz READ excitatoryRateHz NOTIFY telemetryChanged)
    Q_PROPERTY(double inhibitoryRateHz READ inhibitoryRateHz NOTIFY telemetryChanged)
    Q_PROPERTY(double meanVoltageMv READ meanVoltageMv NOTIFY telemetryChanged)
    Q_PROPERTY(bool finiteState READ finiteState NOTIFY telemetryChanged)

    Q_PROPERTY(QVariantMap requestedParameters READ requestedParameters
               NOTIFY configurationChanged)
    Q_PROPERTY(QVariantMap activeParameters READ activeParameters
               NOTIFY configurationChanged)
    Q_PROPERTY(bool hasPendingReinitialization READ hasPendingReinitialization
               NOTIFY configurationChanged)
    Q_PROPERTY(QString parameterVersion READ parameterVersion
               NOTIFY configurationChanged)
    Q_PROPERTY(QString simulationGeneration READ simulationGeneration
               NOTIFY configurationChanged)

    Q_PROPERTY(QVariantList spikeEvents READ spikeEvents
               NOTIFY spikeEventsChanged)
    Q_PROPERTY(QVariantList visualSpikeEvents READ visualSpikeEvents
               NOTIFY visualSpikeEventsChanged)
    Q_PROPERTY(QString spikeEventCount READ spikeEventCount
               NOTIFY spikeEventsChanged)
    Q_PROPERTY(QString spikeEventsDropped READ spikeEventsDropped
               NOTIFY spikeEventsChanged)
    Q_PROPERTY(QString spikeEventsFirstSequence READ spikeEventsFirstSequence
               NOTIFY spikeEventsChanged)
    Q_PROPERTY(QString spikeEventsGeneration READ spikeEventsGeneration
               NOTIFY spikeEventsChanged)

    Q_PROPERTY(QVariantMap tissueSnapshot READ tissueSnapshot NOTIFY tissueChanged)
    Q_PROPERTY(QVariantMap topologySnapshot READ topologySnapshot NOTIFY topologyChanged)
    Q_PROPERTY(QString tissueRevision READ tissueRevision NOTIFY tissueChanged)

public:
    explicit SimulationBridge(QObject* parent = nullptr);

    bool connected() const;
    QString connectionStatus() const;
    QString simulationStatus() const;
    QString statusMessage() const;
    QString errorMessage() const;

    QString completedSteps() const;
    QString excitatorySpikes() const;
    QString inhibitorySpikes() const;
    double excitatoryRateHz() const;
    double inhibitoryRateHz() const;
    double meanVoltageMv() const;
    bool finiteState() const;

    QVariantMap requestedParameters() const;
    QVariantMap activeParameters() const;
    bool hasPendingReinitialization() const;
    QString parameterVersion() const;
    QString simulationGeneration() const;

    QVariantList spikeEvents() const;\n    QVariantList visualSpikeEvents() const;
    QString spikeEventCount() const;
    QString spikeEventsDropped() const;
    QString spikeEventsFirstSequence() const;
    QString spikeEventsGeneration() const;

    QVariantMap tissueSnapshot() const;
    QVariantMap topologySnapshot() const;
    QString tissueRevision() const;

    Q_INVOKABLE void connectToServer(const QString& socketPath = {});
    Q_INVOKABLE void disconnectFromServer();
    Q_INVOKABLE void startSimulation();
    Q_INVOKABLE void pauseSimulation();
    Q_INVOKABLE void stepSimulation(int steps = 1);
    Q_INVOKABLE void stimulateNeurons(const QVariantList& neuronIds,
                                      double amplitude, int durationSteps);
    Q_INVOKABLE void refreshConfiguration();
    Q_INVOKABLE void applyParameters(const QVariantMap& parameters);
    Q_INVOKABLE void reinitializeSimulation();
    Q_INVOKABLE void refreshSpikeEvents();
    Q_INVOKABLE void refreshTissue();
    Q_INVOKABLE void refreshTopology();

signals:
    void connectionChanged();
    void simulationStatusChanged();
    void statusMessageChanged();
    void errorMessageChanged();
    void telemetryChanged();
    void configurationChanged();
    void spikeEventsChanged();\n    void visualSpikeEventsChanged();
    void tissueChanged();
    void topologyChanged();
    void errorOccurred(const QString& message);

private:
    void setStatusMessage(const QString& message);
    void setErrorMessage(const QString& message);
    void updateTelemetry(const QJsonObject& telemetry);
    void updateConfiguration(const QJsonObject& configuration);
    void updateSpikeEvents(const QJsonObject& spikeEvents);
    void updateTissue(const QJsonObject& tissue);
    void updateTopology(const QJsonObject& topology);

    SimulationClient client_;
    bool connected_ = false;
    QString connection_status_ = QStringLiteral("Disconnected");
    QString simulation_status_ = QStringLiteral("Unknown");
    QString status_message_ = QStringLiteral("Waiting for connection");
    QString error_message_;
    QJsonObject telemetry_;
    QJsonObject configuration_;
    QVariantList spike_events_;\n    QVariantList visual_spike_events_;
    QString spike_event_count_ = QStringLiteral("0");
    QString spike_events_dropped_ = QStringLiteral("0");
    QString spike_events_first_sequence_ = QStringLiteral("0");
    QString spike_events_generation_ = QStringLiteral("—");
    quint64 last_spike_event_sequence_ = 0;
    QJsonObject tissue_snapshot_;
    QJsonObject topology_snapshot_;
};
