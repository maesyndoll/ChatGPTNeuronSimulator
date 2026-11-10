#pragma once

#include "qt/simulation_client.h"
#include "runtime/graft_package.h"

#include <QObject>
#include <QJsonObject>
#include <QString>
#include <QVariantMap>
#include <QTimer>

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
    Q_PROPERTY(QVariantList telemetryHistory READ telemetryHistory NOTIFY telemetryChanged)

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

    Q_PROPERTY(QVariantMap connectivityStatistics READ connectivityStatistics
               NOTIFY connectivityChanged)
    Q_PROPERTY(QVariantMap topologyVisual READ topologyVisual
               NOTIFY topologyVisualChanged)
    Q_PROPERTY(QVariantMap graftInspection READ graftInspection
               NOTIFY graftInspectionChanged)
    Q_PROPERTY(bool graftInspectionValid READ graftInspectionValid
               NOTIFY graftInspectionChanged)

    Q_PROPERTY(QVariantList spikeEvents READ spikeEvents
               NOTIFY spikeEventsChanged)
    Q_PROPERTY(QString spikeEventCount READ spikeEventCount
               NOTIFY spikeEventsChanged)
    Q_PROPERTY(QString spikeEventsDropped READ spikeEventsDropped
               NOTIFY spikeEventsChanged)
    Q_PROPERTY(QString spikeEventsFirstSequence READ spikeEventsFirstSequence
               NOTIFY spikeEventsChanged)
    Q_PROPERTY(QString spikeEventsGeneration READ spikeEventsGeneration
               NOTIFY spikeEventsChanged)

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
    QVariantList telemetryHistory() const;

    QVariantMap requestedParameters() const;
    QVariantMap activeParameters() const;
    bool hasPendingReinitialization() const;
    QString parameterVersion() const;
    QString simulationGeneration() const;

    QVariantMap connectivityStatistics() const;
    QVariantMap topologyVisual() const;
    QVariantMap selfObservation() const;
    QVariantMap graftInspection() const;
    QVariantMap topologyProposal() const;
    bool graftInspectionValid() const;
    QVariantList spikeEvents() const;
    QString spikeEventCount() const;
    QString spikeEventsDropped() const;
    QString spikeEventsFirstSequence() const;
    QString spikeEventsGeneration() const;

    Q_INVOKABLE void connectToServer(const QString& socketPath = {});
    Q_INVOKABLE void disconnectFromServer();
    Q_INVOKABLE void startSimulation();
    Q_INVOKABLE void pauseSimulation();
    Q_INVOKABLE void stepSimulation(int steps = 1);
    Q_INVOKABLE void refreshConfiguration();
    Q_INVOKABLE void applyParameters(const QVariantMap& parameters);
    Q_INVOKABLE void reinitializeSimulation();
    Q_INVOKABLE void refreshSpikeEvents();
    Q_INVOKABLE void refreshConnectivity();
    Q_INVOKABLE void refreshTopologyVisual();
    Q_INVOKABLE void refreshSelfObservation();
    Q_INVOKABLE void refreshTopologyProposal();
    Q_INVOKABLE void inspectGraftPackage(const QString& packagePath);
    Q_INVOKABLE void attachGraftPackage(const QString& packagePath, bool plasticityEnabled = false);
    Q_INVOKABLE void detachGraftPackage();
    Q_INVOKABLE void refreshGraftStatus();
    Q_INVOKABLE void sendVisualGrid(const QVariantList& intensities);
    Q_INVOKABLE void sendStimulusPulse(int neuronId, double amplitude, int durationSteps);

signals:
    void connectionChanged();
    void simulationStatusChanged();
    void statusMessageChanged();
    void errorMessageChanged();
    void telemetryChanged();
    void configurationChanged();
    void spikeEventsChanged();
    void connectivityChanged();
    void topologyVisualChanged();
    void selfObservationChanged();
    void topologyProposalChanged();
    void graftInspectionChanged();
    void errorOccurred(const QString& message);

private:
    void setStatusMessage(const QString& message);
    void setErrorMessage(const QString& message);
    void updateTelemetry(const QJsonObject& telemetry);
    void updateConfiguration(const QJsonObject& configuration);
    void updateSpikeEvents(const QJsonObject& spikeEvents);

    SimulationClient client_;
    bool connected_ = false;
    QString connection_status_ = QStringLiteral("Disconnected");
    QString simulation_status_ = QStringLiteral("Unknown");
    QString status_message_ = QStringLiteral("Waiting for connection");
    QString error_message_;
    QJsonObject telemetry_;
    QVariantList telemetry_history_;
    quint64 last_history_step_ = 0;
    bool has_history_step_ = false;
    QJsonObject configuration_;
    QVariantMap connectivity_statistics_;
    QVariantMap topology_visual_;
    QVariantMap self_observation_;
    QVariantMap topology_proposal_;
    bool topology_proposal_request_pending_ = false;
    QVariantMap graft_inspection_;
    bool graft_inspection_valid_ = false;
    QTimer topology_visual_timer_;
    QTimer self_observation_timer_;
    bool topology_visual_request_pending_ = false;
    bool self_observation_request_pending_ = false;
    QVariantList spike_events_;
    QString spike_event_count_ = QStringLiteral("0");
    QString spike_events_dropped_ = QStringLiteral("0");
    QString spike_events_first_sequence_ = QStringLiteral("0");
    QString spike_events_generation_ = QStringLiteral("—");
    quint64 last_spike_event_sequence_ = 0;
    QTimer spike_events_update_timer_;
};
