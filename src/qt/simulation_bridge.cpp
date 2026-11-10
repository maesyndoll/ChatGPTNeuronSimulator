#include "qt/simulation_bridge.h"

#include <QJsonValue>
#include <QFileInfo>
#include <exception>
#include <QJsonObject>
#include <QJsonArray>
#include <QStringList>
#include <QVariant>
#include <cmath>
#include <limits>
#include <filesystem>

#include <unistd.h>

SimulationBridge::SimulationBridge(QObject* parent)
    : QObject(parent) {
    // Keep high-rate spike traffic from forcing a full QML raster redraw for
    // every IPC batch. The latest event data is retained; notifications are
    // coalesced to at most 20 updates per second.
    spike_events_update_timer_.setSingleShot(true);
    spike_events_update_timer_.setInterval(50);
    connect(&spike_events_update_timer_, &QTimer::timeout,
            this, &SimulationBridge::spikeEventsChanged);
    topology_visual_timer_.setInterval(5000);
    self_observation_timer_.setInterval(5000);
    connect(&self_observation_timer_, &QTimer::timeout,
            this, &SimulationBridge::refreshSelfObservation);
    connect(&topology_visual_timer_, &QTimer::timeout,
            this, &SimulationBridge::refreshTopologyVisual);
    connect(&client_, &SimulationClient::connected, this, [this]() {
        connected_ = true;
        connection_status_ = QStringLiteral("Connected");
        emit connectionChanged();

        setErrorMessage({});
        setStatusMessage(
            QStringLiteral("Connected; subscribing to telemetry and spike events"));
        client_.sendRequest(QStringLiteral("subscribe_telemetry"));
        client_.sendRequest(QStringLiteral("subscribe_spike_events"));
        refreshConfiguration();
        refreshTopologyVisual();
        refreshSelfObservation();
        refreshGraftStatus();
        topology_visual_timer_.start();
        // Self-observation is on-demand: a full graph snapshot can be expensive.
        // The server's spike subscription starts with a recent bounded
        // snapshot; do not separately request the oldest retained events.
    });

    connect(&client_, &SimulationClient::disconnected, this, [this]() {
        connected_ = false;
        connection_status_ = QStringLiteral("Disconnected");
        emit connectionChanged();
        spike_events_update_timer_.stop();
        topology_visual_timer_.stop();
        self_observation_timer_.stop();
        topology_visual_request_pending_ = false;
        self_observation_request_pending_ = false;
        topology_visual_.clear();
        emit topologyVisualChanged();
        self_observation_.clear();
        emit selfObservationChanged();
        topology_proposal_.clear();
        topology_proposal_request_pending_ = false;
        emit topologyProposalChanged();
        spike_events_.clear();
        spike_event_count_ = QStringLiteral("0");
        spike_events_dropped_ = QStringLiteral("0");
        spike_events_first_sequence_ = QStringLiteral("0");
        spike_events_generation_ = QStringLiteral("—");
        last_spike_event_sequence_ = 0;
        emit spikeEventsChanged();
        connectivity_statistics_.clear();
        emit connectivityChanged();
        setStatusMessage(QStringLiteral("Disconnected from simulation server"));
    });

    connect(&client_, &SimulationClient::transportError,
            this, [this](const QString& message) {
        setErrorMessage(message);
        emit errorOccurred(message);
    });

    connect(&client_, &SimulationClient::telemetryReceived,
            this, [this](const QJsonObject& telemetry) {
        updateTelemetry(telemetry);
    });

    connect(&client_, &SimulationClient::spikeEventsReceived,
            this, [this](const QJsonObject& spikeEvents) {
        updateSpikeEvents(spikeEvents);
    });

    connect(&client_, &SimulationClient::requestSucceeded,
            this,
            [this](const QString&, const QString& command,
                   const QJsonObject& result) {
        if (command == QStringLiteral("get_parameters")) {
            updateConfiguration(result);
        } else if (command == QStringLiteral("get_spike_events")) {
            updateSpikeEvents(result);
        } else if (command == QStringLiteral("get_connectivity")) {
            connectivity_statistics_ = result.toVariantMap();
            emit connectivityChanged();
        } else if (command == QStringLiteral("get_topology_visual")) {
            topology_visual_request_pending_ = false;
            topology_visual_ = result.toVariantMap();
            emit topologyVisualChanged();
        } else if (command == QStringLiteral("get_self_observation")) {
            self_observation_request_pending_ = false;
            self_observation_ = result.toVariantMap();
            emit selfObservationChanged();
        } else if (command == QStringLiteral("get_topology_proposal")) {
            topology_proposal_request_pending_ = false;
            topology_proposal_ = result.toVariantMap();
            emit topologyProposalChanged();
        } else if (command == QStringLiteral("get_graft_status")) {
            const bool attached = result.value(QStringLiteral("attached")).toBool();
            graft_inspection_.insert(QStringLiteral("attached"), attached);
            if (attached) {
                graft_inspection_.insert(QStringLiteral("lifecycle_state"), QStringLiteral("attached"));
                graft_inspection_.insert(QStringLiteral("status"), QStringLiteral("Attached · live topology"));
            } else if (graft_inspection_valid_) {
                const QString previous = graft_inspection_.value(QStringLiteral("lifecycle_state")).toString();
                if (previous != QStringLiteral("ejected")) {
                    graft_inspection_.insert(QStringLiteral("lifecycle_state"), QStringLiteral("validated"));
                    graft_inspection_.insert(QStringLiteral("status"), QStringLiteral("Validated · not attached"));
                }
            }
            emit graftInspectionChanged();
        } else {
            const QJsonValue configurationValue =
                result.value(QStringLiteral("configuration"));
            if (configurationValue.isObject()) {
                updateConfiguration(configurationValue.toObject());
            }
        }

        if (command == QStringLiteral("attach_graft") ||
            command == QStringLiteral("detach_graft")) {
            refreshConfiguration();
            refreshTopologyVisual();
            refreshConnectivity();
            refreshSelfObservation();
            refreshGraftStatus();
            if (command == QStringLiteral("attach_graft")) {
                graft_inspection_.insert(QStringLiteral("attached"), true);
                graft_inspection_.insert(QStringLiteral("lifecycle_state"), QStringLiteral("attached"));
                graft_inspection_.insert(QStringLiteral("status"), QStringLiteral("Attached · live topology"));
            } else {
                graft_inspection_.insert(QStringLiteral("attached"), false);
                graft_inspection_.insert(QStringLiteral("lifecycle_state"), QStringLiteral("ejected"));
                graft_inspection_.insert(QStringLiteral("status"), QStringLiteral("Ejected · original topology restored"));
            }
            emit graftInspectionChanged();
        }

        if (result.contains(QStringLiteral("status"))) {
            const QString nextStatus =
                result.value(QStringLiteral("status")).toString();
            if (simulation_status_ != nextStatus) {
                simulation_status_ = nextStatus;
                emit simulationStatusChanged();
            }
        }

        const QJsonValue telemetryValue =
            result.value(QStringLiteral("telemetry"));
        if (telemetryValue.isObject()) {
            updateTelemetry(telemetryValue.toObject());
        }

        if (command == QStringLiteral("apply_parameters")) {
            const bool accepted =
                result.value(QStringLiteral("accepted")).toBool(false);

            if (!accepted) {
                QStringList errors;
                const QJsonArray errorArray =
                    result.value(QStringLiteral("errors")).toArray();

                for (const QJsonValue& error : errorArray) {
                    if (error.isObject()) {
                        const QJsonObject detail = error.toObject();
                        const QString field =
                            detail.value(QStringLiteral("field")).toString();
                        const QString message =
                            detail.value(QStringLiteral("message")).toString();

                        errors.append(
                            field.isEmpty()
                                ? message
                                : QStringLiteral("%1: %2").arg(field, message));
                    } else {
                        errors.append(error.toString());
                    }
                }

                const QString message = errors.isEmpty()
                    ? QStringLiteral("The server rejected the parameter update")
                    : errors.join(QStringLiteral("; "));

                setErrorMessage(message);
                setStatusMessage(QStringLiteral("Parameter update rejected"));
                emit errorOccurred(message);
                return;
            }

            setErrorMessage({});

            const bool requiresReinitialization =
                result.value(QStringLiteral("requires_reinitialization"))
                    .toBool(false);

            setStatusMessage(
                requiresReinitialization
                    ? QStringLiteral(
                          "Parameters accepted; reinitialization required")
                    : QStringLiteral(
                          "Parameters accepted; applying at a step boundary"));

            refreshConfiguration();
            return;
        }

        setErrorMessage({});
        setStatusMessage(QStringLiteral("Completed: %1").arg(command));
    });

    connect(&client_, &SimulationClient::requestFailed,
            this,
            [this](const QString&, const QString& command,
                   const QString& errorCode, const QString& message) {
        if (command == QStringLiteral("get_topology_visual"))
            topology_visual_request_pending_ = false;
        if (command == QStringLiteral("get_self_observation"))
            self_observation_request_pending_ = false;
        if (command == QStringLiteral("get_topology_proposal"))
            topology_proposal_request_pending_ = false;
        const QString combined =
            QStringLiteral("%1 (%2): %3").arg(command, errorCode, message);
        if (command == QStringLiteral("attach_graft") ||
            command == QStringLiteral("detach_graft")) {
            const bool stillAttached =
                graft_inspection_.value(QStringLiteral("attached")).toBool();
            graft_inspection_.insert(
                QStringLiteral("lifecycle_state"),
                stillAttached ? QStringLiteral("attached") : QStringLiteral("validated"));
            graft_inspection_.insert(
                QStringLiteral("status"),
                QStringLiteral("%1 failed · %2")
                    .arg(command == QStringLiteral("attach_graft")
                             ? QStringLiteral("Attach") : QStringLiteral("Ejection"),
                         message));
            emit graftInspectionChanged();
            refreshGraftStatus();
        }
        setErrorMessage(combined);
        setStatusMessage(QStringLiteral("Command failed: %1").arg(command));
        emit errorOccurred(combined);
    });
}

bool SimulationBridge::connected() const {
    return connected_;
}

QString SimulationBridge::connectionStatus() const {
    return connection_status_;
}

QString SimulationBridge::simulationStatus() const {
    return simulation_status_;
}

QString SimulationBridge::statusMessage() const {
    return status_message_;
}

QString SimulationBridge::errorMessage() const {
    return error_message_;
}

QString SimulationBridge::completedSteps() const {
    return telemetry_.value(QStringLiteral("completed_steps")).toString(
        QStringLiteral("—"));
}

QString SimulationBridge::excitatorySpikes() const {
    return telemetry_.value(QStringLiteral("excitatory_spikes")).toString(
        QStringLiteral("—"));
}

QString SimulationBridge::inhibitorySpikes() const {
    return telemetry_.value(QStringLiteral("inhibitory_spikes")).toString(
        QStringLiteral("—"));
}

double SimulationBridge::excitatoryRateHz() const {
    return telemetry_.value(QStringLiteral("excitatory_rate_hz")).toDouble();
}

double SimulationBridge::inhibitoryRateHz() const {
    return telemetry_.value(QStringLiteral("inhibitory_rate_hz")).toDouble();
}

double SimulationBridge::meanVoltageMv() const {
    return telemetry_.value(QStringLiteral("mean_voltage_mv")).toDouble();
}

bool SimulationBridge::finiteState() const {
    return telemetry_.value(QStringLiteral("finite_state")).toBool(false);
}

QVariantList SimulationBridge::telemetryHistory() const {
    return telemetry_history_;
}

QVariantMap SimulationBridge::requestedParameters() const {
    return configuration_.value(QStringLiteral("requested"))
        .toObject().toVariantMap();
}

QVariantMap SimulationBridge::activeParameters() const {
    return configuration_.value(QStringLiteral("active"))
        .toObject().toVariantMap();
}

bool SimulationBridge::hasPendingReinitialization() const {
    return configuration_
        .value(QStringLiteral("has_pending_reinitialization")).toBool(false);
}

QString SimulationBridge::parameterVersion() const {
    return configuration_
        .value(QStringLiteral("parameter_version")).toString(QStringLiteral("—"));
}

QString SimulationBridge::simulationGeneration() const {
    return configuration_
        .value(QStringLiteral("simulation_generation")).toString(
            QStringLiteral("—"));
}

QVariantMap SimulationBridge::connectivityStatistics() const {
    return connectivity_statistics_;
}

QVariantMap SimulationBridge::topologyVisual() const {
    return topology_visual_;
}

QVariantMap SimulationBridge::selfObservation() const {
    return self_observation_;
}

QVariantMap SimulationBridge::graftInspection() const {
    return graft_inspection_;
}

QVariantMap SimulationBridge::topologyProposal() const {
    return topology_proposal_;
}

bool SimulationBridge::graftInspectionValid() const {
    return graft_inspection_valid_;
}

QVariantList SimulationBridge::spikeEvents() const {
    return spike_events_;
}

QString SimulationBridge::spikeEventCount() const {
    return spike_event_count_;
}

QString SimulationBridge::spikeEventsDropped() const {
    return spike_events_dropped_;
}

QString SimulationBridge::spikeEventsFirstSequence() const {
    return spike_events_first_sequence_;
}

QString SimulationBridge::spikeEventsGeneration() const {
    return spike_events_generation_;
}

void SimulationBridge::connectToServer(const QString& socketPath) {
    if (connected_ || connection_status_ == QStringLiteral("Connecting...")) {
        return;
    }

    QString path = socketPath;
    if (path.isEmpty()) {
        path = QStringLiteral("/tmp/spiking_model-%1.sock")
                   .arg(static_cast<qulonglong>(::getuid()));
    }

    connection_status_ = QStringLiteral("Connecting...");
    emit connectionChanged();
    setStatusMessage(QStringLiteral("Connecting to %1").arg(path));
    setErrorMessage({});
    client_.connectToServer(path);
}

void SimulationBridge::disconnectFromServer() {
    client_.disconnectFromServer();
}

void SimulationBridge::startSimulation() {
    if (connected_) {
        setErrorMessage({});
        client_.sendRequest(QStringLiteral("start"));
    }
}

void SimulationBridge::pauseSimulation() {
    if (connected_) {
        setErrorMessage({});
        client_.sendRequest(QStringLiteral("pause"));
    }
}

void SimulationBridge::stepSimulation(int steps) {
    if (!connected_ || steps <= 0) {
        return;
    }

    setErrorMessage({});
    client_.sendRequest(
        QStringLiteral("step"),
        QJsonObject{{QStringLiteral("steps"), steps}}
    );
}

void SimulationBridge::refreshConfiguration() {
    if (connected_) {
        client_.sendRequest(QStringLiteral("get_parameters"));
    }
}

void SimulationBridge::applyParameters(const QVariantMap& parameters) {
    if (!connected_) {
        setErrorMessage(QStringLiteral("Cannot apply parameters: not connected"));
        return;
    }

    const QJsonObject parameterObject =
        QJsonObject::fromVariantMap(parameters);

    client_.sendRequest(
        QStringLiteral("apply_parameters"),
        QJsonObject{
            {QStringLiteral("parameters"), parameterObject}
        });
}

void SimulationBridge::refreshConnectivity() {
    if (connected_) {
        client_.sendRequest(QStringLiteral("get_connectivity"));
    }
}

void SimulationBridge::refreshTopologyVisual() {
    if (!connected_ || topology_visual_request_pending_)
        return;
    topology_visual_request_pending_ = true;
    client_.sendRequest(QStringLiteral("get_topology_visual"), {}, 10000);
}

void SimulationBridge::refreshSelfObservation() {
    if (!connected_ || self_observation_request_pending_)
        return;
    self_observation_request_pending_ = true;
    client_.sendRequest(QStringLiteral("get_self_observation"), {}, 30000);
}

void SimulationBridge::refreshTopologyProposal() {
    if (!connected_ || topology_proposal_request_pending_)
        return;
    topology_proposal_request_pending_ = true;
    client_.sendRequest(QStringLiteral("get_topology_proposal"), {}, 15000);
}

void SimulationBridge::inspectGraftPackage(const QString& packagePath) {
    graft_inspection_.clear();
    graft_inspection_valid_ = false;
    const QString path = packagePath.trimmed();
    if (path.isEmpty()) {
        graft_inspection_.insert(QStringLiteral("error"),
                                 QStringLiteral("Select a graft package directory or manifest."));
        emit graftInspectionChanged();
        return;
    }
    try {
        const auto info = spiking::GraftPackageInspector::inspect(
            std::filesystem::path(path.toStdString()));
        graft_inspection_.insert(QStringLiteral("id"), QString::fromStdString(info.id));
        graft_inspection_.insert(QStringLiteral("name"), QString::fromStdString(info.name));
        graft_inspection_.insert(QStringLiteral("description"),
                                 QString::fromStdString(info.description));
        graft_inspection_.insert(QStringLiteral("package_directory"),
                                 QString::fromStdString(info.package_directory.string()));
        graft_inspection_.insert(QStringLiteral("input_path"),
                                 QFileInfo(path).absoluteFilePath());
        graft_inspection_.insert(QStringLiteral("excitatory_count"),
                                 static_cast<qulonglong>(info.excitatory_count));
        graft_inspection_.insert(QStringLiteral("inhibitory_count"),
                                 static_cast<qulonglong>(info.inhibitory_count));
        graft_inspection_.insert(QStringLiteral("synapse_count"),
                                 static_cast<qulonglong>(info.synapse_count));
        graft_inspection_.insert(QStringLiteral("module_binary"),
                                 info.has_module_binary
                                     ? QString::fromStdString(info.module_binary_path.string())
                                     : QStringLiteral("No executable module"));
        graft_inspection_.insert(QStringLiteral("input_adapters"),
                                 static_cast<int>(info.input_adapters.size()));
        graft_inspection_.insert(QStringLiteral("output_adapters"),
                                 static_cast<int>(info.output_adapters.size()));
        graft_inspection_.insert(QStringLiteral("attached"), false);
        graft_inspection_.insert(QStringLiteral("lifecycle_state"), QStringLiteral("validated"));
        graft_inspection_.insert(QStringLiteral("status"),
                                 QStringLiteral("Validated · not attached"));
        graft_inspection_valid_ = true;
    } catch (const std::exception& error) {
        graft_inspection_.insert(QStringLiteral("error"), QString::fromUtf8(error.what()));
    }
    emit graftInspectionChanged();
    refreshGraftStatus();
}


void SimulationBridge::attachGraftPackage(
    const QString& packagePath, bool plasticityEnabled) {
    if (!connected_) {
        setErrorMessage(QStringLiteral("Cannot attach graft: backend disconnected"));
        return;
    }
    if (!graft_inspection_valid_ ||
        QFileInfo(packagePath).absoluteFilePath() !=
            QFileInfo(graft_inspection_.value(QStringLiteral("input_path")).toString())
                .absoluteFilePath()) {
        setErrorMessage(QStringLiteral("Inspect this package before attaching it"));
        return;
    }
    graft_inspection_.insert(QStringLiteral("lifecycle_state"), QStringLiteral("attaching"));
    graft_inspection_.insert(QStringLiteral("status"), QStringLiteral("Attaching · waiting for simulation safe point"));
    emit graftInspectionChanged();
    setStatusMessage(QStringLiteral("Attaching graft at the next simulation safe point..."));
    client_.sendRequest(QStringLiteral("attach_graft"), QJsonObject{
        {QStringLiteral("package_path"), packagePath},
        {QStringLiteral("plasticity_enabled"), plasticityEnabled}
    }, 30000);
}

void SimulationBridge::detachGraftPackage() {
    if (!connected_) {
        setErrorMessage(QStringLiteral("Cannot detach graft: backend disconnected"));
        return;
    }
    graft_inspection_.insert(QStringLiteral("lifecycle_state"), QStringLiteral("ejecting"));
    graft_inspection_.insert(QStringLiteral("status"), QStringLiteral("Ejecting · restoring original topology"));
    emit graftInspectionChanged();
    setStatusMessage(QStringLiteral("Detaching graft at the next simulation safe point..."));
    client_.sendRequest(QStringLiteral("detach_graft"), {}, 30000);
}

void SimulationBridge::refreshGraftStatus() {
    if (connected_) client_.sendRequest(QStringLiteral("get_graft_status"));
}

void SimulationBridge::refreshSpikeEvents() {
    if (connected_) {
        client_.sendRequest(
            QStringLiteral("get_spike_events"),
            QJsonObject{{QStringLiteral("after_sequence"), QStringLiteral("0")}}
        );
    }
}

void SimulationBridge::sendVisualGrid(const QVariantList& intensities) {
    if (!connected_) {
        setErrorMessage(QStringLiteral("Cannot send visual input: backend disconnected"));
        return;
    }
    if (intensities.size() != 64) {
        setErrorMessage(QStringLiteral("Visual input must contain exactly 64 cells"));
        return;
    }

    bool stepOk = false;
    const quint64 completed = telemetry_.value(QStringLiteral("completed_steps"))
        .toString().toULongLong(&stepOk);
    if (!stepOk) {
        setErrorMessage(QStringLiteral("Waiting for a valid simulation step counter"));
        return;
    }

    QJsonArray stimuli;
    int activeCells = 0;
    for (int i = 0; i < 64; ++i) {
        bool valueOk = false;
        const double intensity = intensities.at(i).toDouble(&valueOk);
        if (!valueOk || !std::isfinite(intensity) || intensity < 0.0 || intensity > 1.0) {
            setErrorMessage(QStringLiteral("Visual cell values must be finite and in [0, 1]"));
            return;
        }
        if (intensity < 0.12)
            continue;

        stimuli.append(QJsonObject{
            {QStringLiteral("neuron_id"), i},
            {QStringLiteral("start_step"), static_cast<qint64>(completed + 1)},
            {QStringLiteral("duration_steps"), 2},
            {QStringLiteral("amplitude"), 8.0 + 32.0 * intensity}
        });
        ++activeCells;
    }

    client_.sendRequest(QStringLiteral("schedule_stimuli"),
                        QJsonObject{{QStringLiteral("stimuli"), stimuli}});
    setStatusMessage(QStringLiteral("Submitted %1 active visual cells to neurons 0–63")
                         .arg(activeCells));
}

void SimulationBridge::sendStimulusPulse(
    int neuronId, double amplitude, int durationSteps) {
    if (!connected_) {
        setErrorMessage(QStringLiteral("Cannot send stimulus: backend disconnected"));
        return;
    }
    if (simulation_status_.compare(QStringLiteral("running"), Qt::CaseInsensitive) == 0) {
        setErrorMessage(QStringLiteral("Pause the simulation before scheduling a controlled pulse"));
        return;
    }
    if (neuronId < 0 || durationSteps <= 0 ||
        !std::isfinite(amplitude) || amplitude < 0.0 || amplitude > 100.0) {
        setErrorMessage(QStringLiteral("Pulse requires neuron >= 0, duration > 0, and amplitude in [0, 100]"));
        return;
    }

    bool stepOk = false;
    const quint64 completed = telemetry_.value(QStringLiteral("completed_steps"))
        .toString().toULongLong(&stepOk);
    if (!stepOk || completed >= static_cast<quint64>(std::numeric_limits<qint64>::max())) {
        setErrorMessage(QStringLiteral("Waiting for a valid simulation step counter"));
        return;
    }

    const auto neuronCount = connectivity_statistics_
        .value(QStringLiteral("neuron_count")).toULongLong();
    if (neuronCount > 0 && static_cast<quint64>(neuronId) >= neuronCount) {
        setErrorMessage(QStringLiteral("Selected neuron is outside the current network"));
        return;
    }

    const QJsonObject stimulus{
        {QStringLiteral("neuron_id"), neuronId},
        {QStringLiteral("start_step"), static_cast<qint64>(completed + 1)},
        {QStringLiteral("duration_steps"), durationSteps},
        {QStringLiteral("amplitude"), amplitude}
    };
    client_.sendRequest(
        QStringLiteral("schedule_stimuli"),
        QJsonObject{{QStringLiteral("stimuli"), QJsonArray{stimulus}}});
    setStatusMessage(
        QStringLiteral("Controlled pulse queued for neuron %1 at step %2")
            .arg(neuronId).arg(completed + 1));
}

void SimulationBridge::reinitializeSimulation() {
    if (!connected_) {
        setErrorMessage(
            QStringLiteral("Cannot reinitialize: not connected"));
        return;
    }

    if (!hasPendingReinitialization()) {
        setStatusMessage(
            QStringLiteral("No reinitialization is currently required"));
        return;
    }

    setStatusMessage(QStringLiteral("Reinitializing simulation..."));
    client_.sendRequest(QStringLiteral("reinitialize"));
}

void SimulationBridge::setStatusMessage(const QString& message) {
    if (status_message_ == message) {
        return;
    }
    status_message_ = message;
    emit statusMessageChanged();
}

void SimulationBridge::setErrorMessage(const QString& message) {
    if (error_message_ == message) {
        return;
    }
    error_message_ = message;
    emit errorMessageChanged();
}

void SimulationBridge::updateTelemetry(const QJsonObject& telemetry) {
    telemetry_ = telemetry;

    bool step_ok = false;
    const quint64 completed = telemetry.value(QStringLiteral("completed_steps"))
        .toString().toULongLong(&step_ok);
    if (step_ok) {
        if (has_history_step_ && completed < last_history_step_) {
            telemetry_history_.clear();
        }
        if (telemetry_history_.isEmpty() || completed != last_history_step_) {
            QVariantMap sample;
            sample.insert(QStringLiteral("completed_steps"),
                          QString::number(completed));
            sample.insert(QStringLiteral("mean_voltage_mv"),
                          telemetry.value(QStringLiteral("mean_voltage_mv")).toDouble());
            sample.insert(QStringLiteral("excitatory_rate_hz"),
                          telemetry.value(QStringLiteral("excitatory_rate_hz")).toDouble());
            sample.insert(QStringLiteral("inhibitory_rate_hz"),
                          telemetry.value(QStringLiteral("inhibitory_rate_hz")).toDouble());
            telemetry_history_.append(sample);
            constexpr qsizetype kMaximumTelemetrySamples = 120;
            if (telemetry_history_.size() > kMaximumTelemetrySamples) {
                telemetry_history_.remove(
                    0, telemetry_history_.size() - kMaximumTelemetrySamples);
            }
        }
        last_history_step_ = completed;
        has_history_step_ = true;
    }

    if (telemetry.contains(QStringLiteral("status"))) {
        const QString nextStatus =
            telemetry.value(QStringLiteral("status")).toString();
        if (simulation_status_ != nextStatus) {
            simulation_status_ = nextStatus;
            emit simulationStatusChanged();
        }
    }

    emit telemetryChanged();
}

void SimulationBridge::updateSpikeEvents(const QJsonObject& spikeEvents) {
    const QString incomingGeneration =
        spikeEvents.value(QStringLiteral("simulation_generation"))
            .toString(spike_events_generation_);

    if (spike_events_generation_ != QStringLiteral("—") &&
        incomingGeneration != spike_events_generation_) {
        spike_events_.clear();
        last_spike_event_sequence_ = 0;
    }

    spike_events_generation_ = incomingGeneration;
    spike_event_count_ =
        spikeEvents.value(QStringLiteral("total_events")).toString(
            spike_event_count_);
    spike_events_dropped_ =
        spikeEvents.value(QStringLiteral("dropped_events")).toString(
            spike_events_dropped_);
    spike_events_first_sequence_ =
        spikeEvents.value(QStringLiteral("first_sequence")).toString(
            spike_events_first_sequence_);

    const QJsonArray events =
        spikeEvents.value(QStringLiteral("events")).toArray();

    constexpr qsizetype kMaximumDashboardEvents = 12000;
    for (const QJsonValue& value : events) {
        if (!value.isObject()) {
            continue;
        }

        const QJsonObject event = value.toObject();
        bool sequenceOk = false;
        const quint64 sequence =
            event.value(QStringLiteral("sequence")).toString()
                .toULongLong(&sequenceOk);
        if (!sequenceOk || sequence <= last_spike_event_sequence_) {
            continue;
        }

        QVariantMap item;
        item.insert(QStringLiteral("sequence"),
                    event.value(QStringLiteral("sequence")).toString());
        item.insert(QStringLiteral("step"),
                    event.value(QStringLiteral("step")).toString());
        item.insert(QStringLiteral("neuron_id"),
                    event.value(QStringLiteral("neuron_id")).toInt());
        spike_events_.append(item);
        last_spike_event_sequence_ = sequence;
    }

    if (spike_events_.size() > kMaximumDashboardEvents) {
        const qsizetype excess =
            spike_events_.size() - kMaximumDashboardEvents;
        spike_events_.remove(0, excess);
    }

    if (!spike_events_update_timer_.isActive()) {
        spike_events_update_timer_.start();
    }
}

void SimulationBridge::updateConfiguration(
    const QJsonObject& configuration
) {
    configuration_ = configuration;
    emit configurationChanged();
}
