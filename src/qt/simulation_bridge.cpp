#include "qt/simulation_bridge.h"

#include <QJsonValue>
#include <QJsonObject>
#include <QJsonArray>
#include <QStringList>
#include <limits>
#include <cmath>

#include <unistd.h>

SimulationBridge::SimulationBridge(QObject* parent)
    : QObject(parent) {
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
        refreshSpikeEvents();
        refreshTissue();
    });

    connect(&client_, &SimulationClient::disconnected, this, [this]() {
        connected_ = false;
        connection_status_ = QStringLiteral("Disconnected");
        emit connectionChanged();
        spike_events_.clear();
        spike_event_count_ = QStringLiteral("0");
        spike_events_dropped_ = QStringLiteral("0");
        spike_events_first_sequence_ = QStringLiteral("0");
        spike_events_generation_ = QStringLiteral("—");
        last_spike_event_sequence_ = 0;
        emit spikeEventsChanged();
        tissue_snapshot_ = {};
        emit tissueChanged();
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
        } else if (command == QStringLiteral("get_tissue")) {
            if (!result.value(QStringLiteral("unchanged")).toBool(false)) {
                updateTissue(result);
            }
        } else {
            const QJsonValue configurationValue =
                result.value(QStringLiteral("configuration"));
            if (configurationValue.isObject()) {
                updateConfiguration(configurationValue.toObject());
            }
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
        const QString combined =
            QStringLiteral("%1 (%2): %3").arg(command, errorCode, message);
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

QVariantMap SimulationBridge::tissueSnapshot() const {
    return tissue_snapshot_.toVariantMap();
}

QString SimulationBridge::tissueRevision() const {
    return tissue_snapshot_.value(QStringLiteral("revision")).toString(QStringLiteral("0"));
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
        client_.sendRequest(QStringLiteral("pause"), QJsonObject{}, 30000);
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

void SimulationBridge::stimulateNeurons(
    const QVariantList& neuronIds, double amplitude, int durationSteps) {
    if (!connected_ || neuronIds.isEmpty() || durationSteps <= 0 ||
        !std::isfinite(amplitude) || amplitude == 0.0) {
        setErrorMessage(QStringLiteral("Select neurons and provide a non-zero finite stimulus and positive duration"));
        return;
    }
    QJsonArray ids;
    for (const auto& value : neuronIds) {
        bool ok = false;
        const qulonglong id = value.toULongLong(&ok);
        if (!ok || id > std::numeric_limits<quint32>::max()) {
            setErrorMessage(QStringLiteral("Invalid neuron ID in stimulus selection"));
            return;
        }
        ids.append(static_cast<qint64>(id));
    }
    client_.sendRequest(QStringLiteral("stimulate_neurons"), QJsonObject{
        {QStringLiteral("neuron_ids"), ids},
        {QStringLiteral("amplitude"), amplitude},
        {QStringLiteral("duration_steps"), durationSteps}
    });
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

void SimulationBridge::refreshTissue() {
    if (!connected_) {
        return;
    }

    QJsonObject payload;
    if (tissue_snapshot_.contains(QStringLiteral("revision"))) {
        payload.insert(QStringLiteral("after_revision"), tissueRevision());
    }
    client_.sendRequest(QStringLiteral("get_tissue"), payload);
}

void SimulationBridge::refreshSpikeEvents() {
    if (connected_) {
        client_.sendRequest(
            QStringLiteral("get_spike_events"),
            QJsonObject{{QStringLiteral("after_sequence"), QStringLiteral("0")}}
        );
    }
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

    constexpr qsizetype kMaximumDashboardEvents = 50000;
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

    emit spikeEventsChanged();
}

void SimulationBridge::updateTissue(const QJsonObject& tissue) {
    tissue_snapshot_ = tissue;
    emit tissueChanged();
}

void SimulationBridge::updateConfiguration(
    const QJsonObject& configuration
) {
    configuration_ = configuration;
    emit configurationChanged();
}
