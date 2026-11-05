#include "qt/simulation_client.h"

#include <QJsonDocument>
#include <QJsonParseError>
#include <QTimer>

namespace {

constexpr int kProtocolVersion = 1;
constexpr qsizetype kMaximumResponseBytes = 1024 * 1024;

} // namespace

SimulationClient::SimulationClient(QObject* parent)
    : QObject(parent) {

    connect(
        &socket_,
        &QLocalSocket::connected,
        this,
        &SimulationClient::connected
    );

    connect(
        &socket_,
        &QLocalSocket::disconnected,
        this,
        [this]() {
            failPendingRequests(
                QStringLiteral("disconnected"),
                QStringLiteral("Server disconnected before responding")
            );
            emit disconnected();
        }
    );

    connect(
        &socket_,
        &QLocalSocket::readyRead,
        this,
        &SimulationClient::readAvailable
    );

    connect(
        &socket_,
        &QLocalSocket::errorOccurred,
        this,
        [this](QLocalSocket::LocalSocketError) {
            emit transportError(socket_.errorString());
        }
    );
}

SimulationClient::~SimulationClient() {
    // Disconnect callbacks before aborting the socket. This prevents
    // socket shutdown from invoking callbacks against partially destroyed
    // SimulationClient members.
    QObject::disconnect(&socket_, nullptr, this, nullptr);
    socket_.abort();

    for (auto it = pending_requests_.begin();
         it != pending_requests_.end(); ++it) {
        if (it->timeout != nullptr) {
            it->timeout->stop();
            delete it->timeout;
        }
    }

    pending_requests_.clear();
}

void SimulationClient::connectToServer(const QString& socket_path) {
    if (socket_.state() != QLocalSocket::UnconnectedState) {
        socket_.abort();
    }

    read_buffer_.clear();
    socket_.setServerName(socket_path);
    socket_.connectToServer();
}

void SimulationClient::disconnectFromServer() {
    socket_.disconnectFromServer();
}

bool SimulationClient::isConnected() const {
    return socket_.state() == QLocalSocket::ConnectedState;
}

QString SimulationClient::sendRequest(
    const QString& command,
    const QJsonObject& payload,
    int timeout_ms
) {
    const QString request_id =
        QStringLiteral("qt-%1").arg(++next_request_id_);

    if (!isConnected()) {
        emit requestFailed(
            request_id,
            command,
            QStringLiteral("not_connected"),
            QStringLiteral("Cannot send a request while disconnected")
        );
        return request_id;
    }

    QJsonObject request;
    request.insert(QStringLiteral("protocol_version"), kProtocolVersion);
    request.insert(QStringLiteral("request_id"), request_id);
    request.insert(QStringLiteral("command"), command);
    request.insert(QStringLiteral("payload"), payload);

    const QByteArray line =
        QJsonDocument(request).toJson(QJsonDocument::Compact) + '\n';

    auto* timer = new QTimer(this);
    timer->setSingleShot(true);

    pending_requests_.insert(
        request_id,
        PendingRequest{command, timer}
    );

    connect(timer, &QTimer::timeout, this, [this, request_id]() {
        auto it = pending_requests_.find(request_id);
        if (it == pending_requests_.end()) {
            return;
        }

        const QString command = it->command;
        QTimer* timer = it->timeout;
        pending_requests_.erase(it);
        timer->deleteLater();

        emit requestFailed(
            request_id,
            command,
            QStringLiteral("timeout"),
            QStringLiteral("Timed out waiting for the server response")
        );
    });

    const qint64 written = socket_.write(line);
    if (written != line.size()) {
        pending_requests_.remove(request_id);
        timer->deleteLater();

        emit requestFailed(
            request_id,
            command,
            QStringLiteral("write_failed"),
            socket_.errorString()
        );

        socket_.abort();
        return request_id;
    }

    timer->start(qMax(1, timeout_ms));
    return request_id;
}

void SimulationClient::readAvailable() {
    read_buffer_.append(socket_.readAll());

    while (true) {
        const qsizetype newline = read_buffer_.indexOf('\n');

        if (newline < 0) {
            if (read_buffer_.size() > kMaximumResponseBytes) {
                emit transportError(
                    QStringLiteral("Server response exceeded 1 MiB")
                );
                socket_.abort();
            }
            return;
        }

        if (newline > kMaximumResponseBytes) {
            emit transportError(
                QStringLiteral("Server response exceeded 1 MiB")
            );
            socket_.abort();
            return;
        }

        const QByteArray line = read_buffer_.left(newline);
        read_buffer_.remove(0, newline + 1);

        processLine(line);

        if (socket_.state() == QLocalSocket::UnconnectedState) {
            return;
        }
    }
}

void SimulationClient::processLine(const QByteArray& line) {
    QJsonParseError parse_error;
    const QJsonDocument document =
        QJsonDocument::fromJson(line, &parse_error);

    if (parse_error.error != QJsonParseError::NoError ||
        !document.isObject()) {
        emit transportError(
            QStringLiteral("Invalid JSON response: %1")
                .arg(parse_error.errorString())
        );
        socket_.abort();
        return;
    }

    const QJsonObject message = document.object();

    if (message.value(QStringLiteral("protocol_version")).toInt(-1) !=
        kProtocolVersion) {
        emit transportError(
            QStringLiteral("Unsupported server protocol version")
        );
        socket_.abort();
        return;
    }

    if (message.value(QStringLiteral("event")).toString() ==
        QStringLiteral("spike_events")) {
        const QJsonValue spikeEvents =
            message.value(QStringLiteral("spike_events"));

        if (!spikeEvents.isObject()) {
            emit transportError(
                QStringLiteral("Spike event notification has no spike_events object")
            );
            socket_.abort();
            return;
        }

        emit spikeEventsReceived(spikeEvents.toObject());
        return;
    }

    if (message.value(QStringLiteral("event")).toString() ==
        QStringLiteral("telemetry_update")) {

        const QJsonValue telemetry =
            message.value(QStringLiteral("telemetry"));

        if (!telemetry.isObject()) {
            emit transportError(
                QStringLiteral("Telemetry event has no telemetry object")
            );
            socket_.abort();
            return;
        }

        emit telemetryReceived(telemetry.toObject());
        return;
    }

    const QString request_id =
        message.value(QStringLiteral("request_id")).toString();

    auto it = pending_requests_.find(request_id);
    if (request_id.isEmpty() || it == pending_requests_.end()) {
        emit transportError(
            QStringLiteral("Received response for an unknown request ID")
        );
        return;
    }

    const QString command = it->command;
    QTimer* timer = it->timeout;

    pending_requests_.erase(it);
    timer->stop();
    timer->deleteLater();

    if (!message.value(QStringLiteral("ok")).toBool()) {
        const QJsonObject error =
            message.value(QStringLiteral("error")).toObject();

        emit requestFailed(
            request_id,
            command,
            error.value(QStringLiteral("code"))
                .toString(QStringLiteral("command_failed")),
            error.value(QStringLiteral("message"))
                .toString(QStringLiteral("Server rejected the request"))
        );
        return;
    }

    const QJsonValue result = message.value(QStringLiteral("result"));
    if (!result.isObject()) {
        emit requestFailed(
            request_id,
            command,
            QStringLiteral("invalid_response"),
            QStringLiteral("Successful response did not contain a result object")
        );
        return;
    }

    emit requestSucceeded(request_id, command, result.toObject());
}

void SimulationClient::failPendingRequests(
    const QString& error_code,
    const QString& message
) {
    const auto request_ids = pending_requests_.keys();

    for (const QString& request_id : request_ids) {
        auto it = pending_requests_.find(request_id);
        if (it == pending_requests_.end()) {
            continue;
        }

        const QString command = it->command;
        QTimer* timer = it->timeout;

        pending_requests_.erase(it);
        timer->stop();
        timer->deleteLater();

        emit requestFailed(
            request_id,
            command,
            error_code,
            message
        );
    }
}
