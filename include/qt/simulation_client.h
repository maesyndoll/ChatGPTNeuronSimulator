#pragma once

#include <QHash>
#include <QJsonObject>
#include <QLocalSocket>
#include <QObject>
#include <QString>

class QTimer;

class SimulationClient : public QObject {
    Q_OBJECT

public:
    explicit SimulationClient(QObject* parent = nullptr);
    ~SimulationClient() override;

    void connectToServer(const QString& socket_path);
    void disconnectFromServer();

    bool isConnected() const;

    QString sendRequest(
        const QString& command,
        const QJsonObject& payload = {},
        int timeout_ms = 5000
    );

signals:
    void connected();
    void disconnected();

    void transportError(const QString& message);

    void requestSucceeded(
        const QString& request_id,
        const QString& command,
        const QJsonObject& result
    );

    void requestFailed(
        const QString& request_id,
        const QString& command,
        const QString& error_code,
        const QString& message
    );

    void telemetryReceived(const QJsonObject& telemetry);
    void spikeEventsReceived(const QJsonObject& spike_events);

private:
    struct PendingRequest {
        QString command;
        QTimer* timeout = nullptr;
    };

    void readAvailable();
    void processLine(const QByteArray& line);
    void failPendingRequests(
        const QString& error_code,
        const QString& message
    );

    QLocalSocket socket_;
    QByteArray read_buffer_;
    QHash<QString, PendingRequest> pending_requests_;
    quint64 next_request_id_ = 0;
};
