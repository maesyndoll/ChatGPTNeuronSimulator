#include <QCoreApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocalSocket>
#include <QString>
#include <QTextStream>

#include <unistd.h>

namespace {

constexpr int kProtocolVersion = 1;
constexpr int kConnectTimeoutMs = 3000;
constexpr int kIoTimeoutMs = 3000;

QString default_socket_path() {
    return QStringLiteral("/tmp/spiking_model-%1.sock")
        .arg(static_cast<qulonglong>(::getuid()));
}

int fail(const QString& message) {
    QTextStream(stderr) << "Qt IPC client: " << message << Qt::endl;
    return 1;
}

} // namespace

int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);

    const QStringList arguments = app.arguments();
    if (arguments.size() > 2) {
        return fail(QStringLiteral("Usage: spiking_model_qt_client [socket_path]"));
    }

    const QString socket_path =
        arguments.size() == 2 ? arguments.at(1) : default_socket_path();

    QLocalSocket socket;
    socket.connectToServer(socket_path);

    if (!socket.waitForConnected(kConnectTimeoutMs)) {
        return fail(QStringLiteral("Could not connect to %1: %2")
                        .arg(socket_path, socket.errorString()));
    }

    const QString request_id = QStringLiteral("qt-client-status-1");

    QJsonObject request;
    request.insert(QStringLiteral("protocol_version"), kProtocolVersion);
    request.insert(QStringLiteral("request_id"), request_id);
    request.insert(QStringLiteral("command"), QStringLiteral("get_status"));
    request.insert(QStringLiteral("payload"), QJsonObject{});

    const QByteArray request_line =
        QJsonDocument(request).toJson(QJsonDocument::Compact) + '\n';

    if (socket.write(request_line) != request_line.size()) {
        return fail(QStringLiteral("Could not queue the complete request: %1")
                        .arg(socket.errorString()));
    }

    if (!socket.waitForBytesWritten(kIoTimeoutMs)) {
        return fail(QStringLiteral("Request write failed: %1")
                        .arg(socket.errorString()));
    }

    QByteArray response_bytes;
    qsizetype newline_position = -1;

    while ((newline_position = response_bytes.indexOf('\n')) < 0) {
        if (!socket.waitForReadyRead(kIoTimeoutMs)) {
            response_bytes += socket.readAll();
            if (response_bytes.indexOf('\n') < 0) {
                return fail(QStringLiteral("Timed out waiting for the server response: %1")
                                .arg(socket.errorString()));
            }
            break;
        }

        response_bytes += socket.readAll();
    }

    const QByteArray response_line = response_bytes.left(
        response_bytes.indexOf('\n'));

    QJsonParseError parse_error;
    const QJsonDocument response_document =
        QJsonDocument::fromJson(response_line, &parse_error);

    if (parse_error.error != QJsonParseError::NoError ||
        !response_document.isObject()) {
        return fail(QStringLiteral("Server returned invalid JSON: %1")
                        .arg(parse_error.errorString()));
    }

    const QJsonObject response = response_document.object();

    if (response.value(QStringLiteral("request_id")).toString() != request_id) {
        return fail(QStringLiteral("Response request_id did not match the request"));
    }

    if (response.value(QStringLiteral("protocol_version")).toInt(-1) !=
        kProtocolVersion) {
        return fail(QStringLiteral("Server response used an unsupported protocol version"));
    }

    if (!response.value(QStringLiteral("ok")).toBool()) {
        const QJsonObject error = response.value(QStringLiteral("error")).toObject();
        return fail(QStringLiteral("Server rejected request: %1")
                        .arg(QString::fromUtf8(
                            QJsonDocument(error).toJson(QJsonDocument::Compact))));
    }

    const QJsonObject result = response.value(QStringLiteral("result")).toObject();
    const QString status = result.value(QStringLiteral("status")).toString();

    if (status.isEmpty()) {
        return fail(QStringLiteral("Successful response did not contain a status"));
    }

    QTextStream(stdout)
        << "Connected to: " << socket_path << Qt::endl
        << "Simulation status: " << status << Qt::endl
        << "Full response:" << Qt::endl
        << QJsonDocument(response).toJson(QJsonDocument::Indented);

    socket.disconnectFromServer();
    return 0;
}
