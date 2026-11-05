#include "qt/simulation_client.h"

#include <QCoreApplication>
#include <QJsonDocument>
#include <QTextStream>
#include <QTimer>

#include <unistd.h>

namespace {

QString default_socket_path() {
    return QStringLiteral("/tmp/spiking_model-%1.sock")
        .arg(static_cast<qulonglong>(::getuid()));
}

} // namespace

int main(int argc, char* argv[]) {
    QCoreApplication app(argc, argv);

    const QStringList arguments = app.arguments();
    if (arguments.size() > 2) {
        QTextStream(stderr)
            << "Usage: spiking_model_qt_async_client [socket_path]"
            << Qt::endl;
        return 1;
    }

    const QString socket_path =
        arguments.size() == 2 ? arguments.at(1) : default_socket_path();

    SimulationClient client;
    QTimer connection_timeout;
    connection_timeout.setSingleShot(true);

    QObject::connect(
        &connection_timeout,
        &QTimer::timeout,
        &app,
        [&app]() {
            QTextStream(stderr)
                << "Timed out connecting to the simulation server"
                << Qt::endl;
            app.exit(1);
        }
    );

    QObject::connect(
        &client,
        &SimulationClient::connected,
        &app,
        [&client, &connection_timeout]() {
            connection_timeout.stop();

            QTextStream(stdout)
                << "Connected asynchronously to simulation server"
                << Qt::endl;

            client.sendRequest(
                QStringLiteral("get_status"),
                QJsonObject{},
                5000
            );
        }
    );

    QObject::connect(
        &client,
        &SimulationClient::requestSucceeded,
        &app,
        [&app](
            const QString& request_id,
            const QString& command,
            const QJsonObject& result
        ) {
            QTextStream(stdout)
                << "Request ID: " << request_id << Qt::endl
                << "Command: " << command << Qt::endl
                << "Simulation status: "
                << result.value(QStringLiteral("status")).toString()
                << Qt::endl
                << "Result:" << Qt::endl
                << QJsonDocument(result).toJson(QJsonDocument::Indented);

            app.exit(0);
        }
    );

    QObject::connect(
        &client,
        &SimulationClient::requestFailed,
        &app,
        [&app](
            const QString& request_id,
            const QString& command,
            const QString& error_code,
            const QString& message
        ) {
            QTextStream(stderr)
                << "Request failed: " << command
                << " (" << request_id << ") "
                << error_code << ": " << message
                << Qt::endl;

            app.exit(1);
        }
    );

    QObject::connect(
        &client,
        &SimulationClient::transportError,
        &app,
        [&app](const QString& message) {
            QTextStream(stderr)
                << "Transport error: " << message << Qt::endl;
            app.exit(1);
        }
    );

    connection_timeout.start(5000);
    client.connectToServer(socket_path);

    return app.exec();
}
