#include "qt/simulation_client.h"

#include <QCoreApplication>
#include <QJsonDocument>
#include <QJsonObject>
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
            << "Usage: spiking_model_qt_telemetry_probe [socket_path]"
            << Qt::endl;
        return 1;
    }

    const QString socket_path =
        arguments.size() == 2 ? arguments.at(1) : default_socket_path();

    SimulationClient client;
    QTimer connection_timeout;
    QTimer overall_timeout;

    connection_timeout.setSingleShot(true);
    overall_timeout.setSingleShot(true);

    int telemetry_count = 0;
    bool pause_requested = false;

    QObject::connect(
        &connection_timeout, &QTimer::timeout, &app,
        [&app]() {
            QTextStream(stderr)
                << "Timed out connecting to simulation server"
                << Qt::endl;
            app.exit(1);
        }
    );

    QObject::connect(
        &overall_timeout, &QTimer::timeout, &app,
        [&app]() {
            QTextStream(stderr)
                << "Telemetry probe timed out"
                << Qt::endl;
            app.exit(1);
        }
    );

    QObject::connect(
        &client, &SimulationClient::connected, &app,
        [&client, &connection_timeout]() {
            connection_timeout.stop();

            QTextStream(stdout)
                << "Connected to simulation server"
                << Qt::endl
                << "Subscribing to telemetry..."
                << Qt::endl;

            client.sendRequest(
                QStringLiteral("subscribe_telemetry"),
                QJsonObject{},
                5000
            );
        }
    );

    QObject::connect(
        &client, &SimulationClient::requestSucceeded, &app,
        [&client, &app, &telemetry_count](
            const QString& request_id,
            const QString& command,
            const QJsonObject& result
        ) {
            QTextStream(stdout)
                << "Request succeeded: " << command
                << " (" << request_id << ")"
                << Qt::endl;

            if (command == QStringLiteral("subscribe_telemetry")) {
                if (!result.value(QStringLiteral("subscribed")).toBool()) {
                    QTextStream(stderr)
                        << "Server did not confirm telemetry subscription"
                        << Qt::endl;
                    app.exit(1);
                    return;
                }

                QTextStream(stdout)
                    << "Starting simulation..."
                    << Qt::endl;

                client.sendRequest(
                    QStringLiteral("start"),
                    QJsonObject{},
                    5000
                );
            } else if (command == QStringLiteral("pause")) {
                QTextStream(stdout)
                    << "Simulation paused. Unsubscribing..."
                    << Qt::endl;

                client.sendRequest(
                    QStringLiteral("unsubscribe_telemetry"),
                    QJsonObject{},
                    5000
                );
            } else if (command == QStringLiteral("unsubscribe_telemetry")) {
                if (result.value(QStringLiteral("subscribed")).toBool()) {
                    QTextStream(stderr)
                        << "Server did not confirm telemetry unsubscription"
                        << Qt::endl;
                    app.exit(1);
                    return;
                }

                QTextStream(stdout)
                    << "Telemetry unsubscribed successfully"
                    << Qt::endl
                    << "Received telemetry updates: "
                    << telemetry_count
                    << Qt::endl;
                app.exit(0);
            } else if (command == QStringLiteral("start")) {
                QTextStream(stdout)
                    << "Simulation is running; waiting for telemetry..."
                    << Qt::endl;
            }
        }
    );

    QObject::connect(
        &client, &SimulationClient::telemetryReceived, &app,
        [&client, &app, &telemetry_count, &pause_requested](
            const QJsonObject& telemetry
        ) {
            if (pause_requested)
                return;

            ++telemetry_count;

            QTextStream(stdout)
                << "Telemetry update #" << telemetry_count
                << Qt::endl
                << QJsonDocument(telemetry).toJson(QJsonDocument::Indented)
                << Qt::endl;

            if (telemetry_count >= 3) {
                pause_requested = true;

                QTextStream(stdout)
                    << "Received three updates. Pausing simulation..."
                    << Qt::endl;

                client.sendRequest(
                    QStringLiteral("pause"),
                    QJsonObject{},
                    5000
                );
            }
        }
    );

    QObject::connect(
        &client, &SimulationClient::requestFailed, &app,
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
        &client, &SimulationClient::transportError, &app,
        [&app](const QString& message) {
            QTextStream(stderr)
                << "Transport error: " << message
                << Qt::endl;
            app.exit(1);
        }
    );

    QObject::connect(
        &client, &SimulationClient::disconnected, &app,
        [&app]() {
            QTextStream(stderr)
                << "Disconnected unexpectedly during telemetry probe"
                << Qt::endl;
            app.exit(1);
        }
    );

    connection_timeout.start(5000);
    overall_timeout.start(30000);

    client.connectToServer(socket_path);

    const int exit_code = app.exec();
    overall_timeout.stop();
    connection_timeout.stop();

    return exit_code;
}
