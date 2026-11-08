#include "qt/simulation_bridge.h"
#include "qt/connectome_field.h"

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QUrl>
#include <qqml.h>

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);

    qmlRegisterType<ConnectomeField>(
        "NeuralField", 1, 0, "ConnectomeField");

    QQmlApplicationEngine engine;
    SimulationBridge simulation;

    engine.rootContext()->setContextProperty(
        QStringLiteral("simulation"), &simulation);

    engine.load(QUrl(QStringLiteral("qrc:/qml/Main.qml")));

    if (engine.rootObjects().isEmpty()) {
        return 1;
    }

    return app.exec();
}
