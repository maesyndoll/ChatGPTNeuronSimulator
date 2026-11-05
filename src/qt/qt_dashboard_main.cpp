#include "qt/simulation_bridge.h"

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QUrl>

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);

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
