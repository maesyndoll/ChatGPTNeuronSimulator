#include "qt/desktop_capture.h"
#include "qt/simulation_bridge.h"

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QUrl>

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    SimulationBridge simulation;
    DesktopCapture desktop_capture;

    engine.addImageProvider(
        QStringLiteral("desktop"),
        new DesktopFrameProvider(desktop_capture));
    engine.rootContext()->setContextProperty(
        QStringLiteral("simulation"), &simulation);
    engine.rootContext()->setContextProperty(
        QStringLiteral("desktopCapture"), &desktop_capture);

    engine.load(QUrl(QStringLiteral("qrc:/qml/DesktopAgent.qml")));
    if (engine.rootObjects().isEmpty())
        return 1;

    desktop_capture.start();
    return app.exec();
}
