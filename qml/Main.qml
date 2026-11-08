import QtQuick
import QtQuick.Controls
import NeuralField 1.0

ApplicationWindow {
    id: window
    visible: true
    width: 1280
    height: 900
    minimumWidth: 720
    minimumHeight: 520
    title: "Neural Field"
    color: "#020409"

    property real orbitX: -0.18
    property real orbitY: 0.35
    property real distance: 3.15

    ConnectomeField {
        id: field
        anchors.fill: parent
        topology: simulation.topologySnapshot
        simulationGeneration: simulation.spikeEventsGeneration
        spikeEvents: simulation.visualSpikeEvents
        rotationX: window.orbitX
        rotationY: window.orbitY
        cameraDistance: window.distance
    }

    DragHandler {
        id: orbit
        target: null
        acceptedButtons: Qt.LeftButton
        property point previous: Qt.point(0, 0)
        onActiveChanged: previous = Qt.point(0, 0)
        onTranslationChanged: {
            var dx = translation.x - previous.x
            var dy = translation.y - previous.y
            previous = translation
            if (active) {
                window.orbitY += dx * 0.008
                window.orbitX = Math.max(-1.45, Math.min(1.45,
                    window.orbitX + dy * 0.008))
            }
        }
    }

    WheelHandler {
        target: null
        acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
        onWheel: function(event) {
            window.distance = Math.max(1.45, Math.min(7.0,
                window.distance - event.angleDelta.y / 120 * 0.18))
            event.accepted = true
        }
    }

    Text {
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.margins: 20
        text: "CONNECTOME  /  GPU FIELD"
        color: "#71879a"
        font.pixelSize: 10
        font.letterSpacing: 2
    }

    Row {
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 18
        spacing: 8

        Rectangle {
            width: 8
            height: 8
            radius: 4
            anchors.verticalCenter: parent.verticalCenter
            color: simulation.connected ? "#65e5c6" : "#647080"
        }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: simulation.connectionStatus
            color: "#8998a8"
            font.pixelSize: 11
        }
        Rectangle {
            width: 1
            height: 20
            color: "#26313d"
        }
        Rectangle {
            width: 76
            height: 28
            radius: 5
            color: connectMouse.containsMouse ? "#243543" : "#111a24"
            border.color: "#344657"
            Text {
                anchors.centerIn: parent
                text: simulation.connected ? "Disconnect" : "Connect"
                color: "#d4e1eb"
                font.pixelSize: 11
            }
            MouseArea {
                id: connectMouse
                anchors.fill: parent
                hoverEnabled: true
                onClicked: simulation.connected
                    ? simulation.disconnectFromServer()
                    : simulation.connectToServer()
            }
        }
        Rectangle {
            width: 66
            height: 28
            radius: 5
            color: runMouse.containsMouse ? "#24443e" : "#10231f"
            border.color: "#35685d"
            opacity: simulation.connected ? 1 : 0.35
            Text {
                anchors.centerIn: parent
                text: simulation.simulationStatus === "running" ? "Pause" : "Run"
                color: "#a5f4df"
                font.pixelSize: 11
            }
            MouseArea {
                id: runMouse
                anchors.fill: parent
                hoverEnabled: true
                enabled: simulation.connected
                onClicked: simulation.simulationStatus === "running"
                    ? simulation.pauseSimulation()
                    : simulation.startSimulation()
            }
        }
    }

    Text {
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.margins: 20
        text: simulation.topologySnapshot.rendered_neurons + " NODES  /  " +
              simulation.topologySnapshot.rendered_synapses + " SYNAPSES"
        color: "#71879a"
        font.pixelSize: 10
        font.family: "monospace"
    }

    Text {
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 20
        text: "STEP " + simulation.completedSteps
        color: "#71879a"
        font.pixelSize: 10
        font.family: "monospace"
    }

    Timer {
        interval: 33
        repeat: true
        running: simulation.connected &&
                 simulation.simulationStatus === "running"
        onTriggered: field.advanceFrame()
    }

    Timer {
        interval: 1000
        repeat: true
        running: simulation.connected
        onTriggered: simulation.refreshTopology()
    }
}
