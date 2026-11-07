import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    visible: true
    width: 1440
    height: 900
    minimumWidth: 1000
    minimumHeight: 680
    title: "Neuromancer · Neural Vision"
    color: "#080f18"

    property color panelColor: "#101b29"
    property color borderColor: "#26384b"
    property color mutedColor: "#91a2b8"
    property color textColor: "#edf5ff"
    property int rasterWindowSteps: 2000

    function neuronCount() {
        var p = simulation.activeParameters
        var e = Number(p.excitatory_neurons || 64)
        var i = Number(p.inhibitory_neurons || 0)
        return Math.max(64, e + i)
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 14

        RowLayout {
            Layout.fillWidth: true
            ColumnLayout {
                spacing: 2
                Label {
                    text: "NEUROMANCER"
                    color: "#70efc5"
                    font.pixelSize: 12
                    font.bold: true
                    font.letterSpacing: 2
                }
                Label {
                    text: "Neural vision · live spike activity"
                    color: textColor
                    font.pixelSize: 23
                    font.bold: true
                }
            }
            Item { Layout.fillWidth: true }
            Label {
                text: simulation.connected ? "BACKEND CONNECTED" : "BACKEND DISCONNECTED"
                color: simulation.connected ? "#70efc5" : "#e5b66e"
                font.pixelSize: 11
                font.bold: true
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 14

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                radius: 12
                color: "#050a10"
                border.color: borderColor

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "SPIKE RASTER"
                            color: "#70efc5"
                            font.pixelSize: 11
                            font.bold: true
                            font.letterSpacing: 1.4
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: simulation.spikeEventCount + " total · " +
                                  simulation.spikeEvents.length + " retained"
                            color: mutedColor
                            font.pixelSize: 11
                        }
                    }

                    Canvas {
                        id: raster
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        antialiasing: false

                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.fillStyle = "#050a10"
                            ctx.fillRect(0, 0, width, height)

                            var count = window.neuronCount()
                            var current = Number(simulation.completedSteps || 0)
                            var first = Math.max(0, current - window.rasterWindowSteps)
                            var span = Math.max(1, current - first)

                            // Subtle horizontal guides make individual neuron rows readable.
                            ctx.strokeStyle = "#172638"
                            ctx.lineWidth = 1
                            for (var n = 0; n < count; n += Math.max(1, Math.floor(count / 16))) {
                                var gy = (n + 0.5) * height / count
                                ctx.beginPath()
                                ctx.moveTo(0, gy)
                                ctx.lineTo(width, gy)
                                ctx.stroke()
                            }

                            var events = simulation.spikeEvents
                            for (var j = 0; j < events.length; ++j) {
                                var ev = events[j]
                                var step = Number(ev.step)
                                var id = Number(ev.neuron_id)
                                if (step < first || step > current || id < 0 || id >= count)
                                    continue

                                var x = ((step - first) / span) * width
                                var y = ((id + 0.5) / count) * height
                                ctx.fillStyle = id < Number(simulation.activeParameters.excitatory_neurons || 64)
                                               ? "#65e6bd" : "#f2b66d"
                                ctx.fillRect(Math.floor(x), Math.floor(y), 2, Math.max(2, height / count))
                            }

                            // Current-time marker.
                            ctx.strokeStyle = "#dce9f6"
                            ctx.globalAlpha = 0.35
                            ctx.beginPath()
                            ctx.moveTo(width - 1, 0)
                            ctx.lineTo(width - 1, height)
                            ctx.stroke()
                            ctx.globalAlpha = 1
                        }

                        Connections {
                            target: simulation
                            function onSpikeEventsChanged() { raster.requestPaint() }
                            function onTelemetryChanged() { raster.requestPaint() }
                            function onConfigurationChanged() { raster.requestPaint() }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: "● Excitatory"; color: "#65e6bd"; font.pixelSize: 11 }
                        Label { text: "● Inhibitory"; color: "#f2b66d"; font.pixelSize: 11 }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: "Last " + window.rasterWindowSteps + " simulation steps"
                            color: mutedColor
                            font.pixelSize: 11
                        }
                    }
                }
            }

            Rectangle {
                Layout.preferredWidth: 330
                Layout.fillHeight: true
                radius: 12
                color: panelColor
                border.color: borderColor

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 11

                    Label {
                        text: "VISUAL INPUT"
                        color: "#70efc5"
                        font.pixelSize: 11
                        font.bold: true
                        font.letterSpacing: 1.4
                    }
                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        text: "Captured desktop frames are encoded as 8×8 luminance values. The map below shows the 64 input intensities; it is not drawn over your desktop."
                        color: mutedColor
                        font.pixelSize: 11
                    }

                    Grid {
                        Layout.alignment: Qt.AlignHCenter
                        columns: 8
                        rows: 8
                        spacing: 3
                        Repeater {
                            model: desktopCapture.visualGrid
                            delegate: Rectangle {
                                required property var modelData
                                width: 24
                                height: 24
                                radius: 2
                                color: Qt.rgba(0.12 + 0.45 * Number(modelData),
                                               0.18 + 0.55 * Number(modelData),
                                               0.25 + 0.45 * Number(modelData), 1)
                                border.color: "#34465a"
                                border.width: 1
                            }
                        }
                    }

                    Button {
                        Layout.fillWidth: true
                        enabled: simulation.connected &&
                                 simulation.simulationStatus !== "running" &&
                                 desktopCapture.visualGrid.length === 64
                        text: "Encode current desktop frame"
                        onClicked: simulation.sendVisualGrid(desktopCapture.visualGrid)
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: borderColor
                    }

                    Label {
                        text: "SIMULATION"
                        color: "#70efc5"
                        font.pixelSize: 11
                        font.bold: true
                        font.letterSpacing: 1.4
                    }
                    Label {
                        text: "State: " + simulation.simulationStatus
                        color: textColor
                        font.pixelSize: 12
                    }
                    Label {
                        text: "Step: " + simulation.completedSteps
                        color: mutedColor
                        font.pixelSize: 12
                    }
                    Label {
                        text: "E spikes: " + simulation.excitatorySpikes +
                              "   I spikes: " + simulation.inhibitorySpikes
                        color: mutedColor
                        font.pixelSize: 11
                    }
                    Label {
                        text: "E rate: " + simulation.excitatoryRateHz.toFixed(1) +
                              " Hz   I rate: " + simulation.inhibitoryRateHz.toFixed(1) + " Hz"
                        color: mutedColor
                        font.pixelSize: 11
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Button {
                            Layout.fillWidth: true
                            enabled: simulation.connected && simulation.simulationStatus !== "running"
                            text: "Start"
                            onClicked: simulation.startSimulation()
                        }
                        Button {
                            Layout.fillWidth: true
                            enabled: simulation.connected && simulation.simulationStatus === "running"
                            text: "Pause"
                            onClicked: simulation.pauseSimulation()
                        }
                    }
                    Button {
                        Layout.fillWidth: true
                        enabled: simulation.connected && simulation.simulationStatus !== "running"
                        text: "Step 10"
                        onClicked: simulation.stepSimulation(10)
                    }

                    Item { Layout.fillHeight: true }

                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        text: "Spike events dropped: " + simulation.spikeEventsDropped +
                              "\nGeneration: " + simulation.spikeEventsGeneration
                        color: mutedColor
                        font.pixelSize: 10
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Label {
                Layout.fillWidth: true
                text: "Neural activity is plotted from recorded backend spike events. Desktop pixels are not sent anywhere outside the local simulation."
                color: "#71839a"
                font.pixelSize: 10
            }
            Label {
                text: "Raster window: " + window.rasterWindowSteps + " steps"
                color: "#71839a"
                font.pixelSize: 10
            }
        }
    }

    Component.onCompleted: simulation.connectToServer()
}
