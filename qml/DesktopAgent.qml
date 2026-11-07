import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    visible: true
    width: 1500
    height: 920
    minimumWidth: 1000
    minimumHeight: 680
    title: "Neuromancer · Desktop Agent"
    color: "#080f18"

    property color panelColor: "#101b29"
    property color borderColor: "#26384b"
    property color mutedColor: "#91a2b8"
    property color textColor: "#edf5ff"
    property bool visualInputEnabled: false

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 14

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

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
                    text: "Desktop perception"
                    color: textColor
                    font.pixelSize: 23
                    font.bold: true
                }
            }

            Item { Layout.fillWidth: true }

            Label {
                text: desktopCapture.captureStatus
                color: desktopCapture.capturing ? "#70efc5" : mutedColor
                font.pixelSize: 12
            }

            Button {
                text: desktopCapture.capturing ? "Pause capture" : "Resume capture"
                onClicked: desktopCapture.capturing
                           ? desktopCapture.stop()
                           : desktopCapture.start()
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 14

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumWidth: 560
                radius: 12
                color: "#050a10"
                border.color: borderColor
                clip: true

                Item {
                    id: preview
                    anchors.fill: parent
                    anchors.margins: 10

                    property real fitScale: Math.min(
                        width / Math.max(1, desktopCapture.screenWidth),
                        height / Math.max(1, desktopCapture.screenHeight))
                    property real imageOffsetX:
                        (width - desktopCapture.screenWidth * fitScale) / 2
                    property real imageOffsetY:
                        (height - desktopCapture.screenHeight * fitScale) / 2

                    Image {
                        id: screenImage
                        anchors.fill: parent
                        source: "image://desktop/live/" + desktopCapture.frameSequence
                        cache: false
                        asynchronous: false
                        fillMode: Image.PreserveAspectFit
                    }

                    Rectangle {
                        visible: desktopCapture.screenWidth > 0
                        x: preview.imageOffsetX +
                           desktopCapture.cursorX * preview.fitScale - width / 2
                        y: preview.imageOffsetY +
                           desktopCapture.cursorY * preview.fitScale - height / 2
                        width: 22
                        height: 22
                        radius: 11
                        color: "#35e6b0"
                        opacity: 0.88
                        border.color: "#f1fff9"
                        border.width: 2

                        Rectangle {
                            anchors.centerIn: parent
                            width: 4
                            height: 4
                            radius: 2
                            color: "#092018"
                        }
                    }

                    Grid {
                        x: preview.imageOffsetX
                        y: preview.imageOffsetY
                        width: desktopCapture.screenWidth * preview.fitScale
                        height: desktopCapture.screenHeight * preview.fitScale
                        columns: 8
                        rows: 8
                        opacity: 0.42
                        Repeater {
                            model: desktopCapture.visualGrid
                            delegate: Rectangle {
                                required property var modelData
                                width: parent.width / 8
                                height: parent.height / 8
                                color: "#38e8b4"
                                opacity: 0.08 + Number(modelData) * 0.52
                                border.width: 0.5
                                border.color: "#8fffdc"
                            }
                        }
                    }

                    Rectangle {
                        anchors.left: parent.left
                        anchors.top: parent.top
                        radius: 5
                        color: "#cc08111b"
                        width: statusLabel.implicitWidth + 18
                        height: statusLabel.implicitHeight + 12
                        Label {
                            id: statusLabel
                            anchors.centerIn: parent
                            text: "LIVE SCREEN · POINTER OVERLAY"
                            color: "#dce9f6"
                            font.pixelSize: 10
                            font.bold: true
                            font.letterSpacing: 1
                        }
                    }

                    Label {
                        anchors.centerIn: parent
                        visible: desktopCapture.screenWidth === 0
                        text: "Waiting for the first screen frame…"
                        color: mutedColor
                    }
                }
            }

            Rectangle {
                Layout.preferredWidth: 310
                Layout.fillHeight: true
                radius: 12
                color: panelColor
                border.color: borderColor

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 18
                    spacing: 12

                    Label {
                        text: "ENGINE STATUS"
                        color: "#70efc5"
                        font.pixelSize: 11
                        font.bold: true
                        font.letterSpacing: 1.4
                    }
                    Label {
                        text: simulation.connected ? "Backend connected" :
                                                      "Backend disconnected"
                        color: simulation.connected ? "#70efc5" : "#e5b66e"
                        font.pixelSize: 16
                        font.bold: true
                    }
                    Label {
                        text: "Simulation: " + simulation.simulationStatus
                        color: textColor
                        font.pixelSize: 12
                    }
                    Label {
                        text: "Completed steps: " + simulation.completedSteps
                        color: mutedColor
                        font.pixelSize: 12
                    }
                    Label {
                        text: "Excitatory spikes: " + simulation.excitatorySpikes
                        color: mutedColor
                        font.pixelSize: 12
                    }
                    Label {
                        text: "Inhibitory spikes: " + simulation.inhibitorySpikes
                        color: mutedColor
                        font.pixelSize: 12
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: borderColor
                    }

                    Label {
                        text: "VISUAL ATTENTION"
                        color: "#70efc5"
                        font.pixelSize: 11
                        font.bold: true
                        font.letterSpacing: 1.4
                    }
                    Label {
                        Layout.fillWidth: true
                        text: "Pointer position"
                        color: mutedColor
                        font.pixelSize: 11
                    }
                    Label {
                        text: desktopCapture.cursorX + ", " +
                              desktopCapture.cursorY
                        color: textColor
                        font.pixelSize: 22
                        font.family: "monospace"
                    }
                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        text: "8×8 luminance encoder. Each cell maps to one of excitatory neurons 0–63. The grid overlay shows measured pixel brightness, not learned attention."
                        color: mutedColor
                        font.pixelSize: 11
                    }
                    Button {
                        Layout.fillWidth: true
                        enabled: simulation.connected && desktopCapture.visualGrid.length === 64
                        text: "Send current frame to neural input"
                        onClicked: simulation.sendVisualGrid(desktopCapture.visualGrid)
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: borderColor
                    }

                    Label {
                        text: "ACTION OUTPUT"
                        color: "#70efc5"
                        font.pixelSize: 11
                        font.bold: true
                        font.letterSpacing: 1.4
                    }
                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        text: "Visual stimulation enabled. Mouse and keyboard actuation remain disabled. Sending a frame schedules brief current pulses to neurons 0–63; it does not yet mean the network recognizes objects."
                        color: mutedColor
                        font.pixelSize: 12
                    }

                    Item { Layout.fillHeight: true }

                    Button {
                        Layout.fillWidth: true
                        text: simulation.connected ? "Disconnect backend" :
                                                      "Connect backend"
                        onClicked: simulation.connected
                                   ? simulation.disconnectFromServer()
                                   : simulation.connectToServer()
                    }
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Label {
                Layout.fillWidth: true
                text: "Screen capture is local. No screenshots are transmitted by this frontend."
                color: "#71839a"
                font.pixelSize: 11
            }
            Label {
                text: "100 ms capture interval"
                color: "#71839a"
                font.pixelSize: 11
            }
        }
    }

    Component.onCompleted: simulation.connectToServer()
}
