import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window

    visible: true
    width: 1120
    height: 760
    minimumWidth: 820
    minimumHeight: 620
    title: "ChatGPTNeuronSimulator"
    color: "#101722"

    palette.window: "#101722"
    palette.windowText: "#e8eef7"
    palette.button: "#26364a"
    palette.buttonText: "#e8eef7"
    palette.highlight: "#54b8a5"

    property bool parameterDraftInitialized: false


    Connections {
        target: simulation

        function onConfigurationChanged() {
            if (window.parameterDraftInitialized)
                return

            var parameters = simulation.requestedParameters
            var simulationParameters = parameters["simulation"] || {}
            var topologyParameters = parameters["topology"] || {}

            excitatoryCountField.text =
                String(simulationParameters["excitatory_count"] ?? "")
            inhibitoryCountField.text =
                String(simulationParameters["inhibitory_count"] ?? "")
            dtField.text = String(simulationParameters["dt"] ?? "")
            thresholdField.text =
                String(simulationParameters["threshold"] ?? "")
            pEeField.text = String(topologyParameters["p_ee"] ?? "")
            pEiField.text = String(topologyParameters["p_ei"] ?? "")

            window.parameterDraftInitialized = true
        }
    }

    ScrollView {
        id: dashboardScrollView
        anchors.fill: parent
        anchors.margins: 24
        clip: true
        contentWidth: availableWidth
        contentHeight: dashboardContent.height
        ScrollBar.vertical.policy: ScrollBar.AsNeeded

        ColumnLayout {
            id: dashboardContent
            width: dashboardScrollView.availableWidth
            height: implicitHeight
            spacing: 22

        RowLayout {
            Layout.fillWidth: true
            spacing: 16

            ColumnLayout {
                spacing: 4

                Label {
                    text: "ChatGPTNeuronSimulator"
                    color: "#f0f4fa"
                    font.pixelSize: 27
                    font.bold: true
                }

                Label {
                    text: "CUDA simulation dashboard"
                    color: "#91a2b8"
                    font.pixelSize: 14
                }
            }

            Item { Layout.fillWidth: true }

            Rectangle {
                radius: 8
                color: simulation.connected ? "#173d38" : "#432b30"
                border.color: simulation.connected ? "#347c6e" : "#85505a"
                implicitWidth: connectionLabel.implicitWidth + 24
                implicitHeight: 36

                Label {
                    id: connectionLabel
                    anchors.centerIn: parent
                    text: simulation.connectionStatus
                    color: simulation.connected ? "#8de0ca" : "#f0a7ad"
                    font.pixelSize: 13
                    font.bold: true
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#293647"
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: "Simulation"
                color: "#f0f4fa"
                font.pixelSize: 19
                font.bold: true
            }

            Label {
                text: "Status: " + simulation.simulationStatus
                color: "#a9bbcf"
                font.pixelSize: 14
            }

            Item { Layout.fillWidth: true }

            Button {
                text: "Connect"
                enabled: !simulation.connected
                onClicked: simulation.connectToServer()
            }

            Button {
                text: "Disconnect"
                enabled: simulation.connected
                onClicked: simulation.disconnectFromServer()
            }
        }

        Rectangle {
            Layout.fillWidth: true
            radius: 12
            color: "#192433"
            border.color: "#2b3a4d"
            implicitHeight: controlsColumn.implicitHeight + 32

            ColumnLayout {
                id: controlsColumn
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12

                Label {
                    text: "Controls"
                    color: "#dce6f2"
                    font.pixelSize: 16
                    font.bold: true
                }

                RowLayout {
                    spacing: 10

                    Button {
                        text: "Start"
                        enabled: simulation.connected
                        onClicked: simulation.startSimulation()
                    }

                    Button {
                        text: "Pause"
                        enabled: simulation.connected
                        onClicked: simulation.pauseSimulation()
                    }

                    Button {
                        text: "Step (1)"
                        enabled: simulation.connected
                        onClicked: simulation.stepSimulation(1)
                    }
                }

                Label {
                    Layout.fillWidth: true
                    text: simulation.statusMessage
                    color: "#91a2b8"
                    wrapMode: Text.Wrap
                }

                Label {
                    Layout.fillWidth: true
                    visible: simulation.errorMessage.length > 0
                    text: simulation.errorMessage
                    color: "#ff9b9b"
                    wrapMode: Text.Wrap
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            radius: 12
            color: "#192433"
            border.color: "#2b3a4d"
            implicitHeight: configurationColumn.implicitHeight + 32

            ColumnLayout {
                id: configurationColumn
                anchors.fill: parent
                anchors.margins: 16
                spacing: 12

                RowLayout {
                    Layout.fillWidth: true

                    Label {
                        text: "Model configuration"
                        color: "#f0f4fa"
                        font.pixelSize: 17
                        font.bold: true
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: simulation.hasPendingReinitialization
                              ? "Reinitialization pending"
                              : "Configuration synchronized"
                        color: simulation.hasPendingReinitialization
                               ? "#e5bd78" : "#8de0ca"
                        font.pixelSize: 12
                        font.bold: true
                    }

                    Button {
                        text: "Refresh"
                        enabled: simulation.connected
                        onClicked: simulation.refreshConfiguration()
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 24

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 6

                        Label {
                            text: "Requested values"
                            color: "#8fc9ed"
                            font.pixelSize: 14
                            font.bold: true
                        }

                        Label {
                            Layout.fillWidth: true
                            color: "#cbd7e5"
                            wrapMode: Text.Wrap
                            text: "Neurons: E " +
                                  window.parameterValue(
                                      simulation.requestedParameters,
                                      "simulation", "excitatory_count") +
                                  " / I " +
                                  window.parameterValue(
                                      simulation.requestedParameters,
                                      "simulation", "inhibitory_count")
                        }

                        Label {
                            Layout.fillWidth: true
                            color: "#cbd7e5"
                            wrapMode: Text.Wrap
                            text: "Steps: " +
                                  window.parameterValue(
                                      simulation.requestedParameters,
                                      "simulation", "steps") +
                                  "    dt: " +
                                  window.parameterValue(
                                      simulation.requestedParameters,
                                      "simulation", "dt") + " ms"
                        }

                        Label {
                            Layout.fillWidth: true
                            color: "#cbd7e5"
                            wrapMode: Text.Wrap
                            text: "Threshold: " +
                                  window.parameterValue(
                                      simulation.requestedParameters,
                                      "simulation", "threshold") + " mV"
                        }

                        Label {
                            Layout.fillWidth: true
                            color: "#cbd7e5"
                            wrapMode: Text.Wrap
                            text: "Connectivity pEE / pEI: " +
                                  window.parameterValue(
                                      simulation.requestedParameters,
                                      "topology", "p_ee") + " / " +
                                  window.parameterValue(
                                      simulation.requestedParameters,
                                      "topology", "p_ei")
                        }

                        Label {
                            Layout.fillWidth: true
                            color: "#cbd7e5"
                            wrapMode: Text.Wrap
                            text: "Delay distribution: " +
                                  window.parameterValue(
                                      simulation.requestedParameters,
                                      "topology", "delay_distribution")
                        }
                    }

                    Rectangle {
                        width: 1
                        Layout.fillHeight: true
                        color: "#2b3a4d"
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 6

                        Label {
                            text: "Active values"
                            color: "#8de0ca"
                            font.pixelSize: 14
                            font.bold: true
                        }

                        Label {
                            Layout.fillWidth: true
                            color: "#cbd7e5"
                            wrapMode: Text.Wrap
                            text: "Neurons: E " +
                                  window.parameterValue(
                                      simulation.activeParameters,
                                      "simulation", "excitatory_count") +
                                  " / I " +
                                  window.parameterValue(
                                      simulation.activeParameters,
                                      "simulation", "inhibitory_count")
                        }

                        Label {
                            Layout.fillWidth: true
                            color: "#cbd7e5"
                            wrapMode: Text.Wrap
                            text: "Steps: " +
                                  window.parameterValue(
                                      simulation.activeParameters,
                                      "simulation", "steps") +
                                  "    dt: " +
                                  window.parameterValue(
                                      simulation.activeParameters,
                                      "simulation", "dt") + " ms"
                        }

                        Label {
                            Layout.fillWidth: true
                            color: "#cbd7e5"
                            wrapMode: Text.Wrap
                            text: "Threshold: " +
                                  window.parameterValue(
                                      simulation.activeParameters,
                                      "simulation", "threshold") + " mV"
                        }

                        Label {
                            Layout.fillWidth: true
                            color: "#cbd7e5"
                            wrapMode: Text.Wrap
                            text: "Connectivity pEE / pEI: " +
                                  window.parameterValue(
                                      simulation.activeParameters,
                                      "topology", "p_ee") + " / " +
                                  window.parameterValue(
                                      simulation.activeParameters,
                                      "topology", "p_ei")
                        }

                        Label {
                            Layout.fillWidth: true
                            color: "#cbd7e5"
                            wrapMode: Text.Wrap
                            text: "Delay distribution: " +
                                  window.parameterValue(
                                      simulation.activeParameters,
                                      "topology", "delay_distribution")
                        }
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    height: 1
                    color: "#2b3a4d"
                }

                Label {
                    text: "Edit requested parameters"
                    color: "#f0f4fa"
                    font.pixelSize: 14
                    font.bold: true
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 3
                    columnSpacing: 12
                    rowSpacing: 10

                    ColumnLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "Excitatory neurons"
                            color: "#a9bbcf"
                        }
                        TextField {
                            id: excitatoryCountField
                            Layout.fillWidth: true
                            placeholderText: "Excitatory count"
                            validator: IntValidator {
                                bottom: 1
                                top: 1000000
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "Inhibitory neurons"
                            color: "#a9bbcf"
                        }
                        TextField {
                            id: inhibitoryCountField
                            Layout.fillWidth: true
                            placeholderText: "Inhibitory count"
                            validator: IntValidator {
                                bottom: 1
                                top: 250000
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "Time step (ms)"
                            color: "#a9bbcf"
                        }
                        TextField {
                            id: dtField
                            Layout.fillWidth: true
                            placeholderText: "dt"
                            validator: DoubleValidator {
                                bottom: 0.001
                                top: 5.0
                                decimals: 3
                                notation: DoubleValidator.StandardNotation
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "Firing threshold (mV)"
                            color: "#a9bbcf"
                        }
                        TextField {
                            id: thresholdField
                            Layout.fillWidth: true
                            placeholderText: "Threshold"
                            validator: DoubleValidator {
                                bottom: -100.0
                                top: 100.0
                                decimals: 3
                                notation: DoubleValidator.StandardNotation
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "E to E probability"
                            color: "#a9bbcf"
                        }
                        TextField {
                            id: pEeField
                            Layout.fillWidth: true
                            placeholderText: "p_ee"
                            validator: DoubleValidator {
                                bottom: 0.0
                                top: 1.0
                                decimals: 6
                                notation: DoubleValidator.StandardNotation
                            }
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "E to I probability"
                            color: "#a9bbcf"
                        }
                        TextField {
                            id: pEiField
                            Layout.fillWidth: true
                            placeholderText: "p_ei"
                            validator: DoubleValidator {
                                bottom: 0.0
                                top: 1.0
                                decimals: 6
                                notation: DoubleValidator.StandardNotation
                            }
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true

                    Label {
                        Layout.fillWidth: true
                        color: "#91a2b8"
                        wrapMode: Text.Wrap
                        text: "Changes are validated by the simulation server. " +
                              "Neuron counts and connectivity require reinitialization."
                    }

                    Button {
                        text: "Apply parameters"
                        enabled: simulation.connected &&
                                 excitatoryCountField.acceptableInput &&
                                 inhibitoryCountField.acceptableInput &&
                                 dtField.acceptableInput &&
                                 thresholdField.acceptableInput &&
                                 pEeField.acceptableInput &&
                                 pEiField.acceptableInput

                        onClicked: simulation.applyParameters({
                            "simulation": {
                                "excitatory_count":
                                    Number(excitatoryCountField.text),
                                "inhibitory_count":
                                    Number(inhibitoryCountField.text),
                                "dt": Number(dtField.text),
                                "threshold": Number(thresholdField.text)
                            },
                            "topology": {
                                "p_ee": Number(pEeField.text),
                                "p_ei": Number(pEiField.text)
                            }
                        })
                    }

                    Button {
                        text: "Reinitialize simulation"
                        enabled: simulation.connected &&
                                 simulation.hasPendingReinitialization
                        onClicked: reinitializeConfirmation.open()
                    }
                }

                Label {
                    Layout.fillWidth: true
                    color: "#71839a"
                    font.pixelSize: 11
                    wrapMode: Text.Wrap
                    text: "Parameter version: " + simulation.parameterVersion +
                          "    Simulation generation: " +
                          simulation.simulationGeneration
                }
            }
        }

        Label {
            text: "Live telemetry"
            color: "#f0f4fa"
            font.pixelSize: 19
            font.bold: true
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 2
            columnSpacing: 24
            rowSpacing: 8

            Label { text: "Steps: " + simulation.completedSteps; color: "#f0f4fa" }
            Label { text: "Mean voltage: " + simulation.meanVoltageMv.toFixed(3) + " mV"; color: "#f0f4fa" }
            Label { text: "Excitatory rate: " + simulation.excitatoryRateHz.toFixed(3) + " Hz"; color: "#f0f4fa" }
            Label { text: "Inhibitory rate: " + simulation.inhibitoryRateHz.toFixed(3) + " Hz"; color: "#f0f4fa" }
            Label { text: "Excitatory spikes: " + simulation.excitatorySpikes; color: "#f0f4fa" }
            Label { text: "Inhibitory spikes: " + simulation.inhibitorySpikes; color: "#f0f4fa" }
        }

        Label {
            text: "Numerical state: " +
                  (simulation.finiteState ? "Finite" : "Not yet verified")
            color: simulation.finiteState ? "#8de0ca" : "#d5b77b"
            font.pixelSize: 13
        }

        Label {
            Layout.fillWidth: true
            text: "Telemetry is received asynchronously from the simulation server. The dashboard does not execute the CUDA simulation."
            color: "#71839a"
            font.pixelSize: 12
            wrapMode: Text.Wrap
        }
        }
    }

    Dialog {
        id: reinitializeConfirmation
        title: "Reinitialize simulation?"
        modal: true
        anchors.centerIn: Overlay.overlay
        standardButtons: Dialog.Yes | Dialog.No

        Label {
            width: 360
            wrapMode: Text.Wrap
            text: "This will rebuild the simulation using the requested " +
                  "structural parameters and reset simulation progress. " +
                  "Continue?"
        }

        onAccepted: simulation.reinitializeSimulation()
    }

    function parameterValue(parameters, group, field) {
        var groupValues = parameters[group]
        if (groupValues === undefined || groupValues === null)
            return "—"

        var value = groupValues[field]
        return value === undefined || value === null ? "—" : String(value)
    }

    Component.onCompleted: simulation.connectToServer()
}
