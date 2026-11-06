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
    property string visualGeneration: "—"
    property double visualLastSequence: 0


    Connections {
        target: simulation

        function onTopologyVisualChanged() {
            topologyCanvas.requestPaint()
        }

        function onSpikeEventsChanged() {
            var generation = simulation.spikeEventsGeneration
            if (window.visualGeneration !== generation) {
                window.visualGeneration = generation
                window.visualLastSequence = 0
                topologyCanvas.activeNeuronTimes = ({})
            }
            var events = simulation.spikeEvents
            var active = Object.assign({}, topologyCanvas.activeNeuronTimes)
            var now = Date.now()
            for (var i = 0; i < events.length; ++i) {
                var sequence = Number(events[i].sequence)
                if (sequence <= window.visualLastSequence) continue
                window.visualLastSequence = sequence
                active[String(events[i].neuron_id)] = now
            }
            topologyCanvas.activeNeuronTimes = active
            topologyCanvas.requestPaint()
        }

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

    Timer {
        interval: 50
        repeat: true
        running: simulation.connected
        onTriggered: {
            var active = Object.assign({}, topologyCanvas.activeNeuronTimes)
            var now = Date.now()
            var changed = false
            for (var id in active) {
                if (now - active[id] > 520) {
                    delete active[id]
                    changed = true
                }
            }
            if (changed) topologyCanvas.activeNeuronTimes = active
            topologyCanvas.requestPaint()
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

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 510
            radius: 14
            color: "#0c1420"
            border.color: "#26384b"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "Live CSR neural activity"
                        color: "#edf5ff"
                        font.pixelSize: 18
                        font.bold: true
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: simulation.topologyVisual.sampled_neuron_count === undefined
                              ? "Waiting for topology…" :
                                simulation.topologyVisual.sampled_neuron_count +
                                " neurons · " +
                                simulation.topologyVisual.sampled_edge_count +
                                " sampled connections"
                        color: "#91a2b8"
                        font.pixelSize: 11
                    }
                }

                Canvas {
                    id: topologyCanvas
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    contextType: "2d"

                    property var activeNeuronTimes: ({})

                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.clearRect(0, 0, width, height)
                        ctx.fillStyle = "#0a111b"
                        ctx.fillRect(0, 0, width, height)

                        var graph = simulation.topologyVisual
                        var ids = graph.neuron_ids || []
                        var edges = graph.edges || []
                        var excitatoryCount = Number(graph.excitatory_count || 0)
                        var positions = ({})
                        var eIndex = 0
                        var iIndex = 0
                        var eTotal = 0
                        var iTotal = 0
                        for (var n = 0; n < ids.length; ++n) {
                            if (Number(ids[n]) < excitatoryCount) eTotal++
                            else iTotal++
                        }

                        var goldenAngle = 2.399963229728653
                        for (var p = 0; p < ids.length; ++p) {
                            var id = Number(ids[p])
                            var isExcitatory = id < excitatoryCount
                            var index = isExcitatory ? eIndex++ : iIndex++
                            var total = isExcitatory ? eTotal : iTotal
                            var radius = Math.sqrt((index + 0.5) / Math.max(1, total))
                            var angle = index * goldenAngle
                            var centerX = isExcitatory ? width * 0.34 : width * 0.76
                            var spreadX = isExcitatory ? width * 0.265 : width * 0.17
                            var spreadY = height * 0.39
                            positions[id] = {
                                x: centerX + Math.cos(angle) * radius * spreadX,
                                y: height * 0.51 + Math.sin(angle) * radius * spreadY
                            }
                        }

                        // Draw the sampled, real CSR edges first. Weight controls
                        // opacity; active presynaptic neurons illuminate outgoing edges.
                        for (var e = 0; e < edges.length; ++e) {
                            var edge = edges[e]
                            var from = positions[Number(edge.source)]
                            var to = positions[Number(edge.target)]
                            if (from === undefined || to === undefined) continue
                            var weight = Number(edge.weight)
                            var active = activeNeuronTimes[String(edge.source)] !== undefined &&
                                Date.now() - activeNeuronTimes[String(edge.source)] < 520
                            var strength = Math.min(1, Math.abs(weight) / 0.5)
                            ctx.beginPath()
                            ctx.moveTo(from.x, from.y)
                            ctx.lineTo(to.x, to.y)
                            ctx.lineWidth = active ? 1.45 : 0.55
                            if (active)
                                ctx.strokeStyle = "rgba(112,245,201," +
                                                  (0.42 + strength * 0.5) + ")"
                            else if (weight >= 0)
                                ctx.strokeStyle = "rgba(79,163,222," +
                                                  (0.08 + strength * 0.38) + ")"
                            else
                                ctx.strokeStyle = "rgba(226,164,83," +
                                                  (0.12 + strength * 0.48) + ")"
                            ctx.stroke()
                        }

                        // Neurons flash briefly when their spike event arrives.
                        for (var v = 0; v < ids.length; ++v) {
                            var neuronId = Number(ids[v])
                            var point = positions[neuronId]
                            var lastSpike = activeNeuronTimes[String(neuronId)]
                            var age = lastSpike === undefined ? 9999 : Date.now() - lastSpike
                            var isHot = age < 520
                            var alpha = isHot ? Math.max(0, 1 - age / 520) : 0
                            ctx.beginPath()
                            ctx.arc(point.x, point.y, isHot ? 3.2 + alpha * 3.4 : 2.15,
                                    0, Math.PI * 2)
                            ctx.fillStyle = isHot
                                ? "rgba(226,255,247," + (0.75 + alpha * 0.25) + ")"
                                : (neuronId < excitatoryCount ? "#58aee5" : "#e4ad61")
                            ctx.fill()
                            if (isHot) {
                                ctx.beginPath()
                                ctx.arc(point.x, point.y, 7 + alpha * 8, 0, Math.PI * 2)
                                ctx.strokeStyle = "rgba(105,239,198," + (alpha * 0.65) + ")"
                                ctx.lineWidth = 1.2
                                ctx.stroke()
                            }
                        }

                        ctx.font = "11px sans-serif"
                        ctx.fillStyle = "#8fa4bb"
                        ctx.fillText("EXCITATORY", width * 0.25, height - 7)
                        ctx.fillText("INHIBITORY", width * 0.70, height - 7)
                        if (ids.length === 0) {
                            ctx.fillStyle = "#91a2b8"
                            ctx.fillText("Waiting for sampled CSR topology…", 18, height / 2)
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 16
                    Label { text: "● Excitatory"; color: "#58aee5"; font.pixelSize: 11 }
                    Label { text: "● Inhibitory"; color: "#e4ad61"; font.pixelSize: 11 }
                    Label { text: "━ STDP-strengthened / positive"; color: "#65b8e9"; font.pixelSize: 11 }
                    Label { text: "━ Inhibitory"; color: "#e4ad61"; font.pixelSize: 11 }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: simulation.topologyVisual.synapse_count === undefined
                              ? "" : "Full graph: " + simulation.topologyVisual.synapse_count +
                                    " synapses"
                        color: "#71839a"
                        font.pixelSize: 10
                    }
                }
            }
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
