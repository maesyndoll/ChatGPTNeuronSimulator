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

    // Bounded CPU-side history; samples arrive with existing telemetry.
    property var populationRateHistory: []
    property int populationRateHistoryLimit: 180

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
            columns: 3
            columnSpacing: 12
            rowSpacing: 12

            MetricCard {
                Layout.fillWidth: true
                title: "Completed steps"
                value: simulation.completedSteps
                unit: "simulation steps"
            }

            MetricCard {
                Layout.fillWidth: true
                title: "Excitatory firing rate"
                value: simulation.excitatoryRateHz.toFixed(3)
                unit: "Hz"
            }

            MetricCard {
                Layout.fillWidth: true
                title: "Inhibitory firing rate"
                value: simulation.inhibitoryRateHz.toFixed(3)
                unit: "Hz"
            }

            MetricCard {
                Layout.fillWidth: true
                title: "Excitatory spikes"
                value: simulation.excitatorySpikes
                unit: "cumulative"
            }

            MetricCard {
                Layout.fillWidth: true
                title: "Inhibitory spikes"
                value: simulation.inhibitorySpikes
                unit: "cumulative"
            }

            MetricCard {
                Layout.fillWidth: true
                title: "Mean membrane voltage"
                value: simulation.meanVoltageMv.toFixed(3)
                unit: "mV"
            }
        }

        Rectangle {
            Layout.fillWidth: true
            radius: 10
            color: "#192433"
            border.color: "#2b3a4d"
            implicitHeight: 450

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true

                    Label {
                        text: "Live spike raster"
                        color: "#f0f4fa"
                        font.pixelSize: 15
                        font.bold: true
                    }

                    Item { Layout.fillWidth: true }

                    Button {
                        text: "Refresh events"
                        enabled: simulation.connected
                        onClicked: simulation.refreshSpikeEvents()
                    }
                }

                Canvas {
                    id: spikeRaster
                    Layout.fillWidth: true
                    Layout.preferredHeight: 320
                    Layout.minimumHeight: 240
                    antialiasing: false

                    Connections {
                        target: simulation
                        function onSpikeEventsChanged() {
                            spikeRaster.requestPaint()
                        }
                        function onConfigurationChanged() {
                            spikeRaster.requestPaint()
                        }
                    }

                    onPaint: {
                        var ctx = getContext("2d")
                        var w = width
                        var h = height

                        ctx.clearRect(0, 0, w, h)
                        ctx.fillStyle = "#101722"
                        ctx.fillRect(0, 0, w, h)

                        var left = 58
                        var right = 14
                        var top = 14
                        var bottom = 30
                        var plotWidth = Math.max(1, w - left - right)
                        var plotHeight = Math.max(1, h - top - bottom)

                        var events = simulation.spikeEvents
                        var params = simulation.requestedParameters || {}
                        var simParams = params["simulation"] || {}

                        var excitatoryCount =
                                Number(simParams["excitatory_count"] || 0)
                        var inhibitoryCount =
                                Number(simParams["inhibitory_count"] || 0)
                        var totalNeurons = excitatoryCount + inhibitoryCount

                        // Use simulation progress for a stable time axis.
                        var maxStep = Number(simulation.completedSteps)
                        var lastEventStep = events.length > 0
                                ? Number(events[events.length - 1].step) : 0

                        if (!isFinite(maxStep) || maxStep <= 0)
                            maxStep = lastEventStep
                        else if (isFinite(lastEventStep))
                            maxStep = Math.max(maxStep, lastEventStep)

                        if (totalNeurons <= 0) {
                            totalNeurons = 1
                            for (var n = 0; n < events.length; ++n) {
                                var candidate = Number(events[n].neuron_id)
                                if (isFinite(candidate))
                                    totalNeurons = Math.max(
                                                totalNeurons, candidate + 1)
                            }
                        }

                        totalNeurons = Math.max(1, totalNeurons)

                        var windowSteps = 2000
                        var minStep = Math.max(0, maxStep - windowSteps)
                        var maxVisibleStep = Math.max(minStep + 1, maxStep)
                        var stepRange = maxVisibleStep - minStep

                        // Draw the axes and guides.
                        ctx.strokeStyle = "#42536a"
                        ctx.lineWidth = 1
                        ctx.beginPath()
                        ctx.moveTo(left, top)
                        ctx.lineTo(left, top + plotHeight)
                        ctx.lineTo(left + plotWidth, top + plotHeight)
                        ctx.stroke()

                        ctx.strokeStyle = "#263548"
                        ctx.beginPath()
                        for (var grid = 1; grid <= 4; ++grid) {
                            var gy = top + plotHeight * grid / 5
                            ctx.moveTo(left, gy)
                            ctx.lineTo(left + plotWidth, gy)
                        }
                        ctx.stroke()

                        // One visual mark per occupied pixel per population.
                        // The underlying spike-event records are not modified.
                        var pixelWidth = Math.max(1, Math.ceil(plotWidth))
                        var excitatoryPixels = Object.create(null)
                        var inhibitoryPixels = Object.create(null)

                        for (var i = 0; i < events.length; ++i) {
                            var eventStep = Number(events[i].step)
                            var eventNeuron = Number(events[i].neuron_id)

                            if (!isFinite(eventStep) ||
                                !isFinite(eventNeuron) ||
                                eventStep < minStep ||
                                eventStep > maxVisibleStep ||
                                eventNeuron < 0 ||
                                eventNeuron >= totalNeurons) {
                                continue
                            }

                            var px = Math.floor(
                                        ((eventStep - minStep) / stepRange) *
                                        plotWidth)
                            var py = Math.floor(
                                        (eventNeuron / totalNeurons) *
                                        plotHeight)

                            px = Math.max(0, Math.min(pixelWidth - 1, px))
                            py = Math.max(0, Math.min(
                                             Math.ceil(plotHeight) - 1, py))

                            var pixelKey = py * pixelWidth + px

                            if (eventNeuron < excitatoryCount)
                                excitatoryPixels[pixelKey] = true
                            else
                                inhibitoryPixels[pixelKey] = true
                        }

                        // Batch all excitatory marks into one fill operation.
                        ctx.fillStyle = "#54b8a5"
                        ctx.beginPath()
                        for (var e in excitatoryPixels) {
                            if (Object.prototype.hasOwnProperty.call(
                                        excitatoryPixels, e)) {
                                var eKey = Number(e)
                                ctx.rect(
                                            left + (eKey % pixelWidth),
                                            top + Math.floor(eKey / pixelWidth),
                                            1, 1)
                            }
                        }
                        ctx.fill()

                        // Batch all inhibitory marks into one fill operation.
                        ctx.fillStyle = "#e5a45f"
                        ctx.beginPath()
                        for (var inh in inhibitoryPixels) {
                            if (Object.prototype.hasOwnProperty.call(
                                        inhibitoryPixels, inh)) {
                                var iKey = Number(inh)
                                ctx.rect(
                                            left + (iKey % pixelWidth),
                                            top + Math.floor(iKey / pixelWidth),
                                            1, 1)
                            }
                        }
                        ctx.fill()

                        // Axis labels.
                        ctx.fillStyle = "#9fb0c5"
                        ctx.font = "11px sans-serif"
                        ctx.textAlign = "center"
                        ctx.fillText("Simulation step",
                                     left + plotWidth / 2, h - 5)
                        ctx.fillText(String(Math.round(minStep)),
                                     left, top + plotHeight + 17)
                        ctx.fillText(String(Math.round(maxVisibleStep)),
                                     left + plotWidth, top + plotHeight + 17)

                        ctx.textAlign = "right"
                        ctx.fillText("Neuron ID", left - 8, top + 8)
                        ctx.fillText("0", left - 8, top + plotHeight)
                        ctx.fillText(String(totalNeurons - 1),
                                     left - 8, top + 19)

                        if (events.length === 0) {
                            ctx.textAlign = "center"
                            ctx.fillStyle = "#71839a"
                            ctx.font = "13px sans-serif"
                            ctx.fillText(
                                        "Waiting for recorded GPU spike events...",
                                        left + plotWidth / 2,
                                        top + plotHeight / 2)
                        }
                    }
                }

                RowLayout {
                    spacing: 16

                    Label {
                        text: "● Excitatory"
                        color: "#54b8a5"
                        font.pixelSize: 12
                    }

                    Label {
                        text: "● Inhibitory"
                        color: "#e5a45f"
                        font.pixelSize: 12
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: "Last 2,000 steps"
                        color: "#91a2b8"
                        font.pixelSize: 11
                    }
                }

                Label {
                    Layout.fillWidth: true
                    color: "#cbd7e5"
                    text: "Retained in dashboard: " +
                          simulation.spikeEvents.length +
                          "    Total recorded: " + simulation.spikeEventCount
                    wrapMode: Text.Wrap
                }

                Label {
                    Layout.fillWidth: true
                    color: Number(simulation.spikeEventsDropped) > 0
                           ? "#e5bd78" : "#8de0ca"
                    text: "Events no longer retained (GPU ring / history): " +
                          simulation.spikeEventsDropped +
                          "    Generation: " +
                          simulation.spikeEventsGeneration
                    wrapMode: Text.Wrap
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            radius: 10
            color: "#192433"
            border.color: "#2b3a4d"
            implicitHeight: 260

            Connections {
                target: simulation

                function onTelemetryChanged() {
                    var next = window.populationRateHistory.slice()
                    next.push({
                        excitatory: Number(simulation.excitatoryRateHz),
                        inhibitory: Number(simulation.inhibitoryRateHz),
                        steps: String(simulation.completedSteps)
                    })

                    if (next.length > window.populationRateHistoryLimit)
                        next.splice(
                            0,
                            next.length - window.populationRateHistoryLimit
                        )

                    window.populationRateHistory = next
                    populationRateChart.requestPaint()
                }
            }

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true

                    Label {
                        text: "Population firing rate"
                        color: "#f0f4fa"
                        font.pixelSize: 15
                        font.bold: true
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: "Hz · telemetry samples"
                        color: "#91a2b8"
                        font.pixelSize: 11
                    }
                }

                Canvas {
                    id: populationRateChart
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    Layout.minimumHeight: 140
                    antialiasing: false

                    onPaint: {
                        var ctx = getContext("2d")
                        var w = width
                        var h = height
                        ctx.clearRect(0, 0, w, h)
                        ctx.fillStyle = "#101722"
                        ctx.fillRect(0, 0, w, h)

                        var left = 48
                        var right = 12
                        var top = 12
                        var bottom = 22
                        var plotWidth = Math.max(1, w - left - right)
                        var plotHeight = Math.max(1, h - top - bottom)
                        var history = window.populationRateHistory

                        var maximum = 1
                        for (var i = 0; i < history.length; ++i) {
                            var e = Number(history[i].excitatory)
                            var inh = Number(history[i].inhibitory)
                            if (isFinite(e))
                                maximum = Math.max(maximum, e)
                            if (isFinite(inh))
                                maximum = Math.max(maximum, inh)
                        }
                        maximum *= 1.1

                        ctx.strokeStyle = "#42536a"
                        ctx.lineWidth = 1
                        ctx.beginPath()
                        ctx.moveTo(left, top)
                        ctx.lineTo(left, top + plotHeight)
                        ctx.lineTo(left + plotWidth, top + plotHeight)
                        ctx.stroke()

                        ctx.font = "10px sans-serif"
                        ctx.fillStyle = "#91a2b8"
                        ctx.textAlign = "right"

                        for (var grid = 0; grid <= 4; ++grid) {
                            var fraction = grid / 4
                            var y = top + plotHeight * (1 - fraction)
                            ctx.strokeStyle = "#263548"
                            ctx.beginPath()
                            ctx.moveTo(left, y)
                            ctx.lineTo(left + plotWidth, y)
                            ctx.stroke()
                            ctx.fillText(
                                (maximum * fraction).toFixed(1),
                                left - 7,
                                y + 3
                            )
                        }

                        function drawSeries(key, color) {
                            if (history.length === 0)
                                return

                            ctx.strokeStyle = color
                            ctx.fillStyle = color
                            ctx.lineWidth = 1.5
                            ctx.beginPath()

                            for (var n = 0; n < history.length; ++n) {
                                var value = Number(history[n][key])
                                if (!isFinite(value))
                                    continue

                                var x = left + (
                                    history.length <= 1 ? 0 :
                                    n / (history.length - 1) * plotWidth
                                )
                                var y = top + plotHeight * (
                                    1 - Math.max(0, value) / maximum
                                )

                                if (n === 0)
                                    ctx.moveTo(x, y)
                                else
                                    ctx.lineTo(x, y)
                            }

                            ctx.stroke()

                            if (history.length === 1) {
                                var only = Number(history[0][key])
                                if (isFinite(only)) {
                                    ctx.beginPath()
                                    ctx.arc(
                                        left,
                                        top + plotHeight *
                                            (1 - Math.max(0, only) / maximum),
                                        2.5, 0, Math.PI * 2
                                    )
                                    ctx.fill()
                                }
                            }
                        }

                        drawSeries("excitatory", "#54b8a5")
                        drawSeries("inhibitory", "#e5a45f")

                        if (history.length === 0) {
                            ctx.fillStyle = "#71839a"
                            ctx.textAlign = "center"
                            ctx.font = "12px sans-serif"
                            ctx.fillText(
                                "Waiting for telemetry samples...",
                                left + plotWidth / 2,
                                top + plotHeight / 2
                            )
                        }

                        ctx.fillStyle = "#91a2b8"
                        ctx.textAlign = "center"
                        ctx.font = "10px sans-serif"
                        ctx.fillText(
                            "Recent telemetry samples (oldest → newest)",
                            left + plotWidth / 2,
                            h - 4
                        )
                    }
                }

                RowLayout {
                    spacing: 18

                    Label {
                        text: "● Excitatory"
                        color: "#54b8a5"
                        font.pixelSize: 12
                    }

                    Label {
                        text: "● Inhibitory"
                        color: "#e5a45f"
                        font.pixelSize: 12
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: window.populationRateHistory.length +
                              " / " + window.populationRateHistoryLimit +
                              " samples"
                        color: "#91a2b8"
                        font.pixelSize: 11
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            radius: 10
            color: "#192433"
            border.color: "#2b3a4d"
            implicitHeight: 58

            RowLayout {
                anchors.fill: parent
                anchors.margins: 16

                Label {
                    text: "Numerical state"
                    color: "#cbd7e5"
                    font.pixelSize: 14
                    font.bold: true
                }

                Item { Layout.fillWidth: true }

                Label {
                    text: simulation.finiteState ? "Finite" : "Not yet verified"
                    color: simulation.finiteState ? "#8de0ca" : "#d5b77b"
                    font.pixelSize: 14
                    font.bold: true
                }
            }
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

    component MetricCard: Rectangle {
        id: card

        property string title: ""
        property string value: "—"
        property string unit: ""

        radius: 12
        color: "#192433"
        border.color: "#2b3a4d"
        implicitHeight: 122

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 16
            spacing: 7

            Label {
                text: card.title
                color: "#9fb0c5"
                font.pixelSize: 13
            }

            Label {
                Layout.fillWidth: true
                text: card.value
                color: "#f0f4fa"
                font.pixelSize: 24
                font.bold: true
                elide: Text.ElideRight
            }

            Label {
                text: card.unit
                color: "#71839a"
                font.pixelSize: 11
            }
        }
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
