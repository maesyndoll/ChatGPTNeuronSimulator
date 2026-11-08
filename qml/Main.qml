import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    visible: true
    width: 1120
    height: 820
    minimumWidth: 760
    minimumHeight: 680
    title: "ChatGPTNeuronSimulator"
    color: "#101722"

    palette.window: "#101722"
    palette.windowText: "#e8eef7"
    palette.button: "#26364a"
    palette.buttonText: "#e8eef7"
    palette.highlight: "#54b8a5"

    property real cloudRotation: 0
    property var activeNeuronIds: ({})
    property int cloudNodeCount: 180
    property int selectedNeuronId: -1
    property var rateHistory: []
    property int historyLimit: 1200
    property bool parametersInitialized: false
    property bool pausePending: false

    // Logarithmic maps give useful resolution across orders of magnitude.
    // Connectivity has an explicit zero detent, then a logarithmic 1e-5..0.05 range.
    function logMap(t, minimum, maximum) {
        return minimum * Math.pow(maximum / minimum, Math.max(0, Math.min(1, t)))
    }
    function inverseLogMap(value, minimum, maximum) {
        var v = Math.max(minimum, Math.min(maximum, Number(value)))
        return Math.log(v / minimum) / Math.log(maximum / minimum)
    }
    function probabilityMap(t) {
        return t <= 0.0001 ? 0 : logMap(t, 0.00001, 0.05)
    }
    function inverseProbabilityMap(value) {
        var v = Number(value)
        return v <= 0 ? 0 : inverseLogMap(v, 0.00001, 0.05)
    }
    function fmt(value, digits) {
        return Number(value).toFixed(digits)
    }
    function totalNeurons() {
        var p = simulation.activeParameters || {}
        var s = p.simulation || {}
        return Math.max(1, Number(s.excitatory_count || 81920) +
                           Number(s.inhibitory_count || 10240))
    }
    function currentRate() {
        var events = simulation.spikeEvents || []
        var completed = Number(simulation.completedSteps)
        var p = simulation.activeParameters || {}
        var s = p.simulation || {}
        var dt = Number(s.dt || 0.5)
        if (!isFinite(completed) || completed <= 0 || !isFinite(dt) || dt <= 0)
            return 0
        var windowSteps = Math.min(100, Math.max(10, completed))
        var cutoff = completed - windowSteps
        var count = 0
        for (var i = events.length - 1; i >= 0; --i) {
            var step = Number(events[i].step)
            if (step < cutoff)
                break
            if (step <= completed)
                ++count
        }
        var seconds = windowSteps * dt / 1000.0
        return seconds > 0 ? count / (totalNeurons() * seconds) : 0
    }
    function recordRate() {
        var next = rateHistory.slice()
        next.push(currentRate())
        if (next.length > historyLimit)
            next.splice(0, next.length - historyLimit)
        rateHistory = next
        activityScope.requestPaint()
        activityGauge.requestPaint()
    }
    function maximumRate() {
        var maximum = 1
        for (var i = 0; i < rateHistory.length; ++i)
            maximum = Math.max(maximum, Number(rateHistory[i]) || 0)
        return maximum * 1.15
    }
    function refreshActiveNeuronIds() {
        var active = ({})
        var events = simulation.spikeEvents || []
        var completed = Number(simulation.completedSteps)
        var cutoff = Math.max(0, completed - 12)
        var total = totalNeurons()
        for (var i = Math.max(0, events.length - 12000); i < events.length; ++i) {
            var step = Number(events[i].step)
            if (step >= cutoff && step <= completed) {
                // The scene renders a small representative sample of the full
                // population. Bin every real spike into its displayed sample
                // slot so activity is visible even when that exact neuron is
                // not one of the sampled IDs.
                var neuronId = Number(events[i].neuron_id)
                var slot = Math.max(0, Math.min(cloudNodeCount - 1,
                    Math.floor(neuronId * cloudNodeCount / total)))
                active[slot] = true
            }
        }
        activeNeuronIds = active
    }
    function syncKnobs() {
        var requested = simulation.requestedParameters || {}
        var s = requested.simulation || {}
        var t = requested.topology || {}
        dtKnob.value = inverseLogMap(Number(s.dt || 0.5), 0.001, 5.0)
        thresholdKnob.value = Math.max(0, Math.min(1,
            (Number(s.threshold || 30) + 100) / 200))
        peeKnob.value = inverseProbabilityMap(Number(t.p_ee || 0))
        peiKnob.value = inverseProbabilityMap(Number(t.p_ei || 0))
        pieKnob.value = inverseProbabilityMap(Number(t.p_ie || 0))
        piiKnob.value = inverseProbabilityMap(Number(t.p_ii || 0))
        parametersInitialized = true
    }
    function applyKnobs() {
        simulation.applyParameters({
            simulation: {
                dt: logMap(dtKnob.value, 0.001, 5.0),
                threshold: -100.0 + thresholdKnob.value * 200.0
            },
            topology: {
                p_ee: probabilityMap(peeKnob.value),
                p_ei: probabilityMap(peiKnob.value),
                p_ie: probabilityMap(pieKnob.value),
                p_ii: probabilityMap(piiKnob.value)
            }
        })
    }

    Connections {
        target: simulation
        function onConfigurationChanged() {
            window.syncKnobs()
        }
        function onSimulationStatusChanged() {
            if (simulation.simulationStatus === "paused")
                window.pausePending = false
        }
        function onErrorMessageChanged() {
            if (simulation.errorMessage.length > 0)
                window.pausePending = false
        }
        function onTelemetryChanged() {
            window.recordRate()
            window.refreshActiveNeuronIds()
        }
        function onSpikeEventsChanged() {
            window.refreshActiveNeuronIds()
            window.recordRate()
        }
    }

    ScrollView {
        anchors.fill: parent
        anchors.margins: 22
        clip: true
        contentWidth: availableWidth
        ColumnLayout {
            width: parent.width
            spacing: 14

            RowLayout {
                Layout.fillWidth: true
                spacing: 12
                ColumnLayout {
                    spacing: 2
                    Label {
                        text: "ChatGPTNeuronSimulator"
                        color: "#f0f4fa"
                        font.pixelSize: 24
                        font.bold: true
                    }
                    Label {
                        text: "Live neural activity · CUDA"
                        color: "#91a2b8"
                        font.pixelSize: 12
                    }
                }
                Item { Layout.fillWidth: true }
                Label {
                    text: simulation.connectionStatus
                    color: simulation.connected ? "#8de0ca" : "#f0a7ad"
                    font.bold: true
                }
                Button {
                    text: simulation.connected ? "Disconnect" : "Connect"
                    onClicked: simulation.connected
                              ? simulation.disconnectFromServer()
                              : simulation.connectToServer()
                }
                Button {
                    text: simulation.simulationStatus === "paused" ? "Resume" : "Start"
                    enabled: simulation.connected &&
                             simulation.simulationStatus !== "running"
                    onClicked: simulation.startSimulation()
                }
                Button {
                    text: window.pausePending ? "Pausing…" : "Pause"
                    enabled: simulation.connected &&
                             simulation.simulationStatus === "running" &&
                             !window.pausePending
                    onClicked: {
                        window.pausePending = true
                        simulation.pauseSimulation()
                    }
                }
            }

            Label {
                Layout.fillWidth: true
                visible: simulation.errorMessage.length > 0
                text: simulation.errorMessage
                color: "#ff9b9b"
                wrapMode: Text.Wrap
            }

            // The only time-series display: one needle and its corresponding trace.
            Rectangle {
                Layout.fillWidth: true
                radius: 12
                color: "#192433"
                border.color: "#2b3a4d"
                implicitHeight: 270
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 8
                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "Population activity"
                            color: "#f0f4fa"
                            font.pixelSize: 16
                            font.bold: true
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: "recent per-neuron firing rate · Hz"
                            color: "#91a2b8"
                            font.pixelSize: 11
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        spacing: 12
                        Canvas {
                            id: activityGauge
                            Layout.preferredWidth: 210
                            Layout.fillHeight: true
                            Layout.minimumHeight: 180
                            onPaint: {
                                var ctx = getContext("2d")
                                var w = width, h = height
                                ctx.clearRect(0, 0, w, h)
                                var cx = w / 2, cy = h * 0.79
                                var r = Math.min(w * 0.40, h * 0.64)
                                var value = window.rateHistory.length
                                          ? Number(window.rateHistory[window.rateHistory.length - 1]) : 0
                                var maximum = window.maximumRate()
                                var fraction = Math.max(0, Math.min(1, value / maximum))
                                ctx.lineWidth = 11
                                ctx.lineCap = "round"
                                ctx.strokeStyle = "#29394d"
                                ctx.beginPath()
                                ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI)
                                ctx.stroke()
                                ctx.strokeStyle = "#54b8a5"
                                ctx.beginPath()
                                ctx.arc(cx, cy, r, Math.PI, Math.PI + Math.PI * fraction)
                                ctx.stroke()
                                var angle = Math.PI + Math.PI * fraction
                                ctx.strokeStyle = "#f0f4fa"
                                ctx.lineWidth = 3
                                ctx.beginPath()
                                ctx.moveTo(cx, cy)
                                ctx.lineTo(cx + Math.cos(angle) * r * 0.82,
                                           cy + Math.sin(angle) * r * 0.82)
                                ctx.stroke()
                                ctx.fillStyle = "#f0f4fa"
                                ctx.beginPath()
                                ctx.arc(cx, cy, 5, 0, Math.PI * 2)
                                ctx.fill()
                                ctx.textAlign = "center"
                                ctx.fillStyle = "#91a2b8"
                                ctx.font = "10px sans-serif"
                                ctx.fillText("LIVE RATE", cx, 15)
                                ctx.fillStyle = "#f0f4fa"
                                ctx.font = "bold 20px sans-serif"
                                ctx.fillText(value.toFixed(3) + " Hz", cx, cy - 12)
                                ctx.font = "10px sans-serif"
                                ctx.fillStyle = "#91a2b8"
                                ctx.fillText("0", cx - r, cy + 17)
                                ctx.fillText(maximum.toFixed(2), cx + r, cy + 17)
                            }
                        }
                        Canvas {
                            id: activityScope
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            Layout.minimumHeight: 180
                            onPaint: {
                                var ctx = getContext("2d")
                                var w = width, h = height
                                ctx.clearRect(0, 0, w, h)
                                ctx.fillStyle = "#0b1421"
                                ctx.fillRect(0, 0, w, h)
                                var left = 42, right = 10, top = 12, bottom = 24
                                var pw = Math.max(1, w - left - right)
                                var ph = Math.max(1, h - top - bottom)
                                var values = window.rateHistory
                                var maximum = window.maximumRate()
                                ctx.strokeStyle = "#29394d"
                                ctx.lineWidth = 1
                                for (var g = 0; g <= 4; ++g) {
                                    var y = top + ph * g / 4
                                    ctx.beginPath()
                                    ctx.moveTo(left, y)
                                    ctx.lineTo(left + pw, y)
                                    ctx.stroke()
                                }
                                ctx.strokeStyle = "#54cbb5"
                                ctx.lineWidth = 2
                                ctx.beginPath()
                                for (var i = 0; i < values.length; ++i) {
                                    var x = left + (values.length <= 1 ? 0 :
                                        i * pw / (values.length - 1))
                                    var yv = top + ph * (1 -
                                        Math.max(0, Number(values[i])) / maximum)
                                    if (i === 0) ctx.moveTo(x, yv)
                                    else ctx.lineTo(x, yv)
                                }
                                ctx.stroke()
                                ctx.fillStyle = "#91a2b8"
                                ctx.font = "10px sans-serif"
                                ctx.fillText("OLDEST", left, h - 5)
                                ctx.textAlign = "right"
                                ctx.fillText("NOW", left + pw, h - 5)
                                if (values.length < 2) {
                                    ctx.textAlign = "center"
                                    ctx.fillText("Waiting for live spike samples",
                                                 left + pw / 2, top + ph / 2)
                                }
                            }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "Steps: " + simulation.completedSteps
                            color: "#91a2b8"
                            font.pixelSize: 10
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: window.rateHistory.length + " / " +
                                  window.historyLimit + " samples"
                            color: "#91a2b8"
                            font.pixelSize: 10
                        }
                    }
                }
            }

            // Compact 3D-style topology field; nodes brighten on recent GPU spikes.
            Rectangle {
                Layout.fillWidth: true
                radius: 12
                color: "#121d2c"
                border.color: "#30435c"
                implicitHeight: 350
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 8
                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "3D network topology"
                            color: "#f0f4fa"
                            font.pixelSize: 16
                            font.bold: true
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: "E · teal     I · amber     active · bright"
                            color: "#91a2b8"
                            font.pixelSize: 10
                        }
                    }
                    Item {
                        id: neuralCloud
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        property var nodePositions: ({})
                        Rectangle { anchors.fill: parent; radius: 8; color: "#081321" }
                        Canvas {
                            id: connectomeCanvas
                            anchors.fill: parent
                            onPaint: {
                                var ctx = getContext("2d")
                                ctx.clearRect(0, 0, width, height)
                                var graph = simulation.topologySnapshot || ({})
                                var nodes = graph.nodes || []
                                var edges = graph.edges || []
                                var positions = ({})
                                var active = ({})
                                var events = simulation.spikeEvents || []
                                var completed = Number(simulation.completedSteps)
                                for (var e = Math.max(0, events.length - 6000); e < events.length; ++e) {
                                    var ev = events[e]
                                    if (Number(ev.step) >= completed - 8)
                                        active[String(ev.neuron_id)] = true
                                }
                                for (var n = 0; n < nodes.length; ++n) {
                                    var z = 1 - 2 * (n + 0.5) / Math.max(1, nodes.length)
                                    var ring = Math.sqrt(Math.max(0, 1 - z * z))
                                    var angle = n * 2.399963229728653
                                    var perspective = 0.65 + (z + 1) * 0.18
                                    positions[String(nodes[n])] = {
                                        x: width * 0.5 + Math.cos(angle) * ring * width * 0.39 * perspective,
                                        y: height * 0.48 + Math.sin(angle) * ring * height * 0.40 * perspective
                                    }
                                }
                                for (var i = 0; i < edges.length; ++i) {
                                    var edge = edges[i]
                                    var a = positions[String(edge.source)]
                                    var b = positions[String(edge.target)]
                                    if (!a || !b) continue
                                    var firing = active[String(edge.source)] === true
                                    ctx.beginPath()
                                    ctx.moveTo(a.x, a.y)
                                    ctx.lineTo(b.x, b.y)
                                    ctx.strokeStyle = firing ? "rgba(245,250,255,0.95)" :
                                        (edge.excitatory ? "rgba(67,199,177,0.28)" : "rgba(239,157,86,0.30)")
                                    ctx.lineWidth = firing ? 1.8 : 0.55 + Math.min(0.8, Math.abs(Number(edge.weight)) * 0.12)
                                    ctx.stroke()
                                }
                                for (var j = 0; j < nodes.length; ++j) {
                                    var id = nodes[j]
                                    var point = positions[String(id)]
                                    var isActive = active[String(id)] === true
                                    var isSelected = Number(id) === window.selectedNeuronId
                                    ctx.beginPath()
                                    ctx.arc(point.x, point.y, isSelected ? 5.8 : (isActive ? 4.8 : 3.0), 0, Math.PI * 2)
                                    ctx.fillStyle = isActive ? "#ffffff" :
                                        (Number(id) < Number(graph.excitatory_count) ? "#54cbb5" : "#f0a45e")
                                    ctx.fill()
                                    if (isSelected) {
                                        ctx.beginPath()
                                        ctx.arc(point.x, point.y, 9, 0, Math.PI * 2)
                                        ctx.strokeStyle = "#ffffff"
                                        ctx.lineWidth = 1.5
                                        ctx.stroke()
                                    }
                                }
                                neuralCloud.nodePositions = positions
                            }
                            Connections { target: simulation; function onTopologyChanged() { connectomeCanvas.requestPaint() } }
                            Connections { target: simulation; function onSpikeEventsChanged() { connectomeCanvas.requestPaint() } }
                            Connections { target: simulation; function onTelemetryChanged() { connectomeCanvas.requestPaint() } }
                            Connections { target: window; function onSelectedNeuronIdChanged() { connectomeCanvas.requestPaint() } }
                        }
                        MouseArea {
                            anchors.fill: parent
                            onClicked: function(mouse) {
                                var positions = neuralCloud.nodePositions
                                var nodes = simulation.topologySnapshot.nodes || []
                                var best = -1
                                var bestDistance = 14 * 14
                                for (var i = 0; i < nodes.length; ++i) {
                                    var point = positions[String(nodes[i])]
                                    if (!point) continue
                                    var dx = point.x - mouse.x
                                    var dy = point.y - mouse.y
                                    var distance = dx * dx + dy * dy
                                    if (distance < bestDistance) {
                                        bestDistance = distance
                                        best = Number(nodes[i])
                                    }
                                }
                                if (best >= 0) window.selectedNeuronId = best
                            }
                        }
                        Label {
                            anchors.left: parent.left
                            anchors.bottom: parent.bottom
                            anchors.margins: 8
                            text: (simulation.topologySnapshot.rendered_neurons || 0) + " rendered neurons · " +
                                  (simulation.topologySnapshot.rendered_synapses || 0) + " actual CSR synapses · step " +
                                  (simulation.topologySnapshot.completed_steps || "0")
                            color: "#91a2b8"
                            font.pixelSize: 10
                        }
                    }
                }
                NumberAnimation {
                    target: window
                    property: "cloudRotation"
                    from: 0
                    to: 360
                    duration: 45000
                    loops: Animation.Infinite
                    running: true
                }
            }

            Rectangle {
                Layout.fillWidth: true
                radius: 12
                color: "#192433"
                border.color: "#2b3a4d"
                implicitHeight: tissueInspector.implicitHeight + 28
                ColumnLayout {
                    id: tissueInspector
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 8
                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: "Cyberneural tissue"; color: "#f0f4fa"; font.pixelSize: 16; font.bold: true }
                        Item { Layout.fillWidth: true }
                        Label { text: "revision " + simulation.tissueRevision; color: "#91a2b8"; font.pixelSize: 10 }
                        Button { text: "Refresh"; enabled: simulation.connected; onClicked: simulation.refreshTissue() }
                    }
                    Label {
                        text: (simulation.tissueSnapshot.modules || []).length + " modules  ·  " +
                              (simulation.tissueSnapshot.adapters || []).length + " adapters"
                        color: "#8de0ca"
                        font.pixelSize: 12
                    }
                    TextArea {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 190
                        readOnly: true
                        wrapMode: TextEdit.Wrap
                        color: "#a9bbcf"
                        background: Rectangle { color: "#101b29"; radius: 8 }
                        text: JSON.stringify(simulation.tissueSnapshot, null, 2)
                        placeholderText: "Connect to inspect the tissue model"
                        font.family: "monospace"
                        font.pixelSize: 10
                    }
                }
            }
            Timer {
                interval: 1000
                repeat: true
                running: simulation.connected
                onTriggered: { simulation.refreshTissue(); simulation.refreshTopology() }
            }

            // Each Dial is a normalized control; the displayed number is the
            // physical parameter value sent to the server, not the dial fraction.
            Rectangle {
                Layout.fillWidth: true
                radius: 12
                color: "#192433"
                border.color: "#2b3a4d"
                implicitHeight: stimulationColumn.implicitHeight + 24
                ColumnLayout {
                    id: stimulationColumn
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 8
                    Label { text: "Neural intervention"; color: "#f0f4fa"; font.pixelSize: 15; font.bold: true }
                    Label {
                        text: selectedNeuronId >= 0 ? "Selected neuron " + selectedNeuronId : "Click a neuron in the 3D field to select it"
                        color: "#8de0ca"; font.pixelSize: 12
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: "Current"; color: "#a9bbcf" }
                        SpinBox { id: stimulusAmplitude; from: -500; to: 500; value: 100; editable: true }
                        Label { text: "Duration (steps)"; color: "#a9bbcf" }
                        SpinBox { id: stimulusDuration; from: 1; to: 100000; value: 20; editable: true }
                        Button {
                            text: "Inject pulse"
                            enabled: simulation.connected && simulation.simulationStatus === "running" && selectedNeuronId >= 0
                            onClicked: simulation.stimulateNeurons([selectedNeuronId], stimulusAmplitude.value, stimulusDuration.value)
                        }
                    }
                    Label {
                        text: "External input is applied asynchronously at a simulation batch boundary."
                        color: "#71839a"; font.pixelSize: 10
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                radius: 12
                color: "#192433"
                border.color: "#2b3a4d"
                implicitHeight: knobsColumn.implicitHeight + 28
                ColumnLayout {
                    id: knobsColumn
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 10
                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "Model parameters"
                            color: "#f0f4fa"
                            font.pixelSize: 16
                            font.bold: true
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: simulation.hasPendingReinitialization
                                  ? "Topology changes need reinitialization"
                                  : "Live parameters synchronized"
                            color: simulation.hasPendingReinitialization
                                  ? "#e5bd78" : "#8de0ca"
                            font.pixelSize: 10
                        }
                    }
                    GridLayout {
                        Layout.fillWidth: true
                        columns: 3
                        columnSpacing: 12
                        rowSpacing: 8

                        ColumnLayout {
                            Layout.fillWidth: true
                            Label { text: "Time step · ms"; color: "#a9bbcf"; font.pixelSize: 12 }
                            Dial {
                                id: dtKnob
                                Layout.alignment: Qt.AlignHCenter
                                from: 0; to: 1; value: 0.62
                                stepSize: 0.001
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: fmt(logMap(dtKnob.value, 0.001, 5.0), 4)
                                color: "#f0f4fa"; font.bold: true
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: "log scale · 0.001–5"
                                color: "#71839a"; font.pixelSize: 9
                            }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            Label { text: "Spike threshold · mV"; color: "#a9bbcf"; font.pixelSize: 12 }
                            Dial {
                                id: thresholdKnob
                                Layout.alignment: Qt.AlignHCenter
                                from: 0; to: 1; value: 0.65
                                stepSize: 0.001
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: fmt(-100 + thresholdKnob.value * 200, 2)
                                color: "#f0f4fa"; font.bold: true
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: "linear · −100–100"
                                color: "#71839a"; font.pixelSize: 9
                            }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            Label { text: "E → E probability"; color: "#a9bbcf"; font.pixelSize: 12 }
                            Dial {
                                id: peeKnob
                                Layout.alignment: Qt.AlignHCenter
                                from: 0; to: 1; value: 0.5
                                stepSize: 0.001
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: fmt(probabilityMap(peeKnob.value), 6)
                                color: "#f0f4fa"; font.bold: true
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: "log gain · 0–0.05"
                                color: "#71839a"; font.pixelSize: 9
                            }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            Label { text: "E → I probability"; color: "#a9bbcf"; font.pixelSize: 12 }
                            Dial {
                                id: peiKnob
                                Layout.alignment: Qt.AlignHCenter
                                from: 0; to: 1; value: 0.57
                                stepSize: 0.001
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: fmt(probabilityMap(peiKnob.value), 6)
                                color: "#f0f4fa"; font.bold: true
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: "log gain · 0–0.05"
                                color: "#71839a"; font.pixelSize: 9
                            }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            Label { text: "I → E probability"; color: "#a9bbcf"; font.pixelSize: 12 }
                            Dial {
                                id: pieKnob
                                Layout.alignment: Qt.AlignHCenter
                                from: 0; to: 1; value: 0.57
                                stepSize: 0.001
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: fmt(probabilityMap(pieKnob.value), 6)
                                color: "#f0f4fa"; font.bold: true
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: "log gain · 0–0.05"
                                color: "#71839a"; font.pixelSize: 9
                            }
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            Label { text: "I → I probability"; color: "#a9bbcf"; font.pixelSize: 12 }
                            Dial {
                                id: piiKnob
                                Layout.alignment: Qt.AlignHCenter
                                from: 0; to: 1; value: 0.57
                                stepSize: 0.001
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: fmt(probabilityMap(piiKnob.value), 6)
                                color: "#f0f4fa"; font.bold: true
                            }
                            Label {
                                Layout.alignment: Qt.AlignHCenter
                                text: "log gain · 0–0.05"
                                color: "#71839a"; font.pixelSize: 9
                            }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            Layout.fillWidth: true
                            text: simulation.statusMessage
                            color: "#91a2b8"
                            font.pixelSize: 10
                            wrapMode: Text.Wrap
                        }
                        Button {
                            text: "Apply"
                            enabled: simulation.connected && window.parametersInitialized
                            onClicked: window.applyKnobs()
                        }
                        Button {
                            text: "Rebuild topology"
                            enabled: simulation.connected &&
                                     simulation.hasPendingReinitialization
                            onClicked: simulation.reinitializeSimulation()
                        }
                    }
                }
            }
        }
    }
}
