import QtQuick
import QtQuick.Controls

ApplicationWindow {
    id: window
    visible: true
    width: 1280
    height: 900
    minimumWidth: 720
    minimumHeight: 520
    title: "Neural Field"
    color: "#020409"

    property real rotationY: 0.35
    property real rotationX: -0.18
    property real cameraDistance: 3.15
    property real dragX: 0
    property real dragY: 0
    property var lastSpikeByNeuron: ({})
    property double lastSeenSequence: 0
    property string lastGeneration: ""
    property double clockAnchorStep: 0
    property double clockAnchorMs: Date.now()
    property double stepsPerMillisecond: 0

    function updateClock(step) {
        if (!isFinite(step) || step < clockAnchorStep)
            return
        var now = Date.now()
        var elapsed = now - clockAnchorMs
        var advanced = step - clockAnchorStep
        if (elapsed > 0 && advanced > 0) {
            var measured = advanced / elapsed
            stepsPerMillisecond = stepsPerMillisecond > 0
                ? stepsPerMillisecond * 0.65 + measured * 0.35
                : measured
        }
        clockAnchorStep = step
        clockAnchorMs = now
    }

    function ingestSpikeEvents() {
        var generation = String(simulation.spikeEventsGeneration || "")
        if (lastGeneration !== "" && generation !== lastGeneration) {
            lastSeenSequence = 0
            lastSpikeByNeuron = ({})
        }
        lastGeneration = generation
        var events = simulation.spikeEvents || []
        var next = Object.assign({}, lastSpikeByNeuron)
        var newestStep = clockAnchorStep
        for (var i = events.length - 1; i >= 0; --i) {
            var event = events[i]
            var sequence = Number(event.sequence)
            if (!isFinite(sequence) || sequence <= lastSeenSequence)
                break
            var neuron = String(event.neuron_id)
            var step = Number(event.step)
            if (isFinite(step)) {
                next[neuron] = step
                newestStep = Math.max(newestStep, step)
            }
            lastSeenSequence = Math.max(lastSeenSequence, sequence)
        }
        lastSpikeByNeuron = next
        updateClock(newestStep)
    }

    function estimatedStep() {
        if (simulation.simulationStatus !== "running" ||
            stepsPerMillisecond <= 0)
            return clockAnchorStep
        return clockAnchorStep +
            Math.max(0, Date.now() - clockAnchorMs) * stepsPerMillisecond
    }

    function drawField(ctx, w, h) {
        ctx.clearRect(0, 0, w, h)
        ctx.fillStyle = "#020409"
        ctx.fillRect(0, 0, w, h)

        var graph = simulation.topologySnapshot || ({})
        var nodes = graph.nodes || []
        var edges = graph.edges || []
        var currentStep = estimatedStep()
        var positions = Object.create(null)
        var ordered = []
        var extent = Math.min(w, h) * 0.43
        var cy = Math.cos(rotationY), sy = Math.sin(rotationY)
        var cx = Math.cos(rotationX), sx = Math.sin(rotationX)
        var count = Math.max(1, nodes.length)

        // Stable Fibonacci-sphere coordinates give every rendered neuron a
        // persistent 3D location. Camera rotation changes only the projection.
        for (var n = 0; n < nodes.length; ++n) {
            var z0 = 1 - 2 * (n + 0.5) / count
            var ring = Math.sqrt(Math.max(0, 1 - z0 * z0))
            var angle = n * 2.399963229728653
            var x0 = Math.cos(angle) * ring
            var y0 = Math.sin(angle) * ring
            var x1 = x0 * cy + z0 * sy
            var z1 = -x0 * sy + z0 * cy
            var y1 = y0 * cx - z1 * sx
            var z2 = y0 * sx + z1 * cx
            var perspective = cameraDistance /
                Math.max(0.35, cameraDistance - z2)
            var point = {
                x: w * 0.5 + x1 * extent * perspective,
                y: h * 0.5 + y1 * extent * perspective,
                z: z2,
                scale: perspective
            }
            positions[String(nodes[n])] = point
            ordered.push({ id: Number(nodes[n]), point: point })
        }

        // Draw the real CSR edges first. A source spike launches a pulse down
        // each outgoing edge; delay is the pulse's travel time in simulation
        // steps, not a decorative animation parameter.
        var activeEdges = 0
        var excitatory = Number(graph.excitatory_count || 0)
        for (var e = 0; e < edges.length; ++e) {
            var edge = edges[e]
            var from = positions[String(edge.source)]
            var to = positions[String(edge.target)]
            if (!from || !to)
                continue

            var sourceStep = lastSpikeByNeuron[String(edge.source)]
            var delay = Math.max(1, Number(edge.delay) || 1)
            var age = sourceStep === undefined ? -1 : currentStep - Number(sourceStep)
            var inFlight = age >= 0 && age < delay
            var progress = inFlight ? Math.max(0, Math.min(1, age / delay)) : 0
            var depth = Math.max(0, Math.min(1, (from.z + to.z + 2) * 0.25))

            ctx.beginPath()
            ctx.moveTo(from.x, from.y)
            ctx.lineTo(to.x, to.y)
            ctx.lineWidth = inFlight ? 1.25 + depth * 1.3 : 0.35 + depth * 0.42
            ctx.strokeStyle = inFlight
                ? (edge.excitatory ? "rgba(104,255,225,0.92)" : "rgba(255,184,112,0.92)")
                : (edge.excitatory
                    ? "rgba(55,154,153," + (0.045 + depth * 0.13) + ")"
                    : "rgba(177,112,65," + (0.045 + depth * 0.13) + ")")
            if (inFlight) {
                ctx.shadowBlur = 9
                ctx.shadowColor = edge.excitatory ? "#62ffe0" : "#ffb66e"
                ++activeEdges
            }
            ctx.stroke()
            ctx.shadowBlur = 0

            if (inFlight) {
                var px = from.x + (to.x - from.x) * progress
                var py = from.y + (to.y - from.y) * progress
                ctx.beginPath()
                ctx.arc(px, py, 1.7 + depth * 1.4, 0, Math.PI * 2)
                ctx.fillStyle = edge.excitatory ? "#c8fff4" : "#ffe0bd"
                ctx.shadowBlur = 12
                ctx.shadowColor = edge.excitatory ? "#62ffe0" : "#ffb66e"
                ctx.fill()
                ctx.shadowBlur = 0
            }
        }

        // Far-side neurons are painted first; near-side neurons sit on top.
        ordered.sort(function(a, b) { return a.point.z - b.point.z })
        for (var j = 0; j < ordered.length; ++j) {
            var item = ordered[j]
            var id = item.id
            var p = item.point
            var last = lastSpikeByNeuron[String(id)]
            var neuronAge = last === undefined ? 1e9 : currentStep - Number(last)
            var firing = neuronAge >= 0 && neuronAge < 5
            var isExcitatory = id < excitatory
            var radius = (0.85 + (p.z + 1) * 0.75) * p.scale
            ctx.beginPath()
            ctx.arc(p.x, p.y, firing ? radius * 2.15 : radius, 0, Math.PI * 2)
            ctx.fillStyle = firing ? "#ffffff" :
                (isExcitatory ? "rgba(94,205,190,0.88)" : "rgba(237,166,103,0.88)")
            if (firing) {
                ctx.shadowBlur = 15
                ctx.shadowColor = isExcitatory ? "#74ffe5" : "#ffc18a"
            }
            ctx.fill()
            ctx.shadowBlur = 0
        }

        ctx.textAlign = "left"
        ctx.font = "11px monospace"
        ctx.fillStyle = "rgba(190,210,226,0.65)"
        ctx.fillText(nodes.length + " NODES  /  " + edges.length +
                     " SYNAPSES  /  " + activeEdges + " ACTIVE", 22, h - 24)
        if (nodes.length === 0) {
            ctx.textAlign = "center"
            ctx.font = "13px sans-serif"
            ctx.fillStyle = "rgba(190,210,226,0.55)"
            ctx.fillText(simulation.connected
                ? "Waiting for connectome"
                : "Connect to load the live connectome", w * 0.5, h * 0.5)
        }
    }

    Connections {
        target: simulation
        function onTopologyChanged() { fieldCanvas.requestPaint() }
        function onSpikeEventsChanged() {
            window.ingestSpikeEvents()
            fieldCanvas.requestPaint()
        }
        function onTelemetryChanged() {
            window.updateClock(Number(simulation.completedSteps))
            fieldCanvas.requestPaint()
        }
        function onSimulationStatusChanged() { fieldCanvas.requestPaint() }
        function onConnectionChanged() { fieldCanvas.requestPaint() }
    }

    Canvas {
        id: fieldCanvas
        anchors.fill: parent
        renderStrategy: Canvas.Threaded
        onPaint: {
            var ctx = getContext("2d")
            window.drawField(ctx, width, height)
        }
    }

    // The field itself is the interface: drag to rotate, wheel to dolly.
    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton
        hoverEnabled: false
        property real previousX: 0
        property real previousY: 0
        onPressed: function(mouse) {
            previousX = mouse.x
            previousY = mouse.y
        }
        onPositionChanged: function(mouse) {
            if (!(mouse.buttons & Qt.LeftButton))
                return
            window.rotationY += (mouse.x - previousX) * 0.006
            window.rotationX = Math.max(-1.35, Math.min(1.35,
                window.rotationX + (mouse.y - previousY) * 0.006))
            previousX = mouse.x
            previousY = mouse.y
            fieldCanvas.requestPaint()
        }
        onWheel: function(wheel) {
            window.cameraDistance = Math.max(1.65, Math.min(6.0,
                window.cameraDistance - wheel.angleDelta.y / 120 * 0.16))
            fieldCanvas.requestPaint()
            wheel.accepted = true
        }
    }

    // Only connection and run-state controls remain; no dashboard panels.
    Row {
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 18
        spacing: 8

        Rectangle {
            width: 9
            height: 9
            radius: 5
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
        anchors.top: parent.top
        anchors.margins: 20
        text: "CONNECTOME  /  DELAY TRACE"
        color: "#667b8e"
        font.pixelSize: 10
        font.letterSpacing: 2
    }

    Text {
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 20
        text: "STEP " + Math.floor(window.estimatedStep()).toLocaleString(Qt.locale(), "f", 0)
        color: "#667b8e"
        font.pixelSize: 10
        font.family: "monospace"
    }

    Timer {
        interval: 33
        repeat: true
        running: true
        onTriggered: fieldCanvas.requestPaint()
    }

    Timer {
        interval: 3000
        repeat: true
        running: simulation.connected
        onTriggered: simulation.refreshTopology()
    }

    Component.onCompleted: fieldCanvas.requestPaint()
}
