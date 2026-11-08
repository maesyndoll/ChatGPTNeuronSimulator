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
    property var lastSpikeAtMs: ({})
    property double lastSpikeEventMs: 0
    property var cachedPositions: ({})
    property string cachedLayoutKey: ""
    property int activeTraceCount: 0

    function ingestSpikeEvents() {
        var generation = String(simulation.spikeEventsGeneration || "")
        if (lastGeneration !== "" && generation !== lastGeneration) {
            lastSeenSequence = 0
            lastSpikeByNeuron = ({})
            lastSpikeAtMs = ({})
        }
        lastGeneration = generation
        var events = simulation.spikeEvents || []
        var nextSteps = Object.assign({}, lastSpikeByNeuron)
        var nextTimes = Object.assign({}, lastSpikeAtMs)
        var now = Date.now()
        var received = false
        for (var i = events.length - 1; i >= 0; --i) {
            var event = events[i]
            var sequence = Number(event.sequence)
            if (!isFinite(sequence) || sequence <= lastSeenSequence) break
            var neuron = String(event.neuron_id)
            var step = Number(event.step)
            if (isFinite(step)) {
                nextSteps[neuron] = step
                nextTimes[neuron] = now
                received = true
            }
            lastSeenSequence = Math.max(lastSeenSequence, sequence)
        }
        lastSpikeByNeuron = nextSteps
        lastSpikeAtMs = nextTimes
        if (received) lastSpikeEventMs = now
    }

    function invalidateLayout() {
        cachedLayoutKey = ""
        fieldCanvas.requestPaint()
    }

    function drawField(ctx, w, h) {
        ctx.clearRect(0, 0, w, h)
        ctx.fillStyle = "#020409"
        ctx.fillRect(0, 0, w, h)
        var graph = simulation.topologySnapshot || ({})
        var nodes = graph.nodes || []
        var edges = graph.edges || []
        var now = Date.now()
        var positions = cachedPositions
        var layoutKey = String(graph.revision || "") + ":" + w + ":" + h + ":" +
            rotationY.toFixed(3) + ":" + rotationX.toFixed(3) + ":" + cameraDistance.toFixed(2)

        if (cachedLayoutKey !== layoutKey) {
            positions = Object.create(null)
            var extent = Math.min(w, h) * 0.43
            var cy = Math.cos(rotationY), sy = Math.sin(rotationY)
            var cx = Math.cos(rotationX), sx = Math.sin(rotationX)
            var count = Math.max(1, nodes.length)
            for (var n = 0; n < nodes.length; ++n) {
                var z0 = 1 - 2 * (n + 0.5) / count
                var ring = Math.sqrt(Math.max(0, 1 - z0 * z0))
                var angle = n * 2.399963229728653
                var x0 = Math.cos(angle) * ring, y0 = Math.sin(angle) * ring
                var x1 = x0 * cy + z0 * sy, z1 = -x0 * sy + z0 * cy
                var y1 = y0 * cx - z1 * sx, z2 = y0 * sx + z1 * cx
                var perspective = cameraDistance / Math.max(0.35, cameraDistance - z2)
                positions[String(nodes[n])] = {
                    x: w * 0.5 + x1 * extent * perspective,
                    y: h * 0.5 + y1 * extent * perspective,
                    z: z2, scale: perspective
                }
            }
            cachedPositions = positions
            cachedLayoutKey = layoutKey
        }

        var maxDelay = 1
        for (var d = 0; d < edges.length; ++d)
            maxDelay = Math.max(maxDelay, Number(edges[d].delay) || 1)

        var activeCount = 0
        for (var e = 0; e < edges.length; ++e) {
            var edge = edges[e]
            var from = positions[String(edge.source)]
            var to = positions[String(edge.target)]
            if (!from || !to) continue
            var firedAt = lastSpikeAtMs[String(edge.source)]
            var delay = Math.max(1, Number(edge.delay) || 1)
            // Monotonic delay-scaled display time keeps even fast simulation
            // traces visible without changing their relative delay ordering.
            var traceDuration = 100 + 900 * (delay / maxDelay)
            var ageMs = firedAt === undefined ? 1e9 : now - Number(firedAt)
            var inFlight = ageMs >= 0 && ageMs < traceDuration
            var progress = inFlight ? ageMs / traceDuration : 0
            var depth = Math.max(0, Math.min(1, (from.z + to.z + 2) * 0.25))
            ctx.beginPath()
            ctx.moveTo(from.x, from.y)
            ctx.lineTo(to.x, to.y)
            ctx.lineWidth = inFlight ? 0.9 + depth * 0.65 : 0.3 + depth * 0.3
            ctx.strokeStyle = inFlight
                ? (edge.excitatory ? "rgba(104,255,225,0.58)" : "rgba(255,184,112,0.58)")
                : (edge.excitatory
                    ? "rgba(55,154,153," + (0.035 + depth * 0.075) + ")"
                    : "rgba(177,112,65," + (0.035 + depth * 0.075) + ")")
            ctx.stroke()
            if (inFlight) {
                ++activeCount
                var px = from.x + (to.x - from.x) * progress
                var py = from.y + (to.y - from.y) * progress
                ctx.beginPath()
                ctx.arc(px, py, 1.2 + depth * 0.9, 0, Math.PI * 2)
                ctx.fillStyle = edge.excitatory ? "#baffef" : "#ffe0bd"
                ctx.fill()
            }
        }

        var excitatory = Number(graph.excitatory_count || 0)
        for (var j = 0; j < nodes.length; ++j) {
            var id = Number(nodes[j]), p = positions[String(id)]
            if (!p) continue
            var last = lastSpikeAtMs[String(id)]
            var firing = last !== undefined && now - Number(last) < 130
            var radius = (0.8 + (p.z + 1) * 0.65) * p.scale
            ctx.beginPath()
            ctx.arc(p.x, p.y, firing ? radius * 1.8 : radius, 0, Math.PI * 2)
            ctx.fillStyle = firing ? "#ffffff" :
                (id < excitatory ? "rgba(94,205,190,0.82)" : "rgba(237,166,103,0.82)")
            ctx.fill()
        }
        activeTraceCount = activeCount
        ctx.textAlign = "left"
        ctx.font = "11px monospace"
        ctx.fillStyle = "rgba(190,210,226,0.65)"
        ctx.fillText(nodes.length + " NODES  /  " + edges.length +
            " SYNAPSES  /  " + activeCount + " ACTIVE", 22, h - 24)
        if (nodes.length === 0) {
            ctx.textAlign = "center"
            ctx.font = "13px sans-serif"
            ctx.fillStyle = "rgba(190,210,226,0.55)"
            ctx.fillText(simulation.connected ? "Waiting for connectome" :
                "Connect to load the live connectome", w * 0.5, h * 0.5)
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
            window.invalidateLayout()
            window.rotationX = Math.max(-1.35, Math.min(1.35,
                window.rotationX + (mouse.y - previousY) * 0.006))
            window.invalidateLayout()
            previousX = mouse.x
            previousY = mouse.y
            fieldCanvas.requestPaint()
        }
        onWheel: function(wheel) {
            window.cameraDistance = Math.max(1.65, Math.min(6.0,
                window.cameraDistance - wheel.angleDelta.y / 120 * 0.16))
            window.invalidateLayout()
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
        text: simulation.completedSteps === "—" ? "STEP —" : "STEP " + simulation.completedSteps
        color: "#667b8e"
        font.pixelSize: 10
        font.family: "monospace"
    }

    Timer {
        interval: 50
        repeat: true
        running: simulation.connected && (simulation.simulationStatus === "running" || Date.now() - window.lastSpikeEventMs < 1600)
        onTriggered: fieldCanvas.requestPaint()
    }

    Timer {
        interval: 10000
        repeat: true
        running: simulation.connected
        onTriggered: simulation.refreshTopology()
    }

    Component.onCompleted: fieldCanvas.requestPaint()
}
