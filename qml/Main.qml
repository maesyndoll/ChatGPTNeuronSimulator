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

    property var recentSpikeEvents: []

    function ingestSpikeEvents() {
        var generation = String(simulation.spikeEventsGeneration || "")
        if (lastGeneration !== "" && generation !== lastGeneration) {
            lastSeenSequence = 0
            lastSpikeByNeuron = ({})
            lastSpikeAtMs = ({})
            recentSpikeEvents = []
        }
        lastGeneration = generation
        var events = simulation.spikeEvents || []
        var pending = []
        var now = Date.now()
        for (var i = 0; i < events.length; ++i) {
            var event = events[i]
            var sequence = Number(event.sequence)
            if (!isFinite(sequence) || sequence <= lastSeenSequence) continue
            pending.push({
                sequence: sequence,
                neuron: Number(event.neuron_id),
                step: Number(event.step),
                receivedAt: now
            })
            lastSeenSequence = Math.max(lastSeenSequence, sequence)
        }
        if (pending.length > 0) {
            var combined = recentSpikeEvents.concat(pending)
            if (combined.length > 3000)
                combined = combined.slice(combined.length - 3000)
            recentSpikeEvents = combined
            lastSpikeEventMs = now
        }
        var cutoff = now - 2600
        var retained = []
        for (var j = 0; j < recentSpikeEvents.length; ++j)
            if (recentSpikeEvents[j].receivedAt >= cutoff)
                retained.push(recentSpikeEvents[j])
        if (retained.length !== recentSpikeEvents.length)
            recentSpikeEvents = retained
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
        var elapsed = now / 1000
        var recent = recentSpikeEvents
        var activity = Object.create(null)
        var pulsesBySource = Object.create(null)
        var cutoff = now - 2600

        // Build a compact active-set index once per frame. Repeated spikes from
        // one neuron remain distinct pulses instead of overwriting one another.
        for (var s = 0; s < recent.length; ++s) {
            var spike = recent[s]
            var age = now - spike.receivedAt
            if (age < 0 || age > 2600) continue
            var key = String(spike.neuron)
            activity[key] = (activity[key] || 0) + Math.exp(-age / 720)
            if (!pulsesBySource[key]) pulsesBySource[key] = []
            var pulseList = pulsesBySource[key]
            if (pulseList.length < 12) pulseList.push(spike.receivedAt)
        }

        // Rebuild the 3D embedding cheaply each frame. Active neurons expand,
        // twist and drift through the volume; connected edges follow them.
        // IDs remain fixed, so this is a visual embedding of the same CSR graph.
        var positions = Object.create(null)
        var count = Math.max(1, nodes.length)
        var extent = Math.min(w, h) * 0.43
        var cy = Math.cos(rotationY), sy = Math.sin(rotationY)
        var cx = Math.cos(rotationX), sx = Math.sin(rotationX)
        var activeNodes = 0
        for (var n = 0; n < nodes.length; ++n) {
            var id = Number(nodes[n])
            var z0 = 1 - 2 * (n + 0.5) / count
            var ring = Math.sqrt(Math.max(0, 1 - z0 * z0))
            var angle = n * 2.399963229728653
            var intensity = Math.min(2.8, activity[String(id)] || 0)
            if (intensity > 0.12) ++activeNodes
            var wave = elapsed * 1.25 + angle * 1.7 + z0 * 5.0
            var radial = 1.0 + intensity * 0.075 +
                         Math.sin(wave) * (0.012 + intensity * 0.025)
            var x0 = Math.cos(angle) * ring * radial
            var y0 = Math.sin(angle) * ring * radial
            var z = z0 * radial + Math.sin(wave * 0.7) * intensity * 0.045
            var twist = intensity * 0.10 + Math.sin(wave * 0.35) * 0.018
            var tx = x0 * Math.cos(twist) - y0 * Math.sin(twist)
            var ty = x0 * Math.sin(twist) + y0 * Math.cos(twist)
            var x1 = tx * cy + z * sy
            var z1 = -tx * sy + z * cy
            var y1 = ty * cx - z1 * sx
            var z2 = ty * sx + z1 * cx
            var perspective = cameraDistance / Math.max(0.35, cameraDistance - z2)
            positions[String(id)] = {
                x: w * 0.5 + x1 * extent * perspective,
                y: h * 0.5 + y1 * extent * perspective,
                z: z2, scale: perspective, intensity: intensity
            }
        }

        var maxDelay = 1
        for (var d = 0; d < edges.length; ++d)
            maxDelay = Math.max(maxDelay, Number(edges[d].delay) || 1)

        var activeEdges = 0
        for (var e = 0; e < edges.length; ++e) {
            var edge = edges[e]
            var from = positions[String(edge.source)]
            var to = positions[String(edge.target)]
            if (!from || !to) continue
            var pulses = pulsesBySource[String(edge.source)] || []
            var delay = Math.max(1, Number(edge.delay) || 1)
            // Delay controls trace lifetime; long-delay synapses carry slower,
            // more persistent wavefronts through the rendered volume.
            var traceDuration = 240 + 1760 * (delay / maxDelay)
            var strongest = 0
            var progressList = []
            for (var p = 0; p < pulses.length; ++p) {
                var ageMs = now - pulses[p]
                if (ageMs >= 0 && ageMs < traceDuration) {
                    var progress = ageMs / traceDuration
                    progressList.push(progress)
                    strongest = Math.max(strongest, 1 - progress * 0.42)
                }
            }
            var active = progressList.length > 0
            var depth = Math.max(0, Math.min(1, (from.z + to.z + 2) * 0.25))
            ctx.beginPath()
            ctx.moveTo(from.x, from.y)
            ctx.lineTo(to.x, to.y)
            ctx.lineWidth = active ? 0.85 + strongest * 1.25 : 0.28 + depth * 0.25
            ctx.strokeStyle = active
                ? (edge.excitatory ? "rgba(103,255,224," + (0.48 + strongest * 0.45) + ")" :
                                     "rgba(255,180,105," + (0.48 + strongest * 0.45) + ")")
                : (edge.excitatory ? "rgba(55,154,153," + (0.025 + depth * 0.065) + ")" :
                                     "rgba(177,112,65," + (0.025 + depth * 0.065) + ")")
            ctx.stroke()
            for (var q = 0; q < progressList.length; ++q) {
                var t = progressList[q]
                var px = from.x + (to.x - from.x) * t
                var py = from.y + (to.y - from.y) * t
                ctx.beginPath()
                ctx.arc(px, py, 1.25 + depth * 1.0, 0, Math.PI * 2)
                ctx.fillStyle = edge.excitatory ? "#c4fff2" : "#ffe2c3"
                ctx.fill()
                ++activeEdges
            }
        }

        var excitatory = Number(graph.excitatory_count || 0)
        for (var j = 0; j < nodes.length; ++j) {
            var neuronId = Number(nodes[j])
            var point = positions[String(neuronId)]
            if (!point) continue
            var level = point.intensity
            var radius = (0.8 + (point.z + 1) * 0.65) * point.scale
            var glow = Math.min(3.0, level)
            ctx.beginPath()
            ctx.arc(point.x, point.y, radius * (1 + glow * 0.9), 0, Math.PI * 2)
            ctx.fillStyle = level > 0.12 ? "#ffffff" :
                (neuronId < excitatory ? "rgba(94,205,190,0.86)" :
                                         "rgba(237,166,103,0.86)")
            if (level > 0.12) {
                ctx.shadowBlur = 5 + glow * 7
                ctx.shadowColor = neuronId < excitatory ? "#62ffe0" : "#ffb66e"
            }
            ctx.fill()
            ctx.shadowBlur = 0
        }

        ctx.textAlign = "left"
        ctx.font = "11px monospace"
        ctx.fillStyle = "rgba(190,210,226,0.68)"
        ctx.fillText(nodes.length + " NODES  /  " + edges.length +
            " SYNAPSES  /  " + activeNodes + " ACTIVE NODES  /  " +
            activeEdges + " ACTIVE TRACES", 22, h - 24)
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

    // Pointer handlers operate independently of the Canvas input stack.
    // DragHandler avoids a full-window MouseArea swallowing other gestures.
    DragHandler {
        id: orbitDrag
        target: null
        acceptedButtons: Qt.LeftButton
        property point previousTranslation: Qt.point(0, 0)
        onActiveChanged: previousTranslation = Qt.point(0, 0)
        onTranslationChanged: {
            var dx = translation.x - previousTranslation.x
            var dy = translation.y - previousTranslation.y
            previousTranslation = translation
            if (active && (dx !== 0 || dy !== 0)) {
                window.rotationY += dx * 0.008
                window.rotationX = Math.max(-1.45, Math.min(1.45,
                    window.rotationX + dy * 0.008))
                window.invalidateLayout()
            }
        }
    }

    WheelHandler {
        id: zoomHandler
        target: null
        acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
        onWheel: function(event) {
            window.cameraDistance = Math.max(1.45, Math.min(7.0,
                window.cameraDistance - event.angleDelta.y / 120 * 0.18))
            window.invalidateLayout()
            event.accepted = true
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
        interval: 33
        repeat: true
        running: simulation.connected && (simulation.simulationStatus === "running" || Date.now() - window.lastSpikeEventMs < 2800)
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
