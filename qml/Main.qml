import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    visible: true
    width: 1360
    height: 900
    minimumWidth: 920
    minimumHeight: 650
    title: "NEUROMANCER · Neural Observatory"
    color: "#080e16"
    property string visualGeneration: "—"
    property double visualLastSequence: 0
    property bool graftExpanded: false

    function latestSpikeText() {
        var events = simulation.spikeEvents || []
        if (events.length === 0) return "No spike events received yet"
        var event = events[events.length - 1]
        return "LAST EVENT   neuron " + event.neuron_id + "   ·   step " + event.step
    }

    Connections {
        target: simulation
        function onTopologyVisualChanged() { topologyCanvas.requestPaint() }
        function onSpikeEventsChanged() {
            var generation = simulation.spikeEventsGeneration
            if (window.visualGeneration !== generation) {
                window.visualGeneration = generation
                window.visualLastSequence = 0
                topologyCanvas.activeNeuronTimes = ({})
            }
            var events = simulation.spikeEvents || []
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
            sensoryCanvas.requestPaint()
            rasterCanvas.requestPaint()
        }
        function onConnectivityChanged() { topologyCanvas.requestPaint() }
        function onConnectionChanged() {
            if (simulation.connected) {
                simulation.refreshConnectivity()
                simulation.refreshTopologyVisual()
            }
        }
    }

    Timer {
        interval: 40
        repeat: true
        running: simulation.connected
        onTriggered: {
            var active = Object.assign({}, topologyCanvas.activeNeuronTimes)
            var now = Date.now()
            var changed = false
            for (var id in active) {
                if (now - active[id] > 1600) {
                    delete active[id]
                    changed = true
                }
            }
            if (changed) topologyCanvas.activeNeuronTimes = active
            sensoryCanvas.requestPaint()
            topologyCanvas.requestPaint()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 18
        spacing: 12

        RowLayout {
            Layout.fillWidth: true
            spacing: 14
            ColumnLayout {
                spacing: 1
                Label {
                    text: "NEUROMANCER"
                    color: "#e7f3fa"
                    font.pixelSize: 25
                    font.bold: true
                    font.letterSpacing: 2.2
                }
                Label {
                    text: "NEURAL OBSERVATORY  /  LIVE SPIKING FIELD"
                    color: "#6e8a9e"
                    font.pixelSize: 10
                    font.letterSpacing: 1.5
                }
            }
            Item { Layout.fillWidth: true }
            Rectangle {
                radius: 7
                color: simulation.connected ? "#102c2b" : "#302027"
                border.color: simulation.connected ? "#27665b" : "#70404b"
                implicitWidth: connectionText.implicitWidth + 22
                implicitHeight: 30
                Label {
                    id: connectionText
                    anchors.centerIn: parent
                    text: simulation.connectionStatus.toUpperCase()
                    color: simulation.connected ? "#75d9bd" : "#e5a2aa"
                    font.pixelSize: 10
                    font.bold: true
                    font.letterSpacing: 1
                }
            }
            Button {
                text: simulation.connected ? "Disconnect" : "Connect"
                onClicked: simulation.connected ? simulation.disconnectFromServer()
                                                : simulation.connectToServer()
            }
            Button { text: "Start"; enabled: simulation.connected; onClicked: simulation.startSimulation() }
            Button { text: "Pause"; enabled: simulation.connected; onClicked: simulation.pauseSimulation() }
            Button { text: "Step"; enabled: simulation.connected; onClicked: simulation.stepSimulation(1) }
        }

        Rectangle { Layout.fillWidth: true; height: 1; color: "#1c2d3b" }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumWidth: 250
                Layout.preferredWidth: 0.38 * window.width
                radius: 12
                color: "#0c1722"
                border.color: "#203747"
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 9
                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "01  /  NEURAL FIELD"
                            color: "#8bd9ce"
                            font.pixelSize: 12
                            font.bold: true
                            font.letterSpacing: 1.2
                        }
                        Item { Layout.fillWidth: true }
                        Label { text: "EVENT-DERIVED"; color: "#577c89"; font.pixelSize: 9; font.letterSpacing: 1 }
                    }
                    Label {
                        Layout.fillWidth: true
                        text: "A visual projection of recent firing across the population."
                        color: "#7792a3"
                        font.pixelSize: 11
                        wrapMode: Text.Wrap
                    }
                    Canvas {
                        id: sensoryCanvas
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.minimumHeight: 220
                        contextType: "2d"
                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.clearRect(0, 0, width, height)
                            ctx.fillStyle = "#07111b"
                            ctx.fillRect(0, 0, width, height)
                            var cols = 36, rows = 24
                            var cw = width / cols, ch = height / rows
                            var now = Date.now()
                            var active = topologyCanvas.activeNeuronTimes
                            ctx.strokeStyle = "rgba(80,130,151,0.10)"
                            ctx.lineWidth = 0.5
                            for (var x = 0; x <= cols; ++x) {
                                ctx.beginPath(); ctx.moveTo(x*cw,0); ctx.lineTo(x*cw,height); ctx.stroke()
                            }
                            for (var y = 0; y <= rows; ++y) {
                                ctx.beginPath(); ctx.moveTo(0,y*ch); ctx.lineTo(width,y*ch); ctx.stroke()
                            }
                            var hasActivity = false
                            for (var id in active) {
                                var age = now - active[id]
                                if (age > 1600) continue
                                hasActivity = true
                                var cell = Number(id) % (cols * rows)
                                var gx = cell % cols, gy = Math.floor(cell / cols)
                                var decay = Math.max(0, 1 - age / 1600)
                                ctx.fillStyle = "rgba(72,220,190," + (0.12 + decay*0.72) + ")"
                                ctx.fillRect(gx*cw+1,gy*ch+1,Math.max(1,cw-2),Math.max(1,ch-2))
                                ctx.beginPath()
                                ctx.arc((gx+0.5)*cw,(gy+0.5)*ch,2+decay*5,0,Math.PI*2)
                                ctx.fillStyle = "rgba(220,255,247," + (decay*0.85) + ")"
                                ctx.fill()
                            }
                            if (!hasActivity) {
                                ctx.fillStyle = "#526b7d"; ctx.font = "12px sans-serif"; ctx.textAlign = "center"
                                ctx.fillText(simulation.connected ? "Waiting for neural events" : "Connect to the simulation",width/2,height/2)
                            }
                            ctx.textAlign = "left"; ctx.fillStyle = "#66899a"; ctx.font = "9px sans-serif"
                            ctx.fillText("RECENT SPIKES  ·  1.6 s DECAY",10,height-10)
                        }
                    }
                    Label {
                        Layout.fillWidth: true
                        text: window.latestSpikeText()
                        color: "#a7c4ce"
                        font.pixelSize: 10
                        elide: Text.ElideRight
                    }
                    Rectangle { Layout.fillWidth: true; height: 1; color: "#1b3040" }
                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: "02  /  SPIKE RASTER"; color: "#8bd9ce"; font.pixelSize: 11; font.bold: true; font.letterSpacing: 1 }
                        Item { Layout.fillWidth: true }
                        Label { text: simulation.spikeEventCount + " events"; color: "#698596"; font.pixelSize: 10 }
                    }
                    Canvas {
                        id: rasterCanvas
                        Layout.fillWidth: true
                        Layout.preferredHeight: 118
                        Layout.minimumHeight: 90
                        contextType: "2d"
                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.clearRect(0,0,width,height)
                            ctx.fillStyle = "#07111b"; ctx.fillRect(0,0,width,height)
                            var graph = simulation.topologyVisual || {}
                            var eCount = Number(graph.excitatory_count || 0)
                            var total = Math.max(1,eCount+Number(graph.inhibitory_count || 0))
                            var events = simulation.spikeEvents || []
                            var maxStep = Number(simulation.completedSteps || 0)
                            var minStep = Math.max(0,maxStep-1200)
                            var span = Math.max(1,maxStep-minStep)
                            ctx.strokeStyle = "rgba(90,125,145,0.12)"
                            for (var r=0;r<=4;++r) {
                                var yy=8+r*(height-20)/4
                                ctx.beginPath();ctx.moveTo(8,yy);ctx.lineTo(width-8,yy);ctx.stroke()
                            }
                            for (var i=0;i<events.length;++i) {
                                var ev=events[i], step=Number(ev.step), neuron=Number(ev.neuron_id)
                                if(step<minStep||neuron<0||neuron>=total) continue
                                var px=8+((step-minStep)/span)*(width-16)
                                var py=6+(neuron/total)*(height-14)
                                ctx.fillStyle=neuron<eCount?"#65c4ed":"#e6b36e"
                                ctx.fillRect(px,py,2,3)
                            }
                            ctx.fillStyle="#5e7a8d";ctx.font="9px sans-serif";ctx.fillText("OLDER",8,height-3)
                            ctx.textAlign="right";ctx.fillText("LATEST",width-8,height-3);ctx.textAlign="left"
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.minimumWidth: 360
                Layout.preferredWidth: 0.62 * window.width
                radius: 12
                color: "#0b141f"
                border.color: "#203747"
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 8
                    RowLayout {
                        Layout.fillWidth: true
                        Label {
                            text: "03  /  SPARSE ROW TOPOLOGY"
                            color: "#8bd9ce"; font.pixelSize: 12; font.bold: true; font.letterSpacing: 1.1
                        }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: simulation.topologyVisual.sampled_neuron_count === undefined
                                  ? "TOPOLOGY PENDING"
                                  : simulation.topologyVisual.sampled_neuron_count + " nodes  ·  " +
                                    simulation.topologyVisual.sampled_edge_count + " edges shown"
                            color: "#718b9b"; font.pixelSize: 10
                        }
                    }
                    Label {
                        Layout.fillWidth: true
                        text: "CSR edges are drawn from the sampled runtime topology. A node flashes only when its spike event arrives."
                        color: "#718b9b"; font.pixelSize: 11; wrapMode: Text.Wrap
                    }
                    Canvas {
                        id: topologyCanvas
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.minimumHeight: 320
                        contextType: "2d"
                        property var activeNeuronTimes: ({})
                        onPaint: {
                            var ctx=getContext("2d")
                            ctx.clearRect(0,0,width,height);ctx.fillStyle="#07111b";ctx.fillRect(0,0,width,height)
                            var graph=simulation.topologyVisual||{},ids=graph.neuron_ids||[],edges=graph.edges||[]
                            var eCount=Number(graph.excitatory_count||0),iCount=Number(graph.inhibitory_count||0)
                            var total=Math.max(1,eCount+iCount)
                            var columns=Math.max(8,Math.min(48,Math.ceil(Math.sqrt(total*1.8))))
                            var rows=Math.max(1,Math.ceil(total/columns))
                            var left=18,right=18,top=18,bottom=28
                            var plotW=Math.max(1,width-left-right),plotH=Math.max(1,height-top-bottom)
                            var cellW=plotW/columns,cellH=plotH/rows,positions=({})
                            var active=activeNeuronTimes,now=Date.now()
                            function positionFor(neuronId) {
                                var col=neuronId%columns,row=Math.floor(neuronId/columns)
                                return {x:left+(col+0.5)*cellW,y:top+(row+0.5)*cellH}
                            }
                            ctx.strokeStyle="rgba(91,133,153,0.13)";ctx.lineWidth=0.6
                            for(var r=0;r<rows;++r) {
                                var gy=top+(r+0.5)*cellH
                                ctx.beginPath();ctx.moveTo(left,gy);ctx.lineTo(width-right,gy);ctx.stroke()
                            }
                            for(var n=0;n<ids.length;++n) positions[String(ids[n])]=positionFor(Number(ids[n]))
                            for(var e=0;e<edges.length;++e) {
                                var edge=edges[e],from=positions[String(edge.source)],to=positions[String(edge.target)]
                                if(from===undefined||to===undefined) continue
                                var weight=Number(edge.weight)
                                var age=active[String(edge.source)]===undefined?9999:now-active[String(edge.source)]
                                var firing=age<520,strength=Math.min(1,Math.abs(weight)/0.5)
                                ctx.beginPath();ctx.moveTo(from.x,from.y);ctx.lineTo(to.x,to.y)
                                ctx.lineWidth=firing?1.4:0.55
                                if(firing) ctx.strokeStyle="rgba(106,245,204,"+(0.35+strength*0.55)+")"
                                else if(weight>=0) ctx.strokeStyle="rgba(83,171,225,"+(0.08+strength*0.36)+")"
                                else ctx.strokeStyle="rgba(226,169,93,"+(0.10+strength*0.42)+")"
                                ctx.stroke()
                            }
                            for(var v=0;v<ids.length;++v) {
                                var neuronId=Number(ids[v]),point=positions[String(neuronId)]
                                var age=active[String(neuronId)]===undefined?9999:now-active[String(neuronId)]
                                var hot=age<520,decay=hot?Math.max(0,1-age/520):0
                                ctx.beginPath();ctx.arc(point.x,point.y,hot?3+decay*3.8:2.1,0,Math.PI*2)
                                ctx.fillStyle=hot?"#e7fff8":(neuronId<eCount?"#58b5e5":"#e4ad68");ctx.fill()
                                if(hot) {
                                    ctx.beginPath();ctx.arc(point.x,point.y,7+decay*7,0,Math.PI*2)
                                    ctx.strokeStyle="rgba(100,240,199,"+(decay*0.72)+")";ctx.lineWidth=1.2;ctx.stroke()
                                }
                            }
                            var extra=0
                            for(var activeId in active) {
                                if(extra>=1200) break
                                if(positions[activeId]!==undefined) continue
                                var numericId=Number(activeId)
                                if(!isFinite(numericId)||numericId<0||numericId>=total) continue
                                var ageExtra=now-active[activeId]
                                if(ageExtra>=520) continue
                                var p=positionFor(numericId),pulse=Math.max(0,1-ageExtra/520)
                                ctx.beginPath();ctx.arc(p.x,p.y,2.5+pulse*4,0,Math.PI*2)
                                ctx.fillStyle="rgba(225,255,247,"+(0.55+pulse*0.45)+")";ctx.fill();extra++
                            }
                            ctx.fillStyle="#69869a";ctx.font="9px sans-serif";ctx.fillText("E  /  EXCITATORY",18,height-8)
                            ctx.fillStyle="#d9aa68";ctx.fillText("I  /  INHIBITORY",130,height-8)
                            if(ids.length===0) {
                                ctx.fillStyle="#70899a";ctx.font="12px sans-serif"
                                ctx.fillText("Waiting for sampled CSR topology…",20,height/2)
                            }
                        }
                    }
                    RowLayout {
                        Layout.fillWidth: true; spacing: 14
                        Label { text: "● Excitatory"; color: "#58b5e5"; font.pixelSize: 10 }
                        Label { text: "● Inhibitory"; color: "#e4ad68"; font.pixelSize: 10 }
                        Label { text: "━ Positive weight"; color: "#579fce"; font.pixelSize: 10 }
                        Label { text: "━ Negative weight"; color: "#d9a45f"; font.pixelSize: 10 }
                        Item { Layout.fillWidth: true }
                        Label {
                            text: "FULL CSR  " + (simulation.topologyVisual.synapse_count || "—")
                            color: "#7d9aaa"; font.pixelSize: 10
                        }
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            radius: 9
            color: "#0b1722"
            border.color: "#284253"
            implicitHeight: pulseRow.implicitHeight + 16
            RowLayout {
                id: pulseRow
                anchors.fill: parent
                anchors.margins: 8
                spacing: 9
                Label { text: "SENSORY INPUT  /  CONTROLLED PULSE"; color: "#8bd9ce"; font.pixelSize: 10; font.bold: true; font.letterSpacing: 0.7 }
                Item { Layout.fillWidth: true }
                Label { text: "Neuron"; color: "#7895a5"; font.pixelSize: 10 }
                TextField { id: pulseNeuron; text: "0"; Layout.preferredWidth: 66; selectByMouse: true; validator: IntValidator { bottom: 0 } }
                Label { text: "Amplitude"; color: "#7895a5"; font.pixelSize: 10 }
                TextField { id: pulseAmplitude; text: "20"; Layout.preferredWidth: 66; selectByMouse: true; validator: DoubleValidator { bottom: 0; top: 100; decimals: 2 } }
                Label { text: "Steps"; color: "#7895a5"; font.pixelSize: 10 }
                TextField { id: pulseDuration; text: "2"; Layout.preferredWidth: 54; selectByMouse: true; validator: IntValidator { bottom: 1 } }
                Button {
                    text: "Queue pulse"
                    enabled: simulation.connected &&
                             simulation.simulationStatus.toLowerCase() !== "running" &&
                             pulseNeuron.acceptableInput && pulseAmplitude.acceptableInput &&
                             pulseDuration.acceptableInput
                    onClicked: simulation.sendStimulusPulse(
                                   Number(pulseNeuron.text),
                                   Number(pulseAmplitude.text),
                                   Number(pulseDuration.text))
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            radius: 10
            color: "#0d1924"
            border.color: "#203747"
            implicitHeight: statsRow.implicitHeight + 20
            RowLayout {
                id: statsRow
                anchors.fill: parent
                anchors.margins: 10
                spacing: 22
                ColumnLayout {
                    spacing: 2
                    Label { text: "SIMULATION"; color: "#587789"; font.pixelSize: 9 }
                    Label { text: simulation.simulationStatus.toUpperCase(); color: "#dceaf0"; font.pixelSize: 13; font.bold: true }
                }
                Rectangle { width: 1; Layout.fillHeight: true; color: "#243a49" }
                ColumnLayout {
                    spacing: 2
                    Label { text: "NEURONS"; color: "#587789"; font.pixelSize: 9 }
                    Label { text: simulation.connectivityStatistics.neuron_count===undefined?"—":String(simulation.connectivityStatistics.neuron_count); color: "#dceaf0"; font.pixelSize: 13; font.bold: true }
                }
                ColumnLayout {
                    spacing: 2
                    Label { text: "CSR SYNAPSES"; color: "#587789"; font.pixelSize: 9 }
                    Label { text: simulation.connectivityStatistics.synapse_count===undefined?"—":String(simulation.connectivityStatistics.synapse_count); color: "#dceaf0"; font.pixelSize: 13; font.bold: true }
                }
                ColumnLayout {
                    spacing: 2
                    Label { text: "COMPLETED STEPS"; color: "#587789"; font.pixelSize: 9 }
                    Label { text: simulation.completedSteps; color: "#dceaf0"; font.pixelSize: 13; font.bold: true }
                }
                ColumnLayout {
                    spacing: 2
                    Label { text: "E / I RATE"; color: "#587789"; font.pixelSize: 9 }
                    Label { text: simulation.excitatoryRateHz.toFixed(2)+" / "+simulation.inhibitoryRateHz.toFixed(2)+" Hz"; color: "#dceaf0"; font.pixelSize: 13; font.bold: true }
                }
                ColumnLayout {
                    spacing: 2
                    Label { text: "MEAN VOLTAGE"; color: "#587789"; font.pixelSize: 9 }
                    Label { text: simulation.meanVoltageMv.toFixed(2)+" mV"; color: "#dceaf0"; font.pixelSize: 13; font.bold: true }
                }
                Item { Layout.fillWidth: true }
                Button { text: window.graftExpanded?"Hide graft bay":"Graft bay"; onClicked: window.graftExpanded=!window.graftExpanded }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            visible: window.graftExpanded
            radius: 10
            color: "#101d29"
            border.color: "#31506a"
            implicitHeight: graftPanel.implicitHeight + 24
            ColumnLayout {
                id: graftPanel
                anchors.fill: parent
                anchors.margins: 12
                spacing: 8
                RowLayout {
                    Layout.fillWidth: true
                    Label { text: "GRAFT BAY  /  HOT TOPOLOGY MODULES"; color: "#8bd9ce"; font.pixelSize: 11; font.bold: true; font.letterSpacing: 1 }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: simulation.graftInspectionValid
                              ? (simulation.graftInspection.lifecycle_state||"VALIDATED").toUpperCase()
                              : "NO PACKAGE INSPECTED"
                        color: simulation.graftInspection.attached===true?"#8de0ca":"#d9b77d"
                        font.pixelSize: 10; font.bold: true
                    }
                }
                RowLayout {
                    Layout.fillWidth: true
                    TextField { id: graftPackagePath; Layout.fillWidth: true; placeholderText: "/path/to/graft-package"; selectByMouse: true }
                    Button { text: "Inspect"; enabled: graftPackagePath.text.trim().length>0; onClicked: simulation.inspectGraftPackage(graftPackagePath.text) }
                }
                Label {
                    Layout.fillWidth: true
                    visible: Object.keys(simulation.graftInspection).length>0
                    text: simulation.graftInspectionValid
                          ? (simulation.graftInspection.name||"Unnamed graft")+" · E "+
                            simulation.graftInspection.excitatory_count+" / I "+
                            simulation.graftInspection.inhibitory_count+" · "+
                            simulation.graftInspection.synapse_count+" CSR edges"
                          : (simulation.graftInspection.error||"Package inspection failed")
                    color: simulation.graftInspectionValid?"#b9d5df":"#f0a7ad"
                    font.pixelSize: 11; wrapMode: Text.Wrap
                }
                RowLayout {
                    Layout.fillWidth: true
                    Button {
                        text: "Attach frozen graft"
                        enabled: simulation.connected&&simulation.graftInspectionValid&&
                                 !simulation.graftInspection.attached&&
                                 simulation.graftInspection.lifecycle_state!=="attaching"&&
                                 simulation.graftInspection.lifecycle_state!=="ejecting"
                        onClicked: simulation.attachGraftPackage(graftPackagePath.text,false)
                    }
                    Button {
                        text: "Eject graft"
                        enabled: simulation.connected&&simulation.graftInspection.attached===true&&
                                 simulation.graftInspection.lifecycle_state!=="ejecting"
                        onClicked: simulation.detachGraftPackage()
                    }
                    Item { Layout.fillWidth: true }
                    Label { text: simulation.graftInspection.status||""; color: "#8da9b9"; font.pixelSize: 10; wrapMode: Text.Wrap }
                }
            }
        }

        Label {
            Layout.fillWidth: true
            text: simulation.errorMessage.length>0?simulation.errorMessage:simulation.statusMessage
            color: simulation.errorMessage.length>0?"#f0a7ad":"#607d8e"
            font.pixelSize: 10
            elide: Text.ElideRight
        }
    }

    Component.onCompleted: simulation.connectToServer()
}
