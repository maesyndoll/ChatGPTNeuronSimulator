# Neuromancer Neural Observatory Prototype v1

## Purpose

The primary QML surface is now a compact, live neural observatory rather than a configuration-first dashboard. It is intended to make the running sparse spiking network the central visual artifact.

## Panels

- **Neural field:** a time-decaying visual projection of recently observed spike events. Each illuminated cell is keyed to a neuron ID. This is an event-derived projection, not a raw audio/video/sensor display and not a claim about learned sensory perception.
- **Spike raster:** plots retained spike events against simulation step and neuron index, with excitatory and inhibitory populations distinguished.
- **Sparse row topology:** lays neuron IDs out in stable row order and draws the sampled CSR edges returned by the runtime. Positive and negative edge weights use different colors. A neuron flashes only when an actual spike event for that neuron is received. Recently active presynaptic neurons temporarily brighten their outgoing displayed edges.
- **Runtime strip:** shows live simulation status, neuron and CSR synapse counts, completed steps, population firing rates, and mean voltage.
- **Graft bay:** retains package inspection, frozen attach, and eject actions in a collapsible panel.

## Evidence and limitations

All firing flashes and raster marks are driven by the existing spike-event stream. The topology is the sampled CSR visualization supplied by the backend; it is not a complete rendering when the backend reports a sampled subset. The neural-field panel is a deterministic projection of neuron IDs into a display grid, with a short visual decay to make recent events legible.

The current backend does not expose a per-neuron resonance score or phase measurement through this UI contract. The prototype therefore does not label any neuron as resonant. Nor does it yet implement a continuous external sensory-stream adapter. Those are separate backend milestones; visual decoration must not be mistaken for measured neural behavior.

## First validation

Build the existing dashboard target and launch it against the current simulation server. Verify connection, topology population, event-driven node flashes, raster updates, telemetry, and graft bay controls. Run a network with little or no activity as well as one that fires: the display should remain informative without inventing activity.
