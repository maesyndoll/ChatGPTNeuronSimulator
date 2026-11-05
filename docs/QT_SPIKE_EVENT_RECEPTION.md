# Qt spike-event reception

The optional Qt dashboard receives real spike records from the simulation server.
The CUDA runtime remains independent of Qt; the dashboard only consumes IPC data.

## Transport

The Qt asynchronous client recognizes `spike_events` notifications in addition to
`telemetry_update` notifications. It emits `spikeEventsReceived` with the
notification's `spike_events` object. A malformed notification is treated as a
transport error and the connection is closed.

On connection, `SimulationBridge` subscribes to both telemetry and spike events.
It also requests the retained event snapshot so the dashboard can populate after
connecting to an already-running simulation. Event sequence numbers are used to
deduplicate overlap between the snapshot response and asynchronous notifications.

## Dashboard-side retention

The bridge exposes `spikeEvents` as a `QVariantList`. Each item contains:
- `sequence`: decimal string
- `step`: decimal string
- `neuron_id`: integer

The bridge retains at most 50,000 events for the upcoming raster renderer.
When the simulation generation changes, the retained list and sequence cursor
are reset. Decimal strings are preserved for 64-bit sequence and step values.

The dashboard currently displays the retained count, server total, dropped-event
count, and simulation generation. This is an ingestion/transport checkpoint;
the raster renderer is a separate next step.

## Build

With Qt support enabled in the existing build directory:

```bash
cmake --build build-cuda --target spiking_model_dashboard -j2
```

Then run the simulation server and dashboard in separate terminals as described
in the project README.
