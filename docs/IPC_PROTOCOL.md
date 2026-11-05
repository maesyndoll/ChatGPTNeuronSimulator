# IPC Protocol v1

This document describes the local interface provided by
`spiking_model --server`.

## Transport

- POSIX Unix-domain stream socket (`AF_UNIX`, `SOCK_STREAM`).
- Default path: `/tmp/spiking_model-<uid>.sock`.
- Override with `--socket PATH`.
- Socket permissions: owner-only (`0600`).
- One active client connection at a time.
- UTF-8 JSON, one request per newline-terminated line.
- Maximum request line: 1 MiB.
- Multiple requests can share a connection.
- Requests can be fragmented across writes.
- Responses can be fragmented across reads.
- Clients must implement newline framing.

## Request format

Every request is a JSON object containing:

- `protocol_version`: integer, currently `1`.
- `request_id`: non-empty string, echoed in the response.
- `command`: command name.
- `payload`: JSON object, including for commands with no arguments.

## Response format

A successful response contains `protocol_version`, `request_id`,
`ok: true`, and `result`.

An unsuccessful response contains `protocol_version`, `request_id`,
`ok: false`, and an `error` object with `code` and `message`.

Transport errors may use a null `request_id`.

Asynchronous telemetry events share the response stream. Clients
distinguish responses by `request_id` and events by `event`.

## Commands

| Command | Required payload |
|---|---|
| `get_status` | `{}` |
| `get_parameters` | `{}` |
| `get_telemetry` | `{}` |
| `subscribe_telemetry` | `{}` |
| `unsubscribe_telemetry` | `{}` |
| `apply_parameters` | `{"parameters": {...}}` |
| `reset_parameters` | `{}` |
| `reinitialize` | `{}` |
| `start` | `{}` |
| `pause` | `{}` |
| `step` | `{"steps": positive_integer}` |

`step` accepts a positive 32-bit integer.

## Parameters and lifecycle

Parameter names, types, defaults, ranges, units, and mutability are
defined in `config/parameters.schema.json`.

Parameter updates are partial: omitted fields retain their values from
the current requested configuration. The resulting complete configuration
is validated before acceptance.

The parameter update result includes `accepted`,
`requires_reinitialization`, `parameter_version`,
`simulation_generation`, and an `errors` array.

64-bit seeds, memory budgets, counters, sequence numbers, versions,
and generations are represented as decimal strings in JSON. This avoids
precision loss in JavaScript clients.

`reset_parameters` restores requested settings to defaults. It does not
clear neural state or learned weights.

`reinitialize` applies requested initialization settings and rebuilds
simulation state. Resetting settings and reinitializing are distinct
operations.

## Status

The status field can be `ready`, `running`, `paused`, `faulted`, or
`shutting_down`. The `is_running` field is boolean.

- `start` begins continuous execution.
- `pause` pauses continuous execution.
- `step` advances the requested number of steps.
- Controller commands are serialized at command/batch boundaries.

## Telemetry

Telemetry contains:

- `sequence`: publication sequence, decimal string.
- `completed_steps`: completed steps, decimal string.
- `excitatory_spikes`, `inhibitory_spikes`: cumulative counts,
  decimal strings.
- `last_step_excitatory_spikes`, `last_step_inhibitory_spikes`:
  per-step integer counts.
- `mean_voltage_mv`: mean membrane voltage.
- `excitatory_rate_hz`, `inhibitory_rate_hz`: firing rates.
- `finite_state`: boolean.

A telemetry event has `protocol_version: 1`,
`event: "telemetry_update"`, and a `telemetry` object.

Subscriptions are per connection and initially disabled.
Subscribe acknowledgements do not promise an immediate event.
Unsubscribe prevents future event generation, but a frame already
selected or partially transmitted may still complete.

Pending unsent telemetry is coalesced to the latest snapshot. Telemetry
is therefore a live notification stream, not a lossless history.

## Errors

Current error codes include `invalid_request`, `unsupported_protocol`,
`unknown_command`, `invalid_parameters`, `command_failed`,
`internal_error`, `invalid_json`, `request_too_large`, and
`transport_error`.

Clients should use error codes rather than parsing message text.

## Running

Server mode:

    ./build-cuda/spiking_model --server
    ./build-cuda/spiking_model --server --small
    ./build-cuda/spiking_model --server --small --socket /tmp/spiking-test.sock

Original one-shot mode remains available:

    ./build-cuda/spiking_model --small

## Current limitations

- One active client at a time.
- No authentication beyond socket filesystem permissions.
- No telemetry history or replay.
- No request cancellation.
- No TCP transport.
- Protocol version 1.


## Spike-event recording

Spike events are available through the controller's bounded, cached event
history. The CUDA runtime remains owned by the controller worker; IPC clients
never access CUDA buffers directly.

Request commands:

- `get_spike_events` with optional `payload.after_sequence` (a decimal string)
  returns `first_sequence`, `total_events`, `dropped_events`,
  `simulation_generation`, and an `events` array.
- `subscribe_spike_events` enables asynchronous `spike_events` notifications
  for the current connection.
- `unsubscribe_spike_events` disables those notifications.

Each event contains a decimal-string `sequence`, a decimal-string simulation
`step`, and numeric `neuron_id`. The first event sequence in a generation is
zero. `first_sequence` identifies the oldest event still retained by the
controller. A client whose cursor precedes it has missed retained history;
`dropped_events` reports cumulative GPU-ring overwrites and controller-history
evictions for the current simulation generation. Reinitialization starts a
new generation and resets the event sequence.

The controller retains at most 262,144 events. IPC notifications are batched
(up to 4,096 events per frame), and the client can continue from the last
received sequence using `get_spike_events` or the subscription stream.
