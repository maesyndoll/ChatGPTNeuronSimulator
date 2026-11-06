# SpikingModel

Native CUDA C++ spiking-neural-network simulation project targeting NVIDIA
Blackwell (`sm_120` by default).

## Current implementation

- Central JSON parameter schema and generated C++ configuration definitions.
- Parameter validation and explicit apply/reset/reinitialize lifecycle.
- Izhikevich CUDA neuron model with deterministic initialization.
- Seeded random sparse connectivity stored in source-oriented CSR form, with
  delayed spike propagation through a GPU-resident current ring.
- Optional STDP foundation: deterministic incoming-CSR index maps each target's
  incoming synapses back to canonical outgoing-CSR edge indices; GPU weights
  are writable and opt-in GPU pre/post trace buffers are allocated and zeroed.
  This is infrastructure only: no learning rule or weight updates are active,
  and the default runtime allocates none of the additional learning state.
- Bounded GPU spike-event capture with host-side collection.
- `CudaSimulationRuntime` owns GPU allocations, execution stream, stepping,
  reinitialization, graph-batched execution, and metric collection.
- Optional time-windowed experiment telemetry and selected-neuron voltage
  and recovery-state sampling for diagnostic traces.
- Move-only CUDA RAII wrappers provide exception-safe cleanup of partially
  allocated device state.

Pair-based excitatory STDP is implemented as an opt-in CUDA runtime feature.
The runtime maintains GPU-resident pre/post traces and applies trace decay,
presynaptic depression, postsynaptic potentiation, and trace increments at
each event boundary before spike propagation. The learning kernels run in
the same CUDA stream and are included in captured CUDA Graph batches; there
are no per-step host readbacks. Inhibitory-source weights remain fixed.
Learning is disabled by default and can be enabled through the runtime
constructor for development and tests; STDP parameters are currently supplied
through that constructor rather than the JSON/IPC configuration API.
CPU-reference and GPU-kernel tests cover pair ordering, simultaneous events,
and inhibitory-weight preservation. Checkpointing and the Qt UI remain future
milestones. The current
connectivity generator supports seeded random sparse networks; structured
population-to-population connectivity is a planned extension.

## Planned architecture: Meta-topology and ActiveConnectome

The next architectural direction is to preserve each existing CSR network as
an independently executable module, connect modules through a separate
bridge-connectivity layer, and coordinate them through a system-level
`MetaTopology`. An `ActiveConnectome` layer will observe the evolving system
and distinguish:

- **Structural connectivity:** which local and inter-module edges exist.
- **Functional connectivity:** which pathways participate in communication
  during a defined observation window.
- **Plastic connectivity:** how synaptic weights and plasticity state evolve.

This is an architecture proposal, not an implemented feature. The design
prioritizes preserving the validated local CSR and CUDA runtime, deterministic
inter-module event delivery, independently measurable local and bridge
plasticity, and eventual SNN–ANN interoperability. A fly-brain-inspired
organism model is a motivating research use case, subject to explicit dataset
provenance and biological validation.

See [Active Connectome and Meta-Topology Architecture](docs/architecture/active_connectome.md)
for the design, execution semantics, learning boundaries, experimental
requirements, and incremental implementation milestones.

## CPU-only configuration tests

```bash
cmake -S . -B build-cpu -DSPIKINGMODEL_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

## CUDA build and smoke test

On an NVIDIA GPU with a compatible CUDA Toolkit and driver:

```bash
cmake -S . -B build-cuda -DSPIKINGMODEL_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_TESTING=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
./build-cuda/spiking_model --small
```

## Experiment configurations and automatic results log

The standalone executable loads JSON parameter files without recompiling.
The checked-in full-network control configuration fixes the neuron, topology,
and delay seeds:

```bash
./build-cuda/spiking_model --config experiments/deterministic_control.json
```

Every completed standalone run is appended as one JSON object per line to:

```text
experiments/logs/experiment_log.jsonl
```

The log is append-only: running another experiment does not overwrite earlier
records. Each record includes a UTC timestamp, application version, source
configuration path, complete effective parameter configuration, random seeds,
GPU model and compute capability, completed steps, spike counts, firing rates,
mean membrane voltage, numerical validity, and simulation wall time. The
parent directory is created automatically.

To use a different log file:

```bash
./build-cuda/spiking_model --config experiments/deterministic_control.json \
  --log experiments/logs/control_repeats.jsonl
```

To run without writing a record, use `--no-log`. The log is local experiment
data; review and back it up before cleaning the working tree. Keep raw results
with their original configurations so later comparisons remain auditable.

To record membrane-voltage traces for selected neurons, use the
optional voltage-trace arguments:

```bash
./build-cuda/spiking_model --config experiments/isolated_neurons_baseline.json \
  --voltage-trace experiments/logs/voltage_trace.csv \
  --voltage-trace-neuron 0 --voltage-trace-neuron 1 \
  --voltage-trace-interval 1
```

The CSV contains `step,time_ms,neuron_id,voltage_mv,recovery,spiked_this_step`;
the final column is 1 when that neuron spiked during the sampled step. The
`recovery` column records the Izhikevich recovery variable (u), after
integration and any spike-triggered increment (d). It includes the
initialized state at step 0 and post-integration samples at each interval,
including the final partial interval. Voltage is sampled after threshold
handling, so a neuron that spikes on a sampled step is recorded at its reset
voltage; `spiked_this_step` marks that reset directly. Smaller intervals
provide finer temporal resolution but add host/device synchronization and
readback overhead. The trace mode currently cannot be
combined with `--telemetry-window-steps` or server mode. Repeat
`--voltage-trace-neuron` to record multiple neuron IDs.

To export the exact connectivity graph allocated on the GPU, use
`--connectivity-export`:

```bash
./build-cuda/spiking_model --config experiments/deterministic_control.json \
  --connectivity-export experiments/logs/connectivity.csv
```

The export is copied from the runtime's active GPU-resident CSR arrays after
synchronizing the simulation stream; it does not regenerate the graph. The CSV
contains one row per directed synapse with these columns:

- `source_neuron`, `target_neuron`: zero-based neuron IDs;
- `source_population`, `target_population`: `excitatory` or `inhibitory`;
- `weight`: signed synaptic weight (excitatory sources are positive,
  inhibitory sources negative);
- `delay_steps`: integer propagation delay in simulation steps.

Rows are emitted in source-neuron CSR order. The export is performed before
simulation begins and does not alter the graph or simulation state. The output
file is replaced if it already exists; choose a new path when retaining
multiple experiment exports. This option is for standalone mode, not server
mode.

Configuration files use the same `simulation` and `topology` groups as the
IPC parameter API. They may contain partial updates; omitted values retain
the normal defaults. The 64-bit topology and delay seeds are decimal strings
to preserve their full integer precision.

To create a new experiment, copy the example JSON and change only the
parameters under investigation. Keep all other settings and seeds unchanged
when comparing against the control. The current implementation uses the three
explicit random seeds independently; `master_seed` is recorded as metadata
and does not yet derive the other streams.

Fixed seeds guarantee repeatable initialization and generated connectivity
for the same build and configuration. They do not currently promise
bit-for-bit identical CUDA simulation metrics across repeated runs; parallel
floating-point accumulation can vary. The CUDA `determinism_smoke` test
compares two full-size runs: topology arrays are compared exactly by
`topology_tests`, the initial neuron parameter and state arrays are compared
through an exact state fingerprint, and final spike counts/mean voltage are
compared with tight tolerances.
