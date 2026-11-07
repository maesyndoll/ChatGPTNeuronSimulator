# SpikingModel


## Project ownership and AI-assisted development

SpikingModel is an independent project conceived, directed, and maintained by
its project owner. ChatGPT has been used as an AI development assistant during
the project, including for architecture discussions, code drafting and review,
debugging, and experimental analysis. AI-assisted contributions remain subject
to the project owner's review and testing.

References to ChatGPT and OpenAI describe the development tools used; they do
not indicate that this project is an OpenAI product. SpikingModel is not
developed by, affiliated with, sponsored by, or endorsed by OpenAI. ChatGPT and
OpenAI are names and trademarks of their respective owners.

Native CUDA C++ spiking-neural-network simulation project targeting NVIDIA
Blackwell (`sm_120` by default).

## Current implementation

- Central JSON parameter schema and generated C++ configuration definitions.
- Parameter validation and explicit apply/reset/reinitialize lifecycle.
- Izhikevich CUDA neuron model with deterministic initialization.
- Seeded random sparse connectivity stored in source-oriented CSR form, with
  delayed spike propagation through a GPU-resident current ring.
- Optional pair-based excitatory STDP: deterministic incoming-CSR index maps
  each target's incoming synapses back to canonical outgoing-CSR edge indices;
  GPU weights are writable and opt-in GPU pre/post trace buffers are allocated
  and zeroed. Learning is disabled by default, and the default runtime allocates
  none of the additional learning state.
- Bounded GPU spike-event capture with host-side collection.
- `CudaSimulationRuntime` owns GPU allocations, execution stream, stepping,
  reinitialization, graph-batched execution, and metric collection.
- Optional time-windowed experiment telemetry and selected-neuron voltage
  and recovery-state sampling for diagnostic traces.
- Move-only CUDA RAII wrappers provide exception-safe cleanup of partially
  allocated device state.

Pair-based excitatory STDP is implemented as an opt-in CUDA runtime feature.
The standalone executable exposes this feature with `--enable-stdp`. Learning
parameters can be overridden independently:

```bash
./build-cuda/spiking_model --config experiments/control_30s.json \
  --enable-stdp \
  --stdp-a-plus 0.005 --stdp-a-minus 0.005 \
  --stdp-tau-pre-ms 20 --stdp-tau-post-ms 20 \
  --stdp-min-weight 0.001 --stdp-max-weight 1.0 \
  --telemetry-window-steps 1000 \
  --connectivity-export experiments/logs/stdp_weights_before.csv \
  --connectivity-export-after experiments/logs/stdp_weights_after.csv \
  --log experiments/logs/stdp_training.jsonl
```

The defaults remain the reference settings in `StdpParameters`:
A+ = 0.005, A- = 0.00525, both trace time constants = 20 ms, and
excitatory weight bounds = [0.001, 1.0]. Parameter overrides require
`--enable-stdp`; server mode does not yet expose STDP controls. The optional
before/after connectivity exports preserve the exact CSR edge ordering and
allow independent weight analysis. The run summary and JSONL record report
the number of plastic excitatory synapses, how many changed, mean initial and
final weights, mean absolute weight change, and maximum absolute change.
Inhibitory-source weights are not plastic. A zero changed-synapse count is
reported as such rather than being treated as a successful learning result.

At each simulation step, the runtime decays traces, applies depression for
presynaptic excitatory spikes and potentiation for postsynaptic spikes, clamps
excitatory weights, increments traces, propagates the preceding step's spikes,
and integrates the next step. Learning kernels run in the same CUDA stream
and are included in captured CUDA Graph batches; there are no per-step host
readbacks. CPU-reference and GPU-kernel tests cover pair ordering, simultaneous
events, timing curves, and inhibitory-weight preservation. A separate host
test validates before/after weight summaries and confirms connectivity
structure is unchanged during training.

The Qt Quick dashboard and asynchronous IPC bridge already exist;
STDP-specific controls, live plasticity analysis, and optional calibration
workflows remain future integration milestones. Checkpointing is also planned.
The current
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

## Neuromancer CUDA phase-timing assay (V8)

The `neuromancer_phase_runtime` target is an integration experiment that
runs on the actual CUDA Izhikevich runtime, with a sparse E/I CSR graph,
delayed propagation, scheduled external-current pulses, and recorded spike
timestamps. It compares three phase-coded assemblies plus asynchronous
background activity against a control that preserves each cycle's event-time
histogram but shuffles those times across neuron IDs.

Build and run it on a compatible NVIDIA/CUDA system:

```bash
cmake -S . -B build-cuda -DSPIKINGMODEL_ENABLE_CUDA=ON \\
  -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_TESTING=ON
cmake --build build-cuda --target neuromancer_phase_runtime --parallel
./build-cuda/neuromancer_phase_runtime
```

The phase address is computed from each recorded spike timestamp relative to
an externally imposed 10 Hz reference. The address calculation receives no
assembly membership labels; labels are used only afterward to score precision,
recall, and rejection. This validates temporal phase separation through the
spiking backend, not learned phase addresses, autonomous oscillators, or
associative inference. STDP is disabled in this assay to isolate timing.

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


## Measurement note: matched controls for subthreshold voltage effects

The V30 CSR phase-gated E/I transmission assay exposed an important measurement
caveat in some earlier diagnostic experiments. An absolute voltage difference
from an assumed resting value can combine synaptic input with the neuron's
intrinsic membrane evolution. It should not, by itself, be interpreted as the
synapse's isolated contribution.

For future claims about subthreshold synaptic effects, use matched runs with
identical neuron initialization, stimulation, timing, and runtime settings,
differing only in the presence or absence of the tested CSR edge. Estimate the
synaptic contribution as:

`V_with_edge - V_matched_no_edge_control`

The corrected V30 assay demonstrated this control for one-step E and I
transmission: +0.413574 mV for the excitatory edge and -0.411072 mV for the
inhibitory edge; phase-rejected transmission measured 0 mV relative to the
matched control in both cases. This validates the tested signed transmission
and gate behavior under that assay's conditions. It does not establish
autonomous oscillation, learned per-edge phase offsets, or nonlinear E/I phase
coupling.

This caveat applies to interpretation of earlier absolute subthreshold-voltage
diagnostics where a matched no-input control was not used. It does not, by
itself, invalidate network-level classification, held-out accuracy, shuffled-
label controls, spike-count observations, or topology measurements. Those
results should remain tied to their own measured endpoints and experimental
controls. Earlier diagnostic voltage magnitudes are retained as historical
measurements and should not be treated as isolated synaptic-effect estimates
unless independently verified with matched controls.
