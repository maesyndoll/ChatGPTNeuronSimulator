# SpikingModel

Native CUDA C++ spiking-neural-network simulation project targeting NVIDIA
Blackwell (`sm_120` by default).

## Current implementation

- Central JSON parameter schema and generated C++ configuration definitions.
- Parameter validation and explicit apply/reset/reinitialize lifecycle.
- Izhikevich CUDA neuron model with deterministic initialization.
- `CudaSimulationRuntime` owns GPU allocations, execution stream, stepping,
  reinitialization, and batch metric collection.
- Move-only CUDA RAII wrappers provide exception-safe cleanup of partially
  allocated device state.
- Seeded random sparse CSR connectivity with population-specific connection
  probabilities, signed excitatory/inhibitory weights, and configurable delays.
- GPU-resident spike queues, delayed synaptic-current rings, CUDA spike
  propagation, and a bounded GPU spike-event recorder.
- Bounded-capacity GPU CSR connectome with per-row active edge counts. Associative
  co-activity can add missing directed edges; weak edges are pruned in-place.
  Propagation, phase coupling, and STDP all consume the active GPU CSR rows.
- A standalone Neural Observatory experiment that exports a spike raster as
  an HTML report and every captured event as CSV.

The runtime now supports opt-in GPU-resident pair-based spike-timing-dependent
plasticity (STDP). Per-neuron spike traces remain on the GPU, updates preserve
the excitatory/inhibitory sign of each CSR edge, and configurable magnitude
bounds prevent unbounded weight growth. An opt-in structural-plasticity pass
uses those traces to grow connections between co-active neurons and prunes
sub-threshold edges. CSR storage is GPU-resident and reserves bounded per-row
capacity so structural updates do not require a CPU graph rebuild or per-step
host synchronization. Both weight and topology plasticity are disabled by
default. Learned-connectome checkpoint persistence and automatic capacity
reallocation when a row saturates are not yet implemented. The Qt
client/dashboard is optional.

## CPU-only configuration tests

```bash
cmake -S . -B build-cpu -DSPIKINGMODEL_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

## CUDA build and smoke test

On the RTX 5070 Ti with a compatible CUDA Toolkit and driver:

```bash
cmake -S . -B build-cuda -DSPIKINGMODEL_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_TESTING=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
./build-cuda/spiking_model --small
```

The CUDA CTest includes `cuda_runtime_smoke`, which exercises initialization,
step-boundary parameter updates, continued execution, full reinitialization,
phase synchronization, STDP weight adaptation, associative CSR
growth, and structural pruning experiments. The runtime API accepts a per-neuron external-current vector that
is copied to GPU memory and added during neuron integration. The recall test uses an explicit structured CSR graph in which A projects to B
and C has no incoming edges. Training presents A before B to create the causal
pre/post timing required by the current pair-based STDP rule; recall then
stimulates A alone and compares B activity against C and an identically seeded
zero-STDP control. It also checks that training generated spike events. This is
a controlled functional test, not evidence of general intelligence.
The CPU-only tests do not verify CUDA compilation or GPU execution. See
[`docs/RUNTIME_FOUNDATION.md`](docs/RUNTIME_FOUNDATION.md) for implementation
notes and lifecycle semantics.

## Neural Observatory showcase

The standalone showcase exercises the GPU-resident CSR network and
spike-propagation path, then writes a browser-viewable spike raster and raw
event log. The separate `associative_phase_experiment` target exercises
GPU-side weight plasticity plus bounded CSR growth/pruning, and exports
per-epoch edge-count and activity telemetry to CSV.

Build it with the same CUDA configuration:

```bash
cmake --build build-cuda --target neural_observatory --parallel
./build-cuda/neural_observatory
```

Open the generated report:

```bash
xdg-open neural-observatory-output/index.html
```

Optional controls:

```bash
./build-cuda/neural_observatory --steps 4000 --device 0 \
  --output my-neural-experiment
```

The output directory contains `index.html` and `spikes.csv`. The report
includes the raster, excitatory/inhibitory activity over time, firing-rate
metrics, mean membrane voltage, and the detected GPU model. The showcase
currently focuses on visualization; online STDP is available through the
runtime API and is covered by a separate GPU smoke experiment.

## Live neural observatory dashboard

The optional Qt Quick dashboard includes:
- A scene-graph-rendered, rotating 3D-style neural population field with recent
  spike highlights and excitatory/inhibitory coloring.
- A population activity gauge and scrolling oscilloscope trace.
- A radix-2 FFT magnitude view of the recent combined population-rate samples.
  The spectrum is relative magnitude; frequency labels require a calibrated
  telemetry sampling interval.
- The existing live spike raster and simulation controls.

Build the dashboard when Qt 6.4+ development packages are installed:

```bash
cmake -S . -B build-cuda \
  -DSPIKINGMODEL_ENABLE_CUDA=ON \
  -DSPIKINGMODEL_ENABLE_QT_CLIENT=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda --target spiking_model spiking_model_dashboard --parallel
```

Run the simulation server in one terminal and the dashboard in another:

```bash
./build-cuda/spiking_model
./build-cuda/spiking_model_dashboard
```

The Qt Quick scene graph uses the platform's configured graphics backend. This
dashboard is a 3D-style projected population view, not a Qt Quick 3D mesh renderer.
The runtime's STDP option is currently configured through the C++ runtime API;
the dashboard does not yet expose learning controls or learned-weight
checkpoint persistence.
