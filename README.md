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
- A standalone Neural Observatory experiment that exports a spike raster as
  an HTML report and every captured event as CSV.

The runtime now supports opt-in GPU-resident pair-based spike-timing-dependent
plasticity (STDP). Per-neuron spike traces remain on the GPU, updates preserve
the excitatory/inhibitory sign of each CSR edge, and configurable magnitude
bounds prevent unbounded weight growth. Plasticity is disabled by default.
Learned-weight checkpoint persistence and inference-oriented memory management
are not yet implemented. The Qt client/dashboard is optional.

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
phase synchronization, and an STDP learning/control experiment. The STDP
experiment checks that repeated activity changes mean absolute synaptic weight
while a zero-learning control remains unchanged and configured bounds are
respected.
The CPU-only tests do not verify CUDA compilation or GPU execution. See
[`docs/RUNTIME_FOUNDATION.md`](docs/RUNTIME_FOUNDATION.md) for implementation
notes and lifecycle semantics.

## Neural Observatory showcase

The standalone experiment exercises the existing GPU-resident CSR network and
spike-propagation path, then writes a browser-viewable spike raster and raw
event log. It does not claim to train the network; synaptic weights are fixed
for this experiment.

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
