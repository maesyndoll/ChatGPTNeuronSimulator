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

The current simulation still models independent neurons. Sparse synapses,
spike propagation, plasticity, checkpointing, telemetry streaming, and the Qt
UI are future milestones.

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
