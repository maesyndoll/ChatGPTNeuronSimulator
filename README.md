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

On the RTX 5070 Ti with a compatible CUDA Toolkit and driver:

```bash
cmake -S . -B build-cuda -DSPIKINGMODEL_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_TESTING=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
./build-cuda/spiking_model --small
```

The CUDA CTest includes `cuda_runtime_smoke`, which exercises initialization,
step-boundary parameter updates, continued execution, and full reinitialization.
The CPU-only tests do not verify CUDA compilation or GPU execution. See
[`docs/RUNTIME_FOUNDATION.md`](docs/RUNTIME_FOUNDATION.md) for implementation
notes and lifecycle semantics.


## Reproducible experiment configurations

The standalone executable can load a JSON parameter file without recompiling.
The checked-in full-network control configuration fixes the neuron, topology,
and delay seeds:

```bash
./build-cuda/spiking_model --config experiments/deterministic_control.json
```

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
floating-point accumulation can vary. The deterministic-control test suite
will distinguish exact host-side reproducibility from GPU numerical
repeatability.
