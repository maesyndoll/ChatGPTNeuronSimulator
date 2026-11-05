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
