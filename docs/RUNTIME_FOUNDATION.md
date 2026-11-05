# Runtime foundation (Milestone 2)

## What changed

- `CudaSimulationRuntime` now owns the simulation lifecycle and the GPU allocations.
- GPU allocations use move-only `DeviceBuffer<T>` RAII wrappers. A partially
  completed allocation sequence is cleaned up automatically if a later
  allocation fails.
- `CudaStream` owns a non-blocking CUDA stream. Initialization and timestep
  kernels are submitted to that stream, with explicit synchronization at
  initialization and telemetry collection boundaries.
- CUDA launch/API errors are converted to exceptions with expression and source
  location information.
- The executable delegates initialization, stepping, and final metric
  collection to the runtime instead of managing raw device pointers itself.

## Lifecycle semantics

- `apply_parameters(candidate)` validates the candidate. Step-boundary values
  (`steps`, `dt`, `threshold`) become active together. Structural values are
  retained as requested values and reported as pending reinitialization.
- `reset_parameters_to_defaults()` restores requested defaults and immediately
  restores step-boundary values, but does not reinitialize or clear device
  neuron state. Structural defaults remain pending until explicit
  reinitialization.
- `reinitialize_simulation()` first allocates and initializes a replacement
  device state. Only after that succeeds does it commit the requested parameter
  snapshot and swap out the old state. It resets the step/time counters.

## Current limitations

- The current CUDA backend still simulates independent neurons; synaptic
  connectivity and spike propagation are not implemented yet.
- `collect_metrics()` copies the voltage, recovery, and spike-count arrays to
  host memory. This is a final/batch measurement path, not the future streaming
  telemetry path. A decimated GPU reduction and pinned asynchronous telemetry
  buffers are planned separately.
- This milestone has not been compiled with NVCC in the development container.
  Build and run it on the target machine with CUDA Toolkit support and
  `CMAKE_CUDA_ARCHITECTURES=120` before treating GPU execution as verified.
- `DeviceBuffer` deliberately ignores errors in its destructor; operational
  allocation, launch, copy, and synchronization calls are checked and throw.

## Build and test

CPU-only configuration/lifecycle tests:

```bash
cmake -S . -B build-cpu -DSPIKINGMODEL_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

CUDA build on the RTX 5070 Ti:

```bash
 cmake -S . -B build-cuda -DSPIKINGMODEL_ENABLE_CUDA=ON \
   -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_TESTING=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
./build-cuda/spiking_model --small
```
