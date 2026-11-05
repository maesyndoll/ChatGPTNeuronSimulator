# Configuration and simulation lifecycle

## Parameter source of truth

`config/parameters.schema.json` is the authoritative source for parameter
names, C++ types, defaults, ranges, units, mutability, and descriptions.
`tools/generate_parameters.py` generates
`include/core/parameters_generated.h`. Do not edit the generated header by
hand. Regenerate it with:

```bash
python3 tools/generate_parameters.py
```

Check that the generated file is current with:

```bash
python3 tools/generate_parameters.py --check
```

The generated `kParameterDescriptors` table is intended to be consumed by the
future Qt/QML settings model; UI controls should not duplicate ranges/defaults.

## Parameter lifecycle

`ParameterManager` maintains three snapshots:

- **defaults**: schema-derived baseline configuration;
- **requested**: most recently accepted configuration;
- **active**: values currently safe for the running simulation allocation.

`apply_parameters()` validates a candidate atomically. Step-boundary values
(timestep, threshold, run length) are published to the active snapshot;
initialization/topology values are staged and reported as requiring
reinitialization.

`reset_parameters_to_defaults()` restores configuration defaults without
resetting neuron voltages, synapses, learned weights, or plasticity traces.
Structural settings that would change allocated state remain staged until a
full reinitialization.

`reinitialize_simulation()` commits the requested configuration and increments
the simulation generation. The runtime owner must perform the actual GPU-state
rebuild at a safe point using the returned snapshot. The manager itself does
not own GPU allocations or neural state.

## Build and test

CPU-only configuration tests can run without a CUDA toolkit:

```bash
cmake -S . -B build-cpu -DSPIKINGMODEL_ENABLE_CUDA=OFF
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

The CUDA executable remains enabled by default and targets `sm_120`:

```bash
cmake -S . -B build -DSPIKINGMODEL_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
```

A successful CPU-only test run does not verify CUDA compiler, driver, or GPU
compatibility.
