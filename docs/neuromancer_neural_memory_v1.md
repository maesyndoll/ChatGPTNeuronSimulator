# Neuromancer neural memory store — batch 1

This batch introduces persistent, independently addressable neural modules.
It is deliberately data-only: it does not attach a module to a running CUDA
simulation or load executable code.

## Package layout

The memory root contains one directory per module:

\`\`\`text
memory/
  visual.features/
    neuromancer.module.json
    topology.spkcsr
    state.nstate
  association.edges/
    neuromancer.module.json
    topology.spkcsr
    state.nstate
\`\`\`

The manifest records the module ID, display name, population dimensions, and
relative filenames. The topology uses the existing versioned SPKCSR format.
The state file uses the versioned NSTATE format and can preserve optional
per-neuron voltage, recovery, adaptation, phase, and STDP trace arrays, plus
per-edge phase-lag and plasticity-mask arrays.

Optional state arrays may be empty when the source runtime does not expose that
state yet. Every non-empty array is dimension-checked against the CSR graph;
non-finite floats, invalid mask values, malformed files, path traversal, and
manifest/topology dimension mismatches are rejected.

## API

\`\`\`cpp
spiking::NeuralMemoryStore memory("models/neural-memory");
memory.save("visual.features", "Visual Features", topology, state);
auto module = memory.load("visual.features");
auto modules = memory.list();
memory.erase("visual.features");
\`\`\`

Saving builds a complete staging directory and then publishes it by directory
rename. Existing modules are replaced through a backup/rename transaction.
Module IDs are restricted to letters, digits, dot, dash, and underscore.

## Build and test

\`\`\`bash
cmake --build build-cuda --target neural_memory_store_tests -j$(nproc)
./build-cuda/neural_memory_store_tests
\`\`\`

## Scope boundary

This is batch 1: persistent module packages and state round-tripping. Batch 2
will add a runtime module registry, bounded host/GPU residency, synchronized
CSR graph replacement, neuron/edge remapping, and safe attach/detach. Batch 3
will add stimulus-driven retrieval and recurrent inference control.

NVMe-to-GPU direct storage is not assumed. This format is storage-backend
agnostic so later asynchronous host staging or GPUDirect Storage can use the
same module packages.
