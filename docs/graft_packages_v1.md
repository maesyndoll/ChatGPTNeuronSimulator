# Neuromancer graft packages (schema v1)

A graft package is a directory containing a manifest and a validated SPKCSR topology.
The first inspector milestone deliberately treats packages as data: it does not
execute a module binary, change a live CUDA graph, or modify a running simulation.

## Minimal package layout

```text
example-graft/
  neuromancer.graft.json
  topology.spkcsr
  adapters/
    # Optional adapter assets referenced by future schema extensions.
  module/
    # Optional trusted Linux .so module.
```

## Manifest

```json
{
  "schema_version": 1,
  "id": "example.codec_adapter",
  "name": "Example Codec Adapter",
  "description": "A small excitatory adapter population",
  "excitatory_count": 64,
  "inhibitory_count": 8,
  "topology_file": "topology.spkcsr",
  "module_binary": null,
  "input_adapters": ["codec.input.v1"],
  "output_adapters": ["codec.output.v1"]
}
```

- `schema_version` must be 1.
- `id`, `name`, and `topology_file` are required non-empty strings.
- Population counts must be non-negative uint32 values and their sum must be nonzero.
- The declared counts must exactly match the SPKCSR topology.
- `topology_file` and optional `module_binary` must resolve to regular files
  inside the package directory. Absolute paths and parent-directory traversal
  are rejected.
- `module_binary`, when supplied, must have a `.so` extension. It is only
  identified, never loaded, by the inspector.
- Adapter name arrays are optional and default to empty.

## Inspect

```bash
cmake --build build-cuda --target neuromancer_graft_inspect graft_package_tests -j$(nproc)
./build-cuda/neuromancer_graft_inspect ./example-graft
./build-cuda/graft_package_tests
```

## Lifecycle boundary

The next runtime milestone will introduce a registry and worker-thread commands
for staging, compatibility checking, attaching, detaching, and persisting graft
state. The CUDA runtime currently owns one fixed CSR allocation; a safe live graft
therefore requires an explicit topology replacement/merge transaction, preservation
of neuron state, rebuilding reverse indices and graph-captured resources, and a
rollback path. Dynamic module unload must happen only after callbacks and GPU work
using its resources have completed. The UI must report staged/attached/detaching
states from engine acknowledgements, never optimistically animate a successful graft.
