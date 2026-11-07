# Neuromancer model packages (V25)

A model package is a directory containing a manifest and its serialized CSR topology:

```text
model-package/
├── model.nmc       # versioned inference manifest
├── topology.csr    # exact SPKCSR v1 graph and weights
└── inference.csv   # optional benchmark output
```

The manifest is UTF-8 text. Version 1 currently supports these keys:

```text
NEUROMANCER_MODEL 1
task two_class_binary_feature_inference
input_features 64
output_neurons 64 65
dt_ms 0.5
spike_threshold_mv -65
simulation_steps 100
stimulus_start_step 10
stimulus_duration_steps 4
input_current 500
active_feature_threshold 0.5
topology_file topology.csr
readout supervised_class_conditional_feature_frequency
```

`topology_file` is resolved relative to the manifest, so the package can be moved as a directory. The loader rejects unsupported manifest versions, missing required fields, unknown keys, invalid dimensions, invalid input/output neuron IDs, and malformed CSR files.

## Build and create the demonstration package

```bash
cmake -S . -B build-cuda -DSPIKINGMODEL_ENABLE_CUDA=ON
cmake --build build-cuda --target neuromancer_infer neuromancer_inference_prototype_v23 -j$(nproc)
./build-cuda/neuromancer_inference_prototype_v23 ~/projects/neuromancer/models/two_class_demo
```

## Load and test the saved package

```bash
./build-cuda/neuromancer_infer --self-test ~/projects/neuromancer/models/two_class_demo/model.nmc
```

For normal inference, pass the manifest and a one-line CSV containing exactly 64 finite feature values in [0, 1]:

```bash
./build-cuda/neuromancer_infer ~/projects/neuromancer/models/two_class_demo/model.nmc input.csv
```

## Current scope

V25 packages the frozen CSR readout and inference settings used by the current two-output classifier. It does not yet serialize arbitrary neuron parameter sets, learned phase dynamics, plasticity rules, optimizer/training state, or arbitrary output decoders. The CUDA runtime's Izhikevich neuron parameters remain runtime defaults. The demonstration readout is supervised class-conditional feature-frequency weighting, not STDP-trained inference.
