# V16 — Frozen-runtime recruitment curve

V16 isolates training from recall and measures how cue size affects postsynaptic
recruitment.

## Protocol

- Train A→B and C→D with 100 interleaved causal episodes per association using
  pair-based STDP.
- Leave E→F untrained.
- Export the learned CSR weights.
- Destroy the training runtime.
- Construct a new runtime from the learned CSR with plasticity disabled.
- Probe each association with 8, 16, 24, and 32 stimulated source neurons
  (25%, 50%, 75%, and 100% of the assembly).
- Separate probes with 1000 settling steps.
- Record source spikes, target spikes, unique target neurons, and maximum
  postsynaptic voltage.

The recruitment curve is diagnostic, not a pattern-completion pass criterion.
The test asserts STDP potentiation, the untrained pathway remaining unchanged,
frozen recall weights, finite state, and no dropped events.

## Run

```bash
cd ~/projects/ChatGPTNeuronSimulator
git fetch origin
git switch --track origin/work/neuromancer-associative-memory-v16-dose-response
cmake -S . -B build-cuda -DSPIKINGMODEL_ENABLE_CUDA=ON
cmake --build build-cuda --target neuromancer_associative_memory_v16 -j$(nproc)
./build-cuda/neuromancer_associative_memory_v16
ctest --test-dir build-cuda -R neuromancer_associative_memory_v16 --output-on-failure
```

## Interpretation

The separate recall runtime prevents training-time STDP traces from influencing
retrieval. The cue-size sweep distinguishes insufficient partial input from
failure to propagate even a complete assembly cue. It does not yet test
overlapping memories, cross-association interference, or autonomous attractor
dynamics.
