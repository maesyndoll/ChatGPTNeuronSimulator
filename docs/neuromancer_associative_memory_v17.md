# V17 — Sparse randomized associative-memory feasibility

V17 changes one major assumption from V16: the learned pathways are no longer
complete 32×32 bipartite blocks. Each directed edge in each association block
is independently included with seeded probability 0.50. The two directions
are sampled independently. This creates a sparse, irregular graph while
preserving isolated association pairs as controls.

## Protocol

- Train A→B and C→D with 100 interleaved causal episodes per association.
- Leave E→F untrained.
- Export the trained CSR and load it in a separate runtime with plasticity off.
- Measure 8, 16, 24, and 32 neuron cues for all three pathways.
- Record source spikes, target spikes, unique target neurons, and peak target voltage.
- Check that the trained pathways potentiate, the untrained pathway stays at
  baseline, recall weights remain frozen, and no events are dropped.

This is still a hetero-associative pathway assay. It does not yet test
self-sustaining recurrent attractors, overlap between memories, or retrieval
from a noisy cue. Sparse connectivity is a realism step, not proof of those
capabilities.

## Run

```bash
cd ~/projects/ChatGPTNeuronSimulator
git fetch origin
git switch --track origin/work/neuromancer-associative-memory-v17-sparse-connectivity
cmake -S . -B build-cuda -DSPIKINGMODEL_ENABLE_CUDA=ON
cmake --build build-cuda --target neuromancer_associative_memory_v17 -j$(nproc)
./build-cuda/neuromancer_associative_memory_v17
ctest --test-dir build-cuda -R neuromancer_associative_memory_v17 --output-on-failure
```
