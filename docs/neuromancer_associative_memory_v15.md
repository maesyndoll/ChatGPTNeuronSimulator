# V15 — Multi-association STDP associative-memory feasibility test

V15 is a controlled CUDA experiment, not a desktop-agent milestone.

## Question

Can one shared spiking runtime learn two distinct associations through pair-based
STDP, then recruit both target assemblies from simultaneous 50% partial cues
after plasticity has stopped?

## Network

- 193 neurons: 192 excitatory and 1 inhibitory.
- Three 32-neuron reciprocal assembly pairs in one CSR graph:
  - A (0–31) ↔ B (32–63), trained.
  - C (64–95) ↔ D (96–127), trained.
  - E (128–159) ↔ F (160–191), never teacher-paired.
- Every assembly-to-assembly edge begins at weight 0.01 and has delay 2.
- The three pairs are deliberately structurally isolated. This makes the first
  coexistence test interpretable, but does not test cross-association interference.

## Protocol

1. Measure baseline responses to simultaneous 50% cues in A, C, and E.
2. Train A→B and C→D in the same runtime, interleaving 100 causal episodes
   per association. Each episode stimulates the full source assembly and,
   three integration steps later, the full target assembly.
3. Leave E→F untrained.
4. Allow membrane transients and STDP traces to decay.
5. Present simultaneous 50% cues to A, C, and E with the same frozen network.
6. Repeat the protocol with plasticity disabled as a control.

## Measurements and pass criteria

- Mean weights for all three pathways before and after training.
- Training spike counts for A, B, C, and D.
- Target spike counts and unique recruited target neurons during recall.
- At least half of each trained target assembly must spike on partial-cue recall.
- The never-trained F assembly must show no increase over its baseline response.
- Plasticity-disabled weights and target responses must remain unchanged.
- No dropped spike events and finite simulation state.

## Interpretation limits

This is a feasibility assay for coexistence of two learned, partial-cue associations
in a shared runtime. It is not yet a fully recurrent attractor-memory model:
the three reciprocal pathway blocks are structurally isolated, and recall is
measured over a short response window. It does not establish robustness across
random seeds, noisy cues, overlapping memories, capacity, or interference.
A pass is evidence for this narrowly defined mechanism; a failure is useful
diagnostic data, not a reason to tune thresholds until the test passes.

## Build and run

```bash
cd ~/projects/ChatGPTNeuronSimulator
git fetch origin
git switch --track origin/work/neuromancer-associative-memory-v15

cmake -S . -B build-cuda -DSPIKINGMODEL_ENABLE_CUDA=ON
cmake --build build-cuda --target neuromancer_associative_memory_v15 -j$(nproc)
./build-cuda/neuromancer_associative_memory_v15
ctest --test-dir build-cuda -R neuromancer_associative_memory_v15 --output-on-failure
```

The experiment is deterministic for its configured seed and topology. Run the
executable once first; CTest runs it again.
