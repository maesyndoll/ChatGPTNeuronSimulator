# Neuromancer V26 — overnight online learning

V26 is an experiment runner, not a claim of general-purpose learning. It trains a recurrent sparse E/I CUDA network with pair-based excitatory STDP and exports the learned CSR graph in the V25 .nmc model-package format.

## Network and task

- 1,024 excitatory and 256 inhibitory neurons.
- Seeded globally random sparse recurrent E/I connectivity.
- 64 binary input neurons and two excitatory readout neurons (IDs 64 and 65).
- Each input feature has a candidate edge to both readouts; these edges are initialized at 0.02. The rest of the network is randomly connected.
- Two synthetic classes: features 0–31 are class 0 evidence; features 32–63 are class 1 evidence. Each example drops two preferred features and adds two distractor features.
- Training interleaves both classes and directly stimulates the labeled readout as a teacher, with a 1.5 ms onset lag from the input pulse.
- Pair STDP: A+=0.005, A-=0.00525, tau+=tau-=20 ms, weights clamped to [0.001, 1.0].
- Recall uses a fresh runtime with the saved learned CSR, no teacher stimulation, and plasticity disabled.

The teacher signal makes this supervised, teacher-forced STDP association training, not unsupervised learning. The recurrent graph is present, but V26 does not yet establish attractor memory or autonomous task discovery.

## Running

Smoke test:

```bash
./build-cuda/neuromancer_overnight_v26 --smoke-test ~/projects/neuromancer/results/v26-smoke
```

Overnight run (default eight hours):

```bash
systemd-inhibit --what=sleep --mode=block --why="Neuromancer V26 overnight learning" \
  ./build-cuda/neuromancer_overnight_v26 --hours 8 \
  --output ~/projects/neuromancer/models/v26-overnight
```

The runner checkpoints the learned CSR and manifest approximately once per minute and writes training_metrics.csv. The training spike-event recorder is disabled to avoid overflowing its bounded event ring. Held-out evaluation uses event capture in separate fresh runtimes.

## Checkpoint semantics and limitations

A checkpoint contains the current learned synaptic weights/topology and inference manifest. It does not serialize GPU neuron voltage/recovery state, STDP pre/post traces, RNG state, or exact training order position. It is therefore a valid frozen inference model, but not an exact training-resume snapshot. Restarting training from this package would reset traces and simulation state.

The V25 inference API can load the final model.nmc and its relative topology.csr. The smoke test is only a pipeline/integrity check: held-out accuracy is reported, not required to be high. If recall is poor, inspect output recruitment and learned edge weights before committing to an eight-hour run.

## Interpretation

A passing smoke test proves that the runner can train, checkpoint, reload, and evaluate a learned CSR graph without numerical-state or event-recorder errors. It does not prove robust pattern completion, scalable generalization, or biological plausibility.