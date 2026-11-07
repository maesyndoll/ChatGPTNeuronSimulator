# Neuromancer V42 — Integrated learning milestone

## Purpose

V42 is intended to test three capabilities in one recurrent CUDA network:

1. Autonomous, learned phase modulation of synaptic communication.
2. Activity-dependent structural plasticity (dendritic growth and pruning).
3. Learned associative memory with partial-cue recall.

The experiment must not treat the earlier isolated assays as proof of these capabilities. In particular, V30–V36 use a simulation-time phase carrier and demonstrate externally clocked phase-selective transmission, not an autonomous neural oscillator. V29's structural rewiring is random and is not activity-dependent dendritic growth. V10/V19 demonstrate useful STDP association/weight-change foundations, but not robust integrated pattern completion.

## Non-negotiable implementation constraints

- Preserve V41 and all prior experiment branches.
- Use the existing CUDA neuron runtime and CSR topology representation.
- Preserve Dale's law: excitatory-source edges are nonnegative; inhibitory-source edges are nonpositive.
- Structural updates must be based on measured local activity, not random edge selection.
- Existing edges must retain identity across epochs; record every birth, death, and weight change.
- Memory training must not use a teacher during recall.
- Recall must be evaluated in a fresh runtime with learning frozen.
- No test may use the correct class label to select the output during evaluation.
- All controls must receive identical input examples and matched seeds.
- Log enough information to reconstruct phase trajectories, edge histories, and recall outcomes.

## Integrated task

Use three assemblies, A, B, and C, with a sparse recurrent E/I substrate.

- Train A→B and B→C associations through temporally paired activity.
- Keep A→C as an untrained control.
- Train on full patterns, then evaluate using held-out noisy and partial cues.
- After training, export the learned CSR graph and instantiate a fresh plasticity-disabled runtime.
- Evaluate cue completion and association recall without teacher stimulation.
- Include cue-only, shuffled-label, untrained, and randomized-association controls.

A successful memory result requires above-control recall of the associated assembly from a partial cue, not merely potentiation of one edge or a target spike.

## Autonomous phase mechanism

The runtime's current phase gate advances from absolute simulation time. Reusing it unchanged would make the central claim false.

V42 must introduce an endogenous phase state for each neural population. The phase state must:

- Advance from an intrinsic angular velocity.
- Be perturbed/reset by that population's own observed neural activity.
- Couple to other populations through learned phase-coupling parameters.
- Be updated from measured spike events, not a preprogrammed phase schedule.
- Modulate the efficacy of the relevant E/I synaptic pathway.
- Continue evolving when external stimuli are absent.

The implementation must record both population phase and spike-derived activity. The learned phase relationship must be measured after training and must persist during a stimulus-free observation period. Include a phase-scrambled control and a frozen/unlearned phase-coupling control.

A host-side phase controller may initially orchestrate the experiment, but the report must explicitly identify it as a hybrid host/CUDA mechanism. It must not be described as fully autonomous neuron-intrinsic CUDA dynamics unless the phase state actually resides and evolves in the simulation runtime.

## Activity-dependent structural plasticity

At each structural epoch, calculate local pre/post activity statistics for candidate source-target pairs. Use those statistics to:

- Strengthen existing edges when repeated activity is temporally and/or phase-coincident.
- Weaken edges when repeatedly inactive or consistently uncorrelated.
- Prune edges below a documented threshold after a minimum grace period.
- Grow new edges preferentially between repeatedly co-active, phase-compatible populations.
- Respect Dale's law and configured degree/capacity limits.

Growth must be based on activity evidence. The experiment must include a structural-plasticity-off control and a shuffled-activity control. Report per-epoch edge births, deaths, pruned weight, new edge weight, degree distributions, E→E/E→I/I→E/I→I counts, and source-target activity correlation.

Do not call this dendritic growth unless the implementation maintains a per-neuron/per-compartment dendritic allocation or an explicit local growth-resource variable that constrains where new synapses can form. Otherwise report it accurately as activity-dependent structural synaptic plasticity.

## Associative plasticity

Use the existing STDP runtime path where appropriate, but pair-based STDP alone is not sufficient evidence. The association must survive:

- A fresh runtime with plasticity disabled.
- A delay between training and recall.
- Partial cues (for example 25%, 50%, and 75% of the learned cue assembly).
- Controlled noise and shuffled association labels.

Record cue and target population spike counts, unique recruited neurons, pattern overlap/Jaccard similarity, precision, recall, and classification/association accuracy. Include the untrained A→C pathway as a negative control.

## Factorial controls

Run all eight combinations of the three mechanisms:

| Condition | Phase learning | Structural plasticity | Associative plasticity |
|---|---:|---:|---:|
| Full | on | on | on |
| Phase ablation | off | on | on |
| Structure ablation | on | off | on |
| Association ablation | on | on | off |
| Phase + structure | on | on | off |
| Phase + association | on | off | on |
| Structure + association | off | on | on |
| Baseline | off | off | off |

The last three rows duplicate the corresponding single ablations as written above; do not run redundant conditions. The intended full 2^3 factorial has exactly these eight unique combinations:

| Factorial cell | Phase | Structure | Association |
|---|---:|---:|---:|
| 000 | off | off | off |
| 001 | off | off | on |
| 010 | off | on | off |
| 011 | off | on | on |
| 100 | on | off | off |
| 101 | on | off | on |
| 110 | on | on | off |
| 111 | on | on | on |

Use the same generated examples and matched seeds for all cells. Add shuffled-association and phase-scrambled controls to the full condition. At least 10 independent network/topology seeds are preferred for the first campaign; report per-seed results, not only pooled trial counts.

## Pass/fail criteria

### Phase modulation passes only if
- A population phase trajectory is generated from endogenous activity plus autonomous phase evolution, without a supplied external phase schedule.
- Training changes a measurable phase-coupling parameter or preferred phase relationship.
- The learned phase relationship persists during stimulus-free observation.
- Disabling or scrambling learned phase coupling measurably changes pathway transmission or recall.
- The effect is not explained by a common external clock alone.

### Structural plasticity passes only if
- Birth/death/weight changes are caused by measured local activity.
- Growth is enriched among repeatedly co-active, phase-compatible pathways versus shuffled controls.
- Pruning is enriched among inactive/unrelated pathways.
- The structural-off and shuffled-activity controls do not reproduce the full effect.
- The topology remains valid and Dale-consistent.

### Associative memory passes only if
- The trained association is recalled above untrained and shuffled controls.
- Recall works from partial/noisy cues without a teacher.
- Recall persists in a fresh, frozen runtime.
- The untrained A→C pathway does not show equivalent recall.

### Integrated system passes only if
- All three component criteria pass in the full condition.
- The full condition outperforms relevant ablations on at least one preregistered integrated outcome (partial-cue recall and/or retention).
- All states and outputs remain finite; no spike events are dropped; CSR validation passes.

If any component fails, report the component as failed or inconclusive. Do not infer it from the other two.

## Required output artifacts

- `v42_trials.csv`: seed, factorial cell, training/control condition, cue fraction, noise level, expected/observed association, recall metrics.
- `v42_phase.csv`: time, population, endogenous phase, spike-derived activity, learned coupling parameters, phase error.
- `v42_structure.csv`: epoch, edge source/target, action (born/pruned/retained), old/new weight, local activity/correlation, dendritic resource.
- `v42_summary.json`: configuration, seeds, pass/fail status, per-seed metrics, finite-state and dropped-event checks.
- A Python standard-library analyzer (no pandas dependency) that summarizes all outputs and applies the declared criteria.

## Implementation gate

Do not launch the full campaign until small smoke tests establish:
1. Autonomous phase continues and responds to spikes with no external stimulus.
2. Learned phase coupling changes relative phase and pathway efficacy.
3. Activity-driven growth/pruning is deterministic under a fixed seed and differs from shuffled activity.
4. Frozen partial-cue recall beats untrained/shuffled controls.
5. The combined small network passes all instrumentation and CSR integrity checks.

Only after these gates pass should the full 8-cell, multi-seed campaign run.
