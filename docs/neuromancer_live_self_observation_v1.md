# Neuromancer live self-observation v1

The first live self-observation path is read-only.

## Data path

1. `SimulationController` serializes the request on its CUDA-owning worker thread.
2. The runtime synchronizes its stream and copies the exact resident CSR topology and per-neuron cumulative spike counters to host memory.
3. `observe_connectome` analyzes that snapshot without changing graph structure, weights, or neuron state.
4. IPC command `get_self_observation` returns a JSON report.
5. The Qt `SimulationBridge` polls that command every five seconds and exposes the result as its `selfObservation` property.

## Report fields

- `graph_fingerprint`: deterministic identity for the observed topology and quantized weights.
- `neuron_count`, `synapse_count`: graph dimensions.
- `mean_outgoing_degree`, `mean_incoming_degree`: mean connectivity.
- `isolated_neurons`: neurons with no incoming or outgoing edges.
- `high_degree_neurons`: outgoing-degree outliers above mean plus two standard deviations.
- `inactive_neurons`: neurons with zero cumulative spike count in the current runtime generation.
- `reciprocal_pairs`: unordered neuron pairs with edges in both directions.
- `weak_edges`: edges whose absolute weight is at or below the configured observer threshold.
- `ranked_neurons`: bounded list ranked by total in-degree plus out-degree.

The current bridge polls at five-second intervals. Each observation synchronizes the runtime stream and transfers the full CSR graph plus per-neuron counters to the CPU, so this is intentionally a low-rate diagnostic operation rather than a per-step kernel.

## Safety boundary

This version cannot edit the live graph. It is a structural mirror and analyzer only. Structural proposals, staged graph replacement, state migration, and baseline-versus-candidate evaluation are separate later work. A graph fingerprint or connectivity statistic is not evidence of learning or cognition by itself.


## Proposal-only topology advisor v2

The self-observer now also exposes a deterministic, bounded candidate builder:

- It consumes a time-ordered window of retained spike events.
- Near-coincident spike pairs create candidate directed connections, with source-population sign respected (excitatory positive, inhibitory negative).
- Existing connections are retained. Optional dormant-weak pruning is disabled by default and requires both endpoints to have zero activity.
- Candidate edge additions are capped (default 32); the source graph is never mutated.
- Both the original and candidate graphs receive fingerprints for comparison.
- Unit tests verify bounded additions, unchanged source topology, candidate fingerprint changes, activity-aware pruning safeguards, and invalid event rejection.

IPC command `get_topology_proposal` runs the advisor over the most recent retained spike events and returns proposal counts, fingerprints, evidence count, and explicit `applied=false`.

This is not autonomous topology improvement yet. The endpoint is a proposal diagnostic, not a graph writer or live patch mechanism. The next required stage is a controlled evaluator that compares baseline and candidate behavior on matched input, then persists an accepted candidate as a versioned neural-memory module. Only after that should safe-point graph replacement be enabled.


## Dashboard topology proposal inspection

The Neural Observatory Self-model panel now exposes a manual **Propose topology** action. It requests the proposal-only IPC endpoint and displays the bounded candidate edge additions (source, target, weight, delay), evidence spike count, candidate synapse count, and base/candidate graph fingerprints. The endpoint reports `source_unchanged=true` and `applied=false`; it does not mutate the live CSR graph, prune live edges, or persist the candidate. The proposal is a simple temporal co-activity heuristic over the retained spike-event window, not a validated learning rule or a claim of improved inference. Evaluate candidates against matched baseline stimuli before considering any application.
