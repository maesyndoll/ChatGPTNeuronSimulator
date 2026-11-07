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
