#include "runtime/neural_self_observer.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

spiking::CSRTopology sample() {
    spiking::CSRTopology graph;
    graph.excitatory_count = 3;
    graph.inhibitory_count = 1;
    graph.minimum_delay = 1;
    graph.maximum_delay = 2;
    graph.row_offsets = {0, 2, 3, 4, 5};
    graph.target_indices = {1, 2, 0, 3, 2};
    graph.weights = {0.5f, 0.001f, 0.4f, 0.3f, -0.2f};
    graph.delays = {1, 1, 2, 1, 2};
    return graph;
}
}

int main() {
    using namespace spiking;
    const auto graph = sample();
    const std::vector<std::uint64_t> activity = {4, 0, 9, 1};
    const auto report = observe_connectome(graph, activity, 0.01f, 2);

    check(report.neuron_count == 4, "reports neuron count");
    check(report.synapse_count == 5, "reports exact edge count");
    check(report.mean_outgoing_degree == 1.25, "computes mean outgoing degree");
    check(report.mean_incoming_degree == 1.25, "computes mean incoming degree");
    check(report.isolated_neurons == 0, "detects no isolated neurons");
    check(report.inactive_neurons == 1, "detects inactive neurons from activity counts");
    check(report.weak_edges == 1, "counts weak-weight edges");
    check(report.reciprocal_pairs == 2, "detects reciprocal edge pairs");
    check(report.highest_degree_neurons.size() == 2, "bounds ranked neuron output");
    check(report.highest_degree_neurons.front().neuron_id == 0,
          "ranks highest total-degree neuron first");

    const auto repeat = observe_connectome(graph, activity, 0.01f, 2);
    check(report.graph_fingerprint == repeat.graph_fingerprint,
          "graph fingerprint is deterministic");

    auto changed = graph;
    changed.weights[0] = 0.6f;
    const auto changed_report = observe_connectome(changed, activity, 0.01f, 2);
    check(report.graph_fingerprint != changed_report.graph_fingerprint,
          "fingerprint changes when graph weights change");

    bool bad_activity_rejected = false;
    try { (void)observe_connectome(graph, {1, 2}, 0.01f, 2); }
    catch (const std::invalid_argument&) { bad_activity_rejected = true; }
    check(bad_activity_rejected, "rejects activity vector dimension mismatch");

    bool bad_threshold_rejected = false;
    try { (void)observe_connectome(graph, {}, -1.0f, 2); }
    catch (const std::invalid_argument&) { bad_threshold_rejected = true; }
    check(bad_threshold_rejected, "rejects negative weak-edge threshold");


    // Co-active spike pairs produce a bounded candidate edge, never a live edit.
    auto advisor_graph = sample();
    advisor_graph.target_indices = {1, 2, 0, 3, 2};
    const std::vector<ObservedSpike> recent = {
        {10, 0}, {11, 3}, {12, 0}, {13, 3}, {14, 1}
    };
    const std::vector<std::uint64_t> cumulative = {4, 1, 9, 1};
    TopologyAdvisorOptions advisor_options;
    advisor_options.maximum_coactivity_lag = 2;
    advisor_options.maximum_added_edges = 1;
    const auto proposal = propose_topology_update(
        advisor_graph, recent, cumulative, advisor_options);
    check(proposal.proposed_added_edges == 1,
          "advisor creates bounded coactivity proposal");
    check(proposal.candidate.synapse_count() == advisor_graph.synapse_count() + 1,
          "advisor candidate contains proposed edge");
    check(advisor_graph.synapse_count() == 5,
          "advisor leaves source topology unchanged");
    check(proposal.base_graph_fingerprint != proposal.candidate_graph_fingerprint,
          "advisor fingerprints changed candidate");

    auto pruning_options = advisor_options;
    pruning_options.maximum_added_edges = 0;
    pruning_options.prune_dormant_weak_edges = true;
    pruning_options.weak_edge_threshold = 0.01f;
    const auto no_prune = propose_topology_update(
        advisor_graph, {}, {4, 1, 9, 1}, pruning_options);
    check(no_prune.proposed_pruned_edges == 0,
          "advisor preserves weak edges when endpoints have activity");

    bool bad_spike_rejected = false;
    try {
        (void)propose_topology_update(advisor_graph, {{1, 99}}, cumulative, advisor_options);
    } catch (const std::invalid_argument&) { bad_spike_rejected = true; }
    check(bad_spike_rejected, "advisor rejects out-of-range spike events");

    if (failures == 0) {
        std::cout << "NEURAL SELF-OBSERVER TESTS: PASS\n";
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " self-observer test(s) failed\n";
    return EXIT_FAILURE;
}
