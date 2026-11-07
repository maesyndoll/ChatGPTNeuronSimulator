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
    graph.target_indices = {1, 2, 2, 3, 0};
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
    check(report.reciprocal_pairs == 1, "detects reciprocal edge pair");
    check(report.highest_degree_neurons.size() == 2, "bounds ranked neuron output");
    check(report.highest_degree_neurons.front().neuron_id == 2,
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

    if (failures == 0) {
        std::cout << "NEURAL SELF-OBSERVER TESTS: PASS\n";
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " self-observer test(s) failed\n";
    return EXIT_FAILURE;
}
