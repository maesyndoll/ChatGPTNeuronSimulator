#include "topology/topology_generator.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

spiking::SimulationParameters small_simulation() {
    spiking::SimulationParameters p;
    p.excitatory_count = 32;
    p.inhibitory_count = 8;
    return p;
}

spiking::TopologyParameters sparse_topology() {
    spiking::TopologyParameters p;
    p.p_ee = 0.12;
    p.p_ei = 0.25;
    p.p_ie = 0.30;
    p.p_ii = 0.20;
    p.topology_seed = 987654321ULL;
    return p;
}
}

int main() {
    using namespace spiking;

    const auto simulation = small_simulation();
    auto parameters = sparse_topology();

    const auto first = generate_random_sparse_topology(simulation, parameters);
    const auto second = generate_random_sparse_topology(simulation, parameters);

    check(first.row_offsets.size() == simulation.neuron_count() + 1,
          "CSR has one row offset per neuron plus sentinel");
    check(first.target_indices.size() == first.weights.size() &&
          first.target_indices.size() == first.delays.size(),
          "target, weight, and delay arrays have matching lengths");
    check(first.row_offsets.front() == 0,
          "CSR starts at offset zero");
    check(first.row_offsets.back() == first.synapse_count(),
          "final CSR offset equals synapse count");
    check(first.minimum_delay == parameters.minimum_delay &&
          first.maximum_delay == parameters.maximum_delay,
          "CSR records configured delay bounds");

    check(first.row_offsets == second.row_offsets &&
          first.target_indices == second.target_indices &&
          first.weights == second.weights &&
          first.delays == second.delays,
          "same seeds produce identical CSR topology and delays");

    bool different_seed_differs = false;
    parameters.topology_seed += 1;
    const auto third = generate_random_sparse_topology(simulation, parameters);
    different_seed_differs =
        first.target_indices != third.target_indices ||
        first.weights != third.weights;
    check(different_seed_differs,
          "different topology seeds change a nontrivial random topology");

    auto delay_seed_parameters = sparse_topology();
    delay_seed_parameters.delay_seed += 1;
    const auto changed_delays = generate_random_sparse_topology(
        simulation, delay_seed_parameters);

    check(first.row_offsets == changed_delays.row_offsets &&
          first.target_indices == changed_delays.target_indices &&
          first.weights == changed_delays.weights,
          "changing delay_seed leaves connectivity and weights unchanged");

    check(first.delays != changed_delays.delays,
          "changing delay_seed changes generated delays");

    try {
        first.validate();
        check(true, "generated CSR passes structural validation");
    } catch (const std::exception& error) {
        check(false, std::string("generated CSR validation: ") + error.what());
    }

    bool signs_valid = true;
    for (std::uint32_t source = 0; source < first.neuron_count(); ++source) {
        for (std::uint32_t edge = first.row_offsets[source];
             edge < first.row_offsets[source + 1]; ++edge) {
            const float weight = first.weights[edge];
            if (source < first.excitatory_count ? weight <= 0.0f
                                                : weight >= 0.0f) {
                signs_valid = false;
            }
        }
    }
    check(signs_valid, "weight signs follow the source population");

    bool delays_valid = true;
    for (const auto delay : first.delays) {
        if (delay < parameters.minimum_delay ||
            delay > parameters.maximum_delay) {
            delays_valid = false;
            break;
        }
    }
    check(delays_valid, "Fibonacci delays stay within configured bounds");

    bool targets_valid = true;
    bool no_self_connections = true;
    bool rows_sorted = true;
    for (std::uint32_t source = 0; source < first.neuron_count(); ++source) {
        std::uint32_t previous = 0;
        bool is_first = true;
        for (std::uint32_t edge = first.row_offsets[source];
             edge < first.row_offsets[source + 1]; ++edge) {
            const auto target = first.target_indices[edge];
            targets_valid = targets_valid && target < first.neuron_count();
            no_self_connections = no_self_connections && target != source;
            if (!is_first && target <= previous) rows_sorted = false;
            previous = target;
            is_first = false;
        }
    }
    check(targets_valid, "all target indices are in range");
    check(no_self_connections, "generator excludes self-connections");
    check(rows_sorted, "targets are strictly sorted within each CSR row");

    // Zero probabilities must produce a valid graph with no edges.
    auto zero = sparse_topology();
    zero.p_ee = zero.p_ei = zero.p_ie = zero.p_ii = 0.0;
    const auto empty = generate_random_sparse_topology(simulation, zero);
    check(empty.synapse_count() == 0, "zero probabilities produce no edges");
    check(empty.delays.empty(), "empty graph has no delay entries");
    check(empty.row_offsets.back() == 0, "empty graph has valid CSR offsets");

    // Uniform delays must be integers within inclusive configured bounds.
    auto uniform = sparse_topology();
    uniform.delay_distribution = DelayDistribution::Uniform;
    uniform.minimum_delay = 2;
    uniform.maximum_delay = 7;

    const auto uniform_graph =
        generate_random_sparse_topology(simulation, uniform);

    bool uniform_delays_valid = true;
    for (const auto delay : uniform_graph.delays) {
        if (delay < 2 || delay > 7) {
            uniform_delays_valid = false;
            break;
        }
    }
    check(uniform_delays_valid,
          "uniform delays stay within inclusive configured bounds");

    // Probability one connects every allowed source-target pair exactly once.
    auto dense = sparse_topology();
    dense.p_ee = dense.p_ei = dense.p_ie = dense.p_ii = 1.0;
    const auto complete = generate_random_sparse_topology(simulation, dense);
    const std::uint64_t n = simulation.neuron_count();
    check(complete.synapse_count() == n * (n - 1),
          "probability one creates all non-self directed edges");
    check(complete.delays.size() == complete.synapse_count(),
          "every dense-graph synapse has a delay");

    bool fibonacci_values_valid = true;
    for (const auto delay : complete.delays) {
        bool found = false;
        for (const auto allowed : dense.fibonacci_delays) {
            if (delay == allowed) {
                found = true;
                break;
            }
        }
        if (!found) {
            fibonacci_values_valid = false;
            break;
        }
    }
    check(fibonacci_values_valid,
          "Fibonacci-weighted mode samples only configured Fibonacci delays");

    bool rejected_invalid_probability = false;
    auto invalid = sparse_topology();
    invalid.p_ee = 1.1;
    try {
        (void)generate_random_sparse_topology(simulation, invalid);
    } catch (const std::invalid_argument&) {
        rejected_invalid_probability = true;
    }
    check(rejected_invalid_probability,
          "generator rejects out-of-range connection probabilities");

    bool rejected_bad_weights = false;
    try {
        TopologyGeneratorOptions bad;
        bad.minimum_weight = 0.0f;
        (void)generate_random_sparse_topology(simulation, sparse_topology(), bad);
    } catch (const std::invalid_argument&) {
        rejected_bad_weights = true;
    }
    check(rejected_bad_weights, "generator rejects non-positive weight magnitudes");

    bool rejected_bad_delay_bounds = false;
    try {
        auto bad = sparse_topology();
        bad.minimum_delay = 10;
        bad.maximum_delay = 5;
        (void)generate_random_sparse_topology(simulation, bad);
    } catch (const std::invalid_argument&) {
        rejected_bad_delay_bounds = true;
    }
    check(rejected_bad_delay_bounds,
          "generator rejects invalid delay bounds");

    bool rejected_bad_fibonacci_weights = false;
    try {
        auto bad = sparse_topology();
        bad.fibonacci_weights.assign(
            bad.fibonacci_weights.size(), 0.0);
        (void)generate_random_sparse_topology(simulation, bad);
    } catch (const std::invalid_argument&) {
        rejected_bad_fibonacci_weights = true;
    }
    check(rejected_bad_fibonacci_weights,
          "generator rejects Fibonacci weights with zero total");

    if (failures == 0) {
        std::cout << "All topology tests passed.\n";
        return EXIT_SUCCESS;
    }

    std::cerr << failures << " topology test(s) failed.\n";
    return EXIT_FAILURE;
}
