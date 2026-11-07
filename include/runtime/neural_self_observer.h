#pragma once

#include "topology/csr_topology.h"

#include <cstdint>
#include <vector>

namespace spiking {

struct SelfObservedNeuron {
    std::uint32_t neuron_id = 0;
    std::uint32_t outgoing_degree = 0;
    std::uint32_t incoming_degree = 0;
    std::uint64_t activity_count = 0;
};

struct SelfObservationReport {
    std::uint64_t graph_fingerprint = 0;
    std::uint32_t neuron_count = 0;
    std::uint64_t synapse_count = 0;
    double mean_outgoing_degree = 0.0;
    double mean_incoming_degree = 0.0;
    std::uint32_t isolated_neurons = 0;
    std::uint32_t high_degree_neurons = 0;
    std::uint32_t inactive_neurons = 0;
    std::uint64_t reciprocal_pairs = 0;
    std::uint64_t weak_edges = 0;
    std::vector<SelfObservedNeuron> highest_degree_neurons;
};

// Read-only analysis of a validated source-oriented CSR connectome.
// activity_counts, when supplied, must contain one cumulative count per neuron.
// The observer never changes topology or weights.
SelfObservationReport observe_connectome(
    const CSRTopology& topology,
    const std::vector<std::uint64_t>& activity_counts = {},
    float weak_weight_threshold = 0.01f,
    std::size_t maximum_ranked_neurons = 16);

} // namespace spiking
