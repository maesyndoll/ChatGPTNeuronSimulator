#pragma once

#include "topology/csr_topology.h"

#include <cstddef>
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


struct ObservedSpike {
    std::uint64_t step = 0;
    std::uint32_t neuron_id = 0;
};

struct TopologyAdvisorOptions {
    std::uint32_t maximum_coactivity_lag = 2;
    std::size_t maximum_added_edges = 32;
    float proposed_excitatory_weight = 0.05f;
    float proposed_inhibitory_weight = -0.05f;
    bool prune_dormant_weak_edges = false;
    float weak_edge_threshold = 0.005f;
};

struct TopologyProposalReport {
    std::uint64_t base_graph_fingerprint = 0;
    std::uint64_t candidate_graph_fingerprint = 0;
    std::size_t proposed_added_edges = 0;
    std::size_t proposed_pruned_edges = 0;
    CSRTopology candidate;
};

// Builds a bounded candidate from temporally ordered co-activity. This is
// proposal-only: the supplied topology is never modified or applied live.
TopologyProposalReport propose_topology_update(
    const CSRTopology& topology,
    const std::vector<ObservedSpike>& recent_spikes,
    const std::vector<std::uint64_t>& cumulative_activity,
    const TopologyAdvisorOptions& options = {});

} // namespace spiking
