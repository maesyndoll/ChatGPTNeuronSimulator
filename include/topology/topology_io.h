#pragma once

#include "topology/csr_topology.h"

#include <cstdint>
#include <string>

namespace spiking {

struct TopologyDistributionStats {
    std::uint64_t count = 0;
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
};

struct TopologyStatistics {
    std::uint32_t excitatory_count = 0;
    std::uint32_t inhibitory_count = 0;
    std::uint64_t synapse_count = 0;
    double density = 0.0;

    // Source population -> target population.
    std::uint64_t excitatory_to_excitatory = 0;
    std::uint64_t excitatory_to_inhibitory = 0;
    std::uint64_t inhibitory_to_excitatory = 0;
    std::uint64_t inhibitory_to_inhibitory = 0;

    TopologyDistributionStats outgoing_degree;
    TopologyDistributionStats incoming_degree;
    TopologyDistributionStats weights;
    TopologyDistributionStats delays;
};

// Save the exact CSR graph in the versioned SPKCSR binary format.
void save_csr_topology(const CSRTopology& topology,
                       const std::string& filename);

// Load a saved graph. Rejects unsupported versions, malformed arrays,
// truncated files, trailing data, and invalid CSR structure.
CSRTopology load_csr_topology(const std::string& filename);

// Calculate graph density, population-pair counts, degree distributions,
// and signed weight / delay distributions.
TopologyStatistics analyze_csr_topology(const CSRTopology& topology);

} // namespace spiking
