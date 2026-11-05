#pragma once

#include "core/parameters_generated.h"
#include "topology/csr_topology.h"

namespace spiking {

struct TopologyGeneratorOptions {
    // Positive magnitudes. The generator applies the source-population sign.
    float minimum_weight = 0.01f;
    float maximum_weight = 0.10f;
};

// Generates a seeded random sparse graph using the four population-specific
// probabilities in TopologyParameters. Self-connections are excluded.
CSRTopology generate_random_sparse_topology(
    const SimulationParameters& simulation,
    const TopologyParameters& topology,
    const TopologyGeneratorOptions& options = {});

} // namespace spiking
