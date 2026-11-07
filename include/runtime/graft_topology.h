#pragma once

#include "topology/csr_topology.h"

#include <cstdint>
#include <vector>

namespace spiking {

struct GraftTopologyMerge {
    CSRTopology topology;
    std::vector<std::uint8_t> plasticity_mask;
    std::vector<std::uint32_t> base_neuron_map;
    std::vector<std::uint32_t> graft_neuron_map;
};

// Existing edges retain their supplied plasticity policy (or default open).
// Graft edges default frozen: they still transmit, but STDP cannot alter them.
GraftTopologyMerge merge_graft_topology(
    const CSRTopology& base,
    const CSRTopology& graft,
    const std::vector<std::uint8_t>& base_plasticity_mask = {},
    bool graft_plasticity_enabled = false);

} // namespace spiking
