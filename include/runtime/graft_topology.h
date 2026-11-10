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
    std::vector<std::uint32_t> base_edge_map;
    std::vector<std::uint32_t> graft_edge_map;
};

// Existing edges retain their supplied plasticity policy (or default open).
// Graft edges default frozen: they still transmit, but STDP cannot alter them.
GraftTopologyMerge merge_graft_topology(
    const CSRTopology& base,
    const CSRTopology& graft,
    const std::vector<std::uint8_t>& base_plasticity_mask = {},
    bool graft_plasticity_enabled = false);

// Preserve queued synaptic currents by future-step offset when ring capacity changes.
std::vector<float> remap_delayed_current_ring(
    const std::vector<float>& old_ring, std::uint32_t old_neuron_count,
    std::uint32_t old_ring_slots, std::uint32_t new_neuron_count,
    std::uint32_t new_ring_slots, std::uint64_t completed_steps,
    const std::vector<std::uint32_t>& old_to_new_neuron_map);

} // namespace spiking
