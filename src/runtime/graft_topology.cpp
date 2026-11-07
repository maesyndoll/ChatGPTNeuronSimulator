#include "runtime/graft_topology.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <cmath>

namespace spiking {
namespace {
struct Edge {
    std::uint32_t target;
    float weight;
    std::uint32_t delay;
    std::uint8_t plastic;
};
}

GraftTopologyMerge merge_graft_topology(
    const CSRTopology& base,
    const CSRTopology& graft,
    const std::vector<std::uint8_t>& base_plasticity_mask,
    bool graft_plasticity_enabled) {
    base.validate();
    graft.validate();
    if (!base_plasticity_mask.empty() &&
        base_plasticity_mask.size() != base.synapse_count()) {
        throw std::invalid_argument("Base plasticity mask must match base synapse count");
    }

    const std::uint64_t excitatory =
        static_cast<std::uint64_t>(base.excitatory_count) + graft.excitatory_count;
    const std::uint64_t inhibitory =
        static_cast<std::uint64_t>(base.inhibitory_count) + graft.inhibitory_count;
    const std::uint64_t neurons = excitatory + inhibitory;
    const std::uint64_t edges =
        static_cast<std::uint64_t>(base.synapse_count()) + graft.synapse_count();
    if (neurons == 0 || neurons > UINT32_MAX || edges > UINT32_MAX ||
        neurons + 1U > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("Merged graft topology exceeds CSR index limits");
    }

    GraftTopologyMerge result;
    auto& merged = result.topology;
    merged.excitatory_count = static_cast<std::uint32_t>(excitatory);
    merged.inhibitory_count = static_cast<std::uint32_t>(inhibitory);
    merged.minimum_delay = std::min(base.minimum_delay, graft.minimum_delay);
    merged.maximum_delay = std::max(base.maximum_delay, graft.maximum_delay);

    result.base_neuron_map.resize(base.neuron_count());
    result.graft_neuron_map.resize(graft.neuron_count());
    for (std::uint32_t i = 0; i < base.excitatory_count; ++i)
        result.base_neuron_map[i] = i;
    for (std::uint32_t i = 0; i < base.inhibitory_count; ++i)
        result.base_neuron_map[base.excitatory_count + i] =
            merged.excitatory_count + i;
    for (std::uint32_t i = 0; i < graft.excitatory_count; ++i)
        result.graft_neuron_map[i] = base.excitatory_count + i;
    for (std::uint32_t i = 0; i < graft.inhibitory_count; ++i)
        result.graft_neuron_map[graft.excitatory_count + i] =
            merged.excitatory_count + base.inhibitory_count + i;

    std::vector<std::vector<Edge>> rows(static_cast<std::size_t>(neurons));
    std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> row_edge_sources(
        static_cast<std::size_t>(neurons));
    const auto append_graph = [&](const CSRTopology& graph,
                                  const std::vector<std::uint32_t>& mapping,
                                  bool is_graft) {
        for (std::uint32_t source = 0; source < graph.neuron_count(); ++source) {
            auto& row = rows[mapping[source]];
            for (std::uint32_t edge = graph.row_offsets[source];
                 edge < graph.row_offsets[source + 1]; ++edge) {
                const std::uint8_t plastic = is_graft
                    ? static_cast<std::uint8_t>(graft_plasticity_enabled)
                    : (base_plasticity_mask.empty()
                        ? static_cast<std::uint8_t>(1)
                        : base_plasticity_mask[edge]);
                row.push_back({mapping[graph.target_indices[edge]],
                               graph.weights[edge], graph.delays[edge], plastic});
                row_edge_sources[mapping[source]].push_back({edge, is_graft ? 1U : 0U});
            }
        }
    };
    append_graph(base, result.base_neuron_map, false);
    append_graph(graft, result.graft_neuron_map, true);

    merged.row_offsets.reserve(static_cast<std::size_t>(neurons) + 1U);
    merged.row_offsets.push_back(0);
    result.plasticity_mask.reserve(static_cast<std::size_t>(edges));
    result.base_edge_map.resize(base.synapse_count());
    result.graft_edge_map.resize(graft.synapse_count());
    for (auto& row : rows) {
        const auto row_index = static_cast<std::size_t>(&row - rows.data());
        auto& origins = row_edge_sources[row_index];
        std::vector<std::size_t> order(row.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return row[a].target < row[b].target;
        });
        for (const auto index : order) {
            const auto& edge = row[index];
            const auto origin = origins[index];
            const auto merged_edge = static_cast<std::uint32_t>(merged.target_indices.size());
            if (origin.second == 0U) result.base_edge_map[origin.first] = merged_edge;
            else result.graft_edge_map[origin.first] = merged_edge;
            merged.target_indices.push_back(edge.target);
            merged.weights.push_back(edge.weight);
            merged.delays.push_back(edge.delay);
            result.plasticity_mask.push_back(edge.plastic);
        }
        merged.row_offsets.push_back(
            static_cast<std::uint32_t>(merged.target_indices.size()));
    }
    merged.validate();
    return result;
}


std::vector<float> remap_delayed_current_ring(
    const std::vector<float>& old_ring, std::uint32_t old_neuron_count,
    std::uint32_t old_ring_slots, std::uint32_t new_neuron_count,
    std::uint32_t new_ring_slots, std::uint64_t completed_steps,
    const std::vector<std::uint32_t>& old_to_new_neuron_map) {
    if (old_neuron_count == 0 || new_neuron_count == 0 ||
        old_ring_slots == 0 || new_ring_slots < old_ring_slots ||
        old_to_new_neuron_map.size() != old_neuron_count ||
        old_ring.size() != static_cast<std::size_t>(old_neuron_count) * old_ring_slots)
        throw std::invalid_argument("Invalid delayed-current ring dimensions or neuron map");
    for (const auto id : old_to_new_neuron_map)
        if (id != UINT32_MAX && id >= new_neuron_count)
            throw std::out_of_range("Delayed-current neuron map target is out of range");
    std::vector<float> next(
        static_cast<std::size_t>(new_neuron_count) * new_ring_slots, 0.0f);
    const auto old_current = static_cast<std::uint32_t>(completed_steps % old_ring_slots);
    const auto new_current = static_cast<std::uint32_t>(completed_steps % new_ring_slots);
    for (std::uint32_t old_slot = 0; old_slot < old_ring_slots; ++old_slot) {
        const auto offset = (old_slot + old_ring_slots - old_current) % old_ring_slots;
        const auto new_slot = (new_current + offset) % new_ring_slots;
        for (std::uint32_t old_id = 0; old_id < old_neuron_count; ++old_id) {
            const float value = old_ring[static_cast<std::size_t>(old_slot) *
                                         old_neuron_count + old_id];
            if (!std::isfinite(value))
                throw std::invalid_argument("Delayed-current ring contains non-finite current");
            if (old_to_new_neuron_map[old_id] != UINT32_MAX)
                next[static_cast<std::size_t>(new_slot) * new_neuron_count +
                     old_to_new_neuron_map[old_id]] = value;
        }
    }
    return next;
}

} // namespace spiking
