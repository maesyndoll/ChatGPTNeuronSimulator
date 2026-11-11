#pragma once

#include "topology/csr_topology.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace spiking {

struct StdpWeightSummary {
    std::uint64_t plastic_synapses = 0;
    std::uint64_t changed_synapses = 0;
    double mean_initial_weight = 0.0;
    double mean_final_weight = 0.0;
    double mean_absolute_change = 0.0;
    double maximum_absolute_change = 0.0;
};

// Compare the canonical CSR edge weights without regenerating or reordering
// connectivity. Only excitatory-source edges are plastic in this STDP model.
inline StdpWeightSummary summarize_stdp_weight_changes(
    const CSRTopology& before,
    const CSRTopology& after) {
    if (before.excitatory_count != after.excitatory_count ||
        before.inhibitory_count != after.inhibitory_count ||
        before.row_offsets != after.row_offsets ||
        before.target_indices != after.target_indices ||
        before.delays != after.delays ||
        before.weights.size() != after.weights.size())
        throw std::runtime_error("Connectivity structure changed during STDP training");

    StdpWeightSummary summary;
    double initial_sum = 0.0;
    double final_sum = 0.0;
    double absolute_change_sum = 0.0;
    for (std::uint32_t source = 0; source < before.excitatory_count; ++source) {
        for (std::uint32_t edge = before.row_offsets[source];
             edge < before.row_offsets[source + 1]; ++edge) {
            const double initial = before.weights[edge];
            const double final = after.weights[edge];
            const double change = std::abs(final - initial);
            ++summary.plastic_synapses;
            initial_sum += initial;
            final_sum += final;
            absolute_change_sum += change;
            summary.maximum_absolute_change =
                std::max(summary.maximum_absolute_change, change);
            if (final != initial) ++summary.changed_synapses;
        }
    }
    if (summary.plastic_synapses != 0) {
        const double count = static_cast<double>(summary.plastic_synapses);
        summary.mean_initial_weight = initial_sum / count;
        summary.mean_final_weight = final_sum / count;
        summary.mean_absolute_change = absolute_change_sum / count;
    }
    return summary;
}

} // namespace spiking
