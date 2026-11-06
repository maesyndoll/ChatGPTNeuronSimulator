#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace spiking {

struct ActiveNeuronCounts {
    std::uint64_t excitatory = 0;
    std::uint64_t inhibitory = 0;
    double excitatory_percent = 0.0;
    double inhibitory_percent = 0.0;
};

inline ActiveNeuronCounts count_active_neurons(
    const std::vector<std::uint32_t>& previous,
    const std::vector<std::uint32_t>& current,
    std::size_t excitatory_count) {
    if (previous.size() != current.size())
        throw std::invalid_argument("Spike-count snapshots must have equal sizes");
    if (excitatory_count > current.size())
        throw std::invalid_argument("Excitatory population exceeds snapshot size");

    ActiveNeuronCounts result;
    const auto inhibitory_count = current.size() - excitatory_count;
    for (std::size_t i = 0; i < current.size(); ++i) {
        if (current[i] == previous[i]) continue;
        if (i < excitatory_count) ++result.excitatory;
        else ++result.inhibitory;
    }
    if (excitatory_count != 0)
        result.excitatory_percent = 100.0 * result.excitatory / excitatory_count;
    if (inhibitory_count != 0)
        result.inhibitory_percent = 100.0 * result.inhibitory / inhibitory_count;
    return result;
}

} // namespace spiking
