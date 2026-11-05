#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace spiking {

// Source-oriented compressed sparse row connectivity.
// Row i contains the outgoing synapses from source neuron i.
struct CSRTopology {
    std::uint32_t excitatory_count = 0;
    std::uint32_t inhibitory_count = 0;
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> target_indices;
    std::vector<float> weights;
    std::vector<std::uint32_t> delays;

    std::uint32_t minimum_delay = 1;
    std::uint32_t maximum_delay = 1;

    std::uint32_t neuron_count() const noexcept {
        return excitatory_count + inhibitory_count;
    }

    std::size_t synapse_count() const noexcept {
        return target_indices.size();
    }

    // Throws std::invalid_argument when the CSR representation is malformed.
    void validate() const {
        const std::uint64_t total =
            static_cast<std::uint64_t>(excitatory_count) + inhibitory_count;

        if (total == 0 ||
            total > static_cast<std::uint64_t>(UINT32_MAX)) {
            throw std::invalid_argument("Invalid CSR neuron count");
        }

        if (row_offsets.size() != static_cast<std::size_t>(total) + 1) {
            throw std::invalid_argument("CSR row_offsets must have neuron_count + 1 entries");
        }

        if (target_indices.size() != weights.size() ||
            target_indices.size() != delays.size()) {
            throw std::invalid_argument(
                "CSR target, weight, and delay arrays have different sizes");
        }

        if (minimum_delay == 0 || maximum_delay < minimum_delay) {
            throw std::invalid_argument("Invalid CSR delay bounds");
        }

        if (row_offsets.front() != 0) {
            throw std::invalid_argument("CSR row_offsets must start at zero");
        }

        if (row_offsets.back() != target_indices.size()) {
            throw std::invalid_argument("CSR final offset does not equal synapse count");
        }

        for (std::uint32_t source = 0;
             source < static_cast<std::uint32_t>(total); ++source) {
            const auto begin = row_offsets[source];
            const auto end = row_offsets[source + 1];

            if (begin > end || end > target_indices.size()) {
                throw std::invalid_argument("CSR row offsets are not monotonic or in range");
            }

            std::uint32_t previous_target = 0;
            bool first = true;

            for (std::uint32_t edge = begin; edge < end; ++edge) {
                const auto target = target_indices[edge];
                const float weight = weights[edge];
                const std::uint32_t delay = delays[edge];

                if (target >= total) {
                    throw std::invalid_argument("CSR target index is out of range");
                }
                if (target == source) {
                    throw std::invalid_argument("CSR self-connections are not permitted");
                }
                if (!first && target <= previous_target) {
                    throw std::invalid_argument(
                        "CSR targets must be strictly increasing within each row");
                }
                if (!(weight == weight) ||
                    weight > 3.402823466e+38F ||
                    weight < -3.402823466e+38F) {
                    throw std::invalid_argument("CSR contains a non-finite weight");
                }

                if (delay < minimum_delay || delay > maximum_delay) {
                    throw std::invalid_argument(
                        "CSR synaptic delay is outside configured bounds");
                }

                const bool excitatory_source = source < excitatory_count;
                if ((excitatory_source && weight <= 0.0f) ||
                    (!excitatory_source && weight >= 0.0f)) {
                    throw std::invalid_argument(
                        "CSR weight sign does not match source population");
                }

                previous_target = target;
                first = false;
            }
        }
    }
};

} // namespace spiking
