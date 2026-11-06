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

// Incoming adjacency for postsynaptic traversal. Each target row stores the
// presynaptic source and the corresponding edge index in the original
// source-oriented CSR weight/delay arrays.
struct IncomingCSRIndex {
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> source_indices;
    std::vector<std::uint32_t> edge_indices;

    void validate(std::uint32_t neuron_count, std::size_t edge_count) const {
        if (row_offsets.size() != static_cast<std::size_t>(neuron_count) + 1 ||
            source_indices.size() != edge_count ||
            edge_indices.size() != edge_count ||
            row_offsets.empty() || row_offsets.front() != 0 ||
            row_offsets.back() != edge_count) {
            throw std::invalid_argument("Malformed incoming CSR index");
        }
        for (std::uint32_t target = 0; target < neuron_count; ++target) {
            if (row_offsets[target] > row_offsets[target + 1] ||
                row_offsets[target + 1] > edge_count) {
                throw std::invalid_argument("Incoming CSR offsets are not monotonic");
            }
            std::uint32_t previous_source = 0;
            bool first = true;
            for (std::uint32_t entry = row_offsets[target];
                 entry < row_offsets[target + 1]; ++entry) {
                const auto source = source_indices[entry];
                if (source >= neuron_count || source == target ||
                    edge_indices[entry] >= edge_count ||
                    (!first && source <= previous_source)) {
                    throw std::invalid_argument("Invalid incoming CSR entry");
                }
                previous_source = source;
                first = false;
            }
        }
    }
};

// Construct a deterministic reverse index without duplicating synaptic
// weights. edge_indices always address the original outgoing CSR arrays.
inline IncomingCSRIndex build_incoming_csr_index(const CSRTopology& topology) {
    topology.validate();
    const auto neurons = topology.neuron_count();
    if (topology.synapse_count() >
        static_cast<std::size_t>(UINT32_MAX)) {
        throw std::length_error("Incoming CSR supports at most UINT32_MAX edges");
    }

    IncomingCSRIndex incoming;
    incoming.row_offsets.assign(static_cast<std::size_t>(neurons) + 1, 0);
    incoming.source_indices.resize(topology.synapse_count());
    incoming.edge_indices.resize(topology.synapse_count());

    for (const auto target : topology.target_indices) {
        ++incoming.row_offsets[static_cast<std::size_t>(target) + 1];
    }
    for (std::uint32_t target = 0; target < neurons; ++target) {
        incoming.row_offsets[target + 1] += incoming.row_offsets[target];
    }

    auto cursor = incoming.row_offsets;
    for (std::uint32_t source = 0; source < neurons; ++source) {
        for (std::uint32_t edge = topology.row_offsets[source];
             edge < topology.row_offsets[source + 1]; ++edge) {
            const auto target = topology.target_indices[edge];
            const auto slot = cursor[target]++;
            incoming.source_indices[slot] = source;
            incoming.edge_indices[slot] = edge;
        }
    }

    incoming.validate(neurons, topology.synapse_count());
    return incoming;
}

} // namespace spiking
