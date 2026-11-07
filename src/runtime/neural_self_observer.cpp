#include "runtime/neural_self_observer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace spiking {
namespace {

std::uint64_t mix(std::uint64_t hash, std::uint64_t value) noexcept {
    // FNV-1a over each 64-bit value, byte by byte, for a stable graph identity.
    for (unsigned int i = 0; i < 8; ++i) {
        hash ^= (value >> (i * 8U)) & 0xffU;
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

SelfObservationReport observe_connectome(
    const CSRTopology& topology,
    const std::vector<std::uint64_t>& activity_counts,
    float weak_weight_threshold,
    std::size_t maximum_ranked_neurons) {
    topology.validate();
    const auto neurons = topology.neuron_count();
    if (!activity_counts.empty() && activity_counts.size() != neurons) {
        throw std::invalid_argument(
            "Self-observer activity vector must match neuron count");
    }
    if (!std::isfinite(weak_weight_threshold) || weak_weight_threshold < 0.0f) {
        throw std::invalid_argument("Weak-weight threshold must be finite and non-negative");
    }

    SelfObservationReport report;
    report.neuron_count = neurons;
    report.synapse_count = topology.synapse_count();

    std::vector<std::uint32_t> incoming(neurons, 0);
    std::vector<std::uint32_t> outgoing(neurons, 0);
    std::uint64_t hash = 14695981039346656037ULL;
    hash = mix(hash, topology.excitatory_count);
    hash = mix(hash, topology.inhibitory_count);
    hash = mix(hash, topology.synapse_count());

    for (std::uint32_t source = 0; source < neurons; ++source) {
        const auto begin = topology.row_offsets[source];
        const auto end = topology.row_offsets[source + 1];
        outgoing[source] = end - begin;
        hash = mix(hash, source);
        hash = mix(hash, begin);
        hash = mix(hash, end);
        for (std::uint32_t edge = begin; edge < end; ++edge) {
            const auto target = topology.target_indices[edge];
            ++incoming[target];
            hash = mix(hash, target);
            hash = mix(hash, static_cast<std::uint64_t>(
                static_cast<std::int64_t>(std::llround(
                    static_cast<double>(topology.weights[edge]) * 1000000.0))));
            hash = mix(hash, topology.delays[edge]);
            if (std::abs(topology.weights[edge]) <= weak_weight_threshold) {
                ++report.weak_edges;
            }
        }
    }
    report.graph_fingerprint = hash;

    std::uint64_t outgoing_total = 0;
    std::uint64_t incoming_total = 0;
    std::vector<SelfObservedNeuron> ranked;
    ranked.reserve(neurons);
    for (std::uint32_t neuron = 0; neuron < neurons; ++neuron) {
        outgoing_total += outgoing[neuron];
        incoming_total += incoming[neuron];
        if (outgoing[neuron] == 0 && incoming[neuron] == 0) {
            ++report.isolated_neurons;
        }
        if (!activity_counts.empty() && activity_counts[neuron] == 0) {
            ++report.inactive_neurons;
        }
        ranked.push_back({
            neuron, outgoing[neuron], incoming[neuron],
            activity_counts.empty() ? 0 : activity_counts[neuron]
        });
    }

    report.mean_outgoing_degree =
        static_cast<double>(outgoing_total) / neurons;
    report.mean_incoming_degree =
        static_cast<double>(incoming_total) / neurons;

    double squared_deviation = 0.0;
    for (const auto degree : outgoing) {
        const double delta = static_cast<double>(degree) - report.mean_outgoing_degree;
        squared_deviation += delta * delta;
    }
    const double deviation = std::sqrt(squared_deviation / neurons);
    const double hub_threshold = report.mean_outgoing_degree + 2.0 * deviation;
    for (const auto degree : outgoing) {
        if (static_cast<double>(degree) > hub_threshold) {
            ++report.high_degree_neurons;
        }
    }

    for (std::uint32_t source = 0; source < neurons; ++source) {
        for (std::uint32_t edge = topology.row_offsets[source];
             edge < topology.row_offsets[source + 1]; ++edge) {
            const auto target = topology.target_indices[edge];
            const auto reverse_begin = topology.row_offsets[target];
            const auto reverse_end = topology.row_offsets[target + 1];
            if (std::binary_search(
                    topology.target_indices.begin() + reverse_begin,
                    topology.target_indices.begin() + reverse_end,
                    source)) {
                ++report.reciprocal_pairs;
            }
        }
    }
    // Each reciprocal relationship was encountered in both directions.
    report.reciprocal_pairs /= 2;

    std::sort(ranked.begin(), ranked.end(),
        [](const SelfObservedNeuron& a, const SelfObservedNeuron& b) {
            const auto a_total = static_cast<std::uint64_t>(a.outgoing_degree) +
                                 a.incoming_degree;
            const auto b_total = static_cast<std::uint64_t>(b.outgoing_degree) +
                                 b.incoming_degree;
            if (a_total != b_total) return a_total > b_total;
            return a.neuron_id < b.neuron_id;
        });
    if (ranked.size() > maximum_ranked_neurons) {
        ranked.resize(maximum_ranked_neurons);
    }
    report.highest_degree_neurons = std::move(ranked);
    return report;
}

} // namespace spiking
