#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace spiking {

// Reference parameters for pair-based excitatory STDP. Weight values remain
// positive; inhibitory-source synapses are deliberately left unchanged.
struct StdpParameters {
    float potentiation_rate = 0.005f;
    float depression_rate = 0.00525f;
    float presynaptic_tau_ms = 20.0f;
    float postsynaptic_tau_ms = 20.0f;
    float minimum_weight = 0.001f;
    float maximum_weight = 1.0f;
};

struct StdpTraces {
    std::vector<float> presynaptic;
    std::vector<float> postsynaptic;
};

// Deterministic CPU reference for one event boundary. Event flags describe
// spikes emitted in the integration step that just completed. Existing
// traces are decayed to this boundary, weight changes use those old traces,
// and current events are added only after both update directions. Therefore
// simultaneous pre/post spikes have no zero-lag pairing.
inline void apply_pair_stdp_step(
    std::vector<float>& weights,
    const std::vector<std::uint32_t>& sources,
    const std::vector<std::uint32_t>& targets,
    std::uint32_t neuron_count,
    std::uint32_t excitatory_count,
    const std::vector<std::uint8_t>& spike_events,
    StdpTraces& traces,
    float dt_ms,
    const StdpParameters& parameters = {}) {
    if (weights.size() != sources.size() || weights.size() != targets.size())
        throw std::invalid_argument("STDP edge arrays must have equal sizes");
    if (spike_events.size() != neuron_count ||
        traces.presynaptic.size() != neuron_count ||
        traces.postsynaptic.size() != neuron_count)
        throw std::invalid_argument("STDP neuron arrays have invalid sizes");
    if (excitatory_count > neuron_count || !std::isfinite(dt_ms) || dt_ms <= 0.0f ||
        !std::isfinite(parameters.potentiation_rate) || parameters.potentiation_rate < 0.0f ||
        !std::isfinite(parameters.depression_rate) || parameters.depression_rate < 0.0f ||
        !std::isfinite(parameters.presynaptic_tau_ms) || parameters.presynaptic_tau_ms <= 0.0f ||
        !std::isfinite(parameters.postsynaptic_tau_ms) || parameters.postsynaptic_tau_ms <= 0.0f ||
        !std::isfinite(parameters.minimum_weight) || parameters.minimum_weight <= 0.0f ||
        !std::isfinite(parameters.maximum_weight) ||
        parameters.maximum_weight < parameters.minimum_weight)
        throw std::invalid_argument("Invalid STDP parameters");

    const float pre_decay = std::exp(-dt_ms / parameters.presynaptic_tau_ms);
    const float post_decay = std::exp(-dt_ms / parameters.postsynaptic_tau_ms);
    for (std::uint32_t neuron = 0; neuron < neuron_count; ++neuron) {
        traces.presynaptic[neuron] *= pre_decay;
        traces.postsynaptic[neuron] *= post_decay;
    }

    // One canonical edge-order pass is the reference for the two sequential
    // GPU passes. Only excitatory presynaptic edges are plastic.
    for (std::size_t edge = 0; edge < weights.size(); ++edge) {
        const auto source = sources[edge];
        const auto target = targets[edge];
        if (source >= neuron_count || target >= neuron_count || source == target)
            throw std::invalid_argument("STDP edge endpoint is invalid");
        if (source >= excitatory_count) continue;
        float weight = weights[edge];
        if (!std::isfinite(weight) || weight <= 0.0f)
            throw std::invalid_argument("Excitatory STDP weights must be finite and positive");

        if (spike_events[source] != 0)
            weight -= parameters.depression_rate * traces.postsynaptic[target];
        if (spike_events[target] != 0)
            weight += parameters.potentiation_rate * traces.presynaptic[source];
        weights[edge] = std::clamp(
            weight, parameters.minimum_weight, parameters.maximum_weight);
    }

    for (std::uint32_t neuron = 0; neuron < neuron_count; ++neuron) {
        if (spike_events[neuron] != 0) {
            traces.presynaptic[neuron] += 1.0f;
            traces.postsynaptic[neuron] += 1.0f;
        }
    }
}

} // namespace spiking
