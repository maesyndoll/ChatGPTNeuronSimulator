#pragma once

#include "core/parameters_generated.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

namespace spiking {

struct ParameterValidationError {
    std::string field;
    std::string message;
};

struct ModelParameters {
    SimulationParameters simulation{};
    TopologyParameters topology{};
};

inline std::vector<ParameterValidationError> validate_parameters(
    const ModelParameters& p
) {
    std::vector<ParameterValidationError> errors;
    const auto add = [&errors](const char* field, const char* message) {
        errors.push_back({field, message});
    };
    const auto schema_range = [&add](const char* group, const char* name,
                                     long double value) {
        for (const auto& descriptor : kParameterDescriptors) {
            if (std::string(descriptor.group) != group ||
                std::string(descriptor.name) != name) continue;
            if (!std::isfinite(static_cast<double>(value))) {
                add(name, "value must be finite");
                return;
            }
            if (descriptor.minimum[0] != '\0') {
                const long double minimum = std::strtold(descriptor.minimum, nullptr);
                if (value < minimum) add(name, "value is below the schema minimum");
            }
            if (descriptor.maximum[0] != '\0') {
                const long double maximum = std::strtold(descriptor.maximum, nullptr);
                if (value > maximum) add(name, "value exceeds the schema maximum");
            }
            return;
        }
        add(name, "parameter is missing from the authoritative schema");
    };

    const auto& s = p.simulation;
    schema_range("simulation", "excitatory_count", s.excitatory_count);
    schema_range("simulation", "inhibitory_count", s.inhibitory_count);
    schema_range("simulation", "steps", s.steps);
    schema_range("simulation", "dt", s.dt);
    schema_range("simulation", "threshold", s.threshold);
    schema_range("simulation", "random_seed", s.random_seed);
    schema_range("simulation", "excitatory_current_min", s.excitatory_current_min);
    schema_range("simulation", "excitatory_current_max", s.excitatory_current_max);
    schema_range("simulation", "inhibitory_current_min", s.inhibitory_current_min);
    schema_range("simulation", "inhibitory_current_max", s.inhibitory_current_max);
    schema_range("simulation", "initial_voltage_min", s.initial_voltage_min);
    schema_range("simulation", "initial_voltage_max", s.initial_voltage_max);
    if (static_cast<std::uint64_t>(s.excitatory_count) + s.inhibitory_count > 1250000ULL)
        add("neuron_count", "combined population exceeds the supported 1,250,000-neuron limit");
    if (s.excitatory_current_min > s.excitatory_current_max)
        add("excitatory_current_range", "minimum must not exceed maximum");
    if (s.inhibitory_current_min > s.inhibitory_current_max)
        add("inhibitory_current_range", "minimum must not exceed maximum");
    if (s.initial_voltage_min > s.initial_voltage_max)
        add("initial_voltage_range", "minimum must not exceed maximum");

    const auto& t = p.topology;
    schema_range("topology", "p_ee", t.p_ee);
    schema_range("topology", "p_ei", t.p_ei);
    schema_range("topology", "p_ie", t.p_ie);
    schema_range("topology", "p_ii", t.p_ii);
    schema_range("topology", "master_seed", t.master_seed);
    schema_range("topology", "topology_seed", t.topology_seed);
    schema_range("topology", "delay_seed", t.delay_seed);
    schema_range("topology", "minimum_delay", t.minimum_delay);
    schema_range("topology", "maximum_delay", t.maximum_delay);
    schema_range("topology", "memory_budget_bytes", t.memory_budget_bytes);
    if (t.minimum_delay == 0) add("minimum_delay", "must be at least one simulation step");
    if (t.maximum_delay < t.minimum_delay)
        add("delay_range", "maximum_delay must be greater than or equal to minimum_delay");
    if (t.fibonacci_delays.empty()) add("fibonacci_delays", "must contain at least one delay");
    if (t.fibonacci_delays.size() != t.fibonacci_weights.size())
        add("fibonacci_weights", "must have the same number of entries as fibonacci_delays");
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < t.fibonacci_delays.size(); ++i) {
        if (t.fibonacci_delays[i] < t.minimum_delay ||
            t.fibonacci_delays[i] > t.maximum_delay)
            add("fibonacci_delays", "each candidate must lie within the configured delay range");
        if (i > 0 && t.fibonacci_delays[i] <= t.fibonacci_delays[i - 1])
            add("fibonacci_delays", "candidate delays must be strictly increasing");
    }
    for (double weight : t.fibonacci_weights) {
        if (!std::isfinite(weight) || weight < 0.0)
            add("fibonacci_weights", "weights must be finite and non-negative");
        weight_sum += weight;
    }
    if (!t.fibonacci_weights.empty() && weight_sum <= 0.0)
        add("fibonacci_weights", "at least one weight must be positive");
    if (t.memory_budget_bytes < 1024ULL * 1024ULL)
        add("memory_budget_bytes", "must be at least 1 MiB");

    return errors;
}

} // namespace spiking
