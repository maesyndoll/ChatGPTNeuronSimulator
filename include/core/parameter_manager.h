#pragma once

#include "core/parameters.h"

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spiking {

struct ParameterUpdateResult {
    bool accepted = false;
    bool requires_reinitialization = false;
    std::uint64_t parameter_version = 0;
    std::uint64_t simulation_generation = 0;
    std::vector<ParameterValidationError> errors;
};

// Owns desired and active configuration snapshots. State/weights are owned by
// the runtime and are deliberately not touched by parameter resets.
class ParameterManager {
public:
    explicit ParameterManager(ModelParameters defaults = ModelParameters{})
        : defaults_(std::move(defaults)), active_(defaults_), requested_(defaults_) {
        const auto errors = validate_parameters(defaults_);
        if (!errors.empty()) {
            throw std::invalid_argument("ParameterManager constructed with invalid defaults: " +
                                        errors.front().field + ": " + errors.front().message);
        }
    }

    ParameterUpdateResult apply_parameters(const ModelParameters& candidate) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto errors = validate_parameters(candidate);
        if (!errors.empty()) return result(false, errors);

        requested_ = candidate;
        // Step-boundary values are published together as one coherent snapshot.
        active_.simulation.steps = candidate.simulation.steps;
        active_.simulation.dt = candidate.simulation.dt;
        active_.simulation.threshold = candidate.simulation.threshold;
        ++parameter_version_;
        return result(true, {});
    }

    // Reset configuration defaults without resetting neuron state, synapses,
    // learned weights, or plasticity traces. Initialization-only settings are
    // staged and become active only when reinitialize_simulation() is called.
    ParameterUpdateResult reset_parameters_to_defaults() {
        std::lock_guard<std::mutex> lock(mutex_);
        requested_ = defaults_;
        active_.simulation.steps = defaults_.simulation.steps;
        active_.simulation.dt = defaults_.simulation.dt;
        active_.simulation.threshold = defaults_.simulation.threshold;
        ++parameter_version_;
        return result(true, {});
    }

    // Commits staged initialization/topology settings. The runtime must use the
    // returned snapshot to rebuild GPU-owned simulation state at a safe point.
    ModelParameters reinitialize_simulation() {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = requested_;
        ++parameter_version_;
        ++simulation_generation_;
        return active_;
    }

    ModelParameters active_parameters() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_;
    }

    ModelParameters requested_parameters() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return requested_;
    }

    bool has_pending_reinitialization() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !same_initialization_parameters(active_, requested_);
    }

    std::uint64_t parameter_version() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return parameter_version_;
    }

    std::uint64_t simulation_generation() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return simulation_generation_;
    }

private:
    static bool same_initialization_parameters(const ModelParameters& a,
                                              const ModelParameters& b) {
        const auto& x = a.simulation;
        const auto& y = b.simulation;
        const auto& tx = a.topology;
        const auto& ty = b.topology;
        return x.excitatory_count == y.excitatory_count &&
               x.inhibitory_count == y.inhibitory_count &&
               x.random_seed == y.random_seed &&
               x.excitatory_current_min == y.excitatory_current_min &&
               x.excitatory_current_max == y.excitatory_current_max &&
               x.inhibitory_current_min == y.inhibitory_current_min &&
               x.inhibitory_current_max == y.inhibitory_current_max &&
               x.initial_voltage_min == y.initial_voltage_min &&
               x.initial_voltage_max == y.initial_voltage_max &&
               tx.p_ee == ty.p_ee && tx.p_ei == ty.p_ei &&
               tx.p_ie == ty.p_ie && tx.p_ii == ty.p_ii &&
               tx.master_seed == ty.master_seed &&
               tx.topology_seed == ty.topology_seed && tx.delay_seed == ty.delay_seed &&
               tx.delay_distribution == ty.delay_distribution &&
               tx.minimum_delay == ty.minimum_delay && tx.maximum_delay == ty.maximum_delay &&
               tx.fibonacci_delays == ty.fibonacci_delays &&
               tx.fibonacci_weights == ty.fibonacci_weights &&
               tx.memory_budget_bytes == ty.memory_budget_bytes;
    }

    ParameterUpdateResult result(bool accepted,
                                 std::vector<ParameterValidationError> errors) const {
        return {accepted, !same_initialization_parameters(active_, requested_),
                parameter_version_, simulation_generation_, std::move(errors)};
    }

    ModelParameters defaults_;
    ModelParameters active_;
    ModelParameters requested_;
    mutable std::mutex mutex_;
    std::uint64_t parameter_version_ = 0;
    std::uint64_t simulation_generation_ = 0;
};

} // namespace spiking
