#include "core/parameter_manager.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
}

int main() {
    using namespace spiking;

    ModelParameters defaults;
    check(validate_parameters(defaults).empty(), "generated defaults validate");
    check(defaults.simulation.neuron_count() == 92160,
          "default network is within the agreed 10k-100k scale");

    auto invalid = defaults;
    invalid.simulation.dt = 0.0f;
    check(!validate_parameters(invalid).empty(), "invalid timestep is rejected");
    invalid = defaults;
    invalid.topology.fibonacci_weights.pop_back();
    check(!validate_parameters(invalid).empty(), "mismatched delay weights are rejected");
    invalid = defaults;
    invalid.simulation.dt = 10.0f;
    check(!validate_parameters(invalid).empty(), "schema maximum for timestep is enforced");
    invalid = defaults;
    invalid.topology.p_ee = 1.5;
    check(!validate_parameters(invalid).empty(), "schema maximum for connection probability is enforced");

    ParameterManager manager(defaults);
    auto candidate = manager.requested_parameters();
    candidate.simulation.dt = 0.25f;
    candidate.simulation.threshold = 28.0f;
    auto update = manager.apply_parameters(candidate);
    check(update.accepted, "valid runtime parameter update is accepted");
    check(!update.requires_reinitialization,
          "step-boundary parameter changes do not require reinitialization");
    check(manager.active_parameters().simulation.dt == 0.25f,
          "active timestep changes atomically in the manager snapshot");
    check(manager.simulation_generation() == 0,
          "ordinary parameter updates do not increment simulation generation");

    candidate = manager.requested_parameters();
    candidate.simulation.excitatory_count = 40960;
    update = manager.apply_parameters(candidate);
    check(update.accepted && update.requires_reinitialization,
          "population-size change is staged for reinitialization");
    check(manager.active_parameters().simulation.excitatory_count == 81920,
          "staged structural changes do not alter active allocation dimensions");
    check(manager.requested_parameters().simulation.excitatory_count == 40960,
          "requested structural change is retained");

    const auto generation_before_reset = manager.simulation_generation();
    update = manager.reset_parameters_to_defaults();
    check(update.accepted, "reset-to-defaults succeeds");
    check(manager.active_parameters().simulation.dt == defaults.simulation.dt,
          "reset restores runtime parameters");
    check(manager.active_parameters().simulation.excitatory_count == 81920,
          "reset does not resize or reset active neural state");
    check(manager.simulation_generation() == generation_before_reset,
          "parameter reset does not reinitialize simulation");

    candidate = manager.requested_parameters();
    candidate.simulation.excitatory_count = 40960;
    candidate.topology.delay_distribution = DelayDistribution::Uniform;
    update = manager.apply_parameters(candidate);
    check(update.accepted && update.requires_reinitialization,
          "topology changes are staged for reinitialization");
    const auto committed = manager.reinitialize_simulation();
    check(committed.simulation.excitatory_count == 40960,
          "full reinitialization commits requested population size");
    check(committed.topology.delay_distribution == DelayDistribution::Uniform,
          "full reinitialization commits requested topology");
    check(!manager.has_pending_reinitialization(),
          "no structural changes remain pending after reinitialization");
    check(manager.simulation_generation() == 1,
          "full reinitialization increments simulation generation");

    if (failures == 0) {
        std::cout << "All parameter tests passed.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " parameter test(s) failed.\n";
    return EXIT_FAILURE;
}
