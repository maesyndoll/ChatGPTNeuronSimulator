#include "analysis/izhikevich_analysis.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace spiking::analysis;

namespace {
void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
}

int main() {
    NeuronParameters p;

    // An equilibrium should remain stationary under integration.
    const auto equilibrium = simulate_trajectory(
        p, -70.0, -14.0, 0.0, 0.01, 100, 30.0
    );

    check(equilibrium.samples.size() == 101,
          "Expected initial sample plus 100 integration steps");
    check(equilibrium.spike_times.empty(),
          "Equilibrium trajectory should not spike");

    for (const auto& sample : equilibrium.samples) {
        check(std::abs(sample.voltage - (-70.0)) < 1e-9,
              "Equilibrium voltage drifted");
        check(std::abs(sample.recovery - (-14.0)) < 1e-9,
              "Equilibrium recovery drifted");
    }

    // A strong input should cross threshold and trigger reset.
    const auto spiking = simulate_trajectory(
        p, 29.9, 0.0, 1000.0, 0.01, 2, 30.0
    );

    check(!spiking.spike_times.empty(),
          "Expected at least one spike");
    check(spiking.samples[1].spiked,
          "Expected a spike on the first timestep");
    check(std::abs(spiking.samples[1].voltage - p.c) < 1e-10,
          "Voltage was not reset to c");
    check(spiking.samples[1].recovery > p.d,
          "Recovery should include the spike increment d");

    check(std::abs(
        spiking.spike_times[0] - spiking.samples[1].time
    ) < 1e-12, "Spike time should match the threshold-crossing timestep");

    std::cout << "Trajectory tests passed.\n";
    return 0;
}
