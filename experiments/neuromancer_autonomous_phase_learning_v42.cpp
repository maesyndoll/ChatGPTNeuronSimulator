#include "learning/phase_structural_plasticity.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
float wrapped_difference(float a, float b) {
    constexpr float tau = 6.2831853071795864769f;
    return std::remainder(a - b, tau);
}
}

int main() {
    using namespace spiking;
    AutonomousPhaseParameters p;
    p.dt_ms = 0.5f;
    p.intrinsic_frequency_hz = 8.0f;
    p.activity_kick_rad = 0.08f;
    p.coupling_learning_rate = 0.001f;
    p.maximum_coupling = 0.5f;
    p.coupling_decay = 0.0001f;
    AutonomousPopulationPhase paired(2, p);
    AutonomousPopulationPhase unpaired(2, p);

    const float initial_lag = paired.learned_lag(0, 1);
    constexpr int train_steps = 6000;
    constexpr int period = 40;
    constexpr int offset = 5;
    for (int t = 0; t < train_steps; ++t) {
        std::vector<std::uint32_t> paired_spikes(2, 0U);
        if (t % period == 0) paired_spikes[0] = 1;
        if (t % period == offset) paired_spikes[1] = 1;
        paired.step(paired_spikes, true);

        // Negative control: equal spike counts, but independently jittered
        // event times prevent a stable repeated phase relationship.
        std::vector<std::uint32_t> control_spikes(2, 0U);
        if (t % period == 0) control_spikes[0] = 1;
        if ((t * 17 + 11) % period == 0) control_spikes[1] = 1;
        unpaired.step(control_spikes, true);
    }

    const float learned_lag = paired.learned_lag(0, 1);
    const float control_lag = unpaired.learned_lag(0, 1);
    const float learned_coupling = paired.coupling(0, 1);
    const float control_coupling = unpaired.coupling(0, 1);
    const float trained_phase_difference =
        wrapped_difference(paired.phase(0), paired.phase(1));

    // Freeze plasticity and remove all external spikes. Continue autonomous
    // dynamics to determine whether the trained relation persists.
    for (int t = 0; t < 2000; ++t) {
        paired.step({0U, 0U}, false);
        unpaired.step({0U, 0U}, false);
    }
    const float frozen_phase_difference =
        wrapped_difference(paired.phase(0), paired.phase(1));

    std::cout << "V42 AUTONOMOUS PHASE-LEARNING / FROZEN-RETENTION DIAGNOSTIC\n"
              << "training_steps=" << train_steps << '\n'
              << "training_period_steps=" << period << '\n'
              << "paired_event_offset_steps=" << offset << '\n'
              << "initial_lag=" << initial_lag << '\n'
              << "paired_learned_lag=" << learned_lag << '\n'
              << "unpaired_control_lag=" << control_lag << '\n'
              << "paired_coupling=" << learned_coupling << '\n'
              << "unpaired_control_coupling=" << control_coupling << '\n'
              << "trained_phase_difference=" << trained_phase_difference << '\n'
              << "frozen_phase_difference=" << frozen_phase_difference << '\n'
              << "frozen_phase_drift="
              << wrapped_difference(frozen_phase_difference,
                                    trained_phase_difference) << '\n'
              << "plasticity_frozen_during_recall=YES\n"
              << "external_spikes_during_recall=NONE\n"
              << "NOTE: diagnostic only; a nonzero lag alone is not proof of "
                 "associative memory or robust phase learning.\n";
    return std::isfinite(learned_lag) && std::isfinite(frozen_phase_difference)
        ? 0 : 1;
}
