#include "learning/phase_structural_plasticity.h"

#include <array>
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
    p.activity_kick_rad = 0.04f;
    p.coupling_learning_rate = 0.001f;
    p.maximum_coupling = 0.5f;
    p.coupling_decay = 0.0001f;
    p.temporal_association_window_steps = 299;
    p.temporal_lag_learning_rate = 0.10f;
    p.minimum_temporal_confidence = 0.20f;

    // A is the common presynaptic population. B receives a stable 5-step
    // delay; C receives the same number of events with a deliberately varied
    // delay schedule. This is a within-network timing-specificity control.
    AutonomousPopulationPhase phase(3, p);
    constexpr int train_steps = 6000;
    constexpr int period = 300;
    constexpr int paired_offset = 5;
    constexpr std::array<int, 8> control_offsets{20, 55, 90, 125, 160, 195, 230, 265};

    for (int t = 0; t < train_steps; ++t) {
        std::vector<std::uint32_t> spikes(3, 0U);
        if (t % period == 0) {
            spikes[0] = 1U;
        }
        const int cycle = t / period;
        const int within = t % period;
        if (within == paired_offset) spikes[1] = 1U;
        if (within == control_offsets[static_cast<std::size_t>(cycle) %
                                       control_offsets.size()]) {
            spikes[2] = 1U;
        }
        phase.step(spikes, true);
    }

    const float paired_lag = phase.learned_lag(0, 1);
    const float control_lag = phase.learned_lag(0, 2);
    const float paired_confidence = phase.temporal_confidence(0, 1);
    const float control_confidence = phase.temporal_confidence(0, 2);
    const float paired_coupling = phase.coupling(0, 1);
    const float control_coupling = phase.coupling(0, 2);
    const float trained_ab = wrapped_difference(phase.phase(0), phase.phase(1));
    const float trained_ac = wrapped_difference(phase.phase(0), phase.phase(2));

    // Freeze all learning and remove every external event. Check retention of
    // the learned directed lag estimates, not merely convergence of phases.
    for (int t = 0; t < 2000; ++t) phase.step({0U, 0U, 0U}, false);

    const float frozen_ab = wrapped_difference(phase.phase(0), phase.phase(1));
    const float frozen_ac = wrapped_difference(phase.phase(0), phase.phase(2));
    const float frozen_paired_lag = phase.learned_lag(0, 1);
    const float frozen_control_lag = phase.learned_lag(0, 2);
    const float paired_lag_drift = wrapped_difference(frozen_paired_lag, paired_lag);
    const float control_lag_drift = wrapped_difference(frozen_control_lag, control_lag);

    const bool finite = std::isfinite(paired_lag) && std::isfinite(control_lag) &&
        std::isfinite(paired_confidence) && std::isfinite(control_confidence) &&
        std::isfinite(frozen_ab) && std::isfinite(frozen_ac);
    const bool discriminates = paired_confidence > control_confidence + 0.20f;
    const bool retains = std::abs(paired_lag_drift) < 1.0e-6f &&
                         std::abs(control_lag_drift) < 1.0e-6f;

    std::cout << "V42 DIRECTED TEMPORAL PHASE-ASSOCIATION TEST\n"
              << "populations=A,B,C\n"
              << "training_steps=" << train_steps << '\n'
              << "period_steps=" << period << '\n'
              << "A_to_B_fixed_offset_steps=" << paired_offset << '\n'
              << "A_to_C_variable_offsets=";
    for (std::size_t i = 0; i < control_offsets.size(); ++i)
        std::cout << (i ? "," : "") << control_offsets[i];
    std::cout << '\n'
              << "A_to_B_learned_lag_rad=" << paired_lag << '\n'
              << "A_to_C_control_lag_rad=" << control_lag << '\n'
              << "A_to_B_temporal_confidence=" << paired_confidence << '\n'
              << "A_to_C_temporal_confidence=" << control_confidence << '\n'
              << "A_to_B_coupling=" << paired_coupling << '\n'
              << "A_to_C_coupling=" << control_coupling << '\n'
              << "trained_A_B_phase_difference=" << trained_ab << '\n'
              << "trained_A_C_phase_difference=" << trained_ac << '\n'
              << "frozen_A_B_phase_difference=" << frozen_ab << '\n'
              << "frozen_A_C_phase_difference=" << frozen_ac << '\n'
              << "frozen_A_to_B_lag=" << frozen_paired_lag << '\n'
              << "frozen_A_to_C_lag=" << frozen_control_lag << '\n'
              << "paired_lag_drift=" << paired_lag_drift << '\n'
              << "control_lag_drift=" << control_lag_drift << '\n'
              << "plasticity_frozen_during_recall=YES\n"
              << "external_spikes_during_recall=NONE\n"
              << "finite_state=" << (finite ? "PASS" : "FAIL") << '\n'
              << "temporal_discrimination=" << (discriminates ? "PASS" : "FAIL") << '\n'
              << "frozen_lag_retention=" << (retains ? "PASS" : "FAIL") << '\n'
              << "NOTE: host-side spike-timing association test; CUDA transmission "
                 "must still be tested using the learned lag values.\n";

    return finite ? 0 : 1;
}
