#include "core/parameters.h"
#include "learning/phase_structural_plasticity.h"
#include "plasticity/stdp.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr float PI = 3.14159265358979323846f;

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = 2;
    p.simulation.inhibitory_count = 1;
    p.simulation.steps = 1;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 44001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;
    p.topology.p_ee = p.topology.p_ei = p.topology.p_ie = p.topology.p_ii = 0.0;
    p.topology.minimum_delay = p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology make_graph() {
    CSRTopology g;
    g.excitatory_count = 2;
    g.inhibitory_count = 1;
    g.minimum_delay = g.maximum_delay = 1;
    g.row_offsets = {0, 1, 1, 1};
    g.target_indices = {1};
    g.weights = {30.0f};
    g.delays = {1};
    g.validate();
    return g;
}

struct TrialResult {
    std::string condition;
    std::uint64_t source_spikes = 0;
    std::uint64_t target_spikes = 0;
    float target_vmax = -1000.0f;
    bool finite = true;
};

float learn_lag_from_cuda(const std::filesystem::path& dir,
                          float& confidence,
                          std::uint64_t& source_events,
                          std::uint64_t& target_events) {
    const auto topology = dir / "v44_training.csr";
    save_csr_topology(make_graph(), topology.string());
    StdpParameters stdp;
    stdp.potentiation_rate = 0.002f;
    stdp.depression_rate = 0.001f;
    stdp.maximum_weight = 40.0f;
    CudaSimulationRuntime runtime(make_model(), topology.string(), 0, true, true, stdp);
    std::filesystem::remove(topology);

    AutonomousPhaseParameters pp;
    pp.dt_ms = 0.5f;
    pp.intrinsic_frequency_hz = 8.0f;
    pp.temporal_association_window_steps = 20;
    pp.temporal_lag_learning_rate = 0.12f;
    AutonomousPopulationPhase phase(3, pp);

    std::vector<float> phases(3, 0.0f);
    std::vector<float> lags(1, 0.0f);
    runtime.configure_dynamic_phase_gate(phases, lags, PI);

    for (std::uint32_t cycle = 0; cycle < 24; ++cycle) {
        for (int step = 0; step < 80; ++step) {
            if (step == 0)
                runtime.schedule_stimulus(0, runtime.completed_steps() + 1U, 4, 500.0f);
            runtime.run_steps(1);
            const auto batch = runtime.collect_spike_events();
            if (batch.dropped_events != 0)
                throw std::runtime_error("CUDA training dropped spike events");
            std::vector<std::uint32_t> population_spikes(3, 0);
            for (const auto& event : batch.events) {
                if (event.neuron_id == 0) {
                    ++source_events;
                    population_spikes[0] = 1;
                } else if (event.neuron_id == 1) {
                    ++target_events;
                    population_spikes[1] = 1;
                }
            }
            phase.step(population_spikes, true);
            for (std::uint32_t p = 0; p < 3; ++p)
                phases[p] = phase.phase(p);
            lags[0] = phase.temporal_confidence(0, 1) > 0.2f
                ? phase.learned_lag(0, 1) : 0.0f;
            runtime.update_dynamic_phase_state(phases, lags);
        }
        runtime.clear_stimuli();
    }
    const auto metrics = runtime.collect_metrics();
    runtime.collect_connectivity().validate();
    if (!metrics.finite_state)
        throw std::runtime_error("Non-finite state during CUDA phase training");
    confidence = phase.temporal_confidence(0, 1);
    return phase.learned_lag(0, 1);
}

TrialResult run_frozen_trial(const std::filesystem::path& dir,
                             const std::string& condition,
                             float learned_lag) {
    const auto topology = dir / ("v44_" + condition + ".csr");
    save_csr_topology(make_graph(), topology.string());
    // STDP is deliberately disabled in the frozen causal test. The only
    // difference between conditions is whether/how the phase gate is applied.
    CudaSimulationRuntime runtime(make_model(), topology.string(), 0, true, false);
    std::filesystem::remove(topology);

    if (condition != "ungated") {
        const float source_phase = condition == "learned_aligned"
            ? learned_lag : learned_lag + PI;
        runtime.configure_dynamic_phase_gate(
            {source_phase, 0.0f, 0.0f}, {learned_lag}, 0.35f);
    }

    TrialResult result;
    result.condition = condition;
    runtime.schedule_stimulus(0, runtime.completed_steps() + 1U, 4, 500.0f);
    for (std::uint32_t step = 0; step < 40; ++step) {
        runtime.run_steps(1);
        const auto batch = runtime.collect_spike_events();
        if (batch.dropped_events != 0)
            throw std::runtime_error("CUDA frozen trial dropped spike events");
        for (const auto& event : batch.events) {
            if (event.neuron_id == 0) ++result.source_spikes;
            if (event.neuron_id == 1) ++result.target_spikes;
        }
        const auto voltage = runtime.collect_voltage_samples({1});
        if (voltage.size() != 1)
            throw std::runtime_error("Expected one target voltage sample");
        result.target_vmax = std::max(result.target_vmax, voltage.front().voltage_mv);
        result.finite = result.finite && std::isfinite(voltage.front().voltage_mv);
    }
    result.finite = result.finite && runtime.collect_metrics().finite_state;
    runtime.collect_connectivity().validate();
    return result;
}
}

int main(int argc, char** argv) {
    try {
        const std::filesystem::path dir = argc > 1
            ? argv[1] : "experiments/logs/v44_phase_causal_transfer";
        std::filesystem::create_directories(dir);

        float confidence = 0.0f;
        std::uint64_t training_source = 0, training_target = 0;
        const float lag = learn_lag_from_cuda(
            dir, confidence, training_source, training_target);

        const std::array<std::string, 3> conditions{
            "ungated", "learned_aligned", "learned_misaligned"
        };
        std::array<TrialResult, 3> results;
        for (std::size_t i = 0; i < conditions.size(); ++i)
            results[i] = run_frozen_trial(dir, conditions[i], lag);

        std::ofstream csv(dir / "v44_phase_causal_transfer.csv");
        if (!csv) throw std::runtime_error("Cannot create V44 CSV artifact");
        csv << "condition,source_spikes,target_spikes,target_vmax,finite\n";
        for (const auto& r : results)
            csv << r.condition << ',' << r.source_spikes << ','
                << r.target_spikes << ',' << r.target_vmax << ','
                << (r.finite ? 1 : 0) << '\n';

        const bool training_ok = training_source >= 24 &&
            training_target >= 8 && confidence > 0.2f && std::isfinite(lag);
        const bool matched_source =
            results[0].source_spikes == results[1].source_spikes &&
            results[1].source_spikes == results[2].source_spikes;
        const bool aligned_restores =
            results[1].target_spikes > results[2].target_spikes;
        const bool misalignment_suppresses =
            results[2].target_spikes == 0 && results[1].target_spikes > 0;
        const bool finite = results[0].finite && results[1].finite &&
            results[2].finite;

        std::cout << "V44 LEARNED CUDA PHASE CAUSAL-TRANSFER TEST\n"
                  << "training_source_events=" << training_source
                  << " training_target_events=" << training_target
                  << " learned_lag_rad=" << lag
                  << " temporal_confidence=" << confidence << '\n';
        for (const auto& r : results)
            std::cout << "condition=" << r.condition
                      << " source_spikes=" << r.source_spikes
                      << " target_spikes=" << r.target_spikes
                      << " target_vmax_mv=" << r.target_vmax
                      << " finite=" << (r.finite ? "PASS" : "FAIL") << '\n';
        std::cout << "training_from_cuda_events=" << (training_ok ? "PASS" : "FAIL") << '\n'
                  << "matched_source_spikes=" << (matched_source ? "PASS" : "FAIL") << '\n'
                  << "learned_lag_causally_controls_transmission="
                  << (aligned_restores && misalignment_suppresses ? "PASS" : "FAIL") << '\n'
                  << "finite_state=" << (finite ? "PASS" : "FAIL") << '\n'
                  << "artifacts=" << (dir / "v44_phase_causal_transfer.csv").string() << '\n'
                  << "LIMITATION=phase is learned from CUDA spike events by a host controller and transferred to a frozen CUDA gate; this does not establish GPU-resident phase learning, structural learning, or associative recall.\n";

        return training_ok && matched_source && aligned_restores &&
            misalignment_suppresses && finite ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "V44 phase causal-transfer test failed: " << e.what() << '\n';
        return 1;
    }
}
