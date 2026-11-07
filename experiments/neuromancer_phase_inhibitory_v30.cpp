#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/csr_topology.h"
#include "topology/topology_io.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;

constexpr std::uint32_t kExcitatory = 2;
constexpr std::uint32_t kInhibitory = 1;
constexpr std::uint32_t kSourceE = 0;
constexpr std::uint32_t kTargetE = 1;
constexpr std::uint32_t kSourceI = 2;
constexpr std::uint32_t kPulseStart = 20;
constexpr std::uint32_t kPulseDuration = 4;
constexpr float kPulseCurrent = 500.0f;
constexpr float kWeightMagnitude = 1.0f;
constexpr float kDtMs = 0.5f;
constexpr float kFrequencyHz = 10.0f;
constexpr float kGateHalfWidthRad = 20.0f * 3.14159265358979323846f / 180.0f;
constexpr float kTau = 6.28318530717958647692f;

struct Observation {
    std::uint64_t source_spike_step = 0;
    float target_voltage = -65.0f;
    float target_delta_mv = 0.0f;
};

ModelParameters make_parameters() {
    ModelParameters p;
    p.simulation.excitatory_count = kExcitatory;
    p.simulation.inhibitory_count = kInhibitory;
    p.simulation.steps = 100;
    p.simulation.dt = kDtMs;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 30001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;

    p.topology.p_ee = 0.0;
    p.topology.p_ei = 0.0;
    p.topology.p_ie = 0.0;
    p.topology.p_ii = 0.0;
    p.topology.master_seed = 30001;
    p.topology.topology_seed = 30001;
    p.topology.delay_seed = 30002;
    p.topology.minimum_delay = 1;
    p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology make_single_edge(bool inhibitory_source, bool include_edge) {
    CSRTopology graph;
    graph.excitatory_count = kExcitatory;
    graph.inhibitory_count = kInhibitory;
    graph.minimum_delay = 1;
    graph.maximum_delay = 1;
    graph.row_offsets = {0, 0, 0, 0};
    const std::uint32_t source = inhibitory_source ? kSourceI : kSourceE;
    if (include_edge) {
        graph.target_indices = {kTargetE};
        graph.weights = {inhibitory_source ? -kWeightMagnitude : kWeightMagnitude};
        graph.delays = {1};
        for (std::uint32_t row = source + 1; row < graph.row_offsets.size(); ++row)
            graph.row_offsets[row] = 1;
    }
    graph.validate();
    return graph;
}

Observation run_once(const std::filesystem::path& output_dir,
                     bool inhibitory_source,
                     bool include_edge,
                     bool phase_gate,
                     float preferred_target_phase,
                     const std::string& label) {
    const auto topology = make_single_edge(inhibitory_source, include_edge);
    const auto path = (output_dir / (label + ".csr")).string();
    save_csr_topology(topology, path);

    auto parameters = make_parameters();
    CudaSimulationRuntime runtime(parameters, path, 0, true, false);

    if (phase_gate) {
        std::vector<float> preferred( kExcitatory + kInhibitory, 0.0f);
        preferred[kTargetE] = preferred_target_phase;
        runtime.configure_phase_gate(preferred, kFrequencyHz,
                                      kGateHalfWidthRad, 0.0f);
    }

    const std::uint32_t source = inhibitory_source ? kSourceI : kSourceE;
    runtime.schedule_stimulus(source, kPulseStart, kPulseDuration,
                              kPulseCurrent);

    Observation result;
    bool found_source_spike = false;
    for (std::uint32_t step = 0; step < 100; ++step) {
        runtime.run_steps(1);
        const auto events = runtime.collect_spike_events();
        for (const auto& event : events.events) {
            if (event.neuron_id == source) {
                result.source_spike_step = event.step;
                found_source_spike = true;
                break;
            }
        }
        if (found_source_spike) break;
    }

    if (!found_source_spike)
        throw std::runtime_error(label + ": stimulated source did not spike");

    // A one-step CSR delay delivers the source event during the next
    // integration step. Sample immediately after that delivery.
    runtime.run_steps(1);
    const auto sample = runtime.collect_voltage_samples({kTargetE});
    if (sample.size() != 1)
        throw std::runtime_error(label + ": target voltage sample missing");

    result.target_voltage = sample.front().voltage_mv;
    result.target_delta_mv = result.target_voltage;
    const auto metrics = runtime.collect_metrics();
    if (!metrics.finite_state)
        throw std::runtime_error(label + ": non-finite neuron state");
    return result;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void test_source_type(const std::filesystem::path& output_dir,
                      bool inhibitory_source) {
    const std::string kind = inhibitory_source ? "inhibitory" : "excitatory";

    const auto control = run_once(output_dir, inhibitory_source, false, false, 0.0f,
                                  kind + "_no_edge_control");
    const auto baseline = run_once(output_dir, inhibitory_source, true, false, 0.0f,
                                   kind + "_ungated");
    const float baseline_delta = baseline.target_voltage - control.target_voltage;
    const float phase_increment =
        kTau * kFrequencyHz * kDtMs / 1000.0f;
    const float aligned_phase = std::fmod(
        static_cast<float>(baseline.source_spike_step) * phase_increment, kTau);

    const auto aligned = run_once(output_dir, inhibitory_source, true, true,
                                  aligned_phase, kind + "_phase_aligned");
    const auto rejected = run_once(output_dir, inhibitory_source, true, true,
                                    std::fmod(aligned_phase + 3.14159265358979323846f,
                                              kTau),
                                    kind + "_phase_rejected");

    const float aligned_delta = aligned.target_voltage - control.target_voltage;
    const float rejected_delta = rejected.target_voltage - control.target_voltage;
    const float expected_sign = inhibitory_source ? -1.0f : 1.0f;
    require(baseline.source_spike_step == aligned.source_spike_step &&
            baseline.source_spike_step == rejected.source_spike_step,
            kind + ": source spike timing changed between matched conditions");
    require(expected_sign * baseline_delta > 0.01f,
            kind + ": ungated synapse did not produce the expected signed response versus no-edge control");
    require(expected_sign * aligned_delta > 0.01f,
            kind + ": phase-aligned synapse did not transmit with the expected sign");
    require(std::abs(rejected_delta) < 1.0e-4f,
            kind + ": anti-phase gate failed to reject the synaptic event");
    require(std::abs(aligned_delta - baseline_delta) < 1.0e-3f,
            kind + ": aligned gate response differs unexpectedly from ungated response");

    std::cout << kind << " pathway"
              << " | source spike step=" << baseline.source_spike_step
              << " | ungated synaptic delta=" << baseline_delta << " mV"
              << " | aligned synaptic delta=" << aligned_delta << " mV"
              << " | rejected synaptic delta=" << rejected_delta << " mV\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error(
                "usage: neuromancer_phase_inhibitory_v30 <output-directory>");

        const std::filesystem::path output_dir = argv[1];
        std::filesystem::create_directories(output_dir);

        std::cout << "NEUROMANCER V30 — CSR PHASE-GATED E/I TRANSMISSION ASSAY\n";
        std::cout << "Backend: existing CUDA Izhikevich runtime and source-oriented CSR\n";
        std::cout << "Oscillator: " << kFrequencyHz << " Hz; gate half-width: 20 degrees\n";
        std::cout << "Each condition has one signed CSR edge with one-step delay.\n";

        test_source_type(output_dir, false);
        test_source_type(output_dir, true);

        std::cout << "PASS: phase gating selectively admits/rejects both E and I events, "
                     "and the transmitted current preserves Dale-consistent sign.\n";
        std::cout << "LIMIT: this is a target-phase gate with an external oscillator; "
                     "it does not test learned per-edge phase offsets, autonomous phase "
                     "dynamics, or nonlinear E/I phase coupling.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V30 phase/inhibition assay failed: " << error.what() << '\n';
        return 1;
    }
}
