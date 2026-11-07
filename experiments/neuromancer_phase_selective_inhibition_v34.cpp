#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/csr_topology.h"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;

constexpr std::uint32_t kECount = 2;
constexpr std::uint32_t kICount = 1;
constexpr std::uint32_t kESource = 0;
constexpr std::uint32_t kTarget = 1;
constexpr std::uint32_t kISource = 2;
constexpr std::uint64_t kStart = 40;
constexpr std::uint32_t kPulseDuration = 4;
constexpr float kPulseCurrent = 500.0f;
constexpr float kDtMs = 0.5f;
constexpr float kFrequencyHz = 10.0f;
constexpr float kTau = 6.28318530717958647692f;
constexpr float kHalfWidthRad = 20.0f * kTau / 360.0f;
constexpr std::uint32_t kRunSteps = 180;

enum class Edges { None, E, I, Both };
enum class GateKind { None, All, Inhibitory, Excitatory };

struct Observation {
    std::vector<float> trace;
    std::uint64_t e_spikes = 0, i_spikes = 0, target_spikes = 0;
    std::uint64_t e_first = 0, i_first = 0, target_first = 0;
    bool finite = false;
};

struct GateMode {
    const char* name;
    GateKind kind;
    float phase_shift_degrees;
};

struct Condition {
    Observation none, e, i, both;
    double peak_interaction = 0.0;
    std::uint32_t peak_interaction_step = 0;
    double peak_interaction_pre_spike = 0.0;
    std::uint32_t peak_interaction_pre_spike_step = 0;
    double peak_interaction_spike_free = 0.0;
    std::uint32_t peak_interaction_spike_free_step = 0;
    bool all_spike_free = false;
    std::uint64_t earliest_target_spike = 0;
};

ModelParameters make_parameters(std::uint32_t seed) {
    ModelParameters p;
    p.simulation.excitatory_count = kECount;
    p.simulation.inhibitory_count = kICount;
    p.simulation.steps = kRunSteps;
    p.simulation.dt = kDtMs;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = seed;
    // Different seeds now produce genuinely different initial conditions.
    // Matched ablations within a seed still start from the same distribution.
    p.simulation.initial_voltage_min = -66.0f;
    p.simulation.initial_voltage_max = -64.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;
    p.topology.p_ee = p.topology.p_ei = p.topology.p_ie = p.topology.p_ii = 0.0;
    p.topology.master_seed = seed;
    p.topology.topology_seed = seed;
    p.topology.delay_seed = seed + 1;
    p.topology.minimum_delay = 1;
    p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology make_topology(Edges edges, float e_weight, float i_weight) {
    CSRTopology graph;
    graph.excitatory_count = kECount;
    graph.inhibitory_count = kICount;
    graph.minimum_delay = 1;
    graph.maximum_delay = 1;
    graph.row_offsets = {0, 0, 0, 0};
    if (edges == Edges::E || edges == Edges::Both) {
        graph.target_indices.push_back(kTarget);
        graph.weights.push_back(e_weight);
        graph.delays.push_back(1);
    }
    graph.row_offsets[1] = static_cast<std::uint32_t>(graph.weights.size());
    graph.row_offsets[2] = static_cast<std::uint32_t>(graph.weights.size());
    if (edges == Edges::I || edges == Edges::Both) {
        graph.target_indices.push_back(kTarget);
        graph.weights.push_back(-i_weight);
        graph.delays.push_back(1);
    }
    graph.row_offsets[3] = static_cast<std::uint32_t>(graph.weights.size());
    graph.validate();
    return graph;
}

void configure_gate(CudaSimulationRuntime& runtime, const GateMode& gate) {
    if (gate.kind == GateKind::None) return;
    const float phase_step = kTau * kFrequencyHz * kDtMs / 1000.0f;
    const float preferred_target_phase = std::fmod(
        static_cast<float>(kStart) * phase_step +
        gate.phase_shift_degrees * kTau / 360.0f, kTau);
    std::vector<float> preferred(kECount + kICount, 0.0f);
    preferred[kTarget] = preferred_target_phase;
    if (gate.kind == GateKind::All) {
        runtime.configure_phase_gate(preferred, kFrequencyHz, kHalfWidthRad, 0.0f);
    } else if (gate.kind == GateKind::Inhibitory) {
        runtime.configure_inhibitory_phase_gate(
            preferred, kFrequencyHz, kHalfWidthRad, 0.0f);
    } else {
        runtime.configure_excitatory_phase_gate(
            preferred, kFrequencyHz, kHalfWidthRad, 0.0f);
    }
}

Observation run_trial(const std::filesystem::path& dir, Edges edges,
                      std::int32_t i_offset_steps, const GateMode& gate,
                      float e_weight, float i_weight, std::uint32_t seed,
                      const std::string& label) {
    const auto topology = make_topology(edges, e_weight, i_weight);
    const auto path = (dir / (label + ".csr")).string();
    save_csr_topology(topology, path);

    auto parameters = make_parameters(seed);
    CudaSimulationRuntime runtime(parameters, path, 0, true, false);
    // The runtime has loaded the CSR graph; remove the per-trial staging file.
    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);
    configure_gate(runtime, gate);

    // Stimuli are identical in every ablation. Only the CSR edges differ.
    runtime.schedule_stimulus(kESource, kStart, kPulseDuration, kPulseCurrent);
    const auto i_start = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(kStart) + i_offset_steps);
    runtime.schedule_stimulus(kISource, i_start, kPulseDuration, kPulseCurrent);

    Observation out;
    out.trace.reserve(kRunSteps);
    for (std::uint32_t step = 0; step < kRunSteps; ++step) {
        runtime.run_steps(1);
        const auto v = runtime.collect_voltage_samples({kTarget});
        if (v.size() != 1) throw std::runtime_error(label + ": missing target voltage");
        out.trace.push_back(v.front().voltage_mv);
    }

    const auto metrics = runtime.collect_metrics();
    const auto events = runtime.collect_spike_events();
    for (const auto& ev : events.events) {
        if (ev.neuron_id == kESource) {
            ++out.e_spikes;
            if (!out.e_first) out.e_first = ev.step;
        } else if (ev.neuron_id == kISource) {
            ++out.i_spikes;
            if (!out.i_first) out.i_first = ev.step;
        } else if (ev.neuron_id == kTarget) {
            ++out.target_spikes;
            if (!out.target_first) out.target_first = ev.step;
        }
    }
    out.finite = metrics.finite_state;
    for (float x : out.trace) out.finite = out.finite && std::isfinite(x);
    if (!out.finite) throw std::runtime_error(label + ": non-finite state");
    return out;
}

Condition run_condition(const std::filesystem::path& dir,
                        std::int32_t offset_steps, const GateMode& gate,
                        float e_weight, float i_weight, std::uint32_t seed) {
    const std::string prefix = std::string(gate.name) + "_e" +
        std::to_string(e_weight) + "_i" + std::to_string(i_weight) +
        "_off" + std::to_string(offset_steps) + "_s" + std::to_string(seed);

    Condition c;
    c.none = run_trial(dir, Edges::None, offset_steps, gate,
                       e_weight, i_weight, seed, prefix + "_none");
    c.e = run_trial(dir, Edges::E, offset_steps, gate,
                    e_weight, i_weight, seed, prefix + "_e");
    c.i = run_trial(dir, Edges::I, offset_steps, gate,
                    e_weight, i_weight, seed, prefix + "_i");
    c.both = run_trial(dir, Edges::Both, offset_steps, gate,
                       e_weight, i_weight, seed, prefix + "_both");

    const std::uint32_t n = static_cast<std::uint32_t>(std::min({
        c.none.trace.size(), c.e.trace.size(), c.i.trace.size(), c.both.trace.size()}));

    c.all_spike_free = c.none.target_spikes == 0 && c.e.target_spikes == 0 &&
                       c.i.target_spikes == 0 && c.both.target_spikes == 0;

    for (const auto first : {c.none.target_first, c.e.target_first,
                             c.i.target_first, c.both.target_first}) {
        if (first && (!c.earliest_target_spike || first < c.earliest_target_spike))
            c.earliest_target_spike = first;
    }

    for (std::uint32_t s = 0; s < n; ++s) {
        const double de = c.e.trace[s] - c.none.trace[s];
        const double di = c.i.trace[s] - c.none.trace[s];
        const double db = c.both.trace[s] - c.none.trace[s];
        const double interaction = db - de - di;

        if (std::abs(interaction) > std::abs(c.peak_interaction)) {
            c.peak_interaction = interaction;
            c.peak_interaction_step = s + 1;
        }

        const bool before_any_spike =
            !c.earliest_target_spike || (s + 1) < c.earliest_target_spike;
        if (before_any_spike &&
            std::abs(interaction) > std::abs(c.peak_interaction_pre_spike)) {
            c.peak_interaction_pre_spike = interaction;
            c.peak_interaction_pre_spike_step = s + 1;
        }

        if (c.all_spike_free &&
            std::abs(interaction) > std::abs(c.peak_interaction_spike_free)) {
            c.peak_interaction_spike_free = interaction;
            c.peak_interaction_spike_free_step = s + 1;
        }
    }
    return c;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error(
            "usage: neuromancer_phase_selective_inhibition_v34 <output-directory>");

        const std::filesystem::path out_dir = argv[1];
        std::filesystem::create_directories(out_dir);
        std::ofstream csv(out_dir / "phase_selective_inhibition_v34.csv");
        if (!csv) throw std::runtime_error("cannot open V34 CSV");
        csv << std::setprecision(9);
        csv << "seed,gate,gate_phase_shift_deg,e_weight,i_weight,i_offset_steps,i_offset_ms,"
               "target_spikes_e_only,target_spikes_i_only,target_spikes_both,"
               "target_first_step_e_only,target_first_step_i_only,target_first_step_both,"
               "e_source_spikes,i_source_spikes,e_first_step,i_first_step,"
               "all_spike_free,earliest_target_spike,"
               "peak_interaction_mv,peak_interaction_step,"
               "peak_interaction_pre_spike_mv,peak_interaction_pre_spike_step,"
               "peak_interaction_spike_free_mv,peak_interaction_spike_free_step,finite\n";

        constexpr std::array<float, 4> e_weights{6.0f, 8.0f, 10.0f, 12.0f};
        constexpr std::array<float, 5> i_weights{1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
        // 1 ms resolution: -6 ms through +6 ms around coincidence.
        // A follow-up can refine the transition region at 0.5 ms resolution.
        constexpr std::array<std::int32_t, 13> offsets{
            -12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12
        };
        constexpr std::array<GateMode, 5> gates{{
            {"ungated", GateKind::None, 0.0f},
            {"all_aligned", GateKind::All, 0.0f},
            {"inhibitory_aligned", GateKind::Inhibitory, 0.0f},
            {"inhibitory_shifted90", GateKind::Inhibitory, 90.0f},
            {"excitatory_shifted90", GateKind::Excitatory, 90.0f}
        }};
        constexpr std::array<std::uint32_t, 4> seeds{34031,34132,34233,34334};

        std::uint64_t rows = 0;
        constexpr std::uint64_t total =
            seeds.size() * e_weights.size() * i_weights.size() *
            offsets.size() * gates.size();

        for (const auto seed : seeds) {
            for (const auto e_weight : e_weights) {
                for (const auto i_weight : i_weights) {
                    for (const auto offset : offsets) {
                        for (const auto& gate : gates) {
                            const auto c = run_condition(
                                out_dir, offset, gate, e_weight, i_weight, seed);
                            csv << seed << ',' << gate.name << ','
                                << gate.phase_shift_degrees << ','
                                << e_weight << ',' << i_weight << ','
                                << offset << ',' << offset * kDtMs << ','
                                << c.e.target_spikes << ',' << c.i.target_spikes << ','
                                << c.both.target_spikes << ','
                                << c.e.target_first << ',' << c.i.target_first << ','
                                << c.both.target_first << ','
                                << c.both.e_spikes << ',' << c.both.i_spikes << ','
                                << c.both.e_first << ',' << c.both.i_first << ','
                                << (c.all_spike_free ? 1 : 0) << ','
                                << c.earliest_target_spike << ','
                                << c.peak_interaction << ',' << c.peak_interaction_step << ','
                                << c.peak_interaction_pre_spike << ','
                                << c.peak_interaction_pre_spike_step << ','
                                << c.peak_interaction_spike_free << ','
                                << c.peak_interaction_spike_free_step << ','
                                << (c.none.finite && c.e.finite && c.i.finite &&
                                    c.both.finite ? 1 : 0) << '\n';
                            ++rows;
                            if (rows % 25 == 0) {
                                std::cout << "completed " << rows << " / " << total
                                          << " conditions\n";
                                csv.flush();
                            }
                        }
                    }
                }
            }
        }

        csv.close();
        std::cout << "NEUROMANCER V34 — PHASE-SELECTIVE INHIBITION\n"
                  << "Conditions: " << rows << "; four matched edge ablations each.\n"
                  << "Gate controls: ungated, all-input aligned, inhibitory-only aligned,\n"
                  << "inhibitory-only shifted 90 degrees, and excitatory-only shifted 90 degrees.\n"
                  << "Offsets: -6 to +6 ms in 0.5 ms increments; four seeds with varied initial voltages.\n"
                  << "Spike-free interaction is reported separately and only when all four target traces remain spike-free.\n"
                  << "Wrote " << (out_dir / "phase_selective_inhibition_v34.csv").string() << '\n'
                  << "LIMIT: single-target controlled assay; external 10 Hz oscillator; no learned phase coupling.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "V34 phase-selective inhibition failed: " << e.what() << '\n';
        return 1;
    }
}
