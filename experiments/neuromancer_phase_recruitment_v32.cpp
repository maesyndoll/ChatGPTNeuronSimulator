#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/csr_topology.h"
#include "topology/topology_io.h"

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
constexpr std::uint64_t kStart = 20;
constexpr std::uint32_t kPulseDuration = 4;
constexpr float kPulseCurrent = 500.0f;
constexpr float kDtMs = 0.5f;
constexpr float kFrequencyHz = 10.0f;
constexpr float kTau = 6.28318530717958647692f;
constexpr float kHalfWidthRad = 20.0f * kTau / 360.0f;
constexpr std::uint32_t kRunSteps = 160;

enum class Edges { None, E, I, Both };

struct Observation {
    std::vector<float> trace;
    std::uint64_t e_spikes = 0, i_spikes = 0, target_spikes = 0;
    std::uint64_t e_first = 0, i_first = 0, target_first = 0;
    bool finite = false;
};

struct Condition {
    Observation none, e, i, both;
    double peak_e = 0, peak_i = 0, peak_both = 0, peak_interaction = 0;
    std::uint32_t peak_e_step = 0, peak_i_step = 0;
    std::uint32_t peak_both_step = 0, peak_interaction_step = 0;
    double final_e = 0, final_i = 0, final_both = 0, final_interaction = 0;
};

struct GateMode {
    const char* name;
    bool enabled;
    float phase_shift_degrees;
};

ModelParameters make_parameters(std::uint32_t seed) {
    ModelParameters p;
    p.simulation.excitatory_count = kECount;
    p.simulation.inhibitory_count = kICount;
    p.simulation.steps = kRunSteps;
    p.simulation.dt = kDtMs;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = seed;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
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

Observation run_trial(const std::filesystem::path& dir, Edges edges,
                      std::uint32_t offset, bool gated, float preferred_phase,
                      float e_weight, float i_weight, std::uint32_t seed,
                      const std::string& label) {
    const auto topology = make_topology(edges, e_weight, i_weight);
    const auto path = (dir / (label + ".csr")).string();
    save_csr_topology(topology, path);
    auto parameters = make_parameters(seed);
    CudaSimulationRuntime runtime(parameters, path, 0, true, false);
    if (gated) {
        std::vector<float> preferred(kECount + kICount, 0.0f);
        preferred[kTarget] = preferred_phase;
        runtime.configure_phase_gate(preferred, kFrequencyHz, kHalfWidthRad, 0.0f);
    }
    if (edges == Edges::E || edges == Edges::Both)
        runtime.schedule_stimulus(kESource, kStart, kPulseDuration, kPulseCurrent);
    if (edges == Edges::I || edges == Edges::Both)
        runtime.schedule_stimulus(kISource, kStart + offset, kPulseDuration, kPulseCurrent);

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

Condition run_condition(const std::filesystem::path& dir, std::uint32_t offset,
                        const GateMode& gate, float e_weight, float i_weight,
                        std::uint32_t seed) {
    const std::string prefix = std::string(gate.name) + "_e" +
        std::to_string(e_weight) + "_i" + std::to_string(i_weight) +
        "_off" + std::to_string(offset) + "_s" + std::to_string(seed);
    const float phase_step = kTau * kFrequencyHz * kDtMs / 1000.0f;
    const float preferred = std::fmod(static_cast<float>(kStart) * phase_step +
        gate.phase_shift_degrees * kTau / 360.0f, kTau);
    Condition c;
    c.none = run_trial(dir, Edges::None, offset, gate.enabled, preferred,
                       e_weight, i_weight, seed, prefix + "_none");
    c.e = run_trial(dir, Edges::E, offset, gate.enabled, preferred,
                    e_weight, i_weight, seed, prefix + "_e");
    c.i = run_trial(dir, Edges::I, offset, gate.enabled, preferred,
                    e_weight, i_weight, seed, prefix + "_i");
    c.both = run_trial(dir, Edges::Both, offset, gate.enabled, preferred,
                       e_weight, i_weight, seed, prefix + "_both");
    const std::uint32_t n = std::min({static_cast<std::uint32_t>(c.none.trace.size()),
        static_cast<std::uint32_t>(c.e.trace.size()),
        static_cast<std::uint32_t>(c.i.trace.size()),
        static_cast<std::uint32_t>(c.both.trace.size())});
    c.final_e = c.e.trace[n-1] - c.none.trace[n-1];
    c.final_i = c.i.trace[n-1] - c.none.trace[n-1];
    c.final_both = c.both.trace[n-1] - c.none.trace[n-1];
    c.final_interaction = c.final_both - c.final_e - c.final_i;
    for (std::uint32_t s = 0; s < n; ++s) {
        const double de = c.e.trace[s] - c.none.trace[s];
        const double di = c.i.trace[s] - c.none.trace[s];
        const double db = c.both.trace[s] - c.none.trace[s];
        const double inter = db - de - di;
        if (std::abs(de) > std::abs(c.peak_e)) { c.peak_e = de; c.peak_e_step = s+1; }
        if (std::abs(di) > std::abs(c.peak_i)) { c.peak_i = di; c.peak_i_step = s+1; }
        if (std::abs(db) > std::abs(c.peak_both)) { c.peak_both = db; c.peak_both_step = s+1; }
        if (std::abs(inter) > std::abs(c.peak_interaction)) {
            c.peak_interaction = inter; c.peak_interaction_step = s+1;
        }
    }
    return c;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error(
            "usage: neuromancer_phase_recruitment_v32 <output-directory>");
        const std::filesystem::path out_dir = argv[1];
        std::filesystem::create_directories(out_dir);
        std::ofstream csv(out_dir / "phase_recruitment_v32.csv");
        if (!csv) throw std::runtime_error("cannot open V32 CSV");
        csv << std::setprecision(9);
        csv << "seed,gate,gate_phase_shift_deg,e_weight,i_weight,i_offset_steps,i_offset_ms,"
               "peak_e_delta_mv,peak_e_step,peak_i_delta_mv,peak_i_step,peak_both_delta_mv,"
               "peak_both_step,peak_interaction_mv,peak_interaction_step,final_interaction_mv,"
               "e_source_spikes,i_source_spikes,target_spikes_e_only,target_spikes_i_only,"
               "target_spikes_both,target_first_step_both,e_first_step,i_first_step,finite\n";

        constexpr std::array<float,5> e_weights{8.0f,16.0f,32.0f,64.0f,96.0f};
        constexpr std::array<float,3> i_weights{0.5f,1.0f,2.0f};
        constexpr std::array<std::uint32_t,6> offsets{0,4,8,12,16,20};
        constexpr std::array<GateMode,3> gates{{
            {"ungated", false, 0.0f},
            {"aligned", true, 0.0f},
            {"shifted90", true, 90.0f}
        }};
        constexpr std::array<std::uint32_t,2> seeds{32031,32132};

        std::uint64_t rows = 0;
        for (const auto seed : seeds) {
            for (const auto e_weight : e_weights) {
                for (const auto i_weight : i_weights) {
                    for (const auto offset : offsets) {
                        for (const auto& gate : gates) {
                            const auto c = run_condition(out_dir, offset, gate,
                                e_weight, i_weight, seed);
                            csv << seed << ',' << gate.name << ','
                                << gate.phase_shift_degrees << ','
                                << e_weight << ',' << i_weight << ','
                                << offset << ',' << offset*kDtMs << ','
                                << c.peak_e << ',' << c.peak_e_step << ','
                                << c.peak_i << ',' << c.peak_i_step << ','
                                << c.peak_both << ',' << c.peak_both_step << ','
                                << c.peak_interaction << ',' << c.peak_interaction_step << ','
                                << c.final_interaction << ','
                                << c.both.e_spikes << ',' << c.both.i_spikes << ','
                                << c.e.target_spikes << ',' << c.i.target_spikes << ','
                                << c.both.target_spikes << ',' << c.both.target_first << ','
                                << c.both.e_first << ',' << c.both.i_first << ','
                                << (c.none.finite && c.e.finite && c.i.finite && c.both.finite ? 1 : 0)
                                << '\n';
                            ++rows;
                            if (rows % 30 == 0) {
                                std::cout << "completed " << rows << " / 540 conditions\n";
                                csv.flush();
                            }
                        }
                    }
                }
            }
        }
        csv.close();
        std::cout << "NEUROMANCER V32 — PHASE-GATED RECRUITMENT AND PRELIMINARY CONTROL BATTERY\n"
                  << "Conditions: " << rows << " (two seeds); 4 matched edge conditions each.\n"
                  << "E weight sweep tests recruitment; I weight/timing sweeps test suppression;\n"
                  << "aligned vs 90-degree-shifted gate tests phase specificity; ungated is control.\n"
                  << "Wrote " << (out_dir / "phase_recruitment_v32.csv").string() << '\n'
                  << "LIMIT: finite-size, two-seed, single-target assay; no autonomous oscillator or learned phase coupling.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "V32 phase recruitment failed: " << e.what() << '\n';
        return 1;
    }
}
