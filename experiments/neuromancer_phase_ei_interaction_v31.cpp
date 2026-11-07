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
constexpr std::uint32_t kExcitatorySource = 0;
constexpr std::uint32_t kTarget = 1;
constexpr std::uint32_t kInhibitorySource = 2;
constexpr std::uint64_t kEStart = 20;
constexpr std::uint32_t kPulseDuration = 4;
constexpr float kPulseCurrent = 500.0f;
constexpr float kWeight = 1.0f;
constexpr float kDtMs = 0.5f;
constexpr float kFrequencyHz = 10.0f;
constexpr float kHalfWidthRad = 20.0f * 3.14159265358979323846f / 180.0f;
constexpr float kTau = 6.28318530717958647692f;
constexpr std::uint32_t kRunSteps = 140;

enum class Edges { None, Excitatory, Inhibitory, Both };

struct Observation {
    float target_voltage = 0.0f;
    std::uint64_t e_spikes = 0;
    std::uint64_t i_spikes = 0;
    std::uint64_t target_spikes = 0;
    std::uint64_t e_source_step = 0;
    std::uint64_t i_source_step = 0;
    std::vector<float> target_trace;
    bool finite = false;
};

ModelParameters make_parameters() {
    ModelParameters p;
    p.simulation.excitatory_count = kECount;
    p.simulation.inhibitory_count = kICount;
    p.simulation.steps = kRunSteps;
    p.simulation.dt = kDtMs;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 31031;
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
    p.topology.master_seed = 31031;
    p.topology.topology_seed = 31031;
    p.topology.delay_seed = 31032;
    p.topology.minimum_delay = 1;
    p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology make_topology(Edges edges) {
    CSRTopology graph;
    graph.excitatory_count = kECount;
    graph.inhibitory_count = kICount;
    graph.minimum_delay = 1;
    graph.maximum_delay = 1;
    graph.row_offsets = {0, 0, 0, 0};

    if (edges == Edges::Excitatory || edges == Edges::Both) {
        graph.target_indices.push_back(kTarget);
        graph.weights.push_back(kWeight);
        graph.delays.push_back(1);
    }
    graph.row_offsets[1] = static_cast<std::uint32_t>(graph.weights.size());
    graph.row_offsets[2] = static_cast<std::uint32_t>(graph.weights.size());

    if (edges == Edges::Inhibitory || edges == Edges::Both) {
        graph.target_indices.push_back(kTarget);
        graph.weights.push_back(-kWeight);
        graph.delays.push_back(1);
    }
    graph.row_offsets[3] = static_cast<std::uint32_t>(graph.weights.size());
    graph.validate();
    return graph;
}

Observation run_trial(const std::filesystem::path& directory,
                      Edges edges,
                      std::uint32_t inhibitory_offset,
                      bool phase_gate,
                      float preferred_target_phase,
                      const std::string& label) {
    const auto topology = make_topology(edges);
    const auto path = (directory / (label + ".csr")).string();
    save_csr_topology(topology, path);

    auto parameters = make_parameters();
    CudaSimulationRuntime runtime(parameters, path, 0, true, false);

    if (phase_gate) {
        std::vector<float> preferred(kECount + kICount, 0.0f);
        preferred[kTarget] = preferred_target_phase;
        runtime.configure_phase_gate(preferred, kFrequencyHz,
                                      kHalfWidthRad, 0.0f);
    }

    if (edges == Edges::Excitatory || edges == Edges::Both) {
        runtime.schedule_stimulus(kExcitatorySource, kEStart,
                                  kPulseDuration, kPulseCurrent);
    }
    if (edges == Edges::Inhibitory || edges == Edges::Both) {
        runtime.schedule_stimulus(kInhibitorySource,
                                  kEStart + inhibitory_offset,
                                  kPulseDuration, kPulseCurrent);
    }

    Observation result;
    result.target_trace.reserve(kRunSteps);
    for (std::uint32_t step = 0; step < kRunSteps; ++step) {
        runtime.run_steps(1);
        const auto voltage = runtime.collect_voltage_samples({kTarget});
        if (voltage.size() != 1)
            throw std::runtime_error(label + ": missing target voltage sample");
        result.target_trace.push_back(voltage.front().voltage_mv);
    }
    const auto metrics = runtime.collect_metrics();
    result.target_voltage = result.target_trace.back();
    const auto events = runtime.collect_spike_events();
    for (const auto& event : events.events) {
        if (event.neuron_id == kExcitatorySource) {
            ++result.e_spikes;
            if (result.e_spikes == 1) result.e_source_step = event.step;
        } else if (event.neuron_id == kInhibitorySource) {
            ++result.i_spikes;
            if (result.i_spikes == 1) result.i_source_step = event.step;
        } else if (event.neuron_id == kTarget) {
            ++result.target_spikes;
        }
    }
    result.finite = metrics.finite_state && std::isfinite(result.target_voltage);
    if (!result.finite)
        throw std::runtime_error(label + ": non-finite simulation state");
    return result;
}

struct Condition {
    Observation none;
    Observation e;
    Observation i;
    Observation both;
    double delta_e = 0.0;
    double delta_i = 0.0;
    double delta_both = 0.0;
    double interaction = 0.0;
    double peak_e = 0.0, peak_i = 0.0, peak_both = 0.0, peak_interaction = 0.0;
    std::uint32_t peak_e_step = 0, peak_i_step = 0, peak_both_step = 0, peak_interaction_step = 0;
};

Condition run_condition(const std::filesystem::path& directory,
                        std::uint32_t offset,
                        bool gated,
                        float preferred_phase) {
    const std::string prefix =
        std::string(gated ? "gated_" : "ungated_") +
        "offset_" + std::to_string(offset);
    Condition c;
    c.none = run_trial(directory, Edges::None, offset, gated, preferred_phase,
                       prefix + "_none");
    c.e = run_trial(directory, Edges::Excitatory, offset, gated, preferred_phase,
                    prefix + "_e");
    c.i = run_trial(directory, Edges::Inhibitory, offset, gated, preferred_phase,
                    prefix + "_i");
    c.both = run_trial(directory, Edges::Both, offset, gated, preferred_phase,
                       prefix + "_both");

    c.delta_e = c.e.target_voltage - c.none.target_voltage;
    c.delta_i = c.i.target_voltage - c.none.target_voltage;
    c.delta_both = c.both.target_voltage - c.none.target_voltage;
    c.interaction = c.delta_both - c.delta_e - c.delta_i;
    for (std::uint32_t step = 0; step < kRunSteps; ++step) {
        const double de = c.e.target_trace[step] - c.none.target_trace[step];
        const double di = c.i.target_trace[step] - c.none.target_trace[step];
        const double db = c.both.target_trace[step] - c.none.target_trace[step];
        const double inter = db - de - di;
        if (std::abs(de) > std::abs(c.peak_e)) { c.peak_e = de; c.peak_e_step = step + 1; }
        if (std::abs(di) > std::abs(c.peak_i)) { c.peak_i = di; c.peak_i_step = step + 1; }
        if (std::abs(db) > std::abs(c.peak_both)) { c.peak_both = db; c.peak_both_step = step + 1; }
        if (std::abs(inter) > std::abs(c.peak_interaction)) { c.peak_interaction = inter; c.peak_interaction_step = step + 1; }
    }
    return c;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error(
                "usage: neuromancer_phase_ei_interaction_v31 <output-directory>");

        const std::filesystem::path output_dir = argv[1];
        std::filesystem::create_directories(output_dir);
        const auto csv_path = output_dir / "phase_ei_interaction_v31.csv";
        const auto trace_path = output_dir / "phase_ei_trace_v31.csv";
        std::ofstream csv(csv_path);
        std::ofstream trace_csv(trace_path);
        if (!csv || !trace_csv)
            throw std::runtime_error("could not open V31 CSV output");

        csv << "gate,inh_offset_steps,inh_offset_ms,final_none_mv,final_e_delta_mv,"
               "final_i_delta_mv,final_both_delta_mv,final_interaction_mv,"
               "peak_e_delta_mv,peak_e_step,peak_i_delta_mv,peak_i_step,"
               "peak_both_delta_mv,peak_both_step,peak_interaction_mv,"
               "peak_interaction_step,e_spikes,i_spikes,target_spikes,"
               "first_e_source_step,first_i_source_step,finite\n";
        trace_csv << "gate,inh_offset_steps,step,time_ms,none_mv,e_delta_mv,"
                     "i_delta_mv,both_delta_mv,interaction_mv\n";
        csv << std::setprecision(9);
        trace_csv << std::setprecision(9);

        // With dt=0.5 ms and a 10 Hz oscillator, 5 steps correspond to 9 degrees.
        // The gate half-width is 20 degrees, so these offsets span admitted and
        // rejected timing relationships while preserving the E event as reference.
        constexpr std::array<std::uint32_t, 6> offsets{0, 5, 10, 15, 20, 40};
        const float phase_increment =
            kTau * kFrequencyHz * kDtMs / 1000.0f;
        const float preferred_phase =
            std::fmod(static_cast<float>(kEStart) * phase_increment, kTau);

        std::cout << "NEUROMANCER V31 — PHASE-RELATIVE E/I INTERACTION\n";
        std::cout << "CUDA CSR runtime; E->target weight +" << kWeight
                  << "; I->target weight -" << kWeight << "\n";
        std::cout << "Oscillator " << kFrequencyHz << " Hz; gate half-width 20 deg; "
                  << "dt " << kDtMs << " ms\n";
        std::cout << "Interaction = (E+I response) - (E-only response) - "
                     "(I-only response), all relative to matched no-edge control.\n";

        for (const bool gated : {false, true}) {
            for (const auto offset : offsets) {
                const auto c = run_condition(output_dir, offset, gated,
                                              preferred_phase);
                const double offset_ms = offset * kDtMs;
                csv << (gated ? "gated" : "ungated") << ','
                    << offset << ',' << offset_ms << ','
                    << c.none.target_voltage << ','
                    << c.delta_e << ',' << c.delta_i << ','
                    << c.delta_both << ',' << c.interaction << ','
                    << c.peak_e << ',' << c.peak_e_step << ','
                    << c.peak_i << ',' << c.peak_i_step << ','
                    << c.peak_both << ',' << c.peak_both_step << ','
                    << c.peak_interaction << ',' << c.peak_interaction_step << ','
                    << c.both.e_spikes << ',' << c.both.i_spikes << ','
                    << c.both.target_spikes << ','
                    << c.both.e_source_step << ',' << c.both.i_source_step << ','
                    << (c.none.finite && c.e.finite && c.i.finite &&
                        c.both.finite ? 1 : 0) << '\n';

                for (std::uint32_t step = 0; step < kRunSteps; ++step) {
                    const double de = c.e.target_trace[step] - c.none.target_trace[step];
                    const double di = c.i.target_trace[step] - c.none.target_trace[step];
                    const double db = c.both.target_trace[step] - c.none.target_trace[step];
                    trace_csv << (gated ? "gated" : "ungated") << ','
                              << offset << ',' << (step + 1) << ','
                              << (step + 1) * kDtMs << ','
                              << c.none.target_trace[step] << ','
                              << de << ',' << di << ',' << db << ','
                              << (db - de - di) << '\n';
                }

                std::cout << (gated ? "gated" : "ungated")
                          << " | I offset=" << offset << " steps (" << offset_ms
                          << " ms)"
                          << " | dE=" << c.delta_e << " mV"
                          << " | dI=" << c.delta_i << " mV"
                          << " | final interaction=" << c.interaction << " mV"
                          << " | peak E=" << c.peak_e << " mV @step " << c.peak_e_step
                          << " | peak I=" << c.peak_i << " mV @step " << c.peak_i_step
                          << " | peak interaction=" << c.peak_interaction
                          << " mV @step " << c.peak_interaction_step
                          << " | E source spikes=" << c.both.e_spikes
                          << " first@ " << c.both.e_source_step
                          << " | I source spikes=" << c.both.i_spikes
                          << " first@ " << c.both.i_source_step
                          << " | target spikes=" << c.both.target_spikes << '\n';
            }
        }
        csv.close();
        trace_csv.close();

        std::cout << "Wrote " << csv_path.string() << '\n';
        std::cout << "Wrote " << trace_path.string() << '\n';
        std::cout << "INTERPRETATION LIMIT: a nonzero interaction is evidence of "
                     "non-additive target response in this assay, not by itself "
                     "autonomous oscillation, learned phase offsets, or biological "
                     "inhibitory regulation. Check source-spike counts and compare "
                     "gated with ungated rows before interpreting the timing curve.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V31 phase-relative E/I interaction failed: "
                  << error.what() << '\n';
        return 1;
    }
}
