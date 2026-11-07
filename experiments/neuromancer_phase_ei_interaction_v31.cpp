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

const char* edge_name(Edges edges) {
    switch (edges) {
        case Edges::None: return "none";
        case Edges::Excitatory: return "e_only";
        case Edges::Inhibitory: return "i_only";
        case Edges::Both: return "e_plus_i";
    }
    return "invalid";
}

struct Observation {
    float target_voltage = 0.0f;
    std::uint64_t e_spikes = 0;
    std::uint64_t i_spikes = 0;
    std::uint64_t target_spikes = 0;
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

    runtime.run_steps(kRunSteps);
    const auto voltage = runtime.collect_voltage_samples({kTarget});
    if (voltage.size() != 1)
        throw std::runtime_error(label + ": missing target voltage sample");

    const auto metrics = runtime.collect_metrics();
    Observation result;
    result.target_voltage = voltage.front().voltage_mv;
    result.e_spikes = metrics.excitatory_spikes;
    result.i_spikes = metrics.inhibitory_spikes;
    result.target_spikes = metrics.excitatory_spikes + metrics.inhibitory_spikes;
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
        std::ofstream csv(csv_path);
        if (!csv)
            throw std::runtime_error("could not open V31 CSV output");

        csv << "gate,inh_offset_steps,inh_offset_ms,none_mv,e_only_delta_mv,"
               "i_only_delta_mv,both_delta_mv,interaction_mv,e_spikes,i_spikes,"
               "target_spikes,finite\n";
        csv << std::setprecision(9);

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
                    << c.both.e_spikes << ',' << c.both.i_spikes << ','
                    << c.both.target_spikes << ','
                    << (c.none.finite && c.e.finite && c.i.finite &&
                        c.both.finite ? 1 : 0) << '\n';

                std::cout << (gated ? "gated" : "ungated")
                          << " | I offset=" << offset << " steps (" << offset_ms
                          << " ms)"
                          << " | dE=" << c.delta_e << " mV"
                          << " | dI=" << c.delta_i << " mV"
                          << " | d(E+I)=" << c.delta_both << " mV"
                          << " | interaction=" << c.interaction << " mV"
                          << " | E spikes=" << c.both.e_spikes
                          << " | I spikes=" << c.both.i_spikes << '\n';
            }
        }
        csv.close();

        std::cout << "Wrote " << csv_path.string() << '\n';
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
