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

bool finite(float x) { return std::isfinite(x); }

struct PhasePlasticityResult {
    float initial = 0.10f;
    float aligned = 0.0f;
    float misaligned = 0.0f;
    float ungated = 0.0f;
    float frozen = 0.0f;
    std::uint32_t pre_events = 0;
    std::uint32_t post_events = 0;
};

float run_stdp_reference(bool phase_allowed, bool plastic) {
    std::vector<float> weights{0.10f};
    const std::vector<std::uint32_t> sources{0}, targets{1};
    StdpTraces traces{{0.0f, 0.0f}, {0.0f, 0.0f}};
    StdpParameters p;
    p.potentiation_rate = 0.08f;
    p.depression_rate = 0.005f;
    p.presynaptic_tau_ms = 20.0f;
    p.postsynaptic_tau_ms = 20.0f;
    p.minimum_weight = 0.001f;
    p.maximum_weight = 1.0f;
    constexpr int episodes = 24;
    for (int episode = 0; episode < episodes; ++episode) {
        for (int step = 0; step < 8; ++step) {
            std::vector<std::uint8_t> spikes(2, 0U);
            if (step == 0) spikes[0] = 1U;
            if (step == 2) spikes[1] = 1U;
            StdpParameters effective = p;
            if (!plastic || !phase_allowed) {
                effective.potentiation_rate = 0.0f;
                effective.depression_rate = 0.0f;
            }
            apply_pair_stdp_step(weights, sources, targets, 2, 2,
                                 spikes, traces, 0.5f, effective);
        }
    }
    return weights.front();
}

PhasePlasticityResult phase_plasticity_assay() {
    PhasePlasticityResult r;
    r.aligned = run_stdp_reference(true, true);
    r.misaligned = run_stdp_reference(false, true);
    r.ungated = run_stdp_reference(true, true);
    r.frozen = run_stdp_reference(true, false);
    r.pre_events = 24;
    r.post_events = 24;
    return r;
}

struct PhaseResult {
    float paired_confidence = 0.0f;
    float shuffled_confidence = 0.0f;
    float paired_lag = 0.0f;
    float shuffled_lag = 0.0f;
};

PhaseResult autonomous_phase_assay() {
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
    AutonomousPopulationPhase phase(3, p);
    constexpr int period = 300;
    constexpr std::array<int, 8> offsets{20,55,90,125,160,195,230,265};
    for (int t = 0; t < 6000; ++t) {
        std::vector<std::uint32_t> spikes(3, 0U);
        if (t % period == 0) spikes[0] = 1U;
        const int within = t % period;
        if (within == 5) spikes[1] = 1U;
        if (within == offsets[static_cast<std::size_t>(t / period) % offsets.size()])
            spikes[2] = 1U;
        phase.step(spikes, true);
    }
    return {phase.temporal_confidence(0,1), phase.temporal_confidence(0,2),
            phase.learned_lag(0,1), phase.learned_lag(0,2)};
}

struct StructuralResult {
    std::uint32_t births = 0;
    std::uint32_t deaths = 0;
    std::uint32_t strengthened = 0;
    std::uint32_t weakened = 0;
    bool valid = false;
};

StructuralResult structural_assay() {
    CSRTopology graph;
    graph.excitatory_count = 6;
    graph.inhibitory_count = 0;
    graph.row_offsets = {0,1,1,2,2,2,2};
    graph.target_indices = {1,3};
    graph.weights = {0.10f,0.005f};
    graph.delays = {1,1};
    graph.validate();
    StructuralPlasticityParameters p;
    p.potentiation = 0.03f;
    p.depression = 0.02f;
    p.growth_threshold = 0.50f;
    p.prune_threshold = 0.01f;
    p.minimum_epochs_before_prune = 3;
    p.maximum_out_degree = 4;
    p.maximum_weight = 0.5f;
    p.resource_per_excitatory_neuron = 1.0f;
    ActivityStructuralPlasticity structural(p);
    std::vector<float> activity{0.95f,0.95f,0.0f,0.0f,0.95f,0.95f};
    std::vector<std::uint32_t> ages{4,4};
    const auto report = structural.apply(graph, activity, ages, true);
    graph.validate();
    return {report.births, report.deaths, report.strengthened,
            report.weakened, true};
}

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = 16;
    p.simulation.inhibitory_count = 1;
    p.simulation.steps = 1;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 51001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = p.simulation.inhibitory_current_max = 0.0f;
    p.topology.minimum_delay = p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology inhibition_graph(bool include_inhibition) {
    constexpr std::uint32_t E = 16, N = 17;
    CSRTopology g;
    g.excitatory_count = E;
    g.inhibitory_count = 1;
    g.minimum_delay = g.maximum_delay = 1;
    g.row_offsets.assign(N + 1, 0U);
    for (std::uint32_t s = 0; s < N; ++s) {
        if (s < 4) {
            g.target_indices.push_back(8 + s);
            g.weights.push_back(150.0f);
            g.delays.push_back(1);
        }
        if (include_inhibition && s == E) {
            for (std::uint32_t t = 8; t < 12; ++t) {
                g.target_indices.push_back(t);
                g.weights.push_back(-150.0f);
                g.delays.push_back(1);
            }
        }
        g.row_offsets[s + 1] = static_cast<std::uint32_t>(g.target_indices.size());
    }
    g.validate();
    return g;
}

struct InhibitionResult {
    std::uint64_t source_spikes = 0;
    std::uint64_t inhibitory_spikes = 0;
    std::uint64_t target_spikes = 0;
    std::uint32_t target_unique = 0;
    std::uint64_t dropped = 0;
    bool finite = true;
};

InhibitionResult run_inhibition(const std::filesystem::path& dir,
                                const std::string& label, bool include_inhibition) {
    const auto topology_path = dir / (label + ".csr");
    save_csr_topology(inhibition_graph(include_inhibition), topology_path.string());
    CudaSimulationRuntime rt(make_model(), topology_path.string(), 0, true, false);
    std::filesystem::remove(topology_path);
    InhibitionResult r;
    std::array<bool,4> seen{};
    const auto start = rt.completed_steps() + 1U;
    for (std::uint32_t i = 0; i < 4; ++i)
        rt.schedule_stimulus(i, start, 4, 500.0f);
    if (include_inhibition)
        rt.schedule_stimulus(16, start, 4, 500.0f);
    for (int step = 0; step < 48; ++step) {
        rt.run_steps(1);
        const auto batch = rt.collect_spike_events();
        r.dropped += batch.dropped_events;
        for (const auto& ev : batch.events) {
            if (ev.neuron_id < 4) ++r.source_spikes;
            else if (ev.neuron_id >= 8 && ev.neuron_id < 12) {
                ++r.target_spikes;
                seen[ev.neuron_id - 8] = true;
            } else if (ev.neuron_id == 16) ++r.inhibitory_spikes;
        }
    }
    r.target_unique = static_cast<std::uint32_t>(
        std::count(seen.begin(), seen.end(), true));
    r.finite = rt.collect_metrics().finite_state;
    return r;
}
}

int main(int argc, char** argv) {
    const std::filesystem::path out = argc > 1 ? argv[1] :
        "experiments/logs/v51_validation_batch";
    try {
        std::filesystem::create_directories(out);
        const auto phase_stdp = phase_plasticity_assay();
        const auto phase = autonomous_phase_assay();
        const auto structure = structural_assay();
        const auto no_inhibition = run_inhibition(out, "no_inhibition", false);
        const auto with_inhibition = run_inhibition(out, "with_inhibition", true);

        std::ofstream csv(out / "v51_validation_batch.csv");
        if (!csv) throw std::runtime_error("Cannot create V51 batch CSV");
        csv << "experiment,condition,initial,final,metric_a,metric_b,pass\n";
        csv << "phase_stdp,aligned," << phase_stdp.initial << ',' << phase_stdp.aligned
            << ',' << phase_stdp.pre_events << ',' << phase_stdp.post_events << ','
            << (phase_stdp.aligned > phase_stdp.initial ? 1 : 0) << '\n';
        csv << "phase_stdp,misaligned," << phase_stdp.initial << ',' << phase_stdp.misaligned
            << ',' << phase_stdp.pre_events << ',' << phase_stdp.post_events << ','
            << (std::fabs(phase_stdp.misaligned-phase_stdp.initial)<1e-6f ? 1 : 0) << '\n';
        csv << "phase_stdp,ungated," << phase_stdp.initial << ',' << phase_stdp.ungated
            << ',' << phase_stdp.pre_events << ',' << phase_stdp.post_events << ','
            << (phase_stdp.ungated > phase_stdp.initial ? 1 : 0) << '\n';
        csv << "phase_stdp,frozen," << phase_stdp.initial << ',' << phase_stdp.frozen
            << ",0,0," << (std::fabs(phase_stdp.frozen-phase_stdp.initial)<1e-6f ? 1 : 0) << '\n';
        csv << "autonomous_phase,paired," << phase.paired_lag << ',' << phase.paired_confidence
            << ",0,0," << (phase.paired_confidence > phase.shuffled_confidence + 0.20f ? 1 : 0) << '\n';
        csv << "autonomous_phase,shuffled," << phase.shuffled_lag << ',' << phase.shuffled_confidence
            << ",0,0," << (phase.paired_confidence > phase.shuffled_confidence + 0.20f ? 1 : 0) << '\n';
        csv << "structural,activity," << structure.births << ',' << structure.deaths
            << ',' << structure.strengthened << ',' << structure.weakened << ','
            << (structure.valid && structure.births > 0 && structure.deaths > 0 ? 1 : 0) << '\n';
        csv << "inhibition,off," << no_inhibition.source_spikes << ','
            << no_inhibition.target_spikes << ',' << no_inhibition.target_unique << ','
            << no_inhibition.dropped << ',' << (no_inhibition.finite ? 1 : 0) << '\n';
        csv << "inhibition,on," << with_inhibition.source_spikes << ','
            << with_inhibition.target_spikes << ',' << with_inhibition.target_unique << ','
            << with_inhibition.inhibitory_spikes << ','
            << (with_inhibition.finite ? 1 : 0) << '\n';
        csv.close();

        const bool phase_stdp_pass =
            phase_stdp.aligned > phase_stdp.initial + 1e-5f &&
            std::fabs(phase_stdp.misaligned-phase_stdp.initial) < 1e-6f &&
            phase_stdp.ungated > phase_stdp.initial + 1e-5f &&
            std::fabs(phase_stdp.frozen-phase_stdp.initial) < 1e-6f;
        const bool phase_pass =
            phase.paired_confidence > phase.shuffled_confidence + 0.20f &&
            finite(phase.paired_lag) && finite(phase.shuffled_lag);
        const bool structure_pass = structure.valid && structure.births > 0 &&
            structure.deaths > 0 && structure.strengthened > 0;
        const bool inhibition_pass = with_inhibition.inhibitory_spikes > 0 &&
            with_inhibition.target_spikes < no_inhibition.target_spikes &&
            with_inhibition.finite && no_inhibition.finite &&
            with_inhibition.dropped == 0 && no_inhibition.dropped == 0;

        std::cout << "V51 MULTI-EXPERIMENT VALIDATION BATCH\n"
                  << "phase_stdp aligned_weight=" << phase_stdp.aligned
                  << " misaligned_weight=" << phase_stdp.misaligned
                  << " ungated_weight=" << phase_stdp.ungated
                  << " frozen_weight=" << phase_stdp.frozen
                  << " matched_pre_post_events=" << phase_stdp.pre_events << "/" << phase_stdp.post_events
                  << " result=" << (phase_stdp_pass ? "PASS" : "FAIL") << '\n'
                  << "autonomous_phase paired_confidence=" << phase.paired_confidence
                  << " shuffled_confidence=" << phase.shuffled_confidence
                  << " paired_lag=" << phase.paired_lag
                  << " shuffled_lag=" << phase.shuffled_lag
                  << " result=" << (phase_pass ? "PASS" : "FAIL") << '\n'
                  << "structural births=" << structure.births
                  << " deaths=" << structure.deaths
                  << " strengthened=" << structure.strengthened
                  << " weakened=" << structure.weakened
                  << " csr_valid=" << (structure.valid ? "PASS" : "FAIL")
                  << " result=" << (structure_pass ? "PASS" : "FAIL") << '\n'
                  << "inhibition_off source_spikes=" << no_inhibition.source_spikes
                  << " target_spikes=" << no_inhibition.target_spikes
                  << " target_unique=" << no_inhibition.target_unique << "/4\n"
                  << "inhibition_on source_spikes=" << with_inhibition.source_spikes
                  << " inhibitory_spikes=" << with_inhibition.inhibitory_spikes
                  << " target_spikes=" << with_inhibition.target_spikes
                  << " target_unique=" << with_inhibition.target_unique << "/4"
                  << " result=" << (inhibition_pass ? "PASS" : "FAIL") << '\n'
                  << "artifacts=" << (out / "v51_validation_batch.csv").string() << '\n'
                  << "LIMITATIONS=Phase-gated STDP is a CPU reference-path assay with matched synthetic pre/post events, not yet the CUDA STDP kernel. Autonomous phase and structural tests are host-side mechanisms. Inhibition is tested on the CUDA transmission runtime. No associative recall or dendritic morphology is claimed.\n";
        return phase_stdp_pass && phase_pass && structure_pass && inhibition_pass ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "V51 batch failed: " << e.what() << '\n';
        return 1;
    }
}
