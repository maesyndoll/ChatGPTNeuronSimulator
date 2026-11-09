#include "learning/phase_structural_plasticity.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace spiking;
constexpr float kTau = 6.2831853071795864769f;

float wrap(float x) {
    x = std::fmod(x, kTau);
    return x < 0.0f ? x + kTau : x;
}

void train_phase_controller(AutonomousPopulationPhase& phase) {
    constexpr int period = 300;
    for (int t = 0; t < 6000; ++t) {
        std::vector<std::uint32_t> spikes(3, 0U);
        const int within = t % period;
        if (within == 0) spikes[0] = 1U;
        if (within == 5) spikes[1] = 1U;
        const int c_offsets[] = {20,55,90,125,160,195,230,265};
        if (within == c_offsets[(t / period) % 8]) spikes[2] = 1U;
        phase.step(spikes, true);
    }
}

CSRTopology make_structural_probe() {
    CSRTopology graph;
    graph.excitatory_count = 3;
    graph.inhibitory_count = 1;
    graph.minimum_delay = graph.maximum_delay = 1;
    graph.row_offsets = {0,1,1,1,1};
    graph.target_indices = {2};
    graph.weights = {0.0005f};
    graph.delays = {1};
    graph.validate();
    return graph;
}

struct PhaseResult {
    float paired_lag = 0.0f;
    float control_lag = 0.0f;
    float paired_confidence = 0.0f;
    float control_confidence = 0.0f;
    float paired_coupling = 0.0f;
    float phase_drift = 0.0f;
    float frozen_lag_drift = 0.0f;
    bool finite = false;
};

PhaseResult test_phase() {
    AutonomousPhaseParameters p;
    p.intrinsic_frequency_hz = 8.0f;
    p.temporal_association_window_steps = 39;
    p.temporal_lag_learning_rate = 0.10f;
    p.minimum_temporal_confidence = 0.20f;
    AutonomousPopulationPhase phase(3, p);
    train_phase_controller(phase);

    PhaseResult r;
    r.paired_lag = phase.learned_lag(0,1);
    r.control_lag = phase.learned_lag(0,2);
    r.paired_confidence = phase.temporal_confidence(0,1);
    r.control_confidence = phase.temporal_confidence(0,2);
    r.paired_coupling = phase.coupling(0,1);
    const float before_a = phase.phase(0);
    const float before_b = phase.phase(1);
    const float before_lag = phase.learned_lag(0,1);

    // Frozen, stimulus-free autonomous evolution. The phase state continues
    // to advance, while learned lag/coupling parameters are held fixed.
    for (int t = 0; t < 1997; ++t)
        phase.step(std::vector<std::uint32_t>(3,0U), false);

    const float after_a = phase.phase(0);
    const float after_b = phase.phase(1);
    r.phase_drift = std::abs(wrap(after_a - before_a));
    r.frozen_lag_drift = std::abs(wrap(phase.learned_lag(0,1) - before_lag));
    const float paired_error = std::abs(wrap(after_b - after_a - r.paired_lag));
    const float control_error = std::abs(wrap(phase.phase(2) - after_a - r.control_lag));
    r.finite = std::isfinite(r.paired_lag) && std::isfinite(r.control_lag) &&
        std::isfinite(r.paired_confidence) && std::isfinite(r.paired_coupling) &&
        std::isfinite(paired_error) && std::isfinite(control_error);
    std::cout << "phase_paired_lag_rad=" << r.paired_lag
              << " phase_control_lag_rad=" << r.control_lag
              << " paired_confidence=" << r.paired_confidence
              << " control_confidence=" << r.control_confidence
              << " learned_coupling=" << r.paired_coupling
              << " stimulus_free_phase_advance_rad=" << r.phase_drift
              << " frozen_lag_drift_rad=" << r.frozen_lag_drift
              << " paired_phase_error_rad=" << paired_error
              << " control_phase_error_rad=" << control_error << '\n';
    return r;
}

struct StructureResult {
    std::uint32_t births = 0;
    std::uint32_t deaths = 0;
    std::uint32_t strengthened = 0;
    std::uint32_t weakened = 0;
    std::size_t initial_edges = 0;
    std::size_t final_edges = 0;
    float grown_weight = 0.0f;
    bool csr_valid = false;
};

StructureResult test_structure() {
    auto graph = make_structural_probe();
    StructureResult r;
    r.initial_edges = graph.synapse_count();
    std::vector<std::uint32_t> ages(graph.synapse_count(), 3U);
    std::vector<float> activity = {1.0f,0.0f,0.0f,1.0f};

    StructuralPlasticityParameters p;
    p.potentiation = 0.02f;
    p.depression = 0.01f;
    p.growth_threshold = 0.50f;
    p.prune_threshold = 0.002f;
    p.minimum_epochs_before_prune = 3;
    p.maximum_out_degree = 3;
    p.maximum_weight = 0.25f;
    p.resource_per_excitatory_neuron = 1.0f;
    ActivityStructuralPlasticity structural(p);

    // One controlled epoch: the old inactive edge should be pruned, and a
    // new edge from active neuron 0 to co-active neuron 1 should be grown.
    auto report = structural.apply(graph, activity, ages, true);
    r.births = report.births;
    r.deaths = report.deaths;
    r.strengthened = report.strengthened;
    r.weakened = report.weakened;
    r.final_edges = graph.synapse_count();
    for (std::uint32_t e = graph.row_offsets[0]; e < graph.row_offsets[1]; ++e)
        if (graph.target_indices[e] == 3U) r.grown_weight = graph.weights[e];
    graph.validate();
    r.csr_valid = true;

    // Off control must preserve the exact topology and weights.
    const auto before_edges = graph.target_indices;
    const auto before_weights = graph.weights;
    const auto off = structural.apply(graph, activity, report.next_edge_ages, false);
    if (off.births != 0 || off.deaths != 0 ||
        graph.target_indices != before_edges || graph.weights != before_weights)
        r.csr_valid = false;

    std::cout << "structural_initial_edges=" << r.initial_edges
              << " structural_births=" << r.births
              << " structural_deaths=" << r.deaths
              << " strengthened=" << r.strengthened
              << " weakened=" << r.weakened
              << " structural_final_edges=" << r.final_edges
              << " grown_coactive_edge_weight=" << r.grown_weight
              << " structural_off_control=" << (r.csr_valid ? "PASS" : "FAIL")
              << " CSR_validation=" << (r.csr_valid ? "PASS" : "FAIL") << '\n';
    return r;
}
}

int main(int argc, char** argv) {
    try {
        const std::filesystem::path out = argc > 1 ? argv[1] :
            "experiments/logs/v42_phase_structural_feasibility";
        std::filesystem::create_directories(out);
        std::ofstream csv(out / "v42_phase_structural_feasibility.csv");
        if (!csv) throw std::runtime_error("Cannot create feasibility CSV");

        std::cout << "V42 PHASE AND STRUCTURAL PLASTICITY FEASIBILITY TEST\n"
                  << "scope=host-side mechanism test; no CUDA neural propagation claim\n";
        const auto phase = test_phase();
        const auto structure = test_structure();

        const bool phase_learning =
            phase.paired_confidence >= 0.80f &&
            phase.paired_confidence > phase.control_confidence + 0.20f &&
            phase.frozen_lag_drift < 1.0e-5f &&
            phase.phase_drift > 0.10f && phase.finite;
        const bool structural_learning =
            structure.births >= 1 && structure.deaths >= 1 &&
            structure.grown_weight > 0.0f && structure.final_edges == 1 &&
            structure.csr_valid;

        csv << "mechanism,pass,paired_confidence,control_confidence,lag_rad,"
               "coupling,births,deaths,initial_edges,final_edges,csr_valid\n";
        csv << "autonomous_phase," << (phase_learning?1:0) << ','
            << phase.paired_confidence << ',' << phase.control_confidence << ','
            << phase.paired_lag << ',' << phase.paired_coupling
            << ",0,0,0,0,1\n";
        csv << "activity_structural," << (structural_learning?1:0)
            << ",0,0,0,0," << structure.births << ',' << structure.deaths
            << ',' << structure.initial_edges << ',' << structure.final_edges
            << ',' << (structure.csr_valid?1:0) << '\n';
        csv.close();

        std::cout << "autonomous_phase_learning=" << (phase_learning?"PASS":"FAIL") << '\n'
                  << "activity_structural_growth_and_pruning="
                  << (structural_learning?"PASS":"FAIL") << '\n'
                  << "artifact=" << (out/"v42_phase_structural_feasibility.csv").string() << '\n'
                  << "LIMITATION=positive results establish host-side mechanism feasibility, "
                     "not yet emergent learning in a CUDA neural network or dendritic morphology.\n";
        return phase_learning && structural_learning ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "V42 phase/structural feasibility test failed: "
                  << e.what() << '\n';
        return 1;
    }
}
