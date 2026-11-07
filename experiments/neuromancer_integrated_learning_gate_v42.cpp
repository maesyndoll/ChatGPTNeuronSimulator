#include "learning/phase_structural_plasticity.h"
#include "plasticity/stdp.h"

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
constexpr std::uint32_t kAssembly = 4;
constexpr std::uint32_t kExcitatory = 12;
constexpr std::uint32_t kInhibitory = 2;
constexpr std::uint32_t kNeurons = kExcitatory + kInhibitory;

struct Cell {
    int phase = 0;
    int structure = 0;
    int association = 0;
    const char* name = "";
};

CSRTopology make_graph(int seed) {
    CSRTopology g;
    g.excitatory_count = kExcitatory;
    g.inhibitory_count = kInhibitory;
    g.minimum_delay = g.maximum_delay = 1;
    g.row_offsets.assign(kNeurons + 1, 0U);
    for (std::uint32_t s = 0; s < kExcitatory; ++s) {
        if (s < kAssembly) {
            const auto i = s;
            const auto b = kAssembly + ((i + static_cast<std::uint32_t>(seed)) % kAssembly);
            g.target_indices.push_back(b);
            g.weights.push_back(0.005f);
            g.delays.push_back(1U);
        } else if (s < 2 * kAssembly) {
            const auto i = s - kAssembly;
            const auto c = 2 * kAssembly +
                ((i + static_cast<std::uint32_t>(2 * seed)) % kAssembly);
            g.target_indices.push_back(c);
            g.weights.push_back(0.005f);
            g.delays.push_back(1U);
        }
        g.row_offsets[s + 1] = static_cast<std::uint32_t>(g.target_indices.size());
    }
    for (std::uint32_t s = kExcitatory; s < kNeurons; ++s)
        g.row_offsets[s + 1] = static_cast<std::uint32_t>(g.target_indices.size());
    g.validate();
    return g;
}

void train_phase(AutonomousPopulationPhase& phase) {
    constexpr int period = 300;
    for (int t = 0; t < 6000; ++t) {
        std::vector<std::uint32_t> spikes(3, 0U);
        const int within = t % period;
        if (within == 0) spikes[0] = 1U;
        if (within == 5) spikes[1] = 1U;
        if (within == 10) spikes[2] = 1U;
        phase.step(spikes, true);
    }
}

struct Metrics {
    float b_recall = 0.0f;
    float c_recall = 0.0f;
    float b_precision = 0.0f;
    float c_precision = 0.0f;
    std::uint32_t births = 0;
    std::uint32_t deaths = 0;
    float ab_confidence = 0.0f;
    float bc_confidence = 0.0f;
    bool csr_valid = false;
};

std::vector<std::uint32_t> propagate(
    const CSRTopology& graph, const std::vector<std::uint32_t>& active,
    std::uint32_t target_begin, std::uint32_t target_end, float threshold,
    bool phase_enabled, const AutonomousPopulationPhase& phase,
    std::uint32_t phase_source, std::uint32_t phase_target) {
    std::vector<std::uint32_t> next;
    if (phase_enabled &&
        phase.temporal_confidence(phase_source, phase_target) < 0.50f)
        return next;
    for (std::uint32_t target = target_begin; target < target_end; ++target) {
        float input = 0.0f;
        for (const auto source : active) {
            for (std::uint32_t e = graph.row_offsets[source];
                 e < graph.row_offsets[source + 1]; ++e) {
                if (graph.target_indices[e] == target)
                    input += std::max(0.0f, graph.weights[e]);
            }
        }
        if (input >= threshold) next.push_back(target);
    }
    return next;
}

Metrics run_cell(const Cell& cell, int seed) {
    auto graph = make_graph(seed);
    std::vector<std::uint32_t> ages(graph.synapse_count(), 0U);
    Metrics m;

    AutonomousPhaseParameters pp;
    pp.intrinsic_frequency_hz = 8.0f;
    pp.temporal_association_window_steps = 39;
    pp.temporal_lag_learning_rate = 0.10f;
    pp.minimum_temporal_confidence = 0.20f;
    AutonomousPopulationPhase phase(3, pp);
    if (cell.phase) train_phase(phase);
    m.ab_confidence = phase.temporal_confidence(0, 1);
    m.bc_confidence = phase.temporal_confidence(1, 2);

    if (cell.association) {
        std::vector<std::uint32_t> sources;
        std::vector<std::uint32_t> targets;
        for (std::uint32_t s = 0; s < kNeurons; ++s) {
            for (std::uint32_t e = graph.row_offsets[s]; e < graph.row_offsets[s + 1]; ++e) {
                sources.push_back(s);
                targets.push_back(graph.target_indices[e]);
            }
        }
        StdpTraces traces;
        traces.presynaptic.assign(kNeurons, 0.0f);
        traces.postsynaptic.assign(kNeurons, 0.0f);
        StdpParameters sp;
        sp.potentiation_rate = 0.08f;
        sp.depression_rate = 0.02f;
        sp.presynaptic_tau_ms = 20.0f;
        sp.postsynaptic_tau_ms = 20.0f;
        sp.minimum_weight = 0.001f;
        sp.maximum_weight = 0.25f;
        for (int trial = 0; trial < 24; ++trial) {
            for (std::uint32_t i = 0; i < kAssembly; ++i) {
                const auto b = kAssembly +
                    ((i + static_cast<std::uint32_t>(seed)) % kAssembly);
                const auto c = 2 * kAssembly +
                    ((i + static_cast<std::uint32_t>(2 * seed)) % kAssembly);
                std::vector<std::uint8_t> spikes(kNeurons, 0U);
                spikes[i] = 1U;
                apply_pair_stdp_step(graph.weights, sources, targets, kNeurons,
                                     kExcitatory, spikes, traces, 0.5f, sp);
                std::fill(spikes.begin(), spikes.end(), 0U);
                spikes[b] = 1U;
                apply_pair_stdp_step(graph.weights, sources, targets, kNeurons,
                                     kExcitatory, spikes, traces, 0.5f, sp);
                // Clear residual traces between the A->B and B->C pairings so
                // an unintended A->C association is not trained by sequence overlap.
                for (int gap = 0; gap < 80; ++gap) {
                    std::fill(spikes.begin(), spikes.end(), 0U);
                    apply_pair_stdp_step(graph.weights, sources, targets, kNeurons,
                                         kExcitatory, spikes, traces, 0.5f, sp);
                }
                spikes[b] = 1U;
                apply_pair_stdp_step(graph.weights, sources, targets, kNeurons,
                                     kExcitatory, spikes, traces, 0.5f, sp);
                std::fill(spikes.begin(), spikes.end(), 0U);
                spikes[c] = 1U;
                apply_pair_stdp_step(graph.weights, sources, targets, kNeurons,
                                     kExcitatory, spikes, traces, 0.5f, sp);
                for (int gap = 0; gap < 80; ++gap) {
                    std::fill(spikes.begin(), spikes.end(), 0U);
                    apply_pair_stdp_step(graph.weights, sources, targets, kNeurons,
                                         kExcitatory, spikes, traces, 0.5f, sp);
                }
            }
        }
    }

    if (cell.structure) {
        StructuralPlasticityParameters p;
        p.potentiation = 0.01f;
        p.depression = 0.001f;
        p.growth_threshold = 0.50f;
        p.prune_threshold = 0.001f;
        p.minimum_epochs_before_prune = 3;
        p.maximum_out_degree = 4;
        p.maximum_weight = 0.25f;
        p.resource_per_excitatory_neuron = 1.0f;
        ActivityStructuralPlasticity structural(p);
        std::vector<float> activity(kNeurons, 0.0f);
        for (std::uint32_t i = 0; i < kAssembly; ++i) {
            activity[i] = 0.95f;
            activity[kAssembly + i] = 0.95f;
        }
        auto first = structural.apply(graph, activity, ages, true);
        m.births += first.births; m.deaths += first.deaths;
        ages = std::move(first.next_edge_ages);
        std::fill(activity.begin(), activity.end(), 0.0f);
        for (std::uint32_t i = kAssembly; i < 3 * kAssembly; ++i)
            activity[i] = 0.95f;
        auto second = structural.apply(graph, activity, ages, true);
        m.births += second.births; m.deaths += second.deaths;
        ages = std::move(second.next_edge_ages);
    }

    // Fresh frozen recall: only the two-neuron (50%) A cue is supplied.
    // No STDP or structural update is called during this evaluation.
    std::vector<std::uint32_t> cue{0U, 1U};
    const auto recalled_b = propagate(graph, cue, kAssembly, 2*kAssembly,
                                      0.03f, cell.phase != 0, phase, 0, 1);
    const auto recalled_c = propagate(graph, recalled_b, 2*kAssembly, 3*kAssembly,
                                      0.03f, cell.phase != 0, phase, 1, 2);
    std::vector<std::uint32_t> expected_b, expected_c;
    for (const auto a : cue) {
        expected_b.push_back(kAssembly +
            ((a + static_cast<std::uint32_t>(seed)) % kAssembly));
        expected_c.push_back(2*kAssembly +
            ((a + static_cast<std::uint32_t>(2*seed)) % kAssembly));
    }
    auto score = [](const std::vector<std::uint32_t>& got,
                    const std::vector<std::uint32_t>& expected,
                    std::uint32_t total) {
        std::uint32_t correct = 0;
        for (const auto x : got)
            if (std::find(expected.begin(), expected.end(), x) != expected.end()) ++correct;
        const float recall = expected.empty() ? 0.0f :
            static_cast<float>(correct) / static_cast<float>(expected.size());
        const float precision = got.empty() ? 0.0f :
            static_cast<float>(correct) / static_cast<float>(got.size());
        (void)total;
        return std::pair<float,float>{recall,precision};
    };
    const auto b = score(recalled_b, expected_b, kAssembly);
    const auto c = score(recalled_c, expected_c, kAssembly);
    m.b_recall=b.first; m.b_precision=b.second;
    m.c_recall=c.first; m.c_precision=c.second;
    graph.validate();
    m.csr_valid = true;
    return m;
}
}

int main(int argc, char** argv) {
    try {
        const std::filesystem::path out = argc > 1 ? argv[1] :
            "experiments/logs/v42_integrated_learning_gate";
        std::filesystem::create_directories(out);
        const std::array<Cell,8> cells{{
            {0,0,0,"000"},{0,0,1,"001"},{0,1,0,"010"},{0,1,1,"011"},
            {1,0,0,"100"},{1,0,1,"101"},{1,1,0,"110"},{1,1,1,"111"}
        }};
        std::ofstream csv(out / "v42_integrated_gate.csv");
        if (!csv) throw std::runtime_error("Cannot create integrated gate CSV");
        csv << "seed,cell,phase,structure,association,b_recall,b_precision,"
               "c_recall,c_precision,births,deaths,ab_confidence,bc_confidence,csr_valid\n";
        std::cout << "V42 INTEGRATED LEARNING IMPLEMENTATION GATE\n"
                  << "scope=host-side reference integration; not CUDA-runtime recall\n"
                  << "seeds=10 cue_fraction=0.50 noise=0.00\n";
        bool all_finite = true;
        for (const auto& cell : cells) {
            double br=0, cr=0, bp=0, cp=0;
            std::uint32_t births=0, deaths=0;
            for (int seed=0; seed<10; ++seed) {
                const auto m=run_cell(cell,seed);
                br+=m.b_recall; cr+=m.c_recall;
                bp+=m.b_precision; cp+=m.c_precision;
                births+=m.births; deaths+=m.deaths;
                all_finite=all_finite && m.csr_valid &&
                    std::isfinite(m.b_recall) && std::isfinite(m.c_recall);
                csv<<seed<<','<<cell.name<<','<<cell.phase<<','<<cell.structure<<','
                   <<cell.association<<','<<m.b_recall<<','<<m.b_precision<<','
                   <<m.c_recall<<','<<m.c_precision<<','<<m.births<<','<<m.deaths<<','
                   <<m.ab_confidence<<','<<m.bc_confidence<<','<<(m.csr_valid?1:0)<<'\n';
            }
            std::cout<<"cell="<<cell.name<<" phase="<<cell.phase
                     <<" structure="<<cell.structure<<" association="<<cell.association
                     <<" mean_B_recall="<<br/10.0<<" mean_C_recall="<<cr/10.0
                     <<" mean_B_precision="<<bp/10.0<<" mean_C_precision="<<cp/10.0
                     <<" total_births="<<births<<" total_deaths="<<deaths<<'\n';
        }
        csv.close();
        std::cout<<"finite_and_CSR_checks="<<(all_finite?"PASS":"FAIL")<<'\n'
                 <<"artifact="<<(out/"v42_integrated_gate.csv").string()<<'\n'
                 <<"LIMITATION=reference gate uses host-side STDP, phase, and CSR "
                   "structural mechanisms with graph-based frozen recall; it does not "
                   "yet demonstrate CUDA-runtime associative recall or dendritic morphology.\n";
        return all_finite ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr<<"V42 integrated learning gate failed: "<<e.what()<<'\n';
        return 1;
    }
}
