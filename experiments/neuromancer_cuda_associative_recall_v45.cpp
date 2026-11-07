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
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr std::uint32_t K = 32;
constexpr std::uint32_t E = 3 * K;
constexpr std::uint32_t N = E + 1;
constexpr float PI = 3.14159265358979323846f;
constexpr float PULSE = 100.0f;
constexpr std::uint32_t TRAINING_EPISODES = 40;
constexpr std::uint32_t PAIR_STEPS = 48;
constexpr std::uint32_t CAUSAL_LAG = 3;
constexpr std::uint32_t CUE_SIZE = 8;
constexpr std::uint32_t RECALL_STEPS = 24;

std::uint32_t population(std::uint32_t id) { return id / K; }

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = E;
    p.simulation.inhibitory_count = 1;
    p.simulation.steps = 1;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = -64.99999f;
    p.simulation.random_seed = 45001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;
    p.topology.minimum_delay = 2;
    p.topology.maximum_delay = 2;
    p.topology.delay_distribution = DelayDistribution::Uniform;
    p.topology.fibonacci_delays = {2};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology make_random_graph(std::uint32_t seed) {
    CSRTopology g;
    g.excitatory_count = E;
    g.inhibitory_count = 1;
    g.minimum_delay = g.maximum_delay = 2;
    g.row_offsets.assign(N + 1, 0);
    std::mt19937 rng(seed);
    std::bernoulli_distribution edge(0.15);
    for (std::uint32_t s = 0; s < E; ++s) {
        for (std::uint32_t t = 0; t < E; ++t) {
            if (s != t && edge(rng)) {
                g.target_indices.push_back(t);
                g.weights.push_back(0.01f);
                g.delays.push_back(2);
            }
        }
        g.row_offsets[s + 1] =
            static_cast<std::uint32_t>(g.target_indices.size());
    }
    g.row_offsets[E + 1] = static_cast<std::uint32_t>(g.target_indices.size());
    g.validate();
    return g;
}

double mean_between(const CSRTopology& g, std::uint32_t source_pop,
                    std::uint32_t target_pop) {
    double sum = 0.0;
    std::uint64_t count = 0;
    for (std::uint32_t s = source_pop * K; s < (source_pop + 1) * K; ++s) {
        for (auto e = g.row_offsets[s]; e < g.row_offsets[s + 1]; ++e) {
            if (population(g.target_indices[e]) == target_pop) {
                sum += g.weights[e];
                ++count;
            }
        }
    }
    return count ? sum / static_cast<double>(count) : 0.0;
}

void stimulate(CudaSimulationRuntime& runtime, std::uint32_t begin,
               std::uint32_t count, std::uint64_t step) {
    for (std::uint32_t i = 0; i < count; ++i)
        runtime.schedule_stimulus(begin + i, step, 1, PULSE);
}

struct TrainingResult {
    CSRTopology learned_graph;
    std::array<double, 3> initial_means{};
    std::array<double, 3> learned_means{};
    std::array<std::uint64_t, 3> population_events{};
    float lag_ab = 0.0f;
    float lag_bc = 0.0f;
    float confidence_ab = 0.0f;
    float confidence_bc = 0.0f;
    std::vector<float> phases;
    std::vector<float> edge_lags;
    std::vector<float> activity;
    StructuralPlasticityReport structural;
};

TrainingResult train(const std::filesystem::path& dir) {
    const auto path = dir / "v45_initial.csr";
    const auto initial = make_random_graph(45001);
    save_csr_topology(initial, path.string());

    StdpParameters stdp;
    stdp.potentiation_rate = 0.005f;
    stdp.depression_rate = 0.00525f;
    stdp.presynaptic_tau_ms = 20.0f;
    stdp.postsynaptic_tau_ms = 20.0f;
    stdp.minimum_weight = 0.001f;
    stdp.maximum_weight = 1.0f;

    TrainingResult out;
    for (std::uint32_t p = 0; p < 3; ++p) {
        out.initial_means[p] = mean_between(initial, p, (p + 1) % 3);
    }
    out.activity.assign(N, 0.0f);
    AutonomousPhaseParameters pp;
    pp.dt_ms = 0.5f;
    pp.intrinsic_frequency_hz = 8.0f;
    pp.temporal_association_window_steps = 12;
    pp.temporal_lag_learning_rate = 0.12f;
    AutonomousPopulationPhase phase(3, pp);

    {
        CudaSimulationRuntime runtime(make_model(), path.string(), 0, true, true, stdp);
        runtime.run_steps(1000);
        for (std::uint32_t episode = 0; episode < TRAINING_EPISODES; ++episode) {
            for (std::uint32_t association = 0; association < 2; ++association) {
                const std::uint32_t source = association;
                const std::uint32_t target = association + 1;
                const auto start = runtime.completed_steps() + 1;
                stimulate(runtime, source * K, K, start);
                stimulate(runtime, target * K, K, start + CAUSAL_LAG);
                for (std::uint32_t step = 0; step < PAIR_STEPS; ++step) {
                    runtime.run_steps(1);
                    const auto batch = runtime.collect_spike_events();
                    if (batch.dropped_events)
                        throw std::runtime_error("V45 training spike recorder overflow");
                    std::vector<std::uint32_t> pop_spikes(3, 0);
                    for (const auto& event : batch.events) {
                        if (event.neuron_id < E) {
                            ++out.population_events[population(event.neuron_id)];
                            out.activity[event.neuron_id] += 1.0f;
                            pop_spikes[population(event.neuron_id)] = 1;
                        }
                    }
                    phase.step(pop_spikes, true);
                }
                runtime.clear_stimuli();
            }
        }
        if (!runtime.collect_metrics().finite_state)
            throw std::runtime_error("Non-finite state during V45 training");
        out.learned_graph = runtime.collect_connectivity();
        out.learned_graph.validate();
        for (std::uint32_t p = 0; p < 3; ++p)
            out.learned_means[p] = mean_between(out.learned_graph, p, (p + 1) % 3);
    }
    std::filesystem::remove(path);

    out.lag_ab = phase.learned_lag(0, 1);
    out.lag_bc = phase.learned_lag(1, 2);
    out.confidence_ab = phase.temporal_confidence(0, 1);
    out.confidence_bc = phase.temporal_confidence(1, 2);
    out.phases.resize(N, 0.0f);
    out.edge_lags.assign(out.learned_graph.synapse_count(), 0.0f);

    // Activity values are measured from CUDA spike events. Normalize them
    // before the explicit host-side structural CSR update.
    const float max_activity = *std::max_element(out.activity.begin(),
                                                  out.activity.begin() + E);
    if (max_activity > 0.0f)
        for (auto& value : out.activity) value /= max_activity;

    StructuralPlasticityParameters structural;
    structural.potentiation = 0.04f;
    structural.depression = 0.015f;
    structural.growth_threshold = 0.70f;
    structural.prune_threshold = 0.002f;
    structural.minimum_epochs_before_prune = 1;
    structural.maximum_out_degree = 24;
    structural.maximum_weight = 1.0f;
    structural.resource_per_excitatory_neuron = 1.0f;
    ActivityStructuralPlasticity updater(structural);
    std::vector<std::uint32_t> ages(out.learned_graph.synapse_count(), 0);
    out.structural = updater.apply(out.learned_graph, out.activity, ages, true);
    out.learned_graph.validate();

    // Use the measured directed lags to align A->B and B->C. The C population
    // is the phase reference; non-trained cross-population edges retain zero lag.
    for (std::uint32_t id = 0; id < E; ++id) {
        const auto p = population(id);
        out.phases[id] = p == 0 ? out.lag_ab + out.lag_bc
                         : p == 1 ? out.lag_bc : 0.0f;
    }
    for (std::uint32_t s = 0; s < E; ++s) {
        for (auto e = out.learned_graph.row_offsets[s];
             e < out.learned_graph.row_offsets[s + 1]; ++e) {
            const auto sp = population(s);
            const auto tp = population(out.learned_graph.target_indices[e]);
            if (sp == 0 && tp == 1) out.edge_lags[e] = out.lag_ab;
            if (sp == 1 && tp == 2) out.edge_lags[e] = out.lag_bc;
        }
    }
    return out;
}

struct RecallResult {
    std::uint64_t source_spikes = 0;
    std::uint64_t b_spikes = 0;
    std::uint64_t c_spikes = 0;
    std::uint32_t b_unique = 0;
    std::uint32_t c_unique = 0;
    std::uint32_t b_false_positive = 0;
    std::uint32_t c_false_positive = 0;
    std::uint32_t false_positive_neurons = 0;
    float precision = 0.0f;
    float recall = 0.0f;
    float jaccard = 0.0f;
    std::uint64_t dropped = 0;
    bool finite = true;
};

RecallResult recall(const std::filesystem::path& topology_path,
                    const TrainingResult& trained, bool phase_gate) {
    save_csr_topology(trained.learned_graph, topology_path.string());
    CudaSimulationRuntime runtime(make_model(), topology_path.string(), 0, true, false);
    std::filesystem::remove(topology_path);
    if (phase_gate)
        runtime.configure_dynamic_phase_gate(trained.phases, trained.edge_lags, 0.35f);

    RecallResult out;
    std::array<bool, K> b_active{}, c_active{};
    std::array<bool, E> any_active{};
    const auto start = runtime.completed_steps() + 1;
    stimulate(runtime, 0, CUE_SIZE, start); // partial A cue only
    for (std::uint32_t step = 0; step < RECALL_STEPS; ++step) {
        runtime.run_steps(1);
        const auto batch = runtime.collect_spike_events();
        out.dropped += batch.dropped_events;
        for (const auto& event : batch.events) {
            if (event.neuron_id < E) any_active[event.neuron_id] = true;
            if (event.neuron_id < CUE_SIZE) ++out.source_spikes;
            if (event.neuron_id >= K && event.neuron_id < 2 * K) {
                ++out.b_spikes;
                b_active[event.neuron_id - K] = true;
            }
            if (event.neuron_id >= 2 * K && event.neuron_id < 3 * K) {
                ++out.c_spikes;
                c_active[event.neuron_id - 2 * K] = true;
            }
        }
    }
    runtime.clear_stimuli();
    out.b_unique = static_cast<std::uint32_t>(
        std::count(b_active.begin(), b_active.end(), true));
    out.c_unique = static_cast<std::uint32_t>(
        std::count(c_active.begin(), c_active.end(), true));
    for (std::uint32_t id = 0; id < E; ++id) {
        const bool expected = id < CUE_SIZE || (id >= K && id < 3 * K);
        if (any_active[id] && !expected) ++out.false_positive_neurons;
    }
    const auto true_positive = out.b_unique + out.c_unique;
    out.precision = static_cast<float>(true_positive) /
        static_cast<float>(std::max(1U, true_positive + out.false_positive_neurons));
    out.recall = static_cast<float>(true_positive) / static_cast<float>(2 * K);
    out.jaccard = static_cast<float>(true_positive) /
        static_cast<float>(2 * K + out.false_positive_neurons);
    // The target assemblies are the expected identities; report all recruited
    // units outside them separately rather than silently counting them as recall.
    const auto metrics = runtime.collect_metrics();
    out.finite = metrics.finite_state;
    runtime.collect_connectivity().validate();
    return out;
}

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path dir = argc > 1 ? argv[1] :
        "experiments/logs/v45_cuda_associative_recall";
    try {
        std::filesystem::create_directories(dir);
        auto trained = train(dir);
        const auto learned_path = dir / "v45_learned_frozen.csr";
        save_csr_topology(trained.learned_graph, learned_path.string());

        const auto ungated = recall(dir / "v45_recall_ungated.csr", trained, false);
        const auto gated = recall(dir / "v45_recall_phase_gated.csr", trained, true);
        const auto final_path = dir / "v45_frozen_check.csr";
        save_csr_topology(trained.learned_graph, final_path.string());
        CudaSimulationRuntime frozen(make_model(), final_path.string(), 0, true, false);
        std::filesystem::remove(final_path);
        const auto before = frozen.collect_connectivity();
        frozen.run_steps(100);
        const auto after = frozen.collect_connectivity();
        bool weights_frozen = before.weights == after.weights;
        const bool csr_valid = (trained.learned_graph.validate(), true);
        const bool phase_learned = trained.confidence_ab > 0.2f &&
                                   trained.confidence_bc > 0.2f &&
                                   std::isfinite(trained.lag_ab) &&
                                   std::isfinite(trained.lag_bc);
        const bool recall_activity = gated.b_unique > 0 && gated.c_unique > 0;
        const bool finite = ungated.finite && gated.finite &&
                            frozen.collect_metrics().finite_state;
        const bool no_drops = ungated.dropped == 0 && gated.dropped == 0;

        std::ofstream csv(dir / "v45_summary.csv");
        if (!csv) throw std::runtime_error("Cannot create V45 summary CSV");
        csv << "condition,source_spikes,b_spikes,b_unique,c_spikes,c_unique,false_positive_neurons,precision,recall,jaccard,dropped,finite\n";
        auto write = [&](const char* name, const RecallResult& r) {
            csv << name << ',' << r.source_spikes << ',' << r.b_spikes << ','
                << r.b_unique << ',' << r.c_spikes << ',' << r.c_unique << ','
                << r.false_positive_neurons << ',' << r.precision << ','
                << r.recall << ',' << r.jaccard << ',' << r.dropped << ','
                << (r.finite ? 1 : 0) << '\n';
        };
        write("ungated", ungated);
        write("learned_phase_gated", gated);

        std::cout << "NEUROMANCER V45 CUDA ASSOCIATIVE RECALL\n"
                  << "topology=global_random_CSR_seed_45001 initial_edges="
                  << make_random_graph(45001).synapse_count()
                  << " learned_edges=" << trained.learned_graph.synapse_count() << '\n'
                  << "training_episodes_per_association=" << TRAINING_EPISODES
                  << " causal_lag_steps=" << CAUSAL_LAG << '\n'
                  << "cuda_spike_events_A=" << trained.population_events[0]
                  << " B=" << trained.population_events[1]
                  << " C=" << trained.population_events[2] << '\n'
                  << "learned_AB_lag=" << trained.lag_ab
                  << " confidence=" << trained.confidence_ab
                  << " learned_BC_lag=" << trained.lag_bc
                  << " confidence=" << trained.confidence_bc << '\n'
                  << "structural_births=" << trained.structural.births
                  << " structural_deaths=" << trained.structural.deaths
                  << " csr_valid=" << (csr_valid ? "PASS" : "FAIL") << '\n';
        for (const auto* item : {&ungated, &gated}) {
            std::cout << "condition=" << (item == &ungated ? "ungated" : "learned_phase_gated")
                      << " source_spikes=" << item->source_spikes
                      << " B_spikes=" << item->b_spikes
                      << " B_unique=" << item->b_unique << "/32"
                      << " C_spikes=" << item->c_spikes
                      << " C_unique=" << item->c_unique << "/32"
                      << " false_positive_neurons=" << item->false_positive_neurons
                      << " precision=" << item->precision
                      << " recall=" << item->recall
                      << " jaccard=" << item->jaccard
                      << " dropped=" << item->dropped
                      << " finite=" << (item->finite ? "PASS" : "FAIL") << '\n';
        }
        std::cout << "phase_learning=" << (phase_learned ? "PASS" : "FAIL") << '\n'
                  << "partial_A_cue_recruits_B_and_C="
                  << (recall_activity ? "PASS" : "FAIL") << '\n'
                  << "frozen_weights=" << (weights_frozen ? "PASS" : "FAIL") << '\n'
                  << "finite_state=" << (finite ? "PASS" : "FAIL") << '\n'
                  << "no_dropped_events=" << (no_drops ? "PASS" : "FAIL") << '\n'
                  << "artifacts=" << (dir / "v45_summary.csv").string() << '\n'
                  << "LIMITATION=CUDA performs STDP training and frozen recall on a global random CSR graph; phase learning and structural CSR remodeling are host-side controllers driven by measured CUDA spike activity. This first integrated assay is a single seeded run and does not establish population-scale generalization or dendritic morphology.\n";

        require(phase_learned, "V45 did not learn both directed phase associations");
        require(csr_valid && finite && no_drops && weights_frozen,
                "V45 integrity check failed");
        require(recall_activity, "Frozen CUDA recall did not recruit both B and C");
        std::filesystem::remove(learned_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "V45 failed: " << e.what() << '\n';
        return 1;
    }
}
