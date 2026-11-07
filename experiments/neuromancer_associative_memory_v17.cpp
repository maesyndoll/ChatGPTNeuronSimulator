#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;

constexpr std::uint32_t kAssembly = 32;
constexpr std::uint32_t kPairs = 3;
constexpr std::uint32_t kExcitatory = 192;
constexpr std::uint32_t kInhibitory = 1;
constexpr std::uint32_t kTrainingEpisodes = 100;
constexpr std::uint32_t kPairInterval = 500;
constexpr std::uint32_t kCausalLag = 3;
constexpr std::uint32_t kObservationSteps = 12;
constexpr float kPulse = 100.0f;
constexpr float kInitialWeight = 0.01f;
constexpr double kConnectionProbability = 0.50;
constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 3> kAssociations{{
    {0, 32}, {64, 96}, {128, 160}
}};
constexpr std::array<std::uint32_t, 4> kCueSizes{{8, 16, 24, 32}};

ModelParameters make_model() {
    ModelParameters m;
    m.simulation.excitatory_count = kExcitatory;
    m.simulation.inhibitory_count = kInhibitory;
    m.simulation.steps = 1000;
    m.simulation.dt = 0.5f;
    m.simulation.threshold = -64.99999f;
    m.simulation.random_seed = 24680;
    m.simulation.initial_voltage_min = -65.0f;
    m.simulation.initial_voltage_max = -65.0f;
    m.simulation.excitatory_current_min = 0.0f;
    m.simulation.excitatory_current_max = 0.0f;
    m.simulation.inhibitory_current_min = 0.0f;
    m.simulation.inhibitory_current_max = 0.0f;
    m.topology.minimum_delay = 2;
    m.topology.maximum_delay = 2;
    m.topology.delay_distribution = DelayDistribution::Uniform;
    m.topology.fibonacci_delays = {2};
    m.topology.fibonacci_weights = {1.0};
    return m;
}

void add_edge(CSRTopology& g, std::uint32_t target) {
    g.target_indices.push_back(target);
    g.weights.push_back(kInitialWeight);
    g.delays.push_back(2);
}

CSRTopology make_topology() {
    CSRTopology g;
    g.excitatory_count = kExcitatory;
    g.inhibitory_count = kInhibitory;
    g.minimum_delay = 2;
    g.maximum_delay = 2;
    g.row_offsets.resize(static_cast<std::size_t>(g.neuron_count()) + 1, 0);

    // Seeded Bernoulli connectivity replaces V16's complete 32x32
    // association blocks. Each direction is sampled independently.
    std::mt19937 rng(24680);
    std::bernoulli_distribution connected(kConnectionProbability);
    for (std::uint32_t source = 0; source < g.neuron_count(); ++source) {
        for (const auto& p : kAssociations) {
            std::uint32_t target_begin = 0;
            if (source >= p.first && source < p.first + kAssembly)
                target_begin = p.second;
            else if (source >= p.second && source < p.second + kAssembly)
                target_begin = p.first;
            else
                continue;

            for (std::uint32_t offset = 0; offset < kAssembly; ++offset) {
                if (connected(rng))
                    add_edge(g, target_begin + offset);
            }
        }
        g.row_offsets[source + 1] =
            static_cast<std::uint32_t>(g.target_indices.size());
    }
    g.validate();
    return g;
}

double mean_weight(const CSRTopology& g, std::uint32_t source_begin,
                   std::uint32_t target_begin) {
    double sum = 0.0;
    std::uint32_t count = 0;
    for (std::uint32_t s = source_begin; s < source_begin + kAssembly; ++s) {
        for (std::uint32_t e = g.row_offsets[s]; e < g.row_offsets[s + 1]; ++e) {
            const auto t = g.target_indices[e];
            if (t >= target_begin && t < target_begin + kAssembly) {
                sum += g.weights[e];
                ++count;
            }
        }
    }
    if (count == 0)
        throw std::runtime_error("Association pathway unexpectedly has no edges");
    return sum / count;
}

void stimulate(CudaSimulationRuntime& r, std::uint32_t begin,
               std::uint32_t count, std::uint64_t start) {
    for (std::uint32_t i = 0; i < count; ++i)
        r.schedule_stimulus(begin + i, start, 1, kPulse);
}

struct Response {
    std::uint64_t source_spikes = 0;
    std::uint64_t target_spikes = 0;
    std::uint32_t unique_target_neurons = 0;
    float max_target_voltage = -1000.0f;
    std::uint64_t dropped_events = 0;
};

Response probe(CudaSimulationRuntime& r, std::uint32_t source_begin,
               std::uint32_t target_begin, std::uint32_t cue_size) {
    const auto start = r.completed_steps() + 1;
    stimulate(r, source_begin, cue_size, start);
    Response out;
    std::array<bool, kAssembly> target_active{};
    std::vector<std::uint32_t> target_ids;
    target_ids.reserve(kAssembly);
    for (std::uint32_t i = 0; i < kAssembly; ++i)
        target_ids.push_back(target_begin + i);

    for (std::uint32_t step = 0; step < kObservationSteps; ++step) {
        r.run_steps(1);
        const auto samples = r.collect_voltage_samples(target_ids);
        for (const auto& sample : samples) {
            out.max_target_voltage =
                std::max(out.max_target_voltage, sample.voltage_mv);
            if (sample.spiked_this_step)
                target_active[sample.neuron_id - target_begin] = true;
        }
    }
    const auto events = r.collect_spike_events();
    out.dropped_events = events.dropped_events;
    for (const auto& e : events.events) {
        if (e.step < start) continue;
        if (e.neuron_id >= source_begin &&
            e.neuron_id < source_begin + cue_size)
            ++out.source_spikes;
        if (e.neuron_id >= target_begin &&
            e.neuron_id < target_begin + kAssembly)
            ++out.target_spikes;
    }
    out.unique_target_neurons = static_cast<std::uint32_t>(
        std::count(target_active.begin(), target_active.end(), true));
    r.clear_stimuli();
    return out;
}

void print_response(const char* name, std::uint32_t cue,
                    const Response& x) {
    std::cout << "  " << name << " cue=" << cue << "/32"
              << " source_spikes=" << x.source_spikes
              << " target_spikes=" << x.target_spikes
              << " target_neurons=" << x.unique_target_neurons << "/32"
              << " max_target_v=" << x.max_target_voltage << " mV\n";
}

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
} // namespace

int main() {
    const std::string initial_path = "/tmp/neuromancer_v16_initial.csr";
    const std::string learned_path = "/tmp/neuromancer_v16_learned.csr";
    try {
        const auto model = make_model();
        const auto topology = make_topology();
        save_csr_topology(topology, initial_path);

        StdpParameters stdp;
        stdp.potentiation_rate = 0.005f;
        stdp.depression_rate = 0.00525f;
        stdp.presynaptic_tau_ms = 20.0f;
        stdp.postsynaptic_tau_ms = 20.0f;
        stdp.minimum_weight = 0.001f;
        stdp.maximum_weight = 1.0f;

        std::array<double, 3> initial_weights{};
        std::array<double, 3> learned_weights{};
        std::uint64_t training_spikes = 0;

        // Train in one runtime, export the learned CSR, then destroy that
        // runtime. Recall below occurs in a new runtime with plasticity off.
        {
            CudaSimulationRuntime training(
                model, initial_path, 0, true, true, stdp);
            const auto initial = training.collect_connectivity();
            for (std::size_t i = 0; i < kAssociations.size(); ++i)
                initial_weights[i] = mean_weight(
                    initial, kAssociations[i].first, kAssociations[i].second);

            training.run_steps(1000);
            for (std::uint32_t episode = 0; episode < kTrainingEpisodes; ++episode) {
                for (std::size_t association = 0; association < 2; ++association) {
                    const auto [source, target] = kAssociations[association];
                    const auto start = training.completed_steps() + 1;
                    stimulate(training, source, kAssembly, start);
                    stimulate(training, target, kAssembly, start + kCausalLag);
                    training.run_steps(kPairInterval);
                    training.clear_stimuli();
                }
            }
            const auto events = training.collect_spike_events();
            training_spikes = events.events.size();
            require(events.dropped_events == 0,
                    "Spike recorder dropped events during training");
            const auto learned = training.collect_connectivity();
            for (std::size_t i = 0; i < kAssociations.size(); ++i)
                learned_weights[i] = mean_weight(
                    learned, kAssociations[i].first, kAssociations[i].second);
            save_csr_topology(learned, learned_path);
            require(training.collect_metrics().finite_state,
                    "Non-finite training state");
        }

        // A distinct runtime loads the learned weights with plasticity disabled.
        CudaSimulationRuntime recall(model, learned_path, 0, true, false, stdp);
        const auto frozen_start = recall.collect_connectivity();
        std::array<double, 3> frozen_weights{};
        for (std::size_t i = 0; i < kAssociations.size(); ++i)
            frozen_weights[i] = mean_weight(
                frozen_start, kAssociations[i].first, kAssociations[i].second);

        std::cout << "\nNEUROMANCER — ASSOCIATIVE MEMORY V17 SPARSE CONNECTIVITY\n"
                  << "--------------------------------------------------\n"
                  << "Network: 193 neurons (192 E, 1 I); shared CSR runtime\n"
                  << "Association connectivity: seeded 50% random directed edges per block\n"
                  << "Training: A->B and C->D; E->F untrained\n"
                  << "STDP episodes: " << kTrainingEpisodes
                  << " interleaved per trained association\n"
                  << "Training spikes recorded: " << training_spikes << '\n'
                  << "Recall runtime: separate instance; plasticity disabled\n\n";
        const char* labels[] = {"A->B", "C->D", "E->F"};
        for (std::size_t i = 0; i < kAssociations.size(); ++i) {
            std::cout << labels[i] << " mean weight: "
                      << initial_weights[i] << " -> " << learned_weights[i]
                      << " -> " << frozen_weights[i] << '\n';
        }

        std::uint64_t total_dropped = 0;
        std::cout << "\nSPARSE RECRUITMENT CURVE (50% edge probability; probes separated by 1000 steps)\n";
        for (const auto cue : kCueSizes) {
            recall.run_steps(1000);
            const auto a = probe(recall, 0, 32, cue);
            recall.run_steps(1000);
            const auto c = probe(recall, 64, 96, cue);
            recall.run_steps(1000);
            const auto e = probe(recall, 128, 160, cue);
            print_response("A->B", cue, a);
            print_response("C->D", cue, c);
            print_response("E->F", cue, e);
            total_dropped += a.dropped_events + c.dropped_events + e.dropped_events;
        }

        const auto final_graph = recall.collect_connectivity();
        bool frozen = true;
        for (std::size_t i = 0; i < kAssociations.size(); ++i) {
            frozen = frozen &&
                std::abs(mean_weight(final_graph, kAssociations[i].first,
                                     kAssociations[i].second) -
                         frozen_weights[i]) < 1.0e-6;
        }
        const bool learned_a = learned_weights[0] > initial_weights[0] + 0.1;
        const bool learned_c = learned_weights[1] > initial_weights[1] + 0.1;
        const bool untouched = std::abs(learned_weights[2] - initial_weights[2]) < 1.0e-6;
        const bool finite = recall.collect_metrics().finite_state;

        std::cout << "\nINTEGRITY CHECKS\n"
                  << "A->B potentiated: " << (learned_a ? "PASS" : "FAIL") << '\n'
                  << "C->D potentiated: " << (learned_c ? "PASS" : "FAIL") << '\n'
                  << "E->F untrained: " << (untouched ? "PASS" : "FAIL") << '\n'
                  << "Weights unchanged in plasticity-disabled recall runtime: "
                  << (frozen ? "PASS" : "FAIL") << '\n'
                  << "Dropped events: " << total_dropped << '\n'
                  << "Finite recall state: " << (finite ? "PASS" : "FAIL") << '\n';

        require(learned_a && learned_c, "STDP did not potentiate both trained pathways");
        require(untouched, "Untrained pathway changed");
        require(frozen, "Weights changed in plasticity-disabled recall runtime");
        require(total_dropped == 0, "Spike recorder dropped events during recall");
        require(finite, "Non-finite recall state");
        std::cout << "\nPASS: STDP learning and genuinely frozen recall setup validated.\n"
                  << "NOTE: Recruitment curve is diagnostic; sparse feedforward association "
                     "is not evidence of pattern completion.\n";
        std::remove(initial_path.c_str());
        std::remove(learned_path.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::remove(initial_path.c_str());
        std::remove(learned_path.c_str());
        std::cerr << "V17 failed: " << e.what() << '\n';
        return 1;
    }
}
