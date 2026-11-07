#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace spiking;

constexpr std::uint32_t kAssemblySize = 32;
constexpr std::uint32_t kCueSize = 16;
constexpr std::uint32_t kPairCount = 3;
constexpr std::uint32_t kExcitatoryCount = kAssemblySize * kPairCount * 2;
constexpr std::uint32_t kInhibitoryCount = 1;
constexpr std::uint32_t kTrainingPairs = 100;
constexpr std::uint32_t kPairIntervalSteps = 500;
constexpr std::uint32_t kCausalLagSteps = 3;
constexpr std::uint32_t kObservationSteps = 6;
constexpr float kPulseAmplitude = 100.0f;
constexpr float kInitialWeight = 0.01f;
constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 3> kPairs{{
    {0, 32}, {64, 96}, {128, 160}
}};

ModelParameters make_model() {
    ModelParameters model;
    model.simulation.excitatory_count = kExcitatoryCount;
    model.simulation.inhibitory_count = kInhibitoryCount;
    model.simulation.steps = 1000;
    model.simulation.dt = 0.5f;
    model.simulation.threshold = -64.99999f;
    model.simulation.random_seed = 24680;
    model.simulation.initial_voltage_min = -65.0f;
    model.simulation.initial_voltage_max = -65.0f;
    model.simulation.excitatory_current_min = 0.0f;
    model.simulation.excitatory_current_max = 0.0f;
    model.simulation.inhibitory_current_min = 0.0f;
    model.simulation.inhibitory_current_max = 0.0f;
    model.topology.minimum_delay = 2;
    model.topology.maximum_delay = 2;
    model.topology.delay_distribution = DelayDistribution::Uniform;
    model.topology.fibonacci_delays = {2};
    model.topology.fibonacci_weights = {1.0};
    return model;
}

void append_edge(CSRTopology& graph, std::uint32_t target) {
    graph.target_indices.push_back(target);
    graph.weights.push_back(kInitialWeight);
    graph.delays.push_back(2);
}

CSRTopology make_topology() {
    CSRTopology graph;
    graph.excitatory_count = kExcitatoryCount;
    graph.inhibitory_count = kInhibitoryCount;
    graph.minimum_delay = 2;
    graph.maximum_delay = 2;
    graph.row_offsets.resize(
        static_cast<std::size_t>(graph.neuron_count()) + 1, 0);

    for (std::uint32_t source = 0; source < graph.neuron_count(); ++source) {
        for (const auto& pair : kPairs) {
            const auto first = pair.first;
            const auto second = pair.second;
            if (source >= first && source < first + kAssemblySize) {
                for (std::uint32_t target = second;
                     target < second + kAssemblySize; ++target)
                    append_edge(graph, target);
            } else if (source >= second && source < second + kAssemblySize) {
                for (std::uint32_t target = first;
                     target < first + kAssemblySize; ++target)
                    append_edge(graph, target);
            }
        }
        graph.row_offsets[static_cast<std::size_t>(source) + 1] =
            static_cast<std::uint32_t>(graph.target_indices.size());
    }
    graph.validate();
    return graph;
}

double mean_path_weight(const CSRTopology& graph,
                        std::uint32_t source_begin,
                        std::uint32_t target_begin) {
    double sum = 0.0;
    std::uint64_t count = 0;
    for (std::uint32_t source = source_begin;
         source < source_begin + kAssemblySize; ++source) {
        for (std::uint32_t edge = graph.row_offsets[source];
             edge < graph.row_offsets[source + 1]; ++edge) {
            const auto target = graph.target_indices[edge];
            if (target >= target_begin && target < target_begin + kAssemblySize) {
                sum += graph.weights[edge];
                ++count;
            }
        }
    }
    if (count != static_cast<std::uint64_t>(kAssemblySize) * kAssemblySize)
        throw std::runtime_error("Association pathway has an invalid CSR edge count");
    return sum / static_cast<double>(count);
}

std::vector<std::uint32_t> neuron_ids(std::uint32_t begin,
                                      std::uint32_t count) {
    std::vector<std::uint32_t> ids;
    ids.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) ids.push_back(begin + i);
    return ids;
}

void stimulate_group(CudaSimulationRuntime& runtime, std::uint32_t begin,
                     std::uint32_t count, std::uint64_t start_step) {
    for (std::uint32_t i = 0; i < count; ++i)
        runtime.schedule_stimulus(
            begin + i, start_step, 1, kPulseAmplitude);
}

struct PopulationResponse {
    std::uint64_t spikes = 0;
    std::uint32_t unique_neurons = 0;
};

struct CueResult {
    PopulationResponse b, d, f;
    std::uint64_t dropped_events = 0;
};

void collect_population_response(const SpikeEventBatch& events,
                                 std::uint64_t first_step,
                                 std::uint32_t begin,
                                 PopulationResponse& response) {
    std::array<bool, kAssemblySize> active{};
    for (const auto& event : events.events) {
        if (event.step < first_step ||
            event.neuron_id < begin ||
            event.neuron_id >= begin + kAssemblySize)
            continue;
        ++response.spikes;
        active[event.neuron_id - begin] = true;
    }
    response.unique_neurons = static_cast<std::uint32_t>(
        std::count(active.begin(), active.end(), true));
}

CueResult present_partial_cues(CudaSimulationRuntime& runtime) {
    const auto start = runtime.completed_steps() + 1;
    // Three equal-size partial cues are presented together: two trained
    // associations (A and C) and one never-trained control (E).
    for (const auto& pair : kPairs)
        stimulate_group(runtime, pair.first, kCueSize, start);
    runtime.run_steps(kObservationSteps);

    const auto events = runtime.collect_spike_events();
    CueResult result;
    result.dropped_events = events.dropped_events;
    collect_population_response(events, start, kPairs[0].second, result.b);
    collect_population_response(events, start, kPairs[1].second, result.d);
    collect_population_response(events, start, kPairs[2].second, result.f);
    runtime.clear_stimuli();
    return result;
}

struct TrialResult {
    bool plasticity_enabled = false;
    std::array<double, 3> initial_weights{};
    std::array<double, 3> trained_weights{};
    std::array<double, 3> final_weights{};
    std::array<std::uint64_t, 4> training_spikes{};
    CueResult baseline;
    CueResult recall;
    bool finite_state = false;
};

TrialResult run_trial(const ModelParameters& model,
                      const std::string& topology_path,
                      const StdpParameters& stdp,
                      bool plasticity_enabled) {
    CudaSimulationRuntime runtime(
        model, topology_path, 0, true, plasticity_enabled, stdp);
    TrialResult result;
    result.plasticity_enabled = plasticity_enabled;

    const auto initial_graph = runtime.collect_connectivity();
    for (std::size_t i = 0; i < kPairs.size(); ++i)
        result.initial_weights[i] = mean_path_weight(
            initial_graph, kPairs[i].first, kPairs[i].second);

    runtime.run_steps(1000);
    result.baseline = present_partial_cues(runtime);
    runtime.run_steps(1000);

    // Interleave both training episodes in the same runtime. The third pair
    // remains anatomically present but is never teacher-paired.
    for (std::uint32_t trial = 0; trial < kTrainingPairs; ++trial) {
        for (std::size_t association = 0; association < 2; ++association) {
            const auto source = kPairs[association].first;
            const auto target = kPairs[association].second;
            const auto start = runtime.completed_steps() + 1;
            stimulate_group(runtime, source, kAssemblySize, start);
            stimulate_group(runtime, target, kAssemblySize,
                            start + kCausalLagSteps);
            runtime.run_steps(kPairIntervalSteps);
            runtime.clear_stimuli();
        }
    }

    const auto training_events = runtime.collect_spike_events();
    if (training_events.dropped_events != 0)
        throw std::runtime_error("Spike recorder dropped events during training");
    for (const auto& event : training_events.events) {
        if (event.neuron_id >= kPairs[0].first &&
            event.neuron_id < kPairs[0].first + kAssemblySize)
            ++result.training_spikes[0];
        else if (event.neuron_id >= kPairs[0].second &&
                 event.neuron_id < kPairs[0].second + kAssemblySize)
            ++result.training_spikes[1];
        else if (event.neuron_id >= kPairs[1].first &&
                 event.neuron_id < kPairs[1].first + kAssemblySize)
            ++result.training_spikes[2];
        else if (event.neuron_id >= kPairs[1].second &&
                 event.neuron_id < kPairs[1].second + kAssemblySize)
            ++result.training_spikes[3];
    }

    const auto trained_graph = runtime.collect_connectivity();
    for (std::size_t i = 0; i < kPairs.size(); ++i)
        result.trained_weights[i] = mean_path_weight(
            trained_graph, kPairs[i].first, kPairs[i].second);

    runtime.run_steps(1000); // let membrane transients and STDP traces decay
    result.recall = present_partial_cues(runtime);

    const auto final_graph = runtime.collect_connectivity();
    for (std::size_t i = 0; i < kPairs.size(); ++i)
        result.final_weights[i] = mean_path_weight(
            final_graph, kPairs[i].first, kPairs[i].second);
    result.finite_state = runtime.collect_metrics().finite_state;
    return result;
}

std::uint64_t evoked(const PopulationResponse& baseline,
                     const PopulationResponse& recall) {
    return recall.spikes > baseline.spikes
        ? recall.spikes - baseline.spikes : 0;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
    const std::string topology_path =
        "/tmp/neuromancer_associative_memory_v15.csr";
    try {
        const auto model = make_model();
        const auto topology = make_topology();
        save_csr_topology(topology, topology_path);

        StdpParameters stdp;
        stdp.potentiation_rate = 0.005f;
        stdp.depression_rate = 0.00525f;
        stdp.presynaptic_tau_ms = 20.0f;
        stdp.postsynaptic_tau_ms = 20.0f;
        stdp.minimum_weight = 0.001f;
        stdp.maximum_weight = 1.0f;

        const auto plastic = run_trial(model, topology_path, stdp, true);
        const auto control = run_trial(model, topology_path, stdp, false);

        const bool learned_a =
            plastic.trained_weights[0] > plastic.initial_weights[0] + 0.1;
        const bool learned_c =
            plastic.trained_weights[1] > plastic.initial_weights[1] + 0.1;
        const bool untouched_untrained =
            std::abs(plastic.final_weights[2] - plastic.initial_weights[2]) < 1.0e-5;
        const bool partial_recall_a = plastic.recall.b.unique_neurons >= kCueSize / 2;
        const bool partial_recall_c = plastic.recall.d.unique_neurons >= kCueSize / 2;
        const bool untrained_pair_silent =
            plastic.recall.f.spikes == plastic.baseline.f.spikes;
        const bool frozen_control_silent =
            control.recall.b.spikes == control.baseline.b.spikes &&
            control.recall.d.spikes == control.baseline.d.spikes &&
            control.recall.f.spikes == control.baseline.f.spikes;
        const bool weights_stable_without_plasticity =
            std::abs(control.final_weights[0] - control.initial_weights[0]) < 1.0e-5 &&
            std::abs(control.final_weights[1] - control.initial_weights[1]) < 1.0e-5 &&
            std::abs(control.final_weights[2] - control.initial_weights[2]) < 1.0e-5;
        const bool no_dropped_events =
            plastic.baseline.dropped_events == 0 &&
            plastic.recall.dropped_events == 0 &&
            control.baseline.dropped_events == 0 &&
            control.recall.dropped_events == 0;
        const bool stable = plastic.finite_state && control.finite_state;

        std::cout << "\nNEUROMANCER — ASSOCIATIVE MEMORY V15\n"
                  << "------------------------------------\n"
                  << "Network: " << model.simulation.neuron_count()
                  << " neurons (E=" << kExcitatoryCount
                  << ", I=" << kInhibitoryCount << ")\n"
                  << "Architecture: three reciprocal assembly pairs in one CSR graph\n"
                  << "Training: A->B and C->D; E->F is never teacher-paired\n"
                  << "Assembly/cue: " << kAssemblySize << '/'
                  << kCueSize << " neurons (50% partial cues)\n"
                  << "Training: " << kTrainingPairs
                  << " interleaved causal pairs per association; lag "
                  << kCausalLagSteps * model.simulation.dt << " ms\n"
                  << "Plasticity: conventional pair STDP; frozen during recall\n\n";

        const char* names[] = {"A->B", "C->D", "E->F (untrained)"};
        for (std::size_t i = 0; i < kPairs.size(); ++i) {
            std::cout << names[i] << " mean weight: "
                      << plastic.initial_weights[i] << " -> "
                      << plastic.trained_weights[i] << " -> "
                      << plastic.final_weights[i] << '\n';
        }
        std::cout << "\nTRAINING SPIKES\n"
                  << "A/B: " << plastic.training_spikes[0] << '/'
                  << plastic.training_spikes[1] << '\n'
                  << "C/D: " << plastic.training_spikes[2] << '/'
                  << plastic.training_spikes[3] << '\n';

        const auto print_response = [](const char* label,
                                       const PopulationResponse& response) {
            std::cout << label << ": " << response.spikes
                      << " spikes across " << response.unique_neurons
                      << '/' << kAssemblySize << " target neurons\n";
        };
        std::cout << "\nPARTIAL-CUE RECALL (50% input assemblies)\n";
        print_response("A cue -> B", plastic.recall.b);
        print_response("C cue -> D", plastic.recall.d);
        print_response("E cue -> F (untrained)", plastic.recall.f);
        std::cout << "\nCONTROLS\n";
        std::cout << "Untrained E->F unchanged: "
                  << (untouched_untrained ? "PASS" : "FAIL") << '\n';
        std::cout << "Plasticity-disabled B/D/F evoked spikes: "
                  << evoked(control.baseline.b, control.recall.b) << '/'
                  << evoked(control.baseline.d, control.recall.d) << '/'
                  << evoked(control.baseline.f, control.recall.f) << '\n';

        std::cout << "\nCRITERIA\n"
                  << "A->B potentiated: " << (learned_a ? "PASS" : "FAIL") << '\n'
                  << "C->D potentiated: " << (learned_c ? "PASS" : "FAIL") << '\n'
                  << "A partial cue recruits B: " << (partial_recall_a ? "PASS" : "FAIL") << '\n'
                  << "C partial cue recruits D: " << (partial_recall_c ? "PASS" : "FAIL") << '\n'
                  << "Untrained pair not recruited: " << (untrained_pair_silent ? "PASS" : "FAIL") << '\n'
                  << "Plasticity-disabled control silent: " << (frozen_control_silent ? "PASS" : "FAIL") << '\n'
                  << "Weights frozen without plasticity: "
                  << (weights_stable_without_plasticity ? "PASS" : "FAIL") << '\n'
                  << "No dropped events: " << (no_dropped_events ? "PASS" : "FAIL") << '\n'
                  << "Finite state: " << (stable ? "PASS" : "FAIL") << '\n';

        require(learned_a && learned_c,
                "STDP failed to potentiate both trained associations");
        require(untouched_untrained,
                "The never-trained E->F pathway changed unexpectedly");
        require(partial_recall_a && partial_recall_c,
                "One or both 50% partial cues failed to recruit their target assembly");
        require(untrained_pair_silent,
                "The never-trained E cue recruited the untrained F assembly");
        require(frozen_control_silent && weights_stable_without_plasticity,
                "Plasticity-disabled control unexpectedly learned or recalled");
        require(no_dropped_events, "Spike recorder dropped events");
        require(stable, "Non-finite neuron state detected");

        std::cout << "\nPASS: two associations coexist in one recurrent network and "
                     "are retrieved from 50% cues with frozen weights.\n"
                  << "LIMIT: the three assembly pairs are structurally isolated; "
                     "cross-association interference is not tested yet.\n";
        std::remove(topology_path.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::remove(topology_path.c_str());
        std::cerr << "V15 associative-memory experiment failed: "
                  << error.what() << '\n';
        return 1;
    }
}
