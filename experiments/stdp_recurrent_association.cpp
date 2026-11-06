#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <algorithm>
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

constexpr std::uint32_t kAssemblySize = 64;
constexpr std::uint32_t kExcitatoryCount = 4 * kAssemblySize;
constexpr std::uint32_t kInhibitoryCount = 1;
constexpr std::uint32_t kCueSize = kAssemblySize / 2;
constexpr std::uint32_t kTrainingPairs = 100;
constexpr std::uint32_t kPairIntervalSteps = 1000;
constexpr std::uint32_t kCausalLagSteps = 3;
constexpr float kPulseAmplitude = 100.0f;
constexpr float kInitialWeight = 0.01f;

constexpr std::uint32_t kA = 0;
constexpr std::uint32_t kB = kAssemblySize;
constexpr std::uint32_t kC = 2 * kAssemblySize;
constexpr std::uint32_t kD = 3 * kAssemblySize;

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

CSRTopology make_association_topology() {
    CSRTopology graph;
    graph.excitatory_count = kExcitatoryCount;
    graph.inhibitory_count = kInhibitoryCount;
    graph.minimum_delay = 2;
    graph.maximum_delay = 2;
    graph.row_offsets.resize(
        static_cast<std::size_t>(graph.neuron_count()) + 1, 0);

    for (std::uint32_t source = 0; source < graph.neuron_count(); ++source) {
        if (source < kAssemblySize) {
            for (std::uint32_t target = kB;
                 target < kB + kAssemblySize; ++target)
                append_edge(graph, target);
        } else if (source < 2 * kAssemblySize) {
            for (std::uint32_t target = kA;
                 target < kA + kAssemblySize; ++target)
                append_edge(graph, target);
        } else if (source < 3 * kAssemblySize) {
            for (std::uint32_t target = kD;
                 target < kD + kAssemblySize; ++target)
                append_edge(graph, target);
        } else if (source < 4 * kAssemblySize) {
            for (std::uint32_t target = kC;
                 target < kC + kAssemblySize; ++target)
                append_edge(graph, target);
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
            if (graph.target_indices[edge] >= target_begin &&
                graph.target_indices[edge] < target_begin + kAssemblySize) {
                sum += graph.weights[edge];
                ++count;
            }
        }
    }
    if (count != static_cast<std::uint64_t>(kAssemblySize) * kAssemblySize)
        throw std::runtime_error("Association topology is missing expected edges");
    return sum / static_cast<double>(count);
}

double mean_voltage(const std::vector<VoltageSample>& samples) {
    if (samples.empty()) throw std::runtime_error("No voltage samples returned");
    double sum = 0.0;
    for (const auto& sample : samples) sum += sample.voltage_mv;
    return sum / static_cast<double>(samples.size());
}

std::vector<std::uint32_t> assembly_ids(std::uint32_t begin,
                                        std::uint32_t count) {
    std::vector<std::uint32_t> ids;
    ids.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) ids.push_back(begin + i);
    return ids;
}

void schedule_group(CudaSimulationRuntime& runtime,
                    std::uint32_t begin, std::uint32_t count,
                    std::uint64_t start_step) {
    for (std::uint32_t i = 0; i < count; ++i)
        runtime.schedule_stimulus(
            begin + i, start_step, 1, kPulseAmplitude);
}

struct CueResult {
    double b_voltage = 0.0;
    double d_voltage = 0.0;
    std::uint64_t b_spikes = 0;
    std::uint64_t d_spikes = 0;
    std::uint64_t dropped_events = 0;
};

CueResult present_matched_cues(CudaSimulationRuntime& runtime) {
    const std::uint64_t start = runtime.completed_steps() + 1;
    schedule_group(runtime, kA, kCueSize, start);
    schedule_group(runtime, kC, kCueSize, start);
    runtime.run_steps(5);

    CueResult result;
    result.b_voltage = mean_voltage(
        runtime.collect_voltage_samples(assembly_ids(kB, kAssemblySize)));
    result.d_voltage = mean_voltage(
        runtime.collect_voltage_samples(assembly_ids(kD, kAssemblySize)));
    const auto events = runtime.collect_spike_events();
    result.dropped_events = events.dropped_events;
    for (const auto& event : events.events) {
        if (event.step < start) continue;
        if (event.neuron_id >= kB &&
            event.neuron_id < kB + kAssemblySize) ++result.b_spikes;
        if (event.neuron_id >= kD &&
            event.neuron_id < kD + kAssemblySize) ++result.d_spikes;
    }
    runtime.clear_stimuli();
    return result;
}

struct TrialResult {
    bool plasticity_enabled = false;
    double initial_ab = 0.0;
    double trained_ab = 0.0;
    double final_ab = 0.0;
    double final_ba = 0.0;
    double final_cd = 0.0;
    std::uint64_t training_a_spikes = 0;
    std::uint64_t training_b_spikes = 0;
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
    result.initial_ab = mean_path_weight(initial_graph, kA, kB);

    runtime.run_steps(1000); // settle all four assemblies
    result.baseline = present_matched_cues(runtime);
    runtime.run_steps(1000);

    for (std::uint32_t pair = 0; pair < kTrainingPairs; ++pair) {
        const std::uint64_t start = runtime.completed_steps() + 1;
        schedule_group(runtime, kA, kAssemblySize, start);
        schedule_group(runtime, kB, kAssemblySize,
                       start + kCausalLagSteps);
        runtime.run_steps(kPairIntervalSteps);
        runtime.clear_stimuli();
    }

    const auto training_events = runtime.collect_spike_events();
    for (const auto& event : training_events.events) {
        if (event.neuron_id >= kA &&
            event.neuron_id < kA + kAssemblySize)
            ++result.training_a_spikes;
        if (event.neuron_id >= kB &&
            event.neuron_id < kB + kAssemblySize)
            ++result.training_b_spikes;
    }
    if (training_events.dropped_events != 0)
        throw std::runtime_error("Spike recorder dropped events during training");

    const auto trained_graph = runtime.collect_connectivity();
    result.trained_ab = mean_path_weight(trained_graph, kA, kB);

    runtime.run_steps(1000); // allow traces and membrane transients to decay
    result.recall = present_matched_cues(runtime);

    const auto final_graph = runtime.collect_connectivity();
    result.final_ab = mean_path_weight(final_graph, kA, kB);
    result.final_ba = mean_path_weight(final_graph, kB, kA);
    result.final_cd = mean_path_weight(final_graph, kC, kD);
    result.finite_state = runtime.collect_metrics().finite_state;
    return result;
}

void write_result(std::ostream& out, const TrialResult& r) {
    out << (r.plasticity_enabled ? "plasticity_on" : "plasticity_off") << ','
        << r.initial_ab << ',' << r.trained_ab << ',' << r.final_ab << ','
        << r.final_ba << ',' << r.final_cd << ','
        << r.training_a_spikes << ',' << r.training_b_spikes << ','
        << r.baseline.b_voltage << ',' << r.recall.b_voltage << ','
        << r.recall.b_voltage - r.baseline.b_voltage << ','
        << r.baseline.d_voltage << ',' << r.recall.d_voltage << ','
        << r.recall.d_voltage - r.baseline.d_voltage << ','
        << r.baseline.b_spikes << ',' << r.recall.b_spikes << ','
        << r.baseline.d_spikes << ',' << r.recall.d_spikes << ','
        << (r.finite_state ? 1 : 0) << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string topology_path = argc > 1
            ? argv[1] : "experiments/logs/stdp_recurrent_association.spkcsr";
        const std::string output_path = argc > 2
            ? argv[2] : "experiments/logs/stdp_recurrent_association.csv";
        std::filesystem::create_directories(
            std::filesystem::path(topology_path).parent_path());
        std::filesystem::create_directories(
            std::filesystem::path(output_path).parent_path());

        const auto topology = make_association_topology();
        save_csr_topology(topology, topology_path);
        const auto model = make_model();

        StdpParameters stdp;
        stdp.potentiation_rate = 0.005f;
        stdp.depression_rate = 0.00525f;
        stdp.presynaptic_tau_ms = 20.0f;
        stdp.postsynaptic_tau_ms = 20.0f;
        stdp.minimum_weight = 0.001f;
        stdp.maximum_weight = 1.0f;

        std::cout << "Controlled recurrent-association experiment\n"
                  << "Neurons: " << model.simulation.neuron_count()
                  << " (E=" << kExcitatoryCount
                  << ", I=" << kInhibitoryCount << ")\n"
                  << "Assemblies: A<->B trained; C<->D matched untrained control\n"
                  << "Assembly size: " << kAssemblySize
                  << "; partial cue size: " << kCueSize << '\n'
                  << "Training pairs: " << kTrainingPairs
                  << "; causal lag: " << kCausalLagSteps * model.simulation.dt
                  << " ms\n"
                  << "STDP: A+=" << stdp.potentiation_rate
                  << ", A-=" << stdp.depression_rate
                  << ", tau_pre/post=" << stdp.presynaptic_tau_ms
                  << '/' << stdp.postsynaptic_tau_ms << " ms\n";

        const auto plastic = run_trial(model, topology_path, stdp, true);
        const auto control = run_trial(model, topology_path, stdp, false);

        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("Cannot create result CSV");
        output << std::setprecision(10)
            << "condition,initial_ab_mean,trained_ab_mean,final_ab_mean,"
               "final_ba_mean,final_cd_mean,training_a_spikes,training_b_spikes,"
               "baseline_b_voltage,recall_b_voltage,b_voltage_delta,"
               "baseline_d_voltage,recall_d_voltage,d_voltage_delta,"
               "baseline_b_spikes,recall_b_spikes,baseline_d_spikes,"
               "recall_d_spikes,finite_state\n";
        write_result(output, plastic);
        write_result(output, control);
        output.close();

        const bool selective_weight_change =
            plastic.trained_ab > plastic.initial_ab + 0.1 &&
            plastic.trained_ab > plastic.final_cd * 2.0;
        const bool spike_recall = plastic.recall.b_spikes > 0;
        const bool selective_voltage =
            plastic.recall.b_voltage - plastic.baseline.b_voltage >
            plastic.recall.d_voltage - plastic.baseline.d_voltage + 0.5;
        const bool control_unchanged =
            std::abs(control.final_ab - control.initial_ab) < 1.0e-5;
        const bool stable = plastic.finite_state && control.finite_state;

        std::cout << std::fixed << std::setprecision(6)
                  << "\nPLASTICITY ENABLED\n"
                  << "A->B mean weight: " << plastic.initial_ab << " -> "
                  << plastic.trained_ab << '\n'
                  << "B->A mean weight: " << plastic.final_ba << '\n'
                  << "Untrained C->D mean weight: " << plastic.final_cd << '\n'
                  << "Training A/B spikes: " << plastic.training_a_spikes
                  << '/' << plastic.training_b_spikes << '\n'
                  << "B baseline/recall voltage: " << plastic.baseline.b_voltage
                  << '/' << plastic.recall.b_voltage << " mV\n"
                  << "B baseline/recall spikes: " << plastic.baseline.b_spikes
                  << '/' << plastic.recall.b_spikes << '\n'
                  << "D baseline/recall voltage: " << plastic.baseline.d_voltage
                  << '/' << plastic.recall.d_voltage << " mV\n"
                  << "D baseline/recall spikes: " << plastic.baseline.d_spikes
                  << '/' << plastic.recall.d_spikes << '\n'
                  << "\nPLASTICITY DISABLED CONTROL\n"
                  << "A->B mean weight: " << control.initial_ab << " -> "
                  << control.final_ab << '\n'
                  << "B recall spikes: " << control.recall.b_spikes << '\n'
                  << "\nCRITERIA\n"
                  << "Selective pathway potentiation: "
                  << (selective_weight_change ? "PASS" : "FAIL") << '\n'
                  << "Partial-cue spike recall: "
                  << (spike_recall ? "PASS" : "FAIL") << '\n'
                  << "Trained-vs-untrained response: "
                  << (selective_voltage ? "PASS" : "FAIL") << '\n'
                  << "Plasticity-disabled control: "
                  << (control_unchanged && control.recall.b_spikes == 0
                      ? "PASS" : "FAIL") << '\n'
                  << "Finite state: " << (stable ? "PASS" : "FAIL") << '\n'
                  << "CSV: " << output_path << '\n';

        return selective_weight_change && spike_recall && selective_voltage &&
               control_unchanged && control.recall.b_spikes == 0 && stable
            ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "STDP recurrent-association experiment failed: "
                  << error.what() << '\n';
        return 1;
    }
}
