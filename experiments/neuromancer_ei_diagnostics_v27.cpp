#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;

constexpr std::uint32_t kSteps = 100;
constexpr std::uint32_t kPulseStart = 1;
constexpr std::uint32_t kPulseDuration = 4;
constexpr float kPulseCurrent = 500.0f;
constexpr float kDtMs = 0.5f;
constexpr float kThresholdMv = 30.0f;

ModelParameters make_model(const CSRTopology& topology) {
    ModelParameters p;
    p.simulation.excitatory_count = topology.excitatory_count;
    p.simulation.inhibitory_count = topology.inhibitory_count;
    p.simulation.steps = kSteps;
    p.simulation.dt = kDtMs;
    p.simulation.threshold = kThresholdMv;
    p.simulation.random_seed = 27001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;
    p.topology.minimum_delay = topology.minimum_delay;
    p.topology.maximum_delay = topology.maximum_delay;
    p.topology.fibonacci_delays = {topology.minimum_delay};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

struct ConditionResult {
    std::string name;
    SimulationMetrics metrics;
    std::uint64_t selected_inhibitory_spikes = 0;
    double inhibitory_voltage_mean = 0.0;
    double inhibitory_voltage_min = 0.0;
    double inhibitory_voltage_max = 0.0;
    float selected_voltage = 0.0f;
    float selected_recovery = 0.0f;
};

ConditionResult run_condition(const std::string& name,
                              const ModelParameters& model,
                              const std::string& topology_path,
                              const std::vector<std::uint32_t>& stimulated_neurons,
                              std::uint32_t selected_inhibitory) {
    CudaSimulationRuntime runtime(model, topology_path, 0, true, false);
    for (const auto neuron : stimulated_neurons) {
        runtime.schedule_stimulus(
            neuron, kPulseStart, kPulseDuration, kPulseCurrent);
    }
    // Capture the inhibitory membrane response immediately after the
    // four-step stimulus, before the remaining quiet interval can erase it.
    runtime.run_steps(kPulseStart + kPulseDuration);

    ConditionResult result;
    result.name = name;

    std::vector<std::uint32_t> inhibitory_ids;
    inhibitory_ids.reserve(model.simulation.inhibitory_count);
    for (std::uint32_t i = 0; i < model.simulation.inhibitory_count; ++i)
        inhibitory_ids.push_back(model.simulation.excitatory_count + i);
    const auto voltages = runtime.collect_voltage_samples(inhibitory_ids);
    if (voltages.empty())
        throw std::runtime_error("No inhibitory voltage samples returned");

    double sum = 0.0;
    result.inhibitory_voltage_min = std::numeric_limits<double>::infinity();
    result.inhibitory_voltage_max = -std::numeric_limits<double>::infinity();
    for (const auto& sample : voltages) {
        sum += sample.voltage_mv;
        result.inhibitory_voltage_min =
            std::min(result.inhibitory_voltage_min,
                     static_cast<double>(sample.voltage_mv));
        result.inhibitory_voltage_max =
            std::max(result.inhibitory_voltage_max,
                     static_cast<double>(sample.voltage_mv));
        if (sample.neuron_id == selected_inhibitory) {
            result.selected_voltage = sample.voltage_mv;
            result.selected_recovery = sample.recovery;
        }
    }
    result.inhibitory_voltage_mean =
        sum / static_cast<double>(voltages.size());

    runtime.run_steps(kSteps - (kPulseStart + kPulseDuration));
    result.metrics = runtime.collect_metrics();
    const auto events = runtime.collect_spike_events();
    if (events.dropped_events != 0)
        throw std::runtime_error("Spike-event recorder dropped events");
    for (const auto& event : events.events) {
        if (event.neuron_id == selected_inhibitory)
            ++result.selected_inhibitory_spikes;
    }
    return result;
}

void write_result(std::ofstream& out, const ConditionResult& r,
                  const TopologyStatistics& topology,
                  std::uint32_t selected_target,
                  std::uint32_t selected_e_inputs) {
    out << r.name << ',' << topology.synapse_count << ','
        << topology.excitatory_to_excitatory << ','
        << topology.excitatory_to_inhibitory << ','
        << topology.inhibitory_to_excitatory << ','
        << topology.inhibitory_to_inhibitory << ','
        << selected_target << ',' << selected_e_inputs << ','
        << r.metrics.excitatory_spikes << ','
        << r.metrics.inhibitory_spikes << ','
        << r.selected_inhibitory_spikes << ','
        << r.selected_voltage << ',' << r.selected_recovery << ','
        << r.inhibitory_voltage_mean << ','
        << r.inhibitory_voltage_min << ','
        << r.inhibitory_voltage_max << ','
        << (r.metrics.finite_state ? 1 : 0) << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr
                << "Usage: neuromancer_ei_diagnostics_v27 <topology.csr> <output.csv>\n";
            return 2;
        }

        const std::string topology_path = argv[1];
        const std::filesystem::path output_path = argv[2];
        const auto graph = load_csr_topology(topology_path);
        const auto stats = analyze_csr_topology(graph);
        if (graph.inhibitory_count == 0)
            throw std::runtime_error("Topology has no inhibitory population");

        const std::uint32_t e_count = graph.excitatory_count;
        const std::uint32_t i_count = graph.inhibitory_count;
        std::vector<std::uint32_t> inhibitory_indegree(i_count, 0U);
        std::vector<std::vector<std::uint32_t>> e_sources_by_i(i_count);
        for (std::uint32_t source = 0; source < e_count; ++source) {
            for (std::uint32_t edge = graph.row_offsets[source];
                 edge < graph.row_offsets[source + 1U]; ++edge) {
                const auto target = graph.target_indices[edge];
                if (target >= e_count) {
                    const auto local_i = target - e_count;
                    ++inhibitory_indegree[local_i];
                    e_sources_by_i[local_i].push_back(source);
                }
            }
        }

        const auto best = std::max_element(
            inhibitory_indegree.begin(), inhibitory_indegree.end());
        const std::uint32_t selected_local_i =
            static_cast<std::uint32_t>(best - inhibitory_indegree.begin());
        const std::uint32_t selected_target = e_count + selected_local_i;
        const auto& e_inputs = e_sources_by_i[selected_local_i];
        if (e_inputs.empty())
            throw std::runtime_error("No E-to-I synapses exist in the supplied topology");

        const auto model = make_model(graph);
        const std::vector<std::uint32_t> no_stimulus;
        const auto baseline = run_condition(
            "baseline", model, topology_path, no_stimulus, selected_target);
        const auto e_driven = run_condition(
            "e_to_i_recruitment", model, topology_path, e_inputs, selected_target);
        const auto direct_i = run_condition(
            "direct_i_excitability", model, topology_path,
            std::vector<std::uint32_t>{selected_target}, selected_target);

        if (!output_path.parent_path().empty())
            std::filesystem::create_directories(output_path.parent_path());
        std::ofstream out(output_path);
        if (!out) throw std::runtime_error("Cannot open diagnostic CSV output");
        out << "condition,synapses,e_to_e,e_to_i,i_to_e,i_to_i,"
               "selected_i_neuron,e_input_sources,excitatory_spikes,"
               "inhibitory_spikes,selected_i_spikes,selected_i_voltage_mv,"
               "selected_i_recovery,inhibitory_mean_voltage_mv,"
               "inhibitory_min_voltage_mv,inhibitory_max_voltage_mv,finite_state\n";
        out << std::setprecision(10);
        write_result(out, baseline, stats, selected_target,
                     static_cast<std::uint32_t>(e_inputs.size()));
        write_result(out, e_driven, stats, selected_target,
                     static_cast<std::uint32_t>(e_inputs.size()));
        write_result(out, direct_i, stats, selected_target,
                     static_cast<std::uint32_t>(e_inputs.size()));
        out.flush();
        if (!out) throw std::runtime_error("Failed writing diagnostic CSV");

        std::cout << "NEUROMANCER V27 — E/I POPULATION DIAGNOSTICS\n"
                  << "Topology: " << topology_path << '\n'
                  << "Neurons: " << e_count << " E + " << i_count << " I\n"
                  << "Synapses: " << stats.synapse_count << '\n'
                  << "E->E: " << stats.excitatory_to_excitatory
                  << " E->I: " << stats.excitatory_to_inhibitory
                  << " I->E: " << stats.inhibitory_to_excitatory
                  << " I->I: " << stats.inhibitory_to_inhibitory << '\n'
                  << "Selected I neuron: " << selected_target
                  << " with " << e_inputs.size() << " E afferents\n";
        const auto print = [](const ConditionResult& r) {
            std::cout << r.name
                      << ": E_spikes=" << r.metrics.excitatory_spikes
                      << " I_spikes=" << r.metrics.inhibitory_spikes
                      << " selected_I_spikes=" << r.selected_inhibitory_spikes
                      << " selected_I_voltage=" << r.selected_voltage << "mV"
                      << " I_voltage_mean=" << r.inhibitory_voltage_mean << "mV"
                      << " I_voltage_range=[" << r.inhibitory_voltage_min
                      << ',' << r.inhibitory_voltage_max << "]mV"
                      << " finite=" << (r.metrics.finite_state ? "yes" : "no")
                      << '\n';
        };
        print(baseline);
        print(e_driven);
        print(direct_i);
        std::cout << "CSV: " << std::filesystem::absolute(output_path).string()
                  << '\n';
        if (!baseline.metrics.finite_state || !e_driven.metrics.finite_state ||
            !direct_i.metrics.finite_state)
            throw std::runtime_error("Non-finite state detected during diagnostics");
        std::cout << "PASS: diagnostic runs completed with finite neuron state.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Neuromancer V27 error: " << error.what() << '\n';
        return 1;
    }
}
