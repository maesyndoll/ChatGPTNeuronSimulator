#include "runtime/cuda_simulation_runtime.cuh"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;

constexpr std::uint32_t kGroup = 32;
constexpr std::uint32_t kExcitatory = 3 * kGroup;
constexpr std::uint32_t kInhibitory = 8;
constexpr std::uint32_t kA = 0;
constexpr std::uint32_t kB = kGroup;
constexpr std::uint32_t kC = 2 * kGroup;

ModelParameters make_parameters() {
    ModelParameters p;
    p.simulation.excitatory_count = kExcitatory;
    p.simulation.inhibitory_count = kInhibitory;
    p.simulation.steps = 1;
    p.simulation.dt = 0.25f;
    p.simulation.threshold = 30.0f;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;
    p.simulation.random_seed = 24681357U;
    p.topology.p_ee = 0.0;
    p.topology.p_ei = 0.0;
    p.topology.p_ie = 0.0;
    p.topology.p_ii = 0.0;
    p.topology.minimum_delay = 1;
    p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology make_association_graph() {
    CSRTopology graph;
    graph.excitatory_count = kExcitatory;
    graph.inhibitory_count = kInhibitory;
    graph.minimum_delay = 1;
    graph.maximum_delay = 1;
    const std::uint32_t total = kExcitatory + kInhibitory;
    graph.row_offsets.resize(static_cast<std::size_t>(total) + 1, 0);

    // A<->B is a bidirectional candidate pathway. C is an unconnected
    // negative-control assembly. A->B is the direction trained by the
    // causal A-then-B protocol; B->A is expected to undergo LTD.
    for (std::uint32_t source = 0; source < total; ++source) {
        if (source < kA + kGroup) {
            const std::uint32_t target = kB + (source - kA);
            graph.target_indices.push_back(target);
            graph.weights.push_back(0.05f);
            graph.delays.push_back(1);
        } else if (source >= kB && source < kB + kGroup) {
            const std::uint32_t target = kA + (source - kB);
            graph.target_indices.push_back(target);
            graph.weights.push_back(0.05f);
            graph.delays.push_back(1);
        }
        graph.row_offsets[source + 1] =
            static_cast<std::uint32_t>(graph.target_indices.size());
    }
    graph.validate();
    return graph;
}

std::vector<float> group_current(std::uint32_t first, float value) {
    std::vector<float> current(kExcitatory + kInhibitory, 0.0f);
    for (std::uint32_t i = first; i < first + kGroup; ++i)
        current[i] = value;
    return current;
}

void append_events(SpikeEventBatch& destination, SpikeEventBatch source) {
    destination.dropped_events += source.dropped_events;
    destination.events.insert(destination.events.end(),
                               source.events.begin(), source.events.end());
}

std::uint64_t count_group(const SpikeEventBatch& batch,
                          std::uint32_t first, std::uint32_t end) {
    std::uint64_t count = 0;
    for (const auto& event : batch.events)
        if (event.neuron_id >= first && event.neuron_id < end) ++count;
    return count;
}

double mean_voltage(const std::vector<float>& voltage,
                    std::uint32_t first, std::uint32_t end) {
    double sum = 0.0;
    for (std::uint32_t i = first; i < end; ++i) sum += voltage.at(i);
    return sum / static_cast<double>(end - first);
}

struct Condition {
    std::string name;
    CudaSimulationRuntime runtime;
    Condition(std::string label, bool plastic, bool phase,
              const ModelParameters& parameters, const CSRTopology& graph)
        : name(std::move(label)), runtime(parameters, graph) {
        SynapticPlasticityConfig learning;
        learning.enabled = plastic;
        learning.trace_time_constant_ms = 30.0f;
        learning.potentiation_rate = plastic ? 0.025f : 0.0f;
        learning.depression_rate = plastic ? 0.002f : 0.0f;
        learning.minimum_weight_magnitude = 0.005f;
        learning.maximum_weight_magnitude = 1.5f;
        runtime.configure_synaptic_plasticity(learning);

        AdaptiveTopologyConfig topology;
        topology.enabled = plastic;
        topology.update_interval_steps = 8;
        topology.growth_trace_threshold = 0.75f;
        topology.prune_weight_threshold = 0.012f;
        topology.growth_weight = 0.025f;
        runtime.configure_adaptive_topology(topology);

        PhaseDynamicsConfig phases;
        phases.enabled = phase;
        phases.coupling_strength = 0.20f;
        phases.modulation_strength = phase ? 0.90f : 0.0f;
        phases.evolve_phases = true;
        runtime.configure_phase_dynamics(phases);
    }
};

void write_record(std::ofstream& csv, Condition& condition,
                  const std::string& stage, int epoch,
                  const SpikeEventBatch& events) {
    const auto metrics = condition.runtime.collect_metrics();
    if (metrics.mean_absolute_synaptic_weight <= 0.0) {
        throw std::runtime_error(
            "Synaptic weight telemetry is zero; expected persistent CSR weights");
    }
    const auto voltage = condition.runtime.collect_neuron_voltages();
    const auto a_spikes = count_group(events, kA, kA + kGroup);
    const auto b_spikes = count_group(events, kB, kB + kGroup);
    const auto c_spikes = count_group(events, kC, kC + kGroup);
    const double a_v = mean_voltage(voltage, kA, kA + kGroup);
    const double b_v = mean_voltage(voltage, kB, kB + kGroup);
    const double c_v = mean_voltage(voltage, kC, kC + kGroup);

    csv << condition.name << ',' << stage << ',' << epoch << ','
        << metrics.completed_steps << ',' << a_spikes << ',' << b_spikes << ','
        << c_spikes << ',' << std::setprecision(9)
        << a_v << ',' << b_v << ',' << c_v << ','
        << metrics.mean_absolute_synaptic_weight << ','
        << metrics.active_synapses << ',' << metrics.synapse_capacity << ','
        << metrics.phase_coherence << ',' << events.dropped_events << '\n';

    std::cout << condition.name << " " << stage << " " << epoch
              << ": A/B/C spikes=" << a_spikes << "/" << b_spikes
              << "/" << c_spikes << ", mean|w|="
              << metrics.mean_absolute_synaptic_weight
              << ", edges=" << metrics.active_synapses << "/" << metrics.synapse_capacity
              << ", phase R=" << metrics.phase_coherence << '\n';
}

void run_batch(Condition& condition, const std::vector<float>& current,
               std::uint32_t steps, SpikeEventBatch& events) {
    condition.runtime.set_external_current(current);
    condition.runtime.run_steps(steps);
    append_events(events, condition.runtime.collect_spike_events());
}

} // namespace

int main() {
    using namespace spiking;
    try {
        const auto parameters = make_parameters();
        const auto graph = make_association_graph();

        Condition plastic_phase("plastic_phase", true, true, parameters, graph);
        Condition plastic_open("plastic_open_loop", true, false, parameters, graph);
        Condition fixed_phase("fixed_phase_control", false, true, parameters, graph);
        Condition pruning_probe("pruning_probe", true, false, parameters, graph);
        auto pruning_learning = pruning_probe.runtime.synaptic_plasticity();
        pruning_learning.potentiation_rate = 0.0f;
        pruning_learning.depression_rate = 0.0f;
        pruning_probe.runtime.configure_synaptic_plasticity(pruning_learning);
        AdaptiveTopologyConfig pruning_config;
        pruning_config.enabled = true;
        pruning_config.update_interval_steps = 8;
        pruning_config.growth_trace_threshold = 100.0f;
        pruning_config.prune_weight_threshold = 0.06f;
        pruning_config.growth_weight = 0.025f;
        pruning_probe.runtime.configure_adaptive_topology(pruning_config);

        const auto a_input = group_current(kA, 100.0f);
        const auto b_input = group_current(kB, 100.0f);
        const std::vector<float> silence(kExcitatory + kInhibitory, 0.0f);

        // Explicit structural-shrink probe: with growth disabled by a high
        // co-activity threshold, sub-threshold edges must be removed on GPU.
        SpikeEventBatch pruning_events;
        run_batch(pruning_probe, a_input, 4, pruning_events);
        run_batch(pruning_probe, b_input, 4, pruning_events);
        const auto pruned_metrics = pruning_probe.runtime.collect_metrics();
        if (pruned_metrics.active_synapses != 0)
            throw std::runtime_error("Adaptive CSR pruning did not remove weak edges");

        std::ofstream csv("associative_phase_experiment.csv");
        if (!csv) throw std::runtime_error("Could not create experiment CSV");
        csv << "condition,stage,epoch,step,A_spikes,B_spikes,C_spikes,"
               "A_mean_mV,B_mean_mV,C_mean_mV,mean_abs_weight,"
               "active_synapses,synapse_capacity,global_phase_coherence,dropped_events\n";

        // Repeated causal pairing: activate A, then B, then allow a quiet
        // interval. This trains A->B rather than forcing both groups at once.
        constexpr int kTrainingEpochs = 24;
        for (int epoch = 1; epoch <= kTrainingEpochs; ++epoch) {
            for (Condition* condition :
                 {&plastic_phase, &plastic_open, &fixed_phase}) {
                SpikeEventBatch events;
                run_batch(*condition, a_input, 4, events);
                run_batch(*condition, b_input, 4, events);
                run_batch(*condition, silence, 20, events);
                write_record(csv, *condition, "training", epoch, events);
            }
        }

        const auto learned_graph = plastic_phase.runtime.collect_topology();
        learned_graph.validate();
        for (std::uint32_t source = kA; source < kA + kGroup; ++source) {
            if (learned_graph.row_offsets[source + 1] -
                    learned_graph.row_offsets[source] <= 1)
                throw std::runtime_error("Causal A-to-B structural growth was not learned");
        }
        for (std::uint32_t source = kB; source < kB + kGroup; ++source) {
            if (learned_graph.row_offsets[source + 1] -
                    learned_graph.row_offsets[source] > 1)
                throw std::runtime_error("Structural growth created an untrained B-to-A pathway");
        }
        const auto learned_topology = plastic_phase.runtime.collect_metrics();
        const auto fixed_topology = fixed_phase.runtime.collect_metrics();
        if (learned_topology.active_synapses <= graph.synapse_count())
            throw std::runtime_error("Associative training did not grow the CSR connectome");
        if (fixed_topology.active_synapses != graph.synapse_count())
            throw std::runtime_error("Fixed-topology control changed its CSR edge count");

        // Freeze plasticity for all recall probes. Phase dynamics remain
        // active in the phase conditions so effective connectivity can still
        // evolve through the learned weighted oscillator coupling.
        for (Condition* condition :
             {&plastic_phase, &plastic_open, &fixed_phase}) {
            auto frozen = condition->runtime.synaptic_plasticity();
            frozen.enabled = false;
            condition->runtime.configure_synaptic_plasticity(frozen);
            SpikeEventBatch baseline;
            run_batch(*condition, silence, 40, baseline);
            write_record(csv, *condition, "pre_recall_baseline", 0, baseline);
        }

        // A brief A-only cue, followed by autonomous evolution. Sample in
        // eight-step batches: spike events retain exact step indices while
        // voltage/weight/coherence snapshots are collected at batch edges.
        for (Condition* condition :
             {&plastic_phase, &plastic_open, &fixed_phase}) {
            SpikeEventBatch cue;
            run_batch(*condition, a_input, 4, cue);
            run_batch(*condition, silence, 4, cue);
            write_record(csv, *condition, "recall_cue", 0, cue);
            for (int window = 1; window <= 10; ++window) {
                SpikeEventBatch events;
                run_batch(*condition, silence, 8, events);
                write_record(csv, *condition, "recall", window, events);
            }
        }

        std::cout << "Associative/phase experiment completed. Data: "
                     "associative_phase_experiment.csv\n"
                  << "Compare plastic_phase with plastic_open_loop and "
                     "fixed_phase_control. B recall must exceed both the "
                     "untrained C response and the zero-STDP B response; "
                     "phase adaptation requires a reproducible change in "
                     "routing-related activity, not coherence alone.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Associative/phase experiment error: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
