#include "runtime/simulation_controller.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace spiking;

constexpr std::uint32_t kSource = 0;
constexpr std::uint32_t kTarget = 1;
constexpr std::uint32_t kTrainingPairs = 20;
constexpr std::uint32_t kPairIntervalSteps = 1000;
constexpr std::uint32_t kObservationSteps = 5;
constexpr float kPulseAmplitude = 20.0f;

ModelParameters make_parameters() {
    ModelParameters p;
    p.simulation.excitatory_count = 2;
    p.simulation.inhibitory_count = 1;
    p.simulation.steps = 1;
    p.simulation.dt = 0.1f;
    p.simulation.threshold = -64.99999f;
    p.simulation.random_seed = 24680;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;

    p.topology.p_ee = 1.0;
    p.topology.p_ei = p.topology.p_ie = p.topology.p_ii = 0.0;
    p.topology.master_seed = 13579;
    p.topology.topology_seed = 97531;
    p.topology.delay_seed = 86420;
    p.topology.delay_distribution = DelayDistribution::Uniform;
    p.topology.minimum_delay = p.topology.maximum_delay = 2;
    p.topology.fibonacci_delays = {2};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

std::uint32_t find_edge(const CSRTopology& graph,
                        std::uint32_t source, std::uint32_t target) {
    for (std::uint32_t edge = graph.row_offsets.at(source);
         edge < graph.row_offsets.at(source + 1); ++edge) {
        if (graph.target_indices.at(edge) == target) return edge;
    }
    throw std::runtime_error("Expected source-to-target CSR edge is missing");
}

struct Response {
    float target_voltage = 0.0f;
    std::uint64_t source_spikes = 0;
    std::uint64_t target_spikes = 0;
};

void advance_in_small_chunks(SimulationController& controller,
                             std::uint32_t steps) {
    // Keep every runtime call below the CUDA-graph capture threshold (100).
    // This experiment is intended to validate the explicit step path first.
    while (steps != 0) {
        const auto chunk = std::min<std::uint32_t>(steps, 99);
        controller.step(chunk);
        steps -= chunk;
    }
}

Response observe_source_only(SimulationController& controller) {
    const auto before = controller.spike_events_snapshot(0, 100000);
    const auto start = controller.telemetry().completed_steps + 1;
    controller.schedule_stimulus(kSource, start, 1, kPulseAmplitude);
    controller.step(kObservationSteps);

    Response result;
    result.target_voltage =
        controller.collect_voltage_samples({kTarget}).at(0).voltage_mv;
    const auto after = controller.spike_events_snapshot(0, 100000);
    for (const auto& event : after.events) {
        if (event.sequence < before.total_events) continue;
        if (event.neuron_id == kSource) ++result.source_spikes;
        if (event.neuron_id == kTarget) ++result.target_spikes;
    }
    return result;
}

std::pair<std::uint64_t, std::uint64_t> count_events(
    const SpikeEventSnapshot& snapshot, std::uint64_t first_sequence) {
    std::pair<std::uint64_t, std::uint64_t> counts{};
    for (const auto& event : snapshot.events) {
        if (event.sequence < first_sequence) continue;
        if (event.neuron_id == kSource) ++counts.first;
        if (event.neuron_id == kTarget) ++counts.second;
    }
    return counts;
}
} // namespace

int main() {
    try {
        auto parameters = make_parameters();
        StdpParameters stdp;
        stdp.potentiation_rate = 0.01f;
        stdp.depression_rate = 0.00525f;
        stdp.presynaptic_tau_ms = 20.0f;
        stdp.postsynaptic_tau_ms = 20.0f;
        stdp.minimum_weight = 0.001f;
        stdp.maximum_weight = 1.0f;

        SimulationController controller(
            parameters, 0, 1, 1000, true, true, stdp);

        const auto initial_graph = controller.connectivity_snapshot();
        const auto edge = find_edge(initial_graph, kSource, kTarget);
        const float initial_weight = initial_graph.weights.at(edge);

        const auto baseline = observe_source_only(controller);
        if (baseline.source_spikes == 0)
            throw std::runtime_error("Baseline A-only pulse produced no A spike");

        // Allow five STDP time constants at dt=0.1 ms, advanced without graph capture.
        advance_in_small_chunks(controller, 1000);

        const auto training_first_sequence =
            controller.spike_events_snapshot(0, 100000).total_events;
        for (std::uint32_t trial = 0; trial < kTrainingPairs; ++trial) {
            // Explicitly execute A, then B three integration steps later.
            // The A->B synaptic delay is two steps.
            auto source_step = controller.telemetry().completed_steps + 1;
            controller.schedule_stimulus(
                kSource, source_step, 1, kPulseAmplitude);
            controller.step(3);

            const auto target_step = controller.telemetry().completed_steps + 1;
            controller.schedule_stimulus(
                kTarget, target_step, 1, kPulseAmplitude);
            controller.step(3);

            // Complete a 1000-step (100 ms) inter-pair interval. This gives
            // regular-spiking neurons time to recover after each forced spike,
            // while STDP traces still evolve naturally.
            advance_in_small_chunks(controller, kPairIntervalSteps - 6);
        }

        const auto training_events =
            controller.spike_events_snapshot(0, 100000);
        const auto training_counts =
            count_events(training_events, training_first_sequence);
        const auto trained_graph = controller.connectivity_snapshot();
        const float trained_weight = trained_graph.weights.at(edge);

        // Clear residual traces by decay only; do not reinitialize the network.
        advance_in_small_chunks(controller, 1000);

        const auto recall = observe_source_only(controller);
        const auto all_events = controller.spike_events_snapshot(0, 100000);
        const auto final_graph = controller.connectivity_snapshot();
        const float final_weight = final_graph.weights.at(edge);
        const auto config = controller.configuration_snapshot();

        const float response_delta =
            recall.target_voltage - baseline.target_voltage;

        std::cout << std::fixed << std::setprecision(7)
                  << "STDP associative-recall diagnostic\n"
                  << "Simulation generation: " << config.simulation_generation << '\n'
                  << "A->B initial weight: " << initial_weight << '\n'
                  << "A->B after training: " << trained_weight << '\n'
                  << "A->B after recall: " << final_weight << '\n'
                  << "Training pairs: " << kTrainingPairs << '\n'
                  << "Training A spikes: " << training_counts.first << '\n'
                  << "Training B spikes: " << training_counts.second << '\n'
                  << "Baseline A/B spikes: " << baseline.source_spikes
                  << '/' << baseline.target_spikes << '\n'
                  << "Recall A/B spikes: " << recall.source_spikes
                  << '/' << recall.target_spikes << '\n'
                  << "Baseline B voltage: " << baseline.target_voltage << " mV\n"
                  << "Recall B voltage: " << recall.target_voltage << " mV\n"
                  << "B voltage delta: " << response_delta << " mV\n"
                  << "Total recorded events: " << all_events.total_events << '\n'
                  << "Dropped events: " << all_events.dropped_events << '\n';

        if (config.simulation_generation != 0)
            throw std::runtime_error("Simulation unexpectedly reinitialized");
        if (all_events.dropped_events != 0)
            throw std::runtime_error("Spike-event recorder dropped events");
        if (training_counts.first < kTrainingPairs ||
            training_counts.second < kTrainingPairs)
            throw std::runtime_error(
                "Training did not produce the expected A and B spike pairs; "
                "inspect the counts above");
        if (trained_weight <= initial_weight)
            throw std::runtime_error(
                "A/B spikes occurred but STDP did not potentiate A->B");
        if (recall.source_spikes == 0)
            throw std::runtime_error("A-only recall pulse produced no A spike");
        if (response_delta <= 1.0e-6f)
            throw std::runtime_error(
                "Weight potentiated, but B voltage did not improve on recall");

        std::cout << "PASS: causal pairing potentiated A->B and improved "
                     "the A-only target response.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "STDP associative-recall experiment failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
