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
constexpr std::uint64_t kPairIntervalSteps = 1000;
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
    p.topology.p_ei = 0.0;
    p.topology.p_ie = 0.0;
    p.topology.p_ii = 0.0;
    p.topology.master_seed = 13579;
    p.topology.topology_seed = 97531;
    p.topology.delay_seed = 86420;
    p.topology.delay_distribution = DelayDistribution::Uniform;
    p.topology.minimum_delay = 2;
    p.topology.maximum_delay = 2;
    p.topology.fibonacci_delays = {2};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

std::uint32_t find_edge(const CSRTopology& graph,
                        std::uint32_t source,
                        std::uint32_t target) {
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

Response observe_source_only(SimulationController& controller,
                              std::uint64_t start_step) {
    const auto before = controller.spike_events_snapshot(0, 100000);
    controller.schedule_stimulus(
        kSource, start_step, 1, kPulseAmplitude);
    controller.step(kObservationSteps);

    const auto samples = controller.collect_voltage_samples({kTarget});
    const auto after = controller.spike_events_snapshot(0, 100000);

    Response result;
    result.target_voltage = samples.at(0).voltage_mv;
    for (const auto& event : after.events) {
        if (event.sequence < before.total_events) continue;
        if (event.neuron_id == kSource) ++result.source_spikes;
        if (event.neuron_id == kTarget) ++result.target_spikes;
    }
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
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

        // A small batch size makes each explicit step boundary observable.
        SimulationController controller(
            parameters, 0, 1, 1000, true, true, stdp);

        const auto initial_graph = controller.connectivity_snapshot();
        const auto edge = find_edge(initial_graph, kSource, kTarget);
        const float initial_weight = initial_graph.weights.at(edge);

        // Untrained A-only response. B receives no direct stimulus.
        const auto baseline = observe_source_only(controller, 1);
        require(baseline.source_spikes > 0,
                "Baseline source pulse did not produce an A spike");

        // Let the baseline presynaptic trace decay before training.
        controller.step(1000);

        const auto training_start =
            controller.telemetry().completed_steps + 2;
        for (std::uint32_t trial = 0; trial < kTrainingPairs; ++trial) {
            const auto source_step =
                training_start + trial * kPairIntervalSteps;
            // A precedes B by three integration steps; the A->B synaptic
            // delay is two steps. This is a causal pre-before-post pairing.
            controller.schedule_stimulus(
                kSource, source_step, 1, kPulseAmplitude);
            controller.schedule_stimulus(
                kTarget, source_step + 3, 1, kPulseAmplitude);
        }

        const auto final_target_step =
            training_start + (kTrainingPairs - 1) * kPairIntervalSteps + 3;
        const auto current_step = controller.telemetry().completed_steps;
        const auto training_steps =
            static_cast<std::uint32_t>(final_target_step + 2 - current_step);
        controller.step(training_steps);

        const auto trained_graph = controller.connectivity_snapshot();
        const float trained_weight = trained_graph.weights.at(edge);

        // More than five STDP time constants: retain weights, clear traces
        // through natural decay, and avoid a residual training association.
        controller.step(1000);

        const auto recall_start = controller.telemetry().completed_steps + 1;
        const auto recall = observe_source_only(controller, recall_start);

        const auto events = controller.spike_events_snapshot(0, 100000);
        const auto final_graph = controller.connectivity_snapshot();
        const float final_weight = final_graph.weights.at(edge);
        const auto config = controller.configuration_snapshot();

        require(config.simulation_generation == 0,
                "Unexpected simulation reinitialization during experiment");
        require(events.dropped_events == 0,
                "Spike-event recorder dropped events");
        require(trained_weight > initial_weight,
                "Paired A-before-B training did not potentiate A->B");
        require(recall.source_spikes > 0,
                "A-only recall pulse did not produce an A spike");

        const float response_delta =
            recall.target_voltage - baseline.target_voltage;
        require(response_delta > 1.0e-6f,
                "Potentiated A->B connection did not measurably improve "
                "the target response to A-only stimulation");

        std::cout << std::fixed << std::setprecision(7)
                  << "STDP associative-recall experiment\n"
                  << "GPU/controller generation: "
                  << config.simulation_generation << '\n'
                  << "A->B initial weight: " << initial_weight << '\n'
                  << "A->B after paired training: " << trained_weight << '\n'
                  << "A->B after recall: " << final_weight << '\n'
                  << "Training pairs: " << kTrainingPairs << '\n'
                  << "Baseline A-only B voltage: "
                  << baseline.target_voltage << " mV\n"
                  << "Trained A-only B voltage: "
                  << recall.target_voltage << " mV\n"
                  << "A-only B voltage change: "
                  << response_delta << " mV\n"
                  << "Baseline A/B spikes in observation: "
                  << baseline.source_spikes << '/'
                  << baseline.target_spikes << '\n'
                  << "Recall A/B spikes in observation: "
                  << recall.source_spikes << '/'
                  << recall.target_spikes << '\n'
                  << "Total recorded spike events: "
                  << events.total_events << '\n'
                  << "Dropped spike events: " << events.dropped_events
                  << '\n';

        if (response_delta > 0.0f) {
            std::cout << "RESULT: learned A->B efficacy increased B's "
                         "response to A-only stimulation.\n";
        } else {
            std::cout << "RESULT: STDP potentiated A->B, but this protocol "
                         "did not produce a larger measured A-only B voltage.\n";
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "STDP associative-recall experiment failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
