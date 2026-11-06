#include "runtime/cuda_simulation_runtime.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace spiking;

struct RunResult {
    float target_voltage_after_delivery = 0.0f;
    float initial_weight = 0.0f;
    float final_weight = 0.0f;
    SpikeEventBatch events;
    std::uint64_t state_hash = 0;
};

ModelParameters make_parameters(bool connected) {
    ModelParameters p;
    p.simulation.excitatory_count = 2;
    p.simulation.inhibitory_count = 1;
    p.simulation.steps = 6;
    p.simulation.dt = 0.01f;
    p.simulation.threshold = -64.99999f;
    p.simulation.random_seed = 24680;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;

    p.topology.p_ee = connected ? 1.0 : 0.0;
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
        if (graph.target_indices.at(edge) == target)
            return edge;
    }
    throw std::runtime_error("Expected seeded connection was not generated");
}

RunResult run_connected_network() {
    auto parameters = make_parameters(true);
    StdpParameters learning;
    learning.potentiation_rate = 0.05f;

    CudaSimulationRuntime runtime(parameters, 0, true, true, learning);

    const auto graph = runtime.collect_connectivity();
    const auto edge = find_edge(graph, 0, 1);
    const float initial_weight = graph.weights.at(edge);

    // Source neuron spikes during integration step 1.
    runtime.schedule_stimulus(0, 1, 1, 20.0f);
    runtime.run_steps(3);

    const auto target_sample = runtime.collect_voltage_samples({1});
    const float target_voltage = target_sample.at(0).voltage_mv;

    // Postsynaptic neuron spikes later, allowing the earlier presynaptic
    // trace to potentiate the 0 -> 1 edge on the following STDP update.
    runtime.schedule_stimulus(1, 4, 1, 20.0f);
    runtime.run_steps(3);

    const auto final_graph = runtime.collect_connectivity();
    const auto events = runtime.collect_spike_events();
    const auto metrics = runtime.collect_metrics();

    if (!metrics.finite_state || metrics.completed_steps != 6)
        throw std::runtime_error("Combined simulation did not complete cleanly");

    return {
        target_voltage,
        initial_weight,
        final_graph.weights.at(edge),
        events,
        runtime.state_fingerprint()
    };
}

float run_disconnected_control() {
    auto parameters = make_parameters(false);
    CudaSimulationRuntime runtime(parameters, 0, false, false);

    runtime.schedule_stimulus(0, 1, 1, 20.0f);
    runtime.run_steps(3);

    return runtime.collect_voltage_samples({1}).at(0).voltage_mv;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        const auto first = run_connected_network();
        const auto second = run_connected_network();
        const float disconnected_voltage = run_disconnected_control();

        require(first.initial_weight > 0.0f,
                "Initial excitatory synaptic weight is not positive");
        require(first.final_weight > first.initial_weight,
                "STDP did not potentiate the expected excitatory connection");
        require(std::abs(first.target_voltage_after_delivery -
                         disconnected_voltage) > 1.0e-7f,
                "Delayed synaptic transmission did not affect the target voltage");

        const auto has_spike = [](const SpikeEventBatch& batch,
                                  std::uint32_t neuron) {
            return std::any_of(batch.events.begin(), batch.events.end(),
                [neuron](const SpikeEvent& event) {
                    return event.neuron_id == neuron;
                });
        };

        require(has_spike(first.events, 0),
                "Externally stimulated presynaptic neuron did not spike");
        require(has_spike(first.events, 1),
                "Externally stimulated postsynaptic neuron did not spike");
        require(first.events.dropped_events == 0,
                "Spike event recorder dropped events");

        require(first.events.events.size() == second.events.events.size(),
                "Repeated seeded runs produced different event counts");
        for (std::size_t i = 0; i < first.events.events.size(); ++i) {
            require(first.events.events[i].step == second.events.events[i].step &&
                    first.events.events[i].neuron_id ==
                        second.events.events[i].neuron_id,
                    "Repeated seeded runs produced different spike events");
        }

        require(first.initial_weight == second.initial_weight &&
                first.final_weight == second.final_weight,
                "Repeated seeded runs produced different synaptic weights");
        require(first.state_hash == second.state_hash,
                "Repeated seeded runs produced different final states");

        std::cout << "Network integration test passed\n"
                  << "Initial 0->1 weight: " << first.initial_weight << '\n'
                  << "Final 0->1 weight: " << first.final_weight << '\n'
                  << "Target voltage with connection: "
                  << first.target_voltage_after_delivery << " mV\n"
                  << "Target voltage without connection: "
                  << disconnected_voltage << " mV\n"
                  << "Recorded spike events: " << first.events.events.size()
                  << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Network integration test failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
