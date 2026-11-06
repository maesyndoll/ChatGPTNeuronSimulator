#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using namespace spiking;
    try {
        // Deterministic one-step spike test.
        ModelParameters spike_parameters;
        spike_parameters.simulation.excitatory_count = 1;
        spike_parameters.simulation.inhibitory_count = 1;
        spike_parameters.simulation.steps = 1;
        spike_parameters.simulation.dt = 0.001f;
        spike_parameters.simulation.threshold = -64.9995f;
        spike_parameters.simulation.initial_voltage_min = -65.0f;
        spike_parameters.simulation.initial_voltage_max = -65.0f;
        spike_parameters.simulation.excitatory_current_min = 4.0f;
        spike_parameters.simulation.excitatory_current_max = 4.0f;
        spike_parameters.simulation.inhibitory_current_min = 4.0f;
        spike_parameters.simulation.inhibitory_current_max = 4.0f;

        spike_parameters.topology.p_ee = 0.0;
        spike_parameters.topology.p_ei = 1.0;
        spike_parameters.topology.p_ie = 1.0;
        spike_parameters.topology.p_ii = 0.0;

        // Use sequential inhibitory timing only for the STDP tests.
        ModelParameters learning_parameters = spike_parameters;
        learning_parameters.simulation.inhibitory_current_min = 3.4f;
        learning_parameters.simulation.inhibitory_current_max = 3.4f;

        CudaSimulationRuntime spike_runtime(spike_parameters);
        if (spike_runtime.plasticity_state_enabled()) {
            std::cerr << "Plasticity support must remain disabled by default\\n";
            return EXIT_FAILURE;
        }
        CudaSimulationRuntime learning_state_runtime(
            learning_parameters, 0, true, true);
        CudaSimulationRuntime disabled_stdp_runtime(learning_parameters);
        if (!learning_state_runtime.plasticity_state_enabled()) {
            std::cerr << "Opt-in plasticity state allocation failed\\n";
            return EXIT_FAILURE;
        }
        const auto learning_initial_topology =
            learning_state_runtime.collect_connectivity();
        const auto disabled_initial_topology =
            disabled_stdp_runtime.collect_connectivity();

        learning_state_runtime.run_steps(2);
        learning_state_runtime.run_steps(100); // Exercise STDP inside CUDA Graph capture/replay.
        disabled_stdp_runtime.run_steps(102);

        const auto learning_state_metrics =
            learning_state_runtime.collect_metrics();
        if (!learning_state_metrics.finite_state ||
            learning_state_metrics.completed_steps != 102) {
            std::cerr << "Opt-in GPU STDP changed simulation validity\\n";
            return EXIT_FAILURE;
        }

        const auto learning_final_topology =
            learning_state_runtime.collect_connectivity();
        const auto disabled_final_topology =
            disabled_stdp_runtime.collect_connectivity();
        if (learning_initial_topology.weights.size() != 2 ||
            learning_final_topology.weights.size() != 2 ||
            disabled_initial_topology.weights != disabled_final_topology.weights) {
            std::cerr << "Default-off STDP modified weights or test topology changed\\n";
            return EXIT_FAILURE;
        }
        if (learning_final_topology.weights[0] ==
                learning_initial_topology.weights[0] ||
            learning_final_topology.weights[1] !=
                learning_initial_topology.weights[1]) {
            std::cerr << "Runtime STDP did not modify only the excitatory-source weight\\n";
            return EXIT_FAILURE;
        }

        // Export must read the exact graph allocated on the GPU, preserving
        // source-oriented CSR offsets, targets, weights, and delays.
        const auto exported_topology =
            disabled_stdp_runtime.collect_connectivity();
        if (exported_topology.excitatory_count != 1 ||
            exported_topology.inhibitory_count != 1 ||
            exported_topology.synapse_count() != 2 ||
            exported_topology.row_offsets != std::vector<std::uint32_t>({0, 1, 2}) ||
            exported_topology.target_indices != std::vector<std::uint32_t>({1, 0}) ||
            exported_topology.weights.size() != 2 ||
            exported_topology.delays.size() != 2) {
            std::cerr << "GPU-resident CSR connectivity readback test failed\n";
            return EXIT_FAILURE;
        }
        try {
            exported_topology.validate();
        } catch (const std::exception& error) {
            std::cerr << "Exported GPU connectivity failed validation: "
                      << error.what() << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "GPU-resident CSR connectivity readback test passed\n";

        const auto initial_voltage_samples =
            spike_runtime.collect_voltage_samples({0U, 1U});
        if (initial_voltage_samples.size() != 2 ||
            initial_voltage_samples[0].step != 0 ||
            initial_voltage_samples[1].step != 0 ||
            initial_voltage_samples[0].time_ms != 0.0 ||
            std::abs(initial_voltage_samples[0].voltage_mv + 65.0f) > 1.0e-6f ||
            std::abs(initial_voltage_samples[1].voltage_mv + 65.0f) > 1.0e-6f ||
            std::abs(initial_voltage_samples[0].recovery + 13.0f) > 1.0e-6f ||
            std::abs(initial_voltage_samples[1].recovery + 13.0f) > 1.0e-6f ||
            initial_voltage_samples[0].spiked_this_step ||
            initial_voltage_samples[1].spiked_this_step) {
            std::cerr << "Initial neuron-state sampling test failed\n";
            return EXIT_FAILURE;
        }
        spike_runtime.run();
        const auto spike_metrics = spike_runtime.collect_metrics();

        if (!spike_metrics.finite_state ||
            spike_metrics.completed_steps != 1 ||
            spike_metrics.excitatory_spikes != 1 ||
            spike_metrics.inhibitory_spikes != 1 ||
            spike_metrics.last_step_excitatory_spikes != 1 ||
            spike_metrics.last_step_inhibitory_spikes != 1) {
            std::cerr << "Deterministic spike-flag test failed\n"
                      << "Cumulative excitatory: "
                      << spike_metrics.excitatory_spikes << '\n'
                      << "Cumulative inhibitory: "
                      << spike_metrics.inhibitory_spikes << '\n'
                      << "Last-step excitatory: "
                      << spike_metrics.last_step_excitatory_spikes << '\n'
                      << "Last-step inhibitory: "
                      << spike_metrics.last_step_inhibitory_spikes << '\n';
            return EXIT_FAILURE;
        }

        // Selected-neuron voltage sampling reports the completed step and
        // post-reset voltage for neurons that spiked during that step.
        const auto voltage_samples =
            spike_runtime.collect_voltage_samples({1U, 0U});
        if (voltage_samples.size() != 2 ||
            voltage_samples[0].step != 1 ||
            voltage_samples[1].step != 1 ||
            std::abs(voltage_samples[0].time_ms - 0.001) > 1.0e-8 ||
            std::abs(voltage_samples[1].time_ms - 0.001) > 1.0e-8 ||
            voltage_samples[0].neuron_id != 1 ||
            voltage_samples[1].neuron_id != 0 ||
            std::abs(voltage_samples[0].voltage_mv + 65.0f) > 1.0e-6f ||
            std::abs(voltage_samples[1].voltage_mv + 65.0f) > 1.0e-6f ||
            std::abs(voltage_samples[0].recovery + 11.0f) > 1.0e-4f ||
            std::abs(voltage_samples[1].recovery + 5.0f) > 1.0e-4f ||
            !voltage_samples[0].spiked_this_step ||
            !voltage_samples[1].spiked_this_step) {
            std::cerr << "Selected-neuron voltage/recovery sampling test failed\n";
            return EXIT_FAILURE;
        }

        bool rejected_invalid_voltage_id = false;
        try {
            (void)spike_runtime.collect_voltage_samples({2U});
        } catch (const std::out_of_range&) {
            rejected_invalid_voltage_id = true;
        }
        if (!rejected_invalid_voltage_id) {
            std::cerr << "Voltage sampling accepted an invalid neuron ID\n";
            return EXIT_FAILURE;
        }
        std::cout << "Selected-neuron voltage/recovery sampling test passed\n";

        // GPU spike-event recorder test: both neurons spike on step 1.
        const auto captured_events = spike_runtime.collect_spike_events();
        bool captured_excitatory = false;
        bool captured_inhibitory = false;
        for (const auto& event : captured_events.events) {
            if (event.step != 1) {
                std::cerr << "Spike recorder returned an incorrect step index\n";
                return EXIT_FAILURE;
            }
            if (event.neuron_id == 0) {
                captured_excitatory = true;
            } else if (event.neuron_id == 1) {
                captured_inhibitory = true;
            }
        }
        if (captured_events.events.size() != 2 ||
            captured_events.first_sequence != 0 ||
            captured_events.total_events != 2 ||
            captured_events.dropped_events != 0 ||
            !captured_excitatory || !captured_inhibitory) {
            std::cerr << "GPU spike-event capture test failed\n"
                      << "Captured: " << captured_events.events.size() << '\n'
                      << "Total: " << captured_events.total_events << '\n'
                      << "Dropped: " << captured_events.dropped_events << '\n';
            return EXIT_FAILURE;
        }

        const auto drained_events = spike_runtime.collect_spike_events();
        if (!drained_events.events.empty() ||
            drained_events.first_sequence != 2 ||
            drained_events.total_events != 2 ||
            drained_events.dropped_events != 0) {
            std::cerr << "Spike-event recorder cursor test failed\n";
            return EXIT_FAILURE;
        }
        std::cout << "GPU spike-event capture and drain tests passed\n";

        // Deterministic GPU synaptic propagation and delay test.
        // One excitatory source connects to one inhibitory target.
        // A delay of two means a step-1 spike first affects integration 3.
        auto make_propagation_parameters = [](bool connected) {
            ModelParameters p;
            p.simulation.excitatory_count = 1;
            p.simulation.inhibitory_count = 1;
            p.simulation.steps = 1;
            p.simulation.dt = 0.01f;
            p.simulation.threshold = -64.99999f;
            p.simulation.initial_voltage_min = -65.0f;
            p.simulation.initial_voltage_max = -65.0f;
            p.simulation.excitatory_current_min = 4.0f;
            p.simulation.excitatory_current_max = 4.0f;
            p.simulation.inhibitory_current_min = 0.0f;
            p.simulation.inhibitory_current_max = 0.0f;
            p.simulation.random_seed = 12345U;

            p.topology.p_ee = 0.0;
            p.topology.p_ei = connected ? 1.0 : 0.0;
            p.topology.p_ie = 0.0;
            p.topology.p_ii = 0.0;
            p.topology.topology_seed = 987654321ULL;
            p.topology.delay_seed = 123456789ULL;
            p.topology.delay_distribution = DelayDistribution::Uniform;
            p.topology.minimum_delay = 2;
            p.topology.maximum_delay = 2;
            p.topology.fibonacci_delays = {2};
            p.topology.fibonacci_weights = {1.0};
            return p;
        };

        CudaSimulationRuntime connected_runtime(
            make_propagation_parameters(true));
        CudaSimulationRuntime disconnected_runtime(
            make_propagation_parameters(false));

        connected_runtime.run_steps(1);
        disconnected_runtime.run_steps(1);

        const auto connected_step1 = connected_runtime.collect_metrics();
        const auto disconnected_step1 = disconnected_runtime.collect_metrics();

        if (connected_step1.last_step_excitatory_spikes != 1 ||
            std::abs(connected_step1.mean_voltage_mv -
                     disconnected_step1.mean_voltage_mv) > 1.0e-7) {
            std::cerr << "Synaptic delay test failed at step 1: "
                         "source spike or initial-state mismatch\n";
            return EXIT_FAILURE;
        }

        connected_runtime.run_steps(1);
        disconnected_runtime.run_steps(1);

        const auto connected_step2 = connected_runtime.collect_metrics();
        const auto disconnected_step2 = disconnected_runtime.collect_metrics();

        if (std::abs(connected_step2.mean_voltage_mv -
                     disconnected_step2.mean_voltage_mv) > 1.0e-7) {
            std::cerr << "Delay-2 synapse affected the target before step 3\n";
            return EXIT_FAILURE;
        }

        connected_runtime.run_steps(1);
        disconnected_runtime.run_steps(1);

        const auto connected_step3 = connected_runtime.collect_metrics();
        const auto disconnected_step3 = disconnected_runtime.collect_metrics();

        const double synaptic_voltage_difference =
            connected_step3.mean_voltage_mv -
            disconnected_step3.mean_voltage_mv;

        if (!connected_step3.finite_state ||
            !disconnected_step3.finite_state ||
            connected_step3.completed_steps != 3 ||
            disconnected_step3.completed_steps != 3 ||
            synaptic_voltage_difference <= 1.0e-7) {
            std::cerr << "GPU synaptic propagation or delay-2 delivery failed\n"
                      << "Connected mean voltage: "
                      << connected_step3.mean_voltage_mv << '\n'
                      << "Disconnected mean voltage: "
                      << disconnected_step3.mean_voltage_mv << '\n'
                      << "Difference: "
                      << synaptic_voltage_difference << '\n';
            return EXIT_FAILURE;
        }

        std::cout << "GPU synaptic delay test passed; "
                  << "step-3 mean voltage difference: "
                  << synaptic_voltage_difference << " mV\n";

        // Fan-in test: two excitatory sources converge on one inhibitory
        // target. Both sources must spike and their currents must accumulate.
        auto make_fan_in_parameters = [](bool connected) {
            ModelParameters p;
            p.simulation.excitatory_count = 2;
            p.simulation.inhibitory_count = 1;
            p.simulation.steps = 1;
            p.simulation.dt = 0.01f;
            p.simulation.threshold = -64.99999f;
            p.simulation.initial_voltage_min = -65.0f;
            p.simulation.initial_voltage_max = -65.0f;
            p.simulation.excitatory_current_min = 4.0f;
            p.simulation.excitatory_current_max = 4.0f;
            p.simulation.inhibitory_current_min = 0.0f;
            p.simulation.inhibitory_current_max = 0.0f;
            p.simulation.random_seed = 12345U;

            p.topology.p_ee = 0.0;
            p.topology.p_ei = connected ? 1.0 : 0.0;
            p.topology.p_ie = 0.0;
            p.topology.p_ii = 0.0;
            p.topology.topology_seed = 987654321ULL;
            p.topology.delay_seed = 123456789ULL;
            p.topology.delay_distribution = DelayDistribution::Uniform;
            p.topology.minimum_delay = 2;
            p.topology.maximum_delay = 2;
            p.topology.fibonacci_delays = {2};
            p.topology.fibonacci_weights = {1.0};
            return p;
        };

        CudaSimulationRuntime fan_in_connected(
            make_fan_in_parameters(true));
        CudaSimulationRuntime fan_in_disconnected(
            make_fan_in_parameters(false));

        fan_in_connected.run_steps(3);
        fan_in_disconnected.run_steps(3);

        const auto fan_in_result = fan_in_connected.collect_metrics();
        const auto fan_in_reference = fan_in_disconnected.collect_metrics();

        if (!fan_in_result.finite_state ||
            fan_in_result.completed_steps != 3 ||
            fan_in_result.last_step_excitatory_spikes > 2 ||
            fan_in_result.excitatory_spikes < 2 ||
            fan_in_result.mean_voltage_mv <=
                fan_in_reference.mean_voltage_mv + 1.0e-7) {
            std::cerr << "GPU fan-in accumulation test failed\n"
                      << "Connected mean voltage: "
                      << fan_in_result.mean_voltage_mv << '\n'
                      << "Disconnected mean voltage: "
                      << fan_in_reference.mean_voltage_mv << '\n'
                      << "Excitatory spikes: "
                      << fan_in_result.excitatory_spikes << '\n';
            return EXIT_FAILURE;
        }

        std::cout << "GPU fan-in accumulation test passed\n";

        // Inhibitory transmission test: an inhibitory source spikes and
        // sends a negative-weight signal to an excitatory target.
        auto make_inhibitory_parameters = [](bool connected) {
            ModelParameters p;
            p.simulation.excitatory_count = 1;
            p.simulation.inhibitory_count = 1;
            p.simulation.steps = 1;
            p.simulation.dt = 0.01f;
            p.simulation.threshold = -64.99999f;
            p.simulation.initial_voltage_min = -65.0f;
            p.simulation.initial_voltage_max = -65.0f;
            p.simulation.excitatory_current_min = 0.0f;
            p.simulation.excitatory_current_max = 0.0f;
            p.simulation.inhibitory_current_min = 4.0f;
            p.simulation.inhibitory_current_max = 4.0f;
            p.simulation.random_seed = 12345U;

            p.topology.p_ee = 0.0;
            p.topology.p_ei = 0.0;
            p.topology.p_ie = connected ? 1.0 : 0.0;
            p.topology.p_ii = 0.0;
            p.topology.topology_seed = 987654321ULL;
            p.topology.delay_seed = 123456789ULL;
            p.topology.delay_distribution = DelayDistribution::Uniform;
            p.topology.minimum_delay = 2;
            p.topology.maximum_delay = 2;
            p.topology.fibonacci_delays = {2};
            p.topology.fibonacci_weights = {1.0};
            return p;
        };

        CudaSimulationRuntime inhibitory_connected(
            make_inhibitory_parameters(true));
        CudaSimulationRuntime inhibitory_disconnected(
            make_inhibitory_parameters(false));

        inhibitory_connected.run_steps(1);
        inhibitory_disconnected.run_steps(1);
        const auto inhibitory_step1 =
            inhibitory_connected.collect_metrics();

        if (inhibitory_step1.last_step_inhibitory_spikes != 1) {
            std::cerr << "Inhibitory source did not spike\n";
            return EXIT_FAILURE;
        }

        inhibitory_connected.run_steps(2);
        inhibitory_disconnected.run_steps(2);

        const auto inhibitory_result =
            inhibitory_connected.collect_metrics();
        const auto inhibitory_reference =
            inhibitory_disconnected.collect_metrics();

        if (!inhibitory_result.finite_state ||
            inhibitory_result.completed_steps != 3 ||
            inhibitory_result.mean_voltage_mv >=
                inhibitory_reference.mean_voltage_mv - 1.0e-7) {
            std::cerr << "Negative inhibitory transmission test failed\n"
                      << "Connected mean voltage: "
                      << inhibitory_result.mean_voltage_mv << '\n'
                      << "Disconnected mean voltage: "
                      << inhibitory_reference.mean_voltage_mv << '\n';
            return EXIT_FAILURE;
        }

        std::cout << "GPU inhibitory transmission test passed\n";

        // Delay-ring reuse test. A delay-3 event is delivered at integration
        // step 4. Continue well beyond the four-slot ring length to exercise
        // repeated slot reuse after the original event has been consumed.
        auto make_ring_parameters = [](bool connected) {
            ModelParameters p;
            p.simulation.excitatory_count = 1;
            p.simulation.inhibitory_count = 1;
            p.simulation.steps = 1;
            p.simulation.dt = 0.01f;
            p.simulation.threshold = -64.99999f;
            p.simulation.initial_voltage_min = -65.0f;
            p.simulation.initial_voltage_max = -65.0f;
            p.simulation.excitatory_current_min = 4.0f;
            p.simulation.excitatory_current_max = 4.0f;
            p.simulation.inhibitory_current_min = 0.0f;
            p.simulation.inhibitory_current_max = 0.0f;
            p.simulation.random_seed = 12345U;

            p.topology.p_ee = 0.0;
            p.topology.p_ei = connected ? 1.0 : 0.0;
            p.topology.p_ie = 0.0;
            p.topology.p_ii = 0.0;
            p.topology.topology_seed = 987654321ULL;
            p.topology.delay_seed = 123456789ULL;
            p.topology.delay_distribution = DelayDistribution::Uniform;
            p.topology.minimum_delay = 3;
            p.topology.maximum_delay = 3;
            p.topology.fibonacci_delays = {3};
            p.topology.fibonacci_weights = {1.0};
            return p;
        };

        CudaSimulationRuntime ring_connected(make_ring_parameters(true));
        CudaSimulationRuntime ring_disconnected(make_ring_parameters(false));

        ring_connected.run_steps(3);
        ring_disconnected.run_steps(3);

        const auto ring_before_delivery = ring_connected.collect_metrics();
        const auto ring_reference_before =
            ring_disconnected.collect_metrics();

        if (std::abs(ring_before_delivery.mean_voltage_mv -
                     ring_reference_before.mean_voltage_mv) > 1.0e-7) {
            std::cerr << "Delay-ring event arrived before its configured delay\n";
            return EXIT_FAILURE;
        }

        ring_connected.run_steps(1);
        ring_disconnected.run_steps(1);

        const auto ring_delivery = ring_connected.collect_metrics();
        const auto ring_reference_delivery =
            ring_disconnected.collect_metrics();

        if (!ring_delivery.finite_state ||
            ring_delivery.completed_steps != 4 ||
            ring_delivery.mean_voltage_mv <=
                ring_reference_delivery.mean_voltage_mv + 1.0e-7) {
            std::cerr << "Delay-ring event was not delivered at step 4\n";
            return EXIT_FAILURE;
        }

        ring_connected.run_steps(8);
        ring_disconnected.run_steps(8);

        const auto ring_after_reuse = ring_connected.collect_metrics();
        const auto ring_reference_after =
            ring_disconnected.collect_metrics();

        if (!ring_after_reuse.finite_state ||
            !ring_reference_after.finite_state ||
            ring_after_reuse.completed_steps != 12 ||
            ring_reference_after.completed_steps != 12) {
            std::cerr << "Delay-ring reuse stability test failed\n";
            return EXIT_FAILURE;
        }

        std::cout << "GPU delay-ring reuse test passed\n";

        // Exercise graph capture after a non-graph prefix, repeated graph
        // replay, and a short non-graph tail. Event timestamps must remain
        // one-based and continuous across the graph boundary.
        ModelParameters graph_parameters;
        graph_parameters.simulation = SimulationParameters::small_test();
        graph_parameters.simulation.steps = 1;
        CudaSimulationRuntime graph_runtime(graph_parameters);
        graph_runtime.run_steps(99);
        graph_runtime.run_steps(206);
        const auto graph_metrics = graph_runtime.collect_metrics();
        if (!graph_metrics.finite_state ||
            graph_metrics.completed_steps != 305) {
            std::cerr << "CUDA graph batch step-count test failed\n";
            return EXIT_FAILURE;
        }
        const auto graph_events = graph_runtime.collect_spike_events();
        for (const auto& event : graph_events.events) {
            if (event.step == 0 || event.step > 305) {
                std::cerr << "CUDA graph spike-event step index failed\n";
                return EXIT_FAILURE;
            }
        }
        if (graph_events.total_events < graph_events.events.size() ||
            graph_events.dropped_events != 0) {
            std::cerr << "CUDA graph spike-event collection failed\n";
            return EXIT_FAILURE;
        }
        std::cout << "CUDA graph replay and event-index tests passed\n";

        // Runtime external-stimulation test. A one-step pulse starts at
        // integration step 2 and targets only the excitatory neuron.
        ModelParameters stimulus_parameters;
        stimulus_parameters.simulation.excitatory_count = 1;
        stimulus_parameters.simulation.inhibitory_count = 1;
        stimulus_parameters.simulation.steps = 2;
        stimulus_parameters.simulation.dt = 0.001f;
        stimulus_parameters.simulation.threshold = -64.99999f;
        stimulus_parameters.simulation.initial_voltage_min = -65.0f;
        stimulus_parameters.simulation.initial_voltage_max = -65.0f;
        stimulus_parameters.simulation.excitatory_current_min = 0.0f;
        stimulus_parameters.simulation.excitatory_current_max = 0.0f;
        stimulus_parameters.simulation.inhibitory_current_min = 0.0f;
        stimulus_parameters.simulation.inhibitory_current_max = 0.0f;
        stimulus_parameters.topology.p_ee = 0.0;
        stimulus_parameters.topology.p_ei = 0.0;
        stimulus_parameters.topology.p_ie = 0.0;
        stimulus_parameters.topology.p_ii = 0.0;

        CudaSimulationRuntime stimulus_runtime(stimulus_parameters);
        stimulus_runtime.schedule_stimulus(0U, 2U, 1U, 20.0f);
        stimulus_runtime.run_steps(1);

        const auto before_stimulus = stimulus_runtime.collect_metrics();
        if (!before_stimulus.finite_state ||
            before_stimulus.completed_steps != 1 ||
            before_stimulus.excitatory_spikes != 0 ||
            before_stimulus.inhibitory_spikes != 0) {
            std::cerr << "External stimulus activated before its start step\\n";
            return EXIT_FAILURE;
        }

        stimulus_runtime.run_steps(1);
        const auto during_stimulus = stimulus_runtime.collect_metrics();
        if (!during_stimulus.finite_state ||
            during_stimulus.completed_steps != 2 ||
            during_stimulus.last_step_excitatory_spikes != 1 ||
            during_stimulus.last_step_inhibitory_spikes != 0) {
            std::cerr << "Runtime external stimulus delivery or target isolation failed\\n"
                      << "Excitatory spikes: "
                      << during_stimulus.excitatory_spikes << '\n'
                      << "Inhibitory spikes: "
                      << during_stimulus.inhibitory_spikes << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "Runtime external-stimulation test passed\\n";

        ModelParameters parameters;
        parameters.simulation = SimulationParameters::small_test();
        parameters.simulation.steps = 2;

        CudaSimulationRuntime runtime(parameters);
        runtime.run();
        auto first = runtime.collect_metrics();
        if (!first.finite_state || first.completed_steps != 2) {
            std::cerr << "Initial CUDA runtime smoke test failed\n";
            return EXIT_FAILURE;
        }

        // Save the graph resident on the GPU, then load it into a new runtime.
        const std::string topology_file =
            "runtime_smoke_topology_roundtrip.csrbin";
        const auto original_graph = runtime.collect_connectivity();
        save_csr_topology(original_graph, topology_file);

        {
            CudaSimulationRuntime loaded_runtime(parameters, topology_file);
            const auto loaded_graph = loaded_runtime.collect_connectivity();

            if (loaded_graph.excitatory_count != original_graph.excitatory_count ||
                loaded_graph.inhibitory_count != original_graph.inhibitory_count ||
                loaded_graph.minimum_delay != original_graph.minimum_delay ||
                loaded_graph.maximum_delay != original_graph.maximum_delay ||
                loaded_graph.row_offsets != original_graph.row_offsets ||
                loaded_graph.target_indices != original_graph.target_indices ||
                loaded_graph.weights != original_graph.weights ||
                loaded_graph.delays != original_graph.delays) {
                std::cerr << "File-backed runtime changed saved CSR connectivity\\n";
                return EXIT_FAILURE;
            }

            // A file-backed runtime must reload the same graph on reinitialization.
            loaded_runtime.reinitialize_simulation();
            const auto reloaded_graph = loaded_runtime.collect_connectivity();

            if (reloaded_graph.row_offsets != original_graph.row_offsets ||
                reloaded_graph.target_indices != original_graph.target_indices ||
                reloaded_graph.weights != original_graph.weights ||
                reloaded_graph.delays != original_graph.delays) {
                std::cerr << "File-backed runtime changed topology on reinitialization\\n";
                return EXIT_FAILURE;
            }
        }

        if (std::remove(topology_file.c_str()) != 0) {
            std::cerr << "Could not remove topology round-trip test file\\n";
            return EXIT_FAILURE;
        }
        std::cout << "Runtime topology persistence test passed\\n";

        const auto first_last_step_spikes =
            first.last_step_excitatory_spikes +
            first.last_step_inhibitory_spikes;
        if (first_last_step_spikes > parameters.simulation.neuron_count()) {
            std::cerr << "Per-step spike flag count is invalid\n";
            return EXIT_FAILURE;
        }

        auto updated = runtime.requested_parameters();
        updated.simulation.dt = 0.25f;
        const auto update_result = runtime.apply_parameters(updated);
        if (!update_result.accepted || update_result.requires_reinitialization) {
            std::cerr << "Step-boundary parameter update failed\n";
            return EXIT_FAILURE;
        }
        runtime.run_steps(2);
        const auto second = runtime.collect_metrics();
        if (!second.finite_state || second.completed_steps != 4) {
            std::cerr << "Runtime continuation after parameter update failed\n";
            return EXIT_FAILURE;
        }

        updated = runtime.requested_parameters();
        updated.simulation.excitatory_count /= 2;
        if (!runtime.apply_parameters(updated).requires_reinitialization) {
            std::cerr << "Structural parameter change was not staged\n";
            return EXIT_FAILURE;
        }
        runtime.reinitialize_simulation();
        const auto reinitialized = runtime.collect_metrics();
        if (runtime.simulation_generation() != 1 ||
            reinitialized.completed_steps != 0 || !reinitialized.finite_state) {
            std::cerr << "Transactional full reinitialization failed\n";
            return EXIT_FAILURE;
        }

        if (reinitialized.last_step_excitatory_spikes != 0 ||
            reinitialized.last_step_inhibitory_spikes != 0) {
            std::cerr << "Spike flags were not cleared by reinitialization\n";
            return EXIT_FAILURE;
        }

        std::cout << "CUDA runtime smoke test passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "CUDA runtime smoke test error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
