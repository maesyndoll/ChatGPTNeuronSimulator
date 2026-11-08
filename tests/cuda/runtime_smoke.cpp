#include "runtime/cuda_simulation_runtime.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
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

        CudaSimulationRuntime spike_runtime(spike_parameters);
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
            p.simulation.threshold = -64.9995f;
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
            p.simulation.threshold = -64.9995f;
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
            p.simulation.threshold = -64.9995f;
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
            p.simulation.threshold = -64.9995f;
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

        // Phase-gated meta-topology experiment. One source projects to two
        // otherwise identical targets. The fixed phases make one route aligned
        // (gain 1) and the other anti-phase (gain 0) at full modulation.
        auto make_meta_topology_parameters = [] {
            ModelParameters p;
            p.simulation.excitatory_count = 3;
            p.simulation.inhibitory_count = 1;
            p.simulation.steps = 1;
            p.simulation.dt = 0.01f;
            p.simulation.threshold = -64.9995f;
            p.simulation.initial_voltage_min = -65.0f;
            p.simulation.initial_voltage_max = -65.0f;
            p.simulation.excitatory_current_min = 4.0f;
            p.simulation.excitatory_current_max = 4.0f;
            p.simulation.inhibitory_current_min = 0.0f;
            p.simulation.inhibitory_current_max = 0.0f;
            p.simulation.random_seed = 12345U;
            p.topology.p_ee = 0.0;
            p.topology.p_ei = 0.0;
            p.topology.p_ie = 0.0;
            p.topology.p_ii = 0.0;
            p.topology.minimum_delay = 1;
            p.topology.maximum_delay = 1;
            p.topology.fibonacci_delays = {1};
            p.topology.fibonacci_weights = {1.0};
            return p;
        };
        auto make_meta_topology = [] {
            CSRTopology graph;
            graph.excitatory_count = 3;
            graph.inhibitory_count = 1;
            graph.minimum_delay = 1;
            graph.maximum_delay = 1;
            graph.row_offsets = {0, 2, 2, 2, 2};
            graph.target_indices = {1, 2};
            graph.weights = {4.0f, 4.0f};
            graph.delays = {1, 1};
            graph.validate();
            return graph;
        };
        CudaSimulationRuntime meta_unmodulated(
            make_meta_topology_parameters(), make_meta_topology());
        CudaSimulationRuntime meta_gated(
            make_meta_topology_parameters(), make_meta_topology());
        std::vector<float> meta_input(4, 0.0f);
        meta_input[0] = 4.0f;
        meta_unmodulated.set_external_current(meta_input);
        meta_gated.set_external_current(meta_input);
        const std::vector<float> aligned_and_opposed_phases{
            0.0f, 0.0f, 3.14159265358979323846f, 0.0f};
        meta_unmodulated.set_neuron_phases(aligned_and_opposed_phases);
        meta_gated.set_neuron_phases(aligned_and_opposed_phases);
        PhaseDynamicsConfig meta_config;
        meta_config.enabled = true;
        meta_config.coupling_strength = 0.0f;
        meta_config.modulation_strength = 1.0f;
        meta_config.evolve_phases = false;
        meta_gated.configure_phase_dynamics(meta_config);
        meta_unmodulated.run_steps(4);
        meta_gated.run_steps(4);
        const auto meta_base_voltage = meta_unmodulated.collect_neuron_voltages();
        const auto meta_gated_voltage = meta_gated.collect_neuron_voltages();
        const auto meta_base_spikes = meta_unmodulated.collect_spike_events();
        const auto meta_gated_spikes = meta_gated.collect_spike_events();
        const float base_target_delta = std::abs(
            meta_base_voltage[1] - meta_base_voltage[2]);
        const float gated_target_delta =
            meta_gated_voltage[1] - meta_gated_voltage[2];
        const auto source_spiked = [](const SpikeEventBatch& events) {
            for (const auto& event : events.events)
                if (event.neuron_id == 0) return true;
            return false;
        };
        if (!source_spiked(meta_base_spikes) ||
            !source_spiked(meta_gated_spikes) ||
            base_target_delta > 1.0e-5f ||
            gated_target_delta < 1.0e-4f ||
            meta_gated_voltage[1] <= meta_gated_voltage[2]) {
            std::cerr << "GPU phase-gated meta-topology experiment failed\\n"
                      << "Ungated target voltage delta: " << base_target_delta << " mV\\n"
                      << "Aligned target voltage: " << meta_gated_voltage[1] << " mV\\n"
                      << "Anti-phase target voltage: " << meta_gated_voltage[2] << " mV\\n"
                      << "Gated target separation: " << gated_target_delta << " mV\\n";
            return EXIT_FAILURE;
        }
        std::cout << "GPU phase-gated meta-topology test passed; "
                  << "ungated target delta=" << base_target_delta
                  << " mV, aligned=" << meta_gated_voltage[1]
                  << " mV, anti-phase=" << meta_gated_voltage[2]
                  << " mV\\n";

        // Phase dynamics test: the same deterministic delayed connection
        // must remain excitatory, while phase gating attenuates its delivered
        // current relative to the unmodulated baseline.
        CudaSimulationRuntime phase_baseline(make_propagation_parameters(true));
        CudaSimulationRuntime phase_gated(make_propagation_parameters(true));
        CudaSimulationRuntime phase_disconnected(make_propagation_parameters(false));
        CudaSimulationRuntime phase_uncoupled(make_propagation_parameters(true));
        PhaseDynamicsConfig phase_config;
        phase_config.enabled = true;
        phase_config.coupling_strength = 50.0f;
        phase_config.modulation_strength = 1.0f;
        phase_gated.configure_phase_dynamics(phase_config);
        auto uncoupled_config = phase_config;
        uncoupled_config.coupling_strength = 0.0f;
        phase_uncoupled.configure_phase_dynamics(uncoupled_config);
        phase_baseline.run_steps(3);
        phase_gated.run_steps(3);
        phase_disconnected.run_steps(3);
        phase_uncoupled.run_steps(3);
        const auto baseline_phase_metrics = phase_baseline.collect_metrics();
        const auto gated_phase_metrics = phase_gated.collect_metrics();
        const auto disconnected_phase_metrics = phase_disconnected.collect_metrics();
        const auto uncoupled_phase_metrics = phase_uncoupled.collect_metrics();
        if (!baseline_phase_metrics.finite_state ||
            !gated_phase_metrics.finite_state ||
            !disconnected_phase_metrics.finite_state ||
            !uncoupled_phase_metrics.finite_state ||
            baseline_phase_metrics.completed_steps != 3 ||
            gated_phase_metrics.completed_steps != 3 ||
            disconnected_phase_metrics.completed_steps != 3 ||
            uncoupled_phase_metrics.completed_steps != 3 ||
            gated_phase_metrics.mean_voltage_mv <=
                disconnected_phase_metrics.mean_voltage_mv + 1.0e-7 ||
            gated_phase_metrics.mean_voltage_mv <=
                uncoupled_phase_metrics.mean_voltage_mv + 1.0e-7 ||
            gated_phase_metrics.mean_voltage_mv >=
                baseline_phase_metrics.mean_voltage_mv - 1.0e-7) {
            std::cerr << "GPU phase-gated synaptic routing test failed\n"
                      << "Baseline mean voltage: "
                      << baseline_phase_metrics.mean_voltage_mv << '\n'
                      << "Phase-gated mean voltage: "
                      << gated_phase_metrics.mean_voltage_mv << '\n';
            return EXIT_FAILURE;
        }

        bool rejected_invalid_phase_config = false;
        try {
            PhaseDynamicsConfig invalid_phase_config;
            invalid_phase_config.enabled = true;
            invalid_phase_config.modulation_strength = 1.5f;
            phase_gated.configure_phase_dynamics(invalid_phase_config);
        } catch (const std::invalid_argument&) {
            rejected_invalid_phase_config = true;
        }
        if (!rejected_invalid_phase_config) {
            std::cerr << "Invalid phase modulation configuration was accepted\n";
            return EXIT_FAILURE;
        }

        // A fully connected excitatory population should synchronize under
        // strong positive coupling. Controls share identical initial phases.
        auto make_coherence_parameters = [] {
            ModelParameters p;
            p.simulation.excitatory_count = 48;
            p.simulation.inhibitory_count = 1;
            p.simulation.steps = 1;
            p.simulation.dt = 0.01f;
            p.simulation.threshold = 100.0f;
            p.simulation.initial_voltage_min = -65.0f;
            p.simulation.initial_voltage_max = -65.0f;
            p.simulation.excitatory_current_min = 0.0f;
            p.simulation.excitatory_current_max = 0.0f;
            p.simulation.inhibitory_current_min = 0.0f;
            p.simulation.inhibitory_current_max = 0.0f;
            p.simulation.random_seed = 424242U;
            p.topology.p_ee = 1.0;
            p.topology.p_ei = 0.0;
            p.topology.p_ie = 0.0;
            p.topology.p_ii = 0.0;
            p.topology.topology_seed = 10101ULL;
            p.topology.delay_seed = 20202ULL;
            p.topology.delay_distribution = DelayDistribution::Uniform;
            p.topology.minimum_delay = 1;
            p.topology.maximum_delay = 1;
            p.topology.fibonacci_delays = {1};
            p.topology.fibonacci_weights = {1.0};
            return p;
        };
        CudaSimulationRuntime coherence_coupled_a(make_coherence_parameters());
        CudaSimulationRuntime coherence_coupled_b(make_coherence_parameters());
        CudaSimulationRuntime coherence_uncoupled(make_coherence_parameters());
        PhaseDynamicsConfig coherence_config;
        coherence_config.enabled = true;
        coherence_config.coupling_strength = 50.0f;
        coherence_config.modulation_strength = 0.0f;
        coherence_coupled_a.configure_phase_dynamics(coherence_config);
        coherence_coupled_b.configure_phase_dynamics(coherence_config);
        auto no_coupling_config = coherence_config;
        no_coupling_config.coupling_strength = 0.0f;
        coherence_uncoupled.configure_phase_dynamics(no_coupling_config);
        coherence_coupled_a.run_steps(1000);
        coherence_coupled_b.run_steps(1000);
        coherence_uncoupled.run_steps(1000);
        const auto coherence_a = coherence_coupled_a.collect_metrics();
        const auto coherence_b = coherence_coupled_b.collect_metrics();
        const auto coherence_control = coherence_uncoupled.collect_metrics();
        if (!coherence_a.finite_state || !coherence_b.finite_state ||
            !coherence_control.finite_state ||
            coherence_a.phase_coherence < 0.0 ||
            coherence_a.phase_coherence > 1.000001 ||
            coherence_a.phase_coherence <= coherence_control.phase_coherence + 0.1 ||
            std::abs(coherence_a.phase_coherence -
                     coherence_b.phase_coherence) > 1.0e-6) {
            std::cerr << "GPU phase synchronization experiment failed\n"
                      << "Coupled coherence: " << coherence_a.phase_coherence << '\n'
                      << "Uncoupled coherence: " << coherence_control.phase_coherence << '\n'
                      << "Repeat coherence: " << coherence_b.phase_coherence << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "GPU phase synchronization test passed; coupled R="
                  << coherence_a.phase_coherence
                  << ", uncoupled R=" << coherence_control.phase_coherence
                  << '\n';

        std::cout << "GPU phase dynamics and phase-gated routing tests passed; "
                  << "baseline=" << baseline_phase_metrics.mean_voltage_mv
                  << " mV, gated=" << gated_phase_metrics.mean_voltage_mv
                  << " mV\n";

        // Pair-based STDP experiment. Identical seeded networks provide an
        // active-learning condition and a zero-learning control.
        auto make_plasticity_parameters = [] {
            ModelParameters p;
            p.simulation.excitatory_count = 2;
            p.simulation.inhibitory_count = 1;
            p.simulation.steps = 1;
            p.simulation.dt = 0.25f;
            p.simulation.threshold = -64.9995f;
            p.simulation.initial_voltage_min = -65.0f;
            p.simulation.initial_voltage_max = -65.0f;
            p.simulation.excitatory_current_min = 4.0f;
            p.simulation.excitatory_current_max = 4.0f;
            p.simulation.inhibitory_current_min = 0.0f;
            p.simulation.inhibitory_current_max = 0.0f;
            p.simulation.random_seed = 7654321U;
            p.topology.p_ee = 1.0;
            p.topology.p_ei = 0.0;
            p.topology.p_ie = 0.0;
            p.topology.p_ii = 0.0;
            p.topology.topology_seed = 1234567ULL;
            p.topology.delay_seed = 7654321ULL;
            p.topology.delay_distribution = DelayDistribution::Uniform;
            p.topology.minimum_delay = 1;
            p.topology.maximum_delay = 1;
            p.topology.fibonacci_delays = {1};
            p.topology.fibonacci_weights = {1.0};
            return p;
        };
        CudaSimulationRuntime plasticity_active(make_plasticity_parameters());
        CudaSimulationRuntime plasticity_control(make_plasticity_parameters());
        SynapticPlasticityConfig learning_config;
        learning_config.enabled = true;
        learning_config.trace_time_constant_ms = 30.0f;
        learning_config.potentiation_rate = 0.02f;
        learning_config.depression_rate = 0.002f;
        learning_config.minimum_weight_magnitude = 0.005f;
        learning_config.maximum_weight_magnitude = 0.20f;
        plasticity_active.configure_synaptic_plasticity(learning_config);
        auto no_learning_config = learning_config;
        no_learning_config.potentiation_rate = 0.0f;
        no_learning_config.depression_rate = 0.0f;
        plasticity_control.configure_synaptic_plasticity(no_learning_config);
        const auto initial_learning_metrics = plasticity_active.collect_metrics();
        const auto initial_control_metrics = plasticity_control.collect_metrics();
        plasticity_active.run_steps(1000);
        plasticity_control.run_steps(1000);
        const auto learned_metrics = plasticity_active.collect_metrics();
        const auto control_learning_metrics = plasticity_control.collect_metrics();
        if (!initial_learning_metrics.finite_state ||
            !initial_control_metrics.finite_state ||
            !learned_metrics.finite_state ||
            !control_learning_metrics.finite_state ||
            learned_metrics.excitatory_spikes == 0 ||
            learned_metrics.mean_absolute_synaptic_weight <=
                initial_learning_metrics.mean_absolute_synaptic_weight + 1.0e-6 ||
            std::abs(control_learning_metrics.mean_absolute_synaptic_weight -
                     initial_control_metrics.mean_absolute_synaptic_weight) > 1.0e-7 ||
            learned_metrics.mean_absolute_synaptic_weight > 0.20 + 1.0e-6) {
            std::cerr << "GPU STDP learning experiment failed\n"
                      << "Initial mean |w|: "
                      << initial_learning_metrics.mean_absolute_synaptic_weight << '\n'
                      << "Learned mean |w|: "
                      << learned_metrics.mean_absolute_synaptic_weight << '\n'
                      << "Control mean |w|: "
                      << control_learning_metrics.mean_absolute_synaptic_weight << '\n'
                      << "Excitatory spikes: " << learned_metrics.excitatory_spikes << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "GPU STDP learning experiment passed; mean |w| "
                  << initial_learning_metrics.mean_absolute_synaptic_weight
                  << " -> " << learned_metrics.mean_absolute_synaptic_weight
                  << ", zero-learning control="
                  << control_learning_metrics.mean_absolute_synaptic_weight
                  << ", spikes=" << learned_metrics.excitatory_spikes << '\n';

        // Associative recall: repeatedly co-activate populations A and B,
        // then stimulate A alone. A learned A->B pathway should recruit B
        // more strongly than an untrained population C and a zero-STDP control.
        auto make_association_parameters = [] {
            ModelParameters p;
            p.simulation.excitatory_count = 24;
            p.simulation.inhibitory_count = 1;
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
            p.topology.p_ee = 1.0;
            p.topology.p_ei = 0.0;
            p.topology.p_ie = 0.0;
            p.topology.p_ii = 0.0;
            p.topology.topology_seed = 975318642ULL;
            p.topology.delay_seed = 864213579ULL;
            p.topology.delay_distribution = DelayDistribution::Uniform;
            p.topology.minimum_delay = 1;
            p.topology.maximum_delay = 1;
            p.topology.fibonacci_delays = {1};
            p.topology.fibonacci_weights = {1.0};
            return p;
        };
        auto make_association_topology = [] {
            CSRTopology graph;
            graph.excitatory_count = 24;
            graph.inhibitory_count = 1;
            graph.minimum_delay = 1;
            graph.maximum_delay = 1;
            graph.row_offsets.resize(26, 0);
            // Only A (0..7) projects to B (8..15). C (16..23) has
            // deliberately no incoming edges and is a true untrained control.
            for (std::uint32_t source = 0; source < 24; ++source) {
                if (source < 8) {
                    for (std::uint32_t target = 8; target < 16; ++target) {
                        graph.target_indices.push_back(target);
                        graph.weights.push_back(0.05f);
                        graph.delays.push_back(1);
                    }
                }
                graph.row_offsets[source + 1] =
                    static_cast<std::uint32_t>(graph.target_indices.size());
            }
            graph.row_offsets[25] = static_cast<std::uint32_t>(graph.target_indices.size());
            graph.validate();
            return graph;
        };
        CudaSimulationRuntime association_active(
            make_association_parameters(), make_association_topology());
        CudaSimulationRuntime association_control(
            make_association_parameters(), make_association_topology());
        SynapticPlasticityConfig association_learning;
        association_learning.enabled = true;
        association_learning.trace_time_constant_ms = 30.0f;
        association_learning.potentiation_rate = 0.25f;
        association_learning.depression_rate = 0.002f;
        association_learning.minimum_weight_magnitude = 0.005f;
        association_learning.maximum_weight_magnitude = 5.0f;
        association_active.configure_synaptic_plasticity(association_learning);
        auto association_no_learning = association_learning;
        association_no_learning.potentiation_rate = 0.0f;
        association_no_learning.depression_rate = 0.0f;
        association_control.configure_synaptic_plasticity(association_no_learning);

        constexpr std::uint32_t association_group_size = 8;
        constexpr float association_input = 100.0f;
        std::vector<float> a_input(25, 0.0f);
        std::vector<float> b_input(25, 0.0f);
        std::vector<float> recall_input(25, 0.0f);
        for (std::uint32_t neuron = 0; neuron < association_group_size; ++neuron) {
            a_input[neuron] = association_input;
            b_input[association_group_size + neuron] = association_input;
            recall_input[neuron] = association_input;
        }

        // The implemented STDP rule potentiates an A->B edge when A's
        // presynaptic trace is present at B's later postsynaptic spike.
        // Train with a causal A-then-B sequence, not simultaneous co-spiking.
        for (int repetition = 0; repetition < 10; ++repetition) {
            association_active.set_external_current(a_input);
            association_control.set_external_current(a_input);
            association_active.run_steps(4);
            association_control.run_steps(4);
            association_active.set_external_current(b_input);
            association_control.set_external_current(b_input);
            association_active.run_steps(4);
            association_control.run_steps(4);
            association_active.set_external_current(std::vector<float>(25, 0.0f));
            association_control.set_external_current(std::vector<float>(25, 0.0f));
            association_active.run_steps(20);
            association_control.run_steps(20);
        }
        // Capture activity and learned-weight diagnostics before freezing.
        const auto active_training_metrics = association_active.collect_metrics();
        const auto control_training_metrics = association_control.collect_metrics();
        const auto active_training = association_active.collect_spike_events();
        const auto control_training = association_control.collect_spike_events();
        if (active_training.events.empty() || control_training.events.empty() ||
            active_training.dropped_events != 0 || control_training.dropped_events != 0) {
            std::cerr << "GPU associative training produced no usable spike events; active="
                      << active_training.events.size() << ", control="
                      << control_training.events.size() << '\n';
            return EXIT_FAILURE;
        }

        // Freeze weights during recall so the probe measures the learned state,
        // rather than additional learning caused by the test stimulus.
        auto frozen_learning = association_learning;
        frozen_learning.enabled = false;
        association_active.configure_synaptic_plasticity(frozen_learning);
        association_control.configure_synaptic_plasticity(frozen_learning);
        association_active.set_external_current(recall_input);
        association_control.set_external_current(recall_input);
        association_active.run_steps(120);
        association_control.run_steps(120);
        const auto active_recall = association_active.collect_spike_events();
        const auto control_recall = association_control.collect_spike_events();
        const auto active_recall_voltage = association_active.collect_neuron_voltages();
        const auto control_recall_voltage = association_control.collect_neuron_voltages();

        const auto count_group_spikes = [](const SpikeEventBatch& batch,
                                           std::uint32_t first,
                                           std::uint32_t last) {
            std::uint64_t count = 0;
            for (const auto& event : batch.events) {
                if (event.neuron_id >= first && event.neuron_id < last)
                    ++count;
            }
            return count;
        };
        const auto active_recall_a = count_group_spikes(active_recall, 0, association_group_size);
        const auto control_recall_a = count_group_spikes(control_recall, 0, association_group_size);
        const auto learned_b = count_group_spikes(
            active_recall, association_group_size, 2 * association_group_size);
        const auto untrained_c = count_group_spikes(
            active_recall, 2 * association_group_size, 3 * association_group_size);
        const auto control_b = count_group_spikes(
            control_recall, association_group_size, 2 * association_group_size);
        const auto control_c = count_group_spikes(
            control_recall, 2 * association_group_size, 3 * association_group_size);
        const auto mean_group_voltage = [](const std::vector<float>& voltage,
                                           std::uint32_t first, std::uint32_t last) {
            double sum = 0.0;
            for (std::uint32_t i = first; i < last; ++i) sum += voltage.at(i);
            return sum / static_cast<double>(last - first);
        };
        const double active_a_voltage = mean_group_voltage(active_recall_voltage, 0, association_group_size);
        const double active_b_voltage = mean_group_voltage(active_recall_voltage, association_group_size, 2 * association_group_size);
        const double control_b_voltage = mean_group_voltage(control_recall_voltage, association_group_size, 2 * association_group_size);
        const auto active_recall_metrics = association_active.collect_metrics();
        const auto control_recall_metrics = association_control.collect_metrics();
        if (!active_recall_metrics.finite_state ||
            !control_recall_metrics.finite_state ||
            active_recall.dropped_events != 0 || control_recall.dropped_events != 0 ||
            learned_b <= untrained_c || learned_b <= control_b) {
            std::cerr << "GPU associative recall experiment failed\n"
                      << "Training A/B spikes: "
                      << count_group_spikes(active_training, 0, association_group_size)
                      << "/" << count_group_spikes(active_training, association_group_size,
                                                     2 * association_group_size) << '\n'
                      << "Training mean |w| active/control: "
                      << active_training_metrics.mean_absolute_synaptic_weight << "/"
                      << control_training_metrics.mean_absolute_synaptic_weight << '\n'
                      << "Recall mean voltage A/B active: " << active_a_voltage << "/" << active_b_voltage << '\n'
                      << "Recall mean voltage B control: " << control_b_voltage << '\n'
                      << "Recall A spikes active/control: " << active_recall_a << "/"
                      << control_recall_a << '\n'
                      << "Learned B response: " << learned_b << '\n'
                      << "Untrained C response: " << untrained_c << '\n'
                      << "Zero-STDP B response: " << control_b << '\n'
                      << "Zero-STDP C response: " << control_c << '\n';
            return EXIT_FAILURE;
        }
        std::cout << "GPU associative recall experiment passed; training events="
                  << active_training.events.size() << ", trained B=" << learned_b
                  << ", untrained C=" << untrained_c
                  << ", zero-STDP B=" << control_b
                  << ", zero-STDP C=" << control_c << '\n';

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
