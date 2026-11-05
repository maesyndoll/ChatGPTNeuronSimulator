#include "runtime/simulation_controller.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

int main() {
    using namespace spiking;
    using namespace std::chrono_literals;

    try {
        ModelParameters parameters;
        parameters.simulation = SimulationParameters::small_test();
        parameters.simulation.steps = 2;

        // The controller must collect CUDA spike events on its worker and
        // expose a cached snapshot without giving IPC callers CUDA access.
        {
            ModelParameters recorder_parameters;
            recorder_parameters.simulation.excitatory_count = 1;
            recorder_parameters.simulation.inhibitory_count = 1;
            recorder_parameters.simulation.steps = 1;
            recorder_parameters.simulation.dt = 0.001f;
            recorder_parameters.simulation.threshold = -64.9995f;
            recorder_parameters.simulation.initial_voltage_min = -65.0f;
            recorder_parameters.simulation.initial_voltage_max = -65.0f;
            recorder_parameters.simulation.excitatory_current_min = 4.0f;
            recorder_parameters.simulation.excitatory_current_max = 4.0f;
            recorder_parameters.simulation.inhibitory_current_min = 4.0f;
            recorder_parameters.simulation.inhibitory_current_max = 4.0f;

            SimulationController recorder_controller(
                recorder_parameters, 0, 1, 1000);
            recorder_controller.step(1);
            const auto events =
                recorder_controller.spike_events_snapshot(0);

            bool saw_excitatory = false;
            bool saw_inhibitory = false;
            for (const auto& event : events.events) {
                if (event.sequence > 1 || event.step != 1) {
                    std::cerr << "Controller spike-event metadata failed\\n";
                    return EXIT_FAILURE;
                }
                saw_excitatory |= event.neuron_id == 0;
                saw_inhibitory |= event.neuron_id == 1;
            }

            if (events.events.size() != 2 ||
                events.first_sequence != 0 ||
                events.total_events != 2 ||
                events.dropped_events != 0 ||
                events.simulation_generation != 0 ||
                !saw_excitatory || !saw_inhibitory) {
                std::cerr << "Controller spike-event collection failed\\n";
                return EXIT_FAILURE;
            }

            const auto drained =
                recorder_controller.spike_events_snapshot(2);
            if (!drained.events.empty() || drained.total_events != 2) {
                std::cerr << "Controller spike-event cursor failed\\n";
                return EXIT_FAILURE;
            }

            recorder_controller.reinitialize_simulation();
            const auto reset_events =
                recorder_controller.spike_events_snapshot(0);
            if (!reset_events.events.empty() ||
                reset_events.total_events != 0 ||
                reset_events.simulation_generation != 1) {
                std::cerr << "Controller spike-event generation reset failed\\n";
                return EXIT_FAILURE;
            }
        }

        std::cout << "Controller spike-event collection/reset tests passed\\n";

        SimulationController controller(parameters, 0, 2, 4);

        if (controller.status() != ControllerStatus::Ready) {
            std::cerr << "Controller did not initialize in Ready state\n";
            return EXIT_FAILURE;
        }

        const auto initial = controller.telemetry_snapshot();
        if (!initial.metrics.finite_state ||
            initial.metrics.completed_steps != 0 ||
            initial.sequence == 0) {
            std::cerr << "Controller initial telemetry failed\n";
            return EXIT_FAILURE;
        }

        const auto initial_configuration = controller.configuration_snapshot();
        if (initial_configuration.has_pending_reinitialization ||
            initial_configuration.requested_parameters.simulation.excitatory_count !=
                parameters.simulation.excitatory_count ||
            initial_configuration.active_parameters.simulation.excitatory_count !=
                parameters.simulation.excitatory_count) {
            std::cerr << "Controller configuration snapshot failed\n";
            return EXIT_FAILURE;
        }

        auto updated = parameters;
        updated.simulation.dt = 0.25f;
        const auto update = controller.apply_parameters(updated);
        if (!update.accepted || update.requires_reinitialization) {
            std::cerr << "Controller live parameter update failed\n";
            return EXIT_FAILURE;
        }

        controller.step(3);
        const auto stepped = controller.telemetry_snapshot();
        if (!stepped.metrics.finite_state ||
            stepped.metrics.completed_steps != 3 ||
            stepped.sequence <= initial.sequence ||
            controller.status() != ControllerStatus::Paused) {
            std::cerr << "Controller single-step command failed\n";
            return EXIT_FAILURE;
        }

        const auto target = stepped.metrics.completed_steps + 4;
        controller.start();

        if (controller.status() != ControllerStatus::Running) {
            std::cerr << "Controller did not enter Running state\n";
            return EXIT_FAILURE;
        }

        if (!controller.wait_until_steps(target, 5s)) {
            std::cerr << "Controller did not reach requested step target\n";
            return EXIT_FAILURE;
        }

        controller.pause();

        if (controller.status() != ControllerStatus::Paused) {
            std::cerr << "Controller did not enter Paused state\n";
            return EXIT_FAILURE;
        }

        const auto paused = controller.telemetry_snapshot();
        if (!paused.metrics.finite_state ||
            paused.metrics.completed_steps < target ||
            paused.sequence <= stepped.sequence) {
            std::cerr << "Controller pause telemetry failed\n";
            return EXIT_FAILURE;
        }

        updated = parameters;
        updated.simulation.excitatory_count /= 2;
        const auto structural_update =
            controller.apply_parameters(updated);

        if (!structural_update.accepted ||
            !structural_update.requires_reinitialization) {
            std::cerr << "Controller did not stage structural update\n";
            return EXIT_FAILURE;
        }

        controller.reinitialize_simulation();
        const auto reinitialized = controller.telemetry_snapshot();

        if (!reinitialized.metrics.finite_state ||
            reinitialized.metrics.completed_steps != 0 ||
            reinitialized.sequence <= paused.sequence ||
            controller.status() != ControllerStatus::Paused) {
            std::cerr << "Controller reinitialization failed\n";
            return EXIT_FAILURE;
        }

        // Previous-generation progress must not satisfy a new wait.
        if (controller.wait_until_steps(1, 20ms)) {
            std::cerr << "Old simulation progress leaked across reinitialization\n";
            return EXIT_FAILURE;
        }

        controller.step(1);
        if (!controller.wait_until_steps(1, 100ms) ||
            controller.telemetry_snapshot().metrics.completed_steps != 1) {
            std::cerr << "Progress tracking did not restart after reinitialization\n";
            return EXIT_FAILURE;
        }

        const auto new_configuration = controller.configuration_snapshot();
        if (new_configuration.simulation_generation <=
                initial_configuration.simulation_generation ||
            new_configuration.has_pending_reinitialization ||
            new_configuration.active_parameters.simulation.excitatory_count !=
                updated.simulation.excitatory_count) {
            std::cerr << "Post-reinitialization configuration snapshot failed\n";
            return EXIT_FAILURE;
        }

        const auto reset = controller.reset_parameters_to_defaults();
        if (!reset.accepted) {
            std::cerr << "Controller parameter reset failed\n";
            return EXIT_FAILURE;
        }

        if (controller.last_execution_error()) {
            std::cerr << "Controller recorded an unexpected execution error\n";
            return EXIT_FAILURE;
        }

        std::cout << "Simulation controller smoke test passed\n";

        // Exercise destruction while continuous execution is active.
        {
            SimulationController shutdown_test(parameters, 0, 2, 1000);
            shutdown_test.start();

            if (!shutdown_test.wait_until_steps(2, 5s)) {
                std::cerr << "Shutdown test simulation did not start\n";
                return EXIT_FAILURE;
            }
        }

        std::cout << "Controller shutdown-while-running test passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Simulation controller smoke test error: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
