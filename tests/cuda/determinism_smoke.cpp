#include "runtime/cuda_simulation_runtime.cuh"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

bool relative_close(std::uint64_t a, std::uint64_t b, double tolerance) {
    const auto high = a > b ? a : b;
    const auto low = a > b ? b : a;
    if (high == 0) return true;
    return static_cast<double>(high - low) / static_cast<double>(high) <= tolerance;
}

} // namespace

int main() {
    using namespace spiking;
    try {
        ModelParameters control;
        control.simulation.excitatory_count = 81920;
        control.simulation.inhibitory_count = 10240;
        control.simulation.steps = 1000;
        control.simulation.random_seed = 12345U;
        control.topology.master_seed = 12345ULL;
        control.topology.topology_seed = 6075162723858990937ULL;
        control.topology.delay_seed = 4919422011527349573ULL;

        // Event capture is disabled: this test compares simulation state and
        // metrics, not the optional live event recorder.
        CudaSimulationRuntime first(control, 0, false);
        CudaSimulationRuntime second(control, 0, false);

        const auto first_initial = first.collect_metrics();
        const auto second_initial = second.collect_metrics();
        const auto first_state_hash = first.state_fingerprint();
        const auto second_state_hash = second.state_fingerprint();
        if (!first_initial.finite_state || !second_initial.finite_state ||
            first_initial.completed_steps != 0 ||
            second_initial.completed_steps != 0 ||
            first_state_hash != second_state_hash) {
            std::cerr << "Same-seed initial-state control check failed\n";
            return EXIT_FAILURE;
        }

        first.run();
        second.run();
        const auto a = first.collect_metrics();
        const auto b = second.collect_metrics();

        if (!a.finite_state || !b.finite_state ||
            a.completed_steps != control.simulation.steps ||
            b.completed_steps != control.simulation.steps) {
            std::cerr << "Deterministic control run did not complete cleanly\n";
            return EXIT_FAILURE;
        }

        // CSR topology reproducibility is checked exactly by topology_tests.
        // CUDA atomics can change floating-point accumulation order, so the
        // full-network spike-count comparison uses a tight 0.1% tolerance.
        if (!relative_close(a.excitatory_spikes, b.excitatory_spikes, 0.001) ||
            !relative_close(a.inhibitory_spikes, b.inhibitory_spikes, 0.001) ||
            std::abs(a.mean_voltage_mv - b.mean_voltage_mv) > 0.1) {
            std::cerr << "Repeated full-network runs exceeded deterministic-control tolerances\n"
                      << "Excitatory spikes: " << a.excitatory_spikes
                      << " vs " << b.excitatory_spikes << '\n'
                      << "Inhibitory spikes: " << a.inhibitory_spikes
                      << " vs " << b.inhibitory_spikes << '\n'
                      << "Mean voltage: " << a.mean_voltage_mv
                      << " vs " << b.mean_voltage_mv << '\n';
            return EXIT_FAILURE;
        }

        std::cout << "Deterministic control passed\n"
                  << "Initial mean voltage: " << first_initial.mean_voltage_mv << " mV\n"
                  << "Excitatory spikes: " << a.excitatory_spikes
                  << " vs " << b.excitatory_spikes << '\n'
                  << "Inhibitory spikes: " << a.inhibitory_spikes
                  << " vs " << b.inhibitory_spikes << '\n'
                  << "Final mean voltage: " << a.mean_voltage_mv
                  << " vs " << b.mean_voltage_mv << " mV\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Deterministic control error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
