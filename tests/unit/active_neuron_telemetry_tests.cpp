#include "runtime/active_neuron_telemetry.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        const auto unchanged = spiking::count_active_neurons(
            {0, 4, 1, 9, 0}, {0, 4, 1, 9, 0}, 3);
        if (unchanged.excitatory != 0 || unchanged.inhibitory != 0)
            throw std::runtime_error("unchanged counters counted as active");

        const auto active = spiking::count_active_neurons(
            {0, 2, 5, 7, 1}, {3, 2, 8, 7, 4}, 3);
        if (active.excitatory != 2 || active.inhibitory != 2 ||
            std::abs(active.excitatory_percent - 200.0 / 3.0) > 1e-9 ||
            std::abs(active.inhibitory_percent - 100.0) > 1e-9)
            throw std::runtime_error("per-neuron activity or percentages incorrect");

        const auto wrapped = spiking::count_active_neurons(
            {0xffffffffU, 0U}, {0U, 1U}, 1);
        if (wrapped.excitatory != 1 || wrapped.inhibitory != 1)
            throw std::runtime_error("counter wrap not recognized");

        const auto empty = spiking::count_active_neurons({}, {}, 0);
        if (empty.excitatory_percent != 0.0 || empty.inhibitory_percent != 0.0)
            throw std::runtime_error("empty population percentage incorrect");

        bool mismatch_threw = false;
        try { (void)spiking::count_active_neurons({0U}, {0U, 1U}, 1); }
        catch (const std::invalid_argument&) { mismatch_threw = true; }
        if (!mismatch_threw) throw std::runtime_error("snapshot size mismatch accepted");

        bool population_threw = false;
        try { (void)spiking::count_active_neurons({0U}, {1U}, 2); }
        catch (const std::invalid_argument&) { population_threw = true; }
        if (!population_threw) throw std::runtime_error("invalid population size accepted");

        std::cout << "Active-neuron telemetry tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Active-neuron telemetry test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
