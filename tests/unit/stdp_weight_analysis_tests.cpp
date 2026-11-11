#include "runtime/stdp_weight_analysis.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
bool near(double a, double b, double tolerance = 1.0e-7) {
    return std::abs(a - b) <= tolerance;
}
}

int main() {
    using namespace spiking;
    try {
        CSRTopology before;
        before.excitatory_count = 2;
        before.inhibitory_count = 1;
        before.minimum_delay = 1;
        before.maximum_delay = 3;
        before.row_offsets = {0, 2, 3, 4};
        before.target_indices = {1, 2, 0, 0};
        before.weights = {0.2f, 0.4f, 0.6f, -0.3f};
        before.delays = {1, 2, 3, 1};

        CSRTopology after = before;
        after.weights = {0.25f, 0.4f, 0.5f, -0.3f};
        const auto summary = summarize_stdp_weight_changes(before, after);
        check(summary.plastic_synapses == 3, "wrong plastic synapse count");
        check(summary.changed_synapses == 2, "wrong changed synapse count");
        check(near(summary.mean_initial_weight, 0.4), "wrong initial mean");
        check(near(summary.mean_final_weight, (0.25 + 0.4 + 0.5) / 3.0),
              "wrong final mean");
        check(near(summary.mean_absolute_change, 0.15 / 3.0),
              "wrong mean absolute change");
        check(near(summary.maximum_absolute_change, 0.1, 1.0e-7),
              "wrong maximum absolute change");

        after = before;
        after.delays[0] = 2;
        bool rejected = false;
        try { (void)summarize_stdp_weight_changes(before, after); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "connectivity delay mutation was not detected");

        std::cout << "All STDP weight analysis tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "STDP weight analysis test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
