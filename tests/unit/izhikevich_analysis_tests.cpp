#include "analysis/izhikevich_analysis.h"

#include <stdexcept>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace spiking::analysis;

static void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    const NeuronParameters regular{
        0.02, 0.20, -65.0, 8.0
    };

    const NeuronParameters fast{
        0.10, 0.20, -65.0, 2.0
    };

    // At this current, the quadratic has no real equilibrium.
    check(analyze_equilibria(regular, 10.0).empty(), "Expected no equilibria at current 10");

    // This current produces two equilibria:
    // approximately -70 mV (stable node) and -50 mV (saddle).
    const auto points = analyze_equilibria(regular, 0.0);
    check(points.size() == 2, "Expected two regular-spiking equilibria");

    check(std::abs(points[0].voltage - (-70.0)) < 1e-8, "Incorrect first equilibrium voltage");
    check(std::abs(points[0].recovery - (-14.0)) < 1e-8, "Incorrect first recovery equilibrium");
    check(points[0].stability == Stability::StableNode, "First equilibrium should be a stable node");

    check(std::abs(points[1].voltage - (-50.0)) < 1e-8, "Incorrect second equilibrium voltage");
    check(std::abs(points[1].recovery - (-10.0)) < 1e-8, "Incorrect second recovery equilibrium");
    check(points[1].stability == Stability::Saddle, "Second equilibrium should be a saddle");

    // The equilibrium branches merge at the critical current.
    const double critical = critical_input_current(regular);
    check(std::abs(critical - 4.0) < 1e-10,
          "Incorrect critical input current");

    const auto critical_points = analyze_equilibria(regular, critical);
    check(critical_points.size() == 1,
          "Expected one merged equilibrium at critical current");
    check(std::abs(critical_points[0].voltage - (-60.0)) < 1e-8,
          "Incorrect critical equilibrium voltage");
    check(std::abs(critical_points[0].recovery - (-12.0)) < 1e-8,
          "Incorrect critical recovery value");
    check(critical_points[0].stability == Stability::NonHyperbolic,
          "Critical equilibrium should be non-hyperbolic");

    check(analyze_equilibria(regular, critical - 0.1).size() == 2,
          "Expected two equilibria below critical current");
    check(analyze_equilibria(regular, critical + 0.1).empty(),
          "Expected no equilibria above critical current");

    // The fast-spiking parameter set must also be analyzable.
    const auto fast_points = analyze_equilibria(fast, 0.0);
    check(fast_points.size() == 2, "Expected two fast-spiking equilibria");
    check(fast_points[0].stability == Stability::StableNode, "Fast-spiking first equilibrium should be stable");
    check(fast_points[1].stability == Stability::Saddle, "Fast-spiking second equilibrium should be a saddle");

    // Recovery equilibrium must satisfy u = b*v.
    for (const auto& point : points) {
        check(std::abs(point.recovery -
                        regular.b * point.voltage) < 1e-10, "Recovery equilibrium must equal b times voltage");
    }

    // Invalid parameters must be rejected.
    bool rejected = false;
    try {
        auto invalid = regular;
        invalid.a = 0.0;
        (void)analyze_equilibria(invalid, 10.0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "Invalid neuron parameter was not rejected");

    std::cout << "Izhikevich analysis tests passed\n";
    return 0;
}
