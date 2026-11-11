#include "analysis/izhikevich_analysis.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace spiking::analysis;

namespace {
void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
}

int main() {
    NeuronParameters p;

    // The nullclines intersect at the known zero-current equilibria.
    for (double voltage : {-70.0, -50.0}) {
        const double voltage_u =
            voltage_nullcline_u(p, voltage, 0.0);
        const double recovery_u =
            recovery_nullcline_u(p, voltage);

        check(std::abs(voltage_u - recovery_u) < 1e-10,
              "Nullclines should intersect at known equilibria");

        const auto field =
            evaluate_vector_field(p, voltage, recovery_u, 0.0);

        check(std::abs(field.voltage_derivative) < 1e-10,
              "Voltage derivative should be zero at equilibrium");
        check(std::abs(field.recovery_derivative) < 1e-10,
              "Recovery derivative should be zero at equilibrium");
    }

    // Check the merged equilibrium at the critical current.
    const double critical_u = voltage_nullcline_u(p, -60.0, 4.0);
    check(std::abs(critical_u - (-12.0)) < 1e-10,
          "Incorrect critical voltage nullcline value");
    check(std::abs(recovery_nullcline_u(p, -60.0) - critical_u) < 1e-10,
          "Nullclines should meet at the critical equilibrium");

    // Check a point away from equilibrium against the differential equations.
    const auto field = evaluate_vector_field(p, -60.0, -10.0, 0.0);
    check(std::abs(field.voltage_derivative - (-6.0)) < 1e-10,
          "Incorrect voltage derivative");
    check(std::abs(field.recovery_derivative - (-0.04)) < 1e-10,
          "Incorrect recovery derivative");

    std::cout << "Phase-plane tests passed.\n";
    return 0;
}
