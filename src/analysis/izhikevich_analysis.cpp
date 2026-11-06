#include "analysis/izhikevich_analysis.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spiking::analysis {
namespace {

constexpr double kVoltageQuadratic = 0.04;
constexpr double kTolerance = 1e-10;

void validate(const NeuronParameters& p, double current) {
    if (!std::isfinite(p.a) || !std::isfinite(p.b) ||
        !std::isfinite(p.c) || !std::isfinite(p.d) ||
        !std::isfinite(current)) {
        throw std::invalid_argument(
            "Izhikevich parameters and input current must be finite"
        );
    }

    if (p.a <= 0.0) {
        throw std::invalid_argument(
            "Izhikevich parameter a must be positive"
        );
    }
}

Stability classify(
    const std::complex<double>& first,
    const std::complex<double>& second
) {
    const double real_1 = first.real();
    const double real_2 = second.real();
    const double imag = std::max(
        std::abs(first.imag()),
        std::abs(second.imag())
    );

    if (std::abs(real_1) <= kTolerance ||
        std::abs(real_2) <= kTolerance) {
        return Stability::NonHyperbolic;
    }

    if (imag > kTolerance) {
        return real_1 < 0.0
            ? Stability::StableFocus
            : Stability::UnstableFocus;
    }

    if (real_1 < 0.0 && real_2 < 0.0) {
        return Stability::StableNode;
    }

    if (real_1 > 0.0 && real_2 > 0.0) {
        return Stability::UnstableNode;
    }

    return Stability::Saddle;
}

Equilibrium analyze_point(
    const NeuronParameters& p,
    double voltage
) {
    Equilibrium result;
    result.voltage = voltage;
    result.recovery = p.b * voltage;

    // Jacobian of:
    // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    // du/dt = a*(b*v - u)
    const double j11 = 2.0 * kVoltageQuadratic * voltage + 5.0;
    const double j12 = -1.0;
    const double j21 = p.a * p.b;
    const double j22 = -p.a;

    result.trace = j11 + j22;
    result.determinant = j11 * j22 - j12 * j21;

    const double discriminant =
        result.trace * result.trace - 4.0 * result.determinant;

    const std::complex<double> root =
        std::sqrt(std::complex<double>(discriminant, 0.0));

    result.eigenvalue_1 = (result.trace + root) / 2.0;
    result.eigenvalue_2 = (result.trace - root) / 2.0;
    result.stability = classify(
        result.eigenvalue_1,
        result.eigenvalue_2
    );

    return result;
}

} // namespace

double critical_input_current(const NeuronParameters& p) {
    validate(p, 0.0);

    const double B = 5.0 - p.b;
    return (B * B) / (4.0 * kVoltageQuadratic) - 140.0;
}

std::vector<Equilibrium> analyze_equilibria(
    const NeuronParameters& p,
    double input_current
) {
    validate(p, input_current);

    // At equilibrium, du/dt = 0 gives u = b*v.
    // Substitution into dv/dt = 0 gives:
    // 0.04*v^2 + (5-b)*v + 140 + I = 0.
    const double A = kVoltageQuadratic;
    const double B = 5.0 - p.b;
    const double C = 140.0 + input_current;

    const double discriminant = B * B - 4.0 * A * C;
    const double tolerance =
        kTolerance * (B * B + std::abs(4.0 * A * C) + 1.0);

    if (discriminant < -tolerance) {
        return {};
    }

    if (std::abs(discriminant) <= tolerance) {
        const double voltage = -B / (2.0 * A);
        return {analyze_point(p, voltage)};
    }

    const double root = std::sqrt(discriminant);
    const double voltage_1 = (-B - root) / (2.0 * A);
    const double voltage_2 = (-B + root) / (2.0 * A);

    std::vector<Equilibrium> result;
    result.reserve(2);
    result.push_back(analyze_point(p, voltage_1));
    result.push_back(analyze_point(p, voltage_2));
    return result;
}

const char* stability_name(Stability stability) noexcept {
    switch (stability) {
        case Stability::StableNode:      return "stable node";
        case Stability::StableFocus:     return "stable focus";
        case Stability::UnstableNode:    return "unstable node";
        case Stability::UnstableFocus:   return "unstable focus";
        case Stability::Saddle:          return "saddle";
        case Stability::NonHyperbolic:   return "non-hyperbolic";
    }
    return "unknown";
}


double voltage_nullcline_u(
    const NeuronParameters& p,
    double voltage,
    double input_current
) {
    validate(p, input_current);
    if (!std::isfinite(voltage)) {
        throw std::invalid_argument("Voltage must be finite");
    }

    return kVoltageQuadratic * voltage * voltage
         + 5.0 * voltage + 140.0 + input_current;
}

double recovery_nullcline_u(
    const NeuronParameters& p,
    double voltage
) {
    validate(p, 0.0);
    if (!std::isfinite(voltage)) {
        throw std::invalid_argument("Voltage must be finite");
    }

    return p.b * voltage;
}

VectorField evaluate_vector_field(
    const NeuronParameters& p,
    double voltage,
    double recovery,
    double input_current
) {
    validate(p, input_current);
    if (!std::isfinite(voltage) || !std::isfinite(recovery)) {
        throw std::invalid_argument("State variables must be finite");
    }

    VectorField field;
    field.voltage_derivative =
        kVoltageQuadratic * voltage * voltage
        + 5.0 * voltage + 140.0 - recovery + input_current;
    field.recovery_derivative =
        p.a * (p.b * voltage - recovery);

    return field;
}


Trajectory simulate_trajectory(
    const NeuronParameters& p,
    double initial_voltage,
    double initial_recovery,
    double input_current,
    double time_step,
    std::size_t number_of_steps,
    double spike_threshold
) {
    validate(p, input_current);

    if (!std::isfinite(initial_voltage) ||
        !std::isfinite(initial_recovery) ||
        !std::isfinite(time_step) ||
        !std::isfinite(spike_threshold) ||
        time_step <= 0.0) {
        throw std::invalid_argument(
            "Initial state and threshold must be finite; timestep must be positive"
        );
    }

    Trajectory trajectory;
    trajectory.samples.reserve(number_of_steps + 1);

    double voltage = initial_voltage;
    double recovery = initial_recovery;
    double time = 0.0;

    trajectory.samples.push_back({
        time, voltage, recovery, false
    });

    for (std::size_t step = 0; step < number_of_steps; ++step) {
        // First derivative at the beginning of the timestep.
        const double dv1 =
            kVoltageQuadratic * voltage * voltage
            + 5.0 * voltage + 140.0 - recovery + input_current;
        const double du1 =
            p.a * (p.b * voltage - recovery);

        // Midpoint state.
        const double midpoint_voltage =
            voltage + 0.5 * time_step * dv1;
        const double midpoint_recovery =
            recovery + 0.5 * time_step * du1;

        // Second derivative at the midpoint.
        const double dv2 =
            kVoltageQuadratic * midpoint_voltage * midpoint_voltage
            + 5.0 * midpoint_voltage + 140.0
            - midpoint_recovery + input_current;
        const double du2 =
            p.a * (p.b * midpoint_voltage - midpoint_recovery);

        // Complete the RK2 step.
        voltage += time_step * dv2;
        recovery += time_step * du2;
        time += time_step;

        if (!std::isfinite(voltage) || !std::isfinite(recovery)) {
            throw std::overflow_error(
                "Trajectory became non-finite during integration"
            );
        }

        bool spiked = false;

        // Match the CUDA threshold/reset ordering.
        if (voltage >= spike_threshold) {
            voltage = p.c;
            recovery += p.d;
            spiked = true;
            trajectory.spike_times.push_back(time);
        }

        trajectory.samples.push_back({
            time, voltage, recovery, spiked
        });
    }

    return trajectory;
}

} // namespace spiking::analysis
