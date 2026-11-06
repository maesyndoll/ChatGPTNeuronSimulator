#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace spiking::analysis {

struct NeuronParameters {
    double a = 0.02;
    double b = 0.20;
    double c = -65.0;
    double d = 8.0;
};

enum class Stability {
    StableNode,
    StableFocus,
    UnstableNode,
    UnstableFocus,
    Saddle,
    NonHyperbolic
};

struct Equilibrium {
    double voltage = 0.0;
    double recovery = 0.0;
    double trace = 0.0;
    double determinant = 0.0;
    std::complex<double> eigenvalue_1;
    std::complex<double> eigenvalue_2;
    Stability stability = Stability::NonHyperbolic;
};

// Analyze the continuous Izhikevich equations for constant input current.
// The discrete threshold/reset operation is intentionally analyzed separately.
double critical_input_current(const NeuronParameters& parameters);

std::vector<Equilibrium> analyze_equilibria(
    const NeuronParameters& parameters,
    double input_current
);

const char* stability_name(Stability stability) noexcept;


// Phase-plane analysis. Both nullclines return u as a function of v.
double voltage_nullcline_u(
    const NeuronParameters& parameters,
    double voltage,
    double input_current
);

double recovery_nullcline_u(
    const NeuronParameters& parameters,
    double voltage
);

struct VectorField {
    double voltage_derivative = 0.0;
    double recovery_derivative = 0.0;
};

VectorField evaluate_vector_field(
    const NeuronParameters& parameters,
    double voltage,
    double recovery,
    double input_current
);


struct TrajectorySample {
    double time = 0.0;
    double voltage = 0.0;
    double recovery = 0.0;
    bool spiked = false;
};

struct Trajectory {
    std::vector<TrajectorySample> samples;
    std::vector<double> spike_times;
};

// Simulate the continuous equations using midpoint RK2.
// Threshold/reset is applied after each completed timestep.
Trajectory simulate_trajectory(
    const NeuronParameters& parameters,
    double initial_voltage,
    double initial_recovery,
    double input_current,
    double time_step,
    std::size_t number_of_steps,
    double spike_threshold
);

} // namespace spiking::analysis

