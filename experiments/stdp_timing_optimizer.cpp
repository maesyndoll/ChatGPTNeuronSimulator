#include "plasticity/stdp_timing_optimizer.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<spiking::StdpTimingObservation> read_observations(
    const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open timing data: " + path);

    std::vector<spiking::StdpTimingObservation> observations;
    std::string line;
    std::getline(input, line); // CSV header
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::stringstream row(line);
        std::string lag, measured;
        if (!std::getline(row, lag, ',') || !std::getline(row, measured, ','))
            throw std::runtime_error("Malformed Stage A CSV row: " + line);
        const double lag_ms = std::stod(lag);
        const double delta_weight = std::stod(measured);
        // Simultaneous spikes intentionally have no zero-lag update in this
        // pair rule and therefore do not constrain either exponential branch.
        if (lag_ms == 0.0) continue;
        observations.push_back({lag_ms, delta_weight});
    }
    return observations;
}
}

int main(int argc, char** argv) {
    try {
        const std::string input_path = argc > 1
            ? argv[1] : "stdp_analysis/stage_a_timing.csv";
        const std::string output_path = argc > 2
            ? argv[2] : "experiments/logs/stdp_optimized_parameters.json";
        const auto observations = read_observations(input_path);

        spiking::StdpParameters initial;
        const auto fit = spiking::fit_stdp_timing_curve(
            observations, initial, 10000.0, 500);

        std::ofstream output(output_path);
        if (!output) throw std::runtime_error(
            "Cannot create optimizer output: " + output_path);
        output << std::setprecision(10)
            << "{\n"
            << "  \"potentiation_rate\": " << fit.parameters.potentiation_rate << ",\n"
            << "  \"depression_rate\": " << fit.parameters.depression_rate << ",\n"
            << "  \"presynaptic_tau_ms\": " << fit.parameters.presynaptic_tau_ms << ",\n"
            << "  \"postsynaptic_tau_ms\": " << fit.parameters.postsynaptic_tau_ms << ",\n"
            << "  \"observations\": " << observations.size() << ",\n"
            << "  \"initial_squared_error\": " << fit.initial_loss << ",\n"
            << "  \"final_squared_error\": " << fit.final_loss << ",\n"
            << "  \"iterations\": " << fit.iterations << ",\n"
            << "  \"converged\": " << (fit.converged ? "true" : "false") << "\n"
            << "}\n";

        std::cout << std::scientific << std::setprecision(8)
                  << "STDP nonlinear timing-curve fit\n"
                  << "Observations: " << observations.size() << '\n'
                  << "A+: " << fit.parameters.potentiation_rate << '\n'
                  << "A-: " << fit.parameters.depression_rate << '\n'
                  << "tau_pre: " << fit.parameters.presynaptic_tau_ms << " ms\n"
                  << "tau_post: " << fit.parameters.postsynaptic_tau_ms << " ms\n"
                  << "Squared error: " << fit.initial_loss << " -> "
                  << fit.final_loss << '\n'
                  << "Iterations: " << fit.iterations << '\n'
                  << "Converged: " << (fit.converged ? "yes" : "NO") << '\n'
                  << "Parameters: " << output_path << '\n';
        if (!fit.converged || !(fit.final_loss < fit.initial_loss))
            throw std::runtime_error("STDP timing fit did not converge");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "STDP timing optimizer failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
