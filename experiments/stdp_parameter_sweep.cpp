#include "ipc/parameter_json.h"
#include "runtime/simulation_controller.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace {
using namespace spiking;

struct Distribution {
    std::uint64_t count = 0;
    double mean = 0.0;
    float minimum = 0.0f, maximum = 0.0f;
    float p05 = 0.0f, median = 0.0f, p95 = 0.0f;
    std::uint64_t at_minimum = 0, at_maximum = 0;
};

Distribution summarize(std::vector<float> values, float lower, float upper) {
    Distribution result;
    if (values.empty()) return result;
    std::sort(values.begin(), values.end());
    result.count = values.size();
    result.minimum = values.front();
    result.maximum = values.back();
    result.p05 = values[static_cast<std::size_t>(0.05 * (values.size() - 1))];
    result.median = values[values.size() / 2];
    result.p95 = values[static_cast<std::size_t>(0.95 * (values.size() - 1))];
    double sum = 0.0;
    for (float value : values) {
        sum += value;
        if (value <= lower + 1.0e-6f) ++result.at_minimum;
        if (value >= upper - 1.0e-6f) ++result.at_maximum;
    }
    result.mean = sum / static_cast<double>(values.size());
    return result;
}

struct TrialResult {
    SimulationMetrics metrics;
    Distribution initial_ee, initial_ei, ee, ei;
    double wall_seconds = 0.0;
};

TrialResult run_trial(const ModelParameters& model, StdpParameters stdp,
                      std::uint32_t steps) {
    const auto start = std::chrono::steady_clock::now();
    SimulationController controller(model, 0, 1000, steps, false, true, stdp);
    TrialResult result;
    // Summarize the seeded starting graph, then release its large host copy.
    {
        const auto initial_graph = controller.connectivity_snapshot();
        std::vector<float> initial_ee, initial_ei;
        for (std::uint32_t source = 0;
             source < initial_graph.excitatory_count; ++source) {
            for (std::uint32_t edge = initial_graph.row_offsets[source];
                 edge < initial_graph.row_offsets[source + 1]; ++edge) {
                if (initial_graph.target_indices[edge] < initial_graph.excitatory_count)
                    initial_ee.push_back(initial_graph.weights[edge]);
                else
                    initial_ei.push_back(initial_graph.weights[edge]);
            }
        }
        result.initial_ee = summarize(std::move(initial_ee),
                                      stdp.minimum_weight, stdp.maximum_weight);
        result.initial_ei = summarize(std::move(initial_ei),
                                      stdp.minimum_weight, stdp.maximum_weight);
    }
    controller.step(steps);
    result.metrics = controller.telemetry();
    const auto graph = controller.connectivity_snapshot();
    std::vector<float> ee, ei;
    ee.reserve(graph.weights.size());
    ei.reserve(graph.weights.size());
    for (std::uint32_t source = 0; source < graph.excitatory_count; ++source) {
        for (std::uint32_t edge = graph.row_offsets[source];
             edge < graph.row_offsets[source + 1]; ++edge) {
            if (graph.target_indices[edge] < graph.excitatory_count)
                ee.push_back(graph.weights[edge]);
            else
                ei.push_back(graph.weights[edge]);
        }
    }
    result.ee = summarize(std::move(ee), stdp.minimum_weight, stdp.maximum_weight);
    result.ei = summarize(std::move(ei), stdp.minimum_weight, stdp.maximum_weight);
    result.wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}

void write_distribution(std::ostream& out, const Distribution& d) {
    out << d.count << ',' << d.mean << ',' << d.minimum << ','
        << d.p05 << ',' << d.median << ',' << d.p95 << ',' << d.maximum << ','
        << (d.count ? static_cast<double>(d.at_minimum) / d.count : 0.0) << ','
        << (d.count ? static_cast<double>(d.at_maximum) / d.count : 0.0);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string config_path = argc > 1 ? argv[1] :
            "experiments/background_soak_2x.json";
        const std::string output_path = argc > 2 ? argv[2] :
            "experiments/logs/stdp_parameter_sweep_2x.csv";
        const std::uint32_t steps = argc > 3 ?
            static_cast<std::uint32_t>(std::stoul(argv[3])) : 40000U;
        if (steps == 0) throw std::invalid_argument("steps must be positive");

        std::ifstream config_file(config_path);
        if (!config_file) throw std::runtime_error("Cannot open config: " + config_path);
        nlohmann::json config;
        config_file >> config;
        const ModelParameters model = ipc::parameters_from_json(config, ModelParameters{});

        struct Candidate { const char* name; float potentiation; };
        const Candidate candidates[] = {
            {"baseline", 0.00500f},
            {"equal_coefficients", 0.00525f},
            {"potentiation_plus_5pct", 0.00550f}
        };

        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("Cannot create CSV: " + output_path);
        output << std::setprecision(10)
            << "candidate,potentiation_rate,depression_rate,pre_tau_ms,post_tau_ms,"
               "steps,wall_seconds,finite_state,excitatory_rate_hz,inhibitory_rate_hz,"
               "mean_voltage_mv,initial_ee_mean,initial_ei_mean,ee_count,ee_mean,ee_min,ee_p05,ee_median,ee_p95,"
               "ee_max,ee_at_min_fraction,ee_at_max_fraction,ei_count,ei_mean,"
               "ei_min,ei_p05,ei_median,ei_p95,ei_max,ei_at_min_fraction,"
               "ei_at_max_fraction\n";

        std::cout << "Scale-matched STDP parameter sweep\n"
                  << "Neurons: " << model.simulation.neuron_count()
                  << " (E=" << model.simulation.excitatory_count
                  << ", I=" << model.simulation.inhibitory_count << ")\n"
                  << "Steps per candidate: " << steps << " (simulated ms="
                  << steps * model.simulation.dt << ")\n"
                  << "Seeded topology and neuron initialization are identical across candidates.\n";

        for (const auto& candidate : candidates) {
            StdpParameters stdp;
            stdp.potentiation_rate = candidate.potentiation;
            stdp.depression_rate = 0.00525f;
            stdp.presynaptic_tau_ms = 20.0f;
            stdp.postsynaptic_tau_ms = 20.0f;
            stdp.minimum_weight = 0.001f;
            stdp.maximum_weight = 1.0f;

            std::cout << "\nRunning " << candidate.name << " (A+="
                      << candidate.potentiation << ", A-="
                      << stdp.depression_rate << ")...\n";
            const auto result = run_trial(model, stdp, steps);
            output << candidate.name << ',' << stdp.potentiation_rate << ','
                   << stdp.depression_rate << ',' << stdp.presynaptic_tau_ms << ','
                   << stdp.postsynaptic_tau_ms << ',' << steps << ','
                   << result.wall_seconds << ','
                   << (result.metrics.finite_state ? 1 : 0) << ','
                   << result.metrics.excitatory_rate_hz << ','
                   << result.metrics.inhibitory_rate_hz << ','
                   << result.metrics.mean_voltage_mv << ','
                   << result.initial_ee.mean << ',' << result.initial_ei.mean << ',';
            write_distribution(output, result.ee);
            output << ',';
            write_distribution(output, result.ei);
            output << '\n';
            output.flush();

            const auto fraction = [](std::uint64_t count, std::uint64_t total) {
                return total ? static_cast<double>(count) / total : 0.0;
            };
            std::cout << std::fixed << std::setprecision(5)
                      << "  wall seconds: " << result.wall_seconds << '\n'
                      << "  E/I rates Hz: " << result.metrics.excitatory_rate_hz
                      << '/' << result.metrics.inhibitory_rate_hz << '\n'
                      << "  EE mean/median: " << result.ee.mean << '/'
                      << result.ee.median << " floor/ceiling="
                      << fraction(result.ee.at_minimum, result.ee.count) << '/'
                      << fraction(result.ee.at_maximum, result.ee.count) << '\n'
                      << "  EI mean/median: " << result.ei.mean << '/'
                      << result.ei.median << '\n'
                      << "  finite state: "
                      << (result.metrics.finite_state ? "yes" : "NO") << '\n';
            if (!result.metrics.finite_state)
                throw std::runtime_error("Non-finite state in " + std::string(candidate.name));
        }
        std::cout << "\nPASS: all candidates completed with finite state.\nCSV: "
                  << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "STDP parameter sweep failed: " << error.what() << '\n';
        return 1;
    }
}
