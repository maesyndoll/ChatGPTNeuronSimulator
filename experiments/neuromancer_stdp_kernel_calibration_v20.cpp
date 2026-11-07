#include "core/parameters.h"
#include "plasticity/stdp.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/csr_topology.h"
#include "topology/topology_io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr std::uint32_t kExcitatory = 2;
constexpr std::uint32_t kInhibitory = 1;
constexpr std::uint32_t kTotal = kExcitatory + kInhibitory;
constexpr std::uint32_t kPairs = 20;
constexpr std::uint32_t kPairIntervalSteps = 200; // 100 ms: isolate successive pairs
constexpr float kDtMs = 0.5f;
constexpr float kInitialWeight = 0.5f;
constexpr std::uint32_t kPulseStart = 50;
constexpr std::uint32_t kMaxAbsLagSteps = 20;
constexpr std::vector<int> kLagSteps{
    -20, -16, -12, -8, -6, -4, -3, -2, -1,
      1,   2,   3,  4,  6,  8, 12, 16, 20
};

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = kExcitatory;
    p.simulation.inhibitory_count = kInhibitory;
    p.simulation.steps = 5000;
    p.simulation.dt = kDtMs;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 2020;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;
    p.topology.minimum_delay = 1;
    p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

StdpParameters make_stdp() {
    StdpParameters p;
    p.potentiation_rate = 0.005f;
    p.depression_rate = 0.00525f;
    p.presynaptic_tau_ms = 20.0f;
    p.postsynaptic_tau_ms = 20.0f;
    p.minimum_weight = 0.001f;
    p.maximum_weight = 1.0f;
    return p;
}

CSRTopology make_topology() {
    CSRTopology g;
    g.excitatory_count = kExcitatory;
    g.inhibitory_count = kInhibitory;
    g.minimum_delay = 1;
    g.maximum_delay = 1;
    g.row_offsets = {0, 1, 1, 1};
    g.target_indices = {1};
    g.weights = {kInitialWeight};
    g.delays = {1};
    g.validate();
    return g;
}

std::uint64_t count_spikes(const SpikeEventBatch& batch, std::uint32_t id) {
    return static_cast<std::uint64_t>(std::count_if(
        batch.events.begin(), batch.events.end(),
        [id](const SpikeEvent& event) { return event.neuron_id == id; }));
}

struct Result {
    int lag_steps = 0;
    double lag_ms = 0.0;
    float final_weight = 0.0f;
    float delta_weight = 0.0f;
    double predicted_delta = 0.0;
    std::uint64_t pre_spikes = 0;
    std::uint64_t post_spikes = 0;
    std::uint64_t dropped = 0;
};

Result run_lag(const ModelParameters& model, const StdpParameters& stdp,
               int lag_steps, const std::string& topology_path) {
    const auto initial = make_topology();
    save_csr_topology(initial, topology_path);
    CudaSimulationRuntime runtime(model, topology_path, 0, true, true, stdp);

    for (std::uint32_t pair = 0; pair < kPairs; ++pair) {
        const std::uint64_t pre_step =
            kPulseStart + static_cast<std::uint64_t>(pair) * kPairIntervalSteps;
        const std::uint64_t post_step =
            static_cast<std::uint64_t>(static_cast<std::int64_t>(pre_step) + lag_steps);
        runtime.schedule_stimulus(0, pre_step, 1, 500.0f);
        runtime.schedule_stimulus(1, post_step, 1, 500.0f);
    }
    const std::uint32_t total_steps =
        kPulseStart + (kPairs - 1) * kPairIntervalSteps +
        kMaxAbsLagSteps + 10;
    runtime.run_steps(total_steps);
    const auto events = runtime.collect_spike_events();
    if (events.dropped_events != 0)
        throw std::runtime_error("Spike recorder dropped events in lag calibration");
    const auto learned = runtime.collect_connectivity();
    if (learned.synapse_count() != 1)
        throw std::runtime_error("Isolated calibration topology changed");

    const double lag_ms = static_cast<double>(lag_steps) * kDtMs;
    const double amplitude = lag_steps > 0
        ? stdp.potentiation_rate : -stdp.depression_rate;
    const double tau = lag_steps > 0
        ? stdp.presynaptic_tau_ms : stdp.postsynaptic_tau_ms;
    const double predicted = static_cast<double>(kPairs) * amplitude *
        std::exp(-std::abs(lag_ms) / tau);

    return {
        lag_steps, lag_ms, learned.weights.front(),
        learned.weights.front() - kInitialWeight, predicted,
        count_spikes(events, 0), count_spikes(events, 1), events.dropped_events
    };
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
    const std::string topology_path = "/tmp/neuromancer_v20_pair.csr";
    const std::string csv_path = "/tmp/neuromancer_v20_stdp_kernel.csv";
    try {
        const auto model = make_model();
        const auto stdp = make_stdp();
        std::ofstream csv(csv_path);
        if (!csv) throw std::runtime_error("Could not create STDP calibration CSV");
        csv << "lag_steps,lag_ms,pre_spikes,post_spikes,initial_weight,final_weight,observed_delta,predicted_isolated_delta,dropped_events\n";
        csv << std::setprecision(10);

        std::vector<Result> results;
        results.reserve(kLagSteps.size());
        for (const int lag : kLagSteps) {
            const auto result = run_lag(model, stdp, lag, topology_path);
            results.push_back(result);
            csv << result.lag_steps << ',' << result.lag_ms << ','
                << result.pre_spikes << ',' << result.post_spikes << ','
                << kInitialWeight << ',' << result.final_weight << ','
                << result.delta_weight << ',' << result.predicted_delta << ','
                << result.dropped << '\n';
        }
        csv.close();

        std::cout << "\nNEUROMANCER — ISOLATED STDP KERNEL CALIBRATION V20\n"
                  << "--------------------------------------------------\n"
                  << "Network: 2 excitatory neurons, 1 directed synapse, 1 isolated pair protocol\n"
                  << "Pairs per lag: " << kPairs
                  << "; inter-pair interval: " << kPairIntervalSteps * kDtMs << " ms\n"
                  << "STDP: A+=" << stdp.potentiation_rate
                  << ", A-=" << stdp.depression_rate
                  << ", tau+=" << stdp.presynaptic_tau_ms
                  << " ms, tau-=" << stdp.postsynaptic_tau_ms << " ms\n"
                  << "No random graph, no recurrent edges, no background input\n"
                  << "CSV: " << csv_path << "\n\n";
        std::cout << "lag_ms  pre_spikes  post_spikes  observed_delta  predicted_delta\n";
        for (const auto& r : results) {
            std::cout << r.lag_ms << "  " << r.pre_spikes << "  "
                      << r.post_spikes << "  " << r.delta_weight << "  "
                      << r.predicted_delta << '\n';
            require(r.pre_spikes == kPairs,
                    "Presynaptic stimulus did not evoke exactly one spike per pair");
            require(r.post_spikes == kPairs,
                    "Postsynaptic stimulus did not evoke exactly one spike per pair");
            require(r.dropped == 0, "Dropped events in calibration");
            if (r.lag_steps > 0)
                require(r.delta_weight > 0.0f,
                        "Causal post-after-pre timing failed to potentiate");
            else
                require(r.delta_weight < 0.0f,
                        "Anti-causal post-before-pre timing failed to depress");
        }
        require(results.size() == kLagSteps.size(), "Not all lag conditions were measured");
        std::remove(topology_path.c_str());
        std::cout << "\nPASS: isolated positive and negative timing offsets produced "
                     "the expected STDP update signs.\n";
        std::cout << "NOTE: Fit the measured lag curve to estimate A+, A-, tau+, and tau-; "
                     "this is a calibration assay, not an associative-recall test.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::remove(topology_path.c_str());
        std::cerr << "V20 calibration failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
