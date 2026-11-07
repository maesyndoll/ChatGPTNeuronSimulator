#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = 2;
    p.simulation.inhibitory_count = 1;
    p.simulation.steps = 1;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 42420;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;
    p.topology.p_ee = p.topology.p_ei = p.topology.p_ie = p.topology.p_ii = 0.0;
    p.topology.minimum_delay = p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology make_graph() {
    CSRTopology g;
    g.excitatory_count = 2;
    g.inhibitory_count = 1;
    g.minimum_delay = g.maximum_delay = 1;
    g.row_offsets = {0, 1, 1, 1};
    g.target_indices = {1};
    g.weights = {30.0f};
    g.delays = {1};
    g.validate();
    return g;
}

struct Result {
    std::string condition;
    std::uint64_t source_spikes = 0;
    std::uint64_t target_spikes = 0;
    float maximum_target_voltage = -1000.0f;
    float minimum_target_voltage = 1000.0f;
    bool finite = true;
};

Result run_condition(const std::string& condition,
                     const std::filesystem::path& topology_path,
                     int gate_mode) {
    save_csr_topology(make_graph(), topology_path.string());
    CudaSimulationRuntime runtime(make_model(), topology_path.string(), 0, true, false);
    std::filesystem::remove(topology_path);

    if (gate_mode != 0) {
        const float pi = 3.14159265358979323846f;
        const std::vector<float> phases =
            gate_mode == 1 ? std::vector<float>{0.0f, 0.0f, 0.0f}
                           : std::vector<float>{0.0f, pi, 0.0f};
        runtime.configure_dynamic_phase_gate(phases, {0.0f}, 0.35f);
    }

    const auto start = runtime.completed_steps() + 1U;
    runtime.schedule_stimulus(0, start, 4, 500.0f);

    Result result;
    result.condition = condition;
    for (std::uint32_t step = 0; step < 40; ++step) {
        runtime.run_steps(1);
        const auto events = runtime.collect_spike_events();
        if (events.dropped_events != 0)
            throw std::runtime_error("Spike recorder dropped events");
        for (const auto& event : events.events) {
            if (event.neuron_id == 0) ++result.source_spikes;
            if (event.neuron_id == 1) ++result.target_spikes;
        }
        const auto samples = runtime.collect_voltage_samples({1});
        if (samples.size() != 1)
            throw std::runtime_error("Expected exactly one target voltage sample");
        const float voltage = samples.front().voltage_mv;
        result.finite = result.finite && std::isfinite(voltage);
        result.maximum_target_voltage =
            std::max(result.maximum_target_voltage, voltage);
        result.minimum_target_voltage =
            std::min(result.minimum_target_voltage, voltage);
    }
    result.finite = result.finite && runtime.collect_metrics().finite_state;
    runtime.collect_connectivity().validate();
    return result;
}
}

int main(int argc, char** argv) {
    try {
        const std::filesystem::path dir = argc > 1 ? argv[1] :
            "experiments/logs/v42_phase_transfer";
        std::filesystem::create_directories(dir);

        const auto ungated = run_condition("ungated", dir / "ungated.csr", 0);
        const auto aligned = run_condition("aligned", dir / "aligned.csr", 1);
        const auto misaligned = run_condition("misaligned", dir / "misaligned.csr", 2);

        std::cout << "V42 CONTROLLED CUDA PHASE-TRANSFER TEST\n";
        for (const auto* r : {&ungated, &aligned, &misaligned}) {
            std::cout << "condition=" << r->condition
                      << " source_spikes=" << r->source_spikes
                      << " target_spikes=" << r->target_spikes
                      << " target_v_min_mv=" << r->minimum_target_voltage
                      << " target_v_max_mv=" << r->maximum_target_voltage
                      << " finite=" << (r->finite ? "PASS" : "FAIL") << '\n';
        }

        const bool matched_sources =
            ungated.source_spikes == aligned.source_spikes &&
            aligned.source_spikes == misaligned.source_spikes;
        const bool aligned_matches_ungated =
            aligned.target_spikes == ungated.target_spikes &&
            std::fabs(aligned.maximum_target_voltage -
                      ungated.maximum_target_voltage) < 1.0e-4f;
        const bool misaligned_suppressed =
            misaligned.target_spikes < aligned.target_spikes ||
            misaligned.maximum_target_voltage < aligned.maximum_target_voltage - 1.0f;
        std::cout << "matched_source_spikes="
                  << (matched_sources ? "PASS" : "FAIL") << '\n'
                  << "aligned_matches_ungated="
                  << (aligned_matches_ungated ? "PASS" : "FAIL") << '\n'
                  << "misaligned_transmission_suppressed="
                  << (misaligned_suppressed ? "PASS" : "FAIL") << '\n'
                  << "interpretation=phase-gate transfer control only; "
                     "no phase learning is claimed\n";
        return matched_sources && aligned_matches_ungated &&
               misaligned_suppressed && ungated.finite &&
               aligned.finite && misaligned.finite ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "V42 phase-transfer test failed: " << error.what() << '\n';
        return 1;
    }
}
