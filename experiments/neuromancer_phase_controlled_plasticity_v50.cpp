#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "runtime/module_manager.h"
#include "topology/topology_io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr std::uint32_t K = 8;
constexpr std::uint32_t E = 2 * K;
constexpr std::uint32_t I = 1;
constexpr std::uint32_t N = E + I;
constexpr float PI = 3.14159265358979323846f;
constexpr float INITIAL_WEIGHT = 150.0f;

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = E;
    p.simulation.inhibitory_count = I;
    p.simulation.steps = 1;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 50001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = p.simulation.inhibitory_current_max = 0.0f;
    p.topology.minimum_delay = p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology make_graph() {
    CSRTopology g;
    g.excitatory_count = E;
    g.inhibitory_count = I;
    g.minimum_delay = g.maximum_delay = 1;
    g.row_offsets.assign(N + 1, 0);
    for (std::uint32_t s = 0; s < N; ++s) {
        if (s < K) {
            g.target_indices.push_back(K + s);
            g.weights.push_back(INITIAL_WEIGHT);
            g.delays.push_back(1);
        }
        g.row_offsets[s + 1] = static_cast<std::uint32_t>(g.target_indices.size());
    }
    g.validate();
    return g;
}

struct HostBridge {
    CudaSimulationRuntime* runtime = nullptr;
    std::uint32_t edge_count = 0;
};

void host_log(void*, int, const char* message) {
    if (message) std::cout << "module=" << message << '\n';
}

int host_set_edge_plasticity(void* opaque, std::uint32_t first,
                             std::uint32_t count, std::uint8_t enabled) {
    auto* bridge = static_cast<HostBridge*>(opaque);
    if (!bridge || !bridge->runtime || first > bridge->edge_count ||
        count > bridge->edge_count - first) return 1;
    auto mask = bridge->runtime->collect_synapse_plasticity_mask();
    if (mask.size() != bridge->edge_count) return 2;
    for (std::uint32_t i = first; i < first + count; ++i)
        mask[i] = enabled ? 1U : 0U;
    bridge->runtime->set_synapse_plasticity_mask(mask);
    return 0;
}

struct Result {
    std::string condition;
    float initial_weight = 0.0f;
    float final_weight = 0.0f;
    std::uint64_t source_spikes = 0;
    std::uint64_t target_spikes = 0;
    std::uint32_t target_unique = 0;
    std::uint32_t masked_edges = 0;
    std::uint64_t dropped = 0;
    float target_vmax = -1000.0f;
    bool finite = true;
    bool detached = false;
};

Result run_condition(const std::filesystem::path& output_dir,
                     const std::string& plugin_path,
                     const std::string& condition,
                     bool aligned, bool plastic) {
    const auto topology_path = output_dir / (condition + ".csr");
    save_csr_topology(make_graph(), topology_path.string());

    StdpParameters stdp;
    stdp.potentiation_rate = 0.08f;
    stdp.depression_rate = 0.01f;
    stdp.presynaptic_tau_ms = 20.0f;
    stdp.postsynaptic_tau_ms = 20.0f;
    stdp.minimum_weight = 0.001f;
    stdp.maximum_weight = 200.0f;

    CudaSimulationRuntime runtime(make_model(), topology_path.string(),
                                  0, true, true, stdp);
    std::filesystem::remove(topology_path);

    Result result;
    result.condition = condition;
    result.initial_weight = runtime.collect_connectivity().weights.at(0);

    // Relative phase is static for this controlled assay: only the target
    // phase differs between aligned and misaligned conditions.
    std::vector<float> phases(N, 0.0f);
    if (!aligned)
        for (std::uint32_t t = K; t < 2 * K; ++t) phases[t] = PI;
    const std::vector<float> edge_lags(K, 0.0f);
    runtime.configure_dynamic_phase_gate(phases, edge_lags, 0.20f);

    HostBridge bridge{&runtime, K};
    NeuromancerHostApiV1 host{
        NEUROMANCER_MODULE_ABI_V1, sizeof(NeuromancerHostApiV1),
        &bridge, host_log, host_set_edge_plasticity
    };
    RuntimeModuleManager modules(host);

    // Attach the hot-swappable policy module to an already-running runtime.
    runtime.run_steps(2);
    modules.load(plugin_path);
    modules.configure("edge_range", plastic ? "0:8:1" : "0:8:0");
    const auto mask = runtime.collect_synapse_plasticity_mask();
    result.masked_edges = static_cast<std::uint32_t>(
        std::count(mask.begin(), mask.end(), 0U));

    std::vector<bool> target_seen(K, false);
    constexpr std::uint32_t EPISODES = 24;
    for (std::uint32_t episode = 0; episode < EPISODES; ++episode) {
        const auto start = runtime.completed_steps() + 1U;
        for (std::uint32_t s = 0; s < 4; ++s)
            runtime.schedule_stimulus(s, start, 4, 500.0f);

        for (std::uint32_t step = 0; step < 12; ++step) {
            runtime.run_steps(1);
            const auto batch = runtime.collect_spike_events();
            result.dropped += batch.dropped_events;
            std::vector<NeuromancerSpikeEventV1> events;
            events.reserve(batch.events.size());
            for (const auto& event : batch.events) {
                events.push_back({event.step, event.neuron_id});
                if (event.neuron_id < K) ++result.source_spikes;
                else if (event.neuron_id < 2 * K) {
                    ++result.target_spikes;
                    target_seen[event.neuron_id - K] = true;
                }
            }
            modules.dispatch_spikes(events.data(),
                static_cast<std::uint32_t>(events.size()));
        }
    }

    for (const auto& sample : runtime.collect_voltage_samples(
             [&] {
                 std::vector<std::uint32_t> ids;
                 for (std::uint32_t t = K; t < 2 * K; ++t) ids.push_back(t);
                 return ids;
             }())) {
        result.target_vmax = std::max(result.target_vmax, sample.voltage_mv);
        result.finite = result.finite && std::isfinite(sample.voltage_mv);
    }

    result.final_weight = runtime.collect_connectivity().weights.at(0);
    result.target_unique = static_cast<std::uint32_t>(
        std::count(target_seen.begin(), target_seen.end(), true));
    result.finite = result.finite && runtime.collect_metrics().finite_state;

    modules.unload();
    result.detached = !modules.loaded();
    return result;
}
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: neuromancer_phase_controlled_plasticity_v50 <module.so> [output-dir]\n";
        return 2;
    }
    const std::string plugin_path = argv[1];
    const std::filesystem::path dir =
        argc > 2 ? argv[2] : "experiments/logs/v50_phase_plasticity";
    try {
        std::filesystem::create_directories(dir);
        std::vector<Result> results;
        results.push_back(run_condition(dir, plugin_path, "aligned_open", true, true));
        results.push_back(run_condition(dir, plugin_path, "misaligned_open", false, true));
        results.push_back(run_condition(dir, plugin_path, "aligned_frozen", true, false));
        results.push_back(run_condition(dir, plugin_path, "misaligned_frozen", false, false));

        std::ofstream csv(dir / "v50_phase_controlled_plasticity.csv");
        if (!csv) throw std::runtime_error("cannot create V50 CSV");
        csv << "condition,initial_weight,final_weight,source_spikes,target_spikes,target_unique,masked_edges,target_vmax_mv,dropped,finite,module_detached\n";
        for (const auto& r : results) {
            csv << r.condition << ',' << r.initial_weight << ',' << r.final_weight
                << ',' << r.source_spikes << ',' << r.target_spikes << ','
                << r.target_unique << ',' << r.masked_edges << ','
                << r.target_vmax << ',' << r.dropped << ','
                << (r.finite ? 1 : 0) << ',' << (r.detached ? 1 : 0) << '\n';
            std::cout << "condition=" << r.condition
                      << " weight=" << r.initial_weight << "->" << r.final_weight
                      << " source_spikes=" << r.source_spikes
                      << " target_spikes=" << r.target_spikes
                      << " target_unique=" << r.target_unique << "/8"
                      << " masked_edges=" << r.masked_edges
                      << " target_vmax_mv=" << r.target_vmax
                      << " dropped=" << r.dropped
                      << " finite=" << (r.finite ? "PASS" : "FAIL")
                      << " module_detached=" << (r.detached ? "PASS" : "FAIL")
                      << '\n';
        }

        const auto& aligned_open = results[0];
        const auto& misaligned_open = results[1];
        const auto& aligned_frozen = results[2];
        const auto& misaligned_frozen = results[3];
        const bool aligned_transmission = aligned_open.target_unique > 0;
        const bool misaligned_suppression = misaligned_open.target_spikes == 0;
        const bool phase_selective_learning =
            aligned_open.final_weight > aligned_open.initial_weight + 1.0e-4f &&
            misaligned_open.final_weight <= misaligned_open.initial_weight + 1.0e-4f;
        const bool frozen_stable =
            std::fabs(aligned_frozen.final_weight - aligned_frozen.initial_weight) < 1.0e-6f &&
            std::fabs(misaligned_frozen.final_weight - misaligned_frozen.initial_weight) < 1.0e-6f;
        const bool frozen_phase_control =
            aligned_frozen.target_unique > 0 && misaligned_frozen.target_spikes == 0;
        bool finite = true, no_drops = true, detached = true;
        for (const auto& r : results) {
            finite = finite && r.finite;
            no_drops = no_drops && r.dropped == 0;
            detached = detached && r.detached;
        }
        std::cout << "aligned_transmission=" << (aligned_transmission ? "PASS" : "FAIL") << '\n'
                  << "misaligned_transmission_suppressed=" << (misaligned_suppression ? "PASS" : "FAIL") << '\n'
                  << "phase_selective_weight_learning=" << (phase_selective_learning ? "PASS" : "FAIL") << '\n'
                  << "frozen_weights_stable=" << (frozen_stable ? "PASS" : "FAIL") << '\n'
                  << "frozen_phase_controls_transmission=" << (frozen_phase_control ? "PASS" : "FAIL") << '\n'
                  << "finite_state=" << (finite ? "PASS" : "FAIL") << '\n'
                  << "no_dropped_events=" << (no_drops ? "PASS" : "FAIL") << '\n'
                  << "module_lifecycle=" << (detached ? "PASS" : "FAIL") << '\n'
                  << "artifacts=" << (dir / "v50_phase_controlled_plasticity.csv").string() << '\n'
                  << "LIMITATION=V50 tests whether phase-gated spike transmission causally changes the postsynaptic events available to CUDA STDP. Phase state is configured by a host test controller and held fixed per condition; this is not autonomous phase learning or GPU-resident plugin execution.\n";
        return aligned_transmission && misaligned_suppression &&
               phase_selective_learning && frozen_stable &&
               frozen_phase_control && finite && no_drops && detached ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "V50 failed: " << e.what() << '\n';
        return 1;
    }
}
