#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "runtime/module_manager.h"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
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
constexpr float EDGE_WEIGHT = 150.0f;

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = E;
    p.simulation.inhibitory_count = I;
    p.simulation.steps = 1;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 49001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = p.simulation.inhibitory_current_max = 0.0f;
    p.topology.minimum_delay = p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology make_graph(bool include_edges) {
    CSRTopology g;
    g.excitatory_count = E;
    g.inhibitory_count = I;
    g.minimum_delay = g.maximum_delay = 1;
    g.row_offsets.assign(N + 1, 0);
    if (include_edges) {
        for (std::uint32_t s = 0; s < K; ++s) {
            g.target_indices.push_back(K + s);
            g.weights.push_back(EDGE_WEIGHT);
            g.delays.push_back(1);
        }
    }
    for (std::uint32_t s = 0; s < N; ++s)
        g.row_offsets[s + 1] = static_cast<std::uint32_t>(g.target_indices.size());
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

struct Trial {
    std::string name;
    std::uint64_t source_spikes = 0;
    std::uint64_t target_spikes = 0;
    std::uint32_t target_unique = 0;
    std::uint64_t dropped = 0;
    float target_vmax = -1000.0f;
    float weight_before = 0.0f;
    float weight_after = 0.0f;
    bool finite = true;
    bool module_detached = true;
    std::uint32_t frozen_mask_edges = 0;
};

Trial run_trial(const std::filesystem::path& dir, const std::string& name,
                bool include_edges, int policy, const std::string& plugin_path) {
    // policy: 1=open, 0=frozen, -1=no plugin/no edges.
    const auto topology_path = dir / (name + ".csr");
    save_csr_topology(make_graph(include_edges), topology_path.string());

    StdpParameters stdp;
    stdp.potentiation_rate = 0.0f;
    stdp.depression_rate = 0.0f;
    stdp.presynaptic_tau_ms = 20.0f;
    stdp.postsynaptic_tau_ms = 20.0f;
    stdp.minimum_weight = 0.001f;
    stdp.maximum_weight = 200.0f;

    CudaSimulationRuntime runtime(make_model(), topology_path.string(),
                                  0, true, true, stdp);
    std::filesystem::remove(topology_path);

    Trial result;
    result.name = name;
    const auto before = runtime.collect_connectivity();
    if (!before.weights.empty()) result.weight_before = before.weights.front();

    if (policy >= 0) {
        HostBridge bridge{&runtime, K};
        NeuromancerHostApiV1 host{
            NEUROMANCER_MODULE_ABI_V1, sizeof(NeuromancerHostApiV1),
            &bridge, host_log, host_set_edge_plasticity
        };
        RuntimeModuleManager modules(host);

        // Attach only after the CUDA simulation has begun executing.
        runtime.run_steps(2);
        modules.load(plugin_path);
        modules.configure("edge_range", policy == 0 ? "0:8:0" : "0:8:1");

        const auto configured_mask = runtime.collect_synapse_plasticity_mask();
        result.frozen_mask_edges = static_cast<std::uint32_t>(
            std::count(configured_mask.begin(), configured_mask.end(), 0U));
        modules.unload();
        result.module_detached = !modules.loaded();
    }

    std::array<bool, K> target_seen{};
    std::vector<std::uint32_t> target_ids;
    for (std::uint32_t id = K; id < 2 * K; ++id) target_ids.push_back(id);

    // Match V47: four simultaneous presynaptic cues, 4-step 500-unit pulses.
    const auto start = runtime.completed_steps() + 1U;
    for (std::uint32_t i = 0; i < 4; ++i)
        runtime.schedule_stimulus(i, start, 4, 500.0f);

    for (std::uint32_t step = 0; step < 48; ++step) {
        runtime.run_steps(1);
        for (const auto& sample : runtime.collect_voltage_samples(target_ids)) {
            result.target_vmax = std::max(result.target_vmax, sample.voltage_mv);
            result.finite = result.finite && std::isfinite(sample.voltage_mv);
        }
        const auto batch = runtime.collect_spike_events();
        result.dropped += batch.dropped_events;
        for (const auto& event : batch.events) {
            if (event.neuron_id < K) ++result.source_spikes;
            else if (event.neuron_id >= K && event.neuron_id < 2 * K) {
                ++result.target_spikes;
                target_seen[event.neuron_id - K] = true;
            }
        }
    }

    result.target_unique = static_cast<std::uint32_t>(
        std::count(target_seen.begin(), target_seen.end(), true));
    const auto after = runtime.collect_connectivity();
    after.validate();
    if (!after.weights.empty()) result.weight_after = after.weights.front();
    result.finite = result.finite && runtime.collect_metrics().finite_state;
    return result;
}
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: neuromancer_frozen_edge_transmission_v49 <module.so> [output-dir]\n";
        return 2;
    }
    const std::string plugin_path = argv[1];
    const std::filesystem::path dir =
        argc > 2 ? argv[2] : "experiments/logs/v49_frozen_edge";
    try {
        std::filesystem::create_directories(dir);
        std::vector<Trial> trials;
        trials.push_back(run_trial(dir, "open_edge_after_detach", true, 1, plugin_path));
        trials.push_back(run_trial(dir, "frozen_edge_after_detach", true, 0, plugin_path));
        trials.push_back(run_trial(dir, "no_edge_negative", false, -1, plugin_path));

        std::ofstream csv(dir / "v49_frozen_edge_transmission.csv");
        if (!csv) throw std::runtime_error("cannot create V49 CSV");
        csv << "condition,source_spikes,target_spikes,target_unique,weight_before,weight_after,frozen_mask_edges,target_vmax_mv,dropped,finite,module_detached\n";
        for (const auto& t : trials) {
            csv << t.name << ',' << t.source_spikes << ',' << t.target_spikes << ','
                << t.target_unique << ',' << t.weight_before << ',' << t.weight_after
                << ',' << t.frozen_mask_edges << ',' << t.target_vmax << ','
                << t.dropped << ',' << (t.finite ? 1 : 0) << ','
                << (t.module_detached ? 1 : 0) << '\n';
            std::cout << "condition=" << t.name
                      << " source_spikes=" << t.source_spikes
                      << " target_spikes=" << t.target_spikes
                      << " target_unique=" << t.target_unique << "/8"
                      << " weight_before=" << t.weight_before
                      << " weight_after=" << t.weight_after
                      << " frozen_mask_edges=" << t.frozen_mask_edges
                      << " target_vmax_mv=" << t.target_vmax
                      << " dropped=" << t.dropped
                      << " finite=" << (t.finite ? "PASS" : "FAIL")
                      << " module_detached=" << (t.module_detached ? "PASS" : "FAIL")
                      << '\n';
        }

        const auto& open = trials[0];
        const auto& frozen = trials[1];
        const auto& negative = trials[2];
        const bool open_pass = open.source_spikes > 0 && open.target_unique > 0;
        const bool frozen_pass = frozen.source_spikes > 0 && frozen.target_unique > 0 &&
                                  frozen.frozen_mask_edges == K &&
                                  std::fabs(frozen.weight_after - frozen.weight_before) < 1.0e-6f;
        const bool negative_pass = negative.target_spikes == 0 &&
                                   negative.target_unique == 0;
        bool finite = true, no_drops = true, detached = true;
        for (const auto& t : trials) {
            finite = finite && t.finite;
            no_drops = no_drops && t.dropped == 0;
            detached = detached && t.module_detached;
        }
        std::cout << "open_edge_transmission=" << (open_pass ? "PASS" : "FAIL") << '\n'
                  << "frozen_edge_transmission_after_unload=" << (frozen_pass ? "PASS" : "FAIL") << '\n'
                  << "no_edge_negative_control=" << (negative_pass ? "PASS" : "FAIL") << '\n'
                  << "finite_state=" << (finite ? "PASS" : "FAIL") << '\n'
                  << "no_dropped_events=" << (no_drops ? "PASS" : "FAIL") << '\n'
                  << "module_detachment=" << (detached ? "PASS" : "FAIL") << '\n'
                  << "artifacts=" << (dir / "v49_frozen_edge_transmission.csv").string() << '\n'
                  << "LIMITATION=V49 isolates whether the per-edge STDP mask affects propagation, using matched V47-style multi-neuron stimulation. It does not test learning or phase-dependent plasticity.\n";
        return open_pass && frozen_pass && negative_pass && finite &&
               no_drops && detached ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "V49 failed: " << e.what() << '\n';
        return 1;
    }
}
