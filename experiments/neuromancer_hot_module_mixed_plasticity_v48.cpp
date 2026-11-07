#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "runtime/module_manager.h"
#include "topology/topology_io.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr std::uint32_t N = 4;

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = N;
    p.simulation.inhibitory_count = 0;
    p.simulation.steps = 1;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 48001;
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
    CSRTopology graph;
    graph.excitatory_count = N;
    graph.inhibitory_count = 0;
    graph.minimum_delay = graph.maximum_delay = 1;
    graph.row_offsets = {0, 1, 1, 2, 2};
    graph.target_indices = {1, 3};
    graph.weights = {30.0f, 30.0f};
    graph.delays = {1, 1};
    graph.validate();
    return graph;
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
    if (bridge == nullptr || bridge->runtime == nullptr ||
        first > bridge->edge_count || count > bridge->edge_count - first)
        return 1;
    // Read-modify-write is committed by the runtime at a CUDA stream boundary.
    auto mask = bridge->runtime->collect_synapse_plasticity_mask();
    for (std::uint32_t i = first; i < first + count; ++i) mask[i] = enabled ? 1U : 0U;
    bridge->runtime->set_synapse_plasticity_mask(mask);
    return 0;
}

void train_pair(CudaSimulationRuntime& runtime, RuntimeModuleManager& modules,
                std::uint32_t pre, std::uint32_t post, std::uint32_t episodes) {
    for (std::uint32_t episode = 0; episode < episodes; ++episode) {
        const auto start = runtime.completed_steps() + 1U;
        runtime.schedule_stimulus(pre, start, 4, 500.0f);
        runtime.schedule_stimulus(post, start + 2U, 4, 500.0f);
        for (std::uint32_t step = 0; step < 12; ++step) {
            runtime.run_steps(1);
            const auto batch = runtime.collect_spike_events();
            std::vector<NeuromancerSpikeEventV1> events;
            events.reserve(batch.events.size());
            for (const auto& event : batch.events)
                events.push_back({event.step, event.neuron_id});
            modules.dispatch_spikes(events.data(), static_cast<std::uint32_t>(events.size()));
            if (batch.dropped_events != 0)
                throw std::runtime_error("Spike event ring overflow during training");
        }
    }
}

bool spike_seen(CudaSimulationRuntime& runtime, std::uint32_t neuron,
                std::uint32_t source, std::uint32_t steps) {
    const auto start = runtime.completed_steps() + 1U;
    runtime.schedule_stimulus(source, start, 4, 500.0f);
    bool seen = false;
    for (std::uint32_t i = 0; i < steps; ++i) {
        runtime.run_steps(1);
        const auto batch = runtime.collect_spike_events();
        for (const auto& event : batch.events)
            if (event.neuron_id == neuron) seen = true;
        if (batch.dropped_events != 0) throw std::runtime_error("Spike event ring overflow");
    }
    return seen;
}
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: neuromancer_hot_module_mixed_plasticity_v48 <module.so> [output-dir]\n";
        return 2;
    }
    const std::filesystem::path module_path = argv[1];
    const std::filesystem::path output_dir =
        argc > 2 ? argv[2] : "experiments/logs/v48_hot_module";
    try {
        std::filesystem::create_directories(output_dir);
        const auto topology_path = output_dir / "v48_two_edge.csr";
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
        HostBridge bridge{&runtime, 2};
        NeuromancerHostApiV1 host{
            NEUROMANCER_MODULE_ABI_V1, sizeof(NeuromancerHostApiV1),
            &bridge, host_log, host_set_edge_plasticity
        };
        RuntimeModuleManager modules(host);

        // The neural runtime is already alive before the plugin is attached.
        runtime.run_steps(3);
        modules.load(module_path.string());
        std::cout << "module_loaded=" << modules.module_name() << '\n';
        // Edge 0 remains open; edge 1 is frozen. Both remain transmissive.
        modules.configure("edge_range", "1:1:0");
        const auto initial = runtime.collect_connectivity();
        train_pair(runtime, modules, 0, 1, 30);
        train_pair(runtime, modules, 2, 3, 30);
        const auto trained = runtime.collect_connectivity();

        const float open_before = initial.weights.at(0);
        const float frozen_before = initial.weights.at(1);
        const float open_after = trained.weights.at(0);
        const float frozen_after = trained.weights.at(1);
        const bool open_changed = open_after > open_before + 1.0e-5f;
        const bool frozen_unchanged = std::fabs(frozen_after - frozen_before) < 1.0e-6f;

        runtime.clear_stimuli();
        modules.unload();
        std::cout << "module_detached=" << (!modules.loaded() ? "PASS" : "FAIL") << '\n';

        // After detachment, the same persistent CUDA runtime still propagates.
        const bool frozen_transmits = spike_seen(runtime, 3, 2, 24);
        const bool open_transmits = spike_seen(runtime, 1, 0, 24);
        const auto final_graph = runtime.collect_connectivity();
        final_graph.validate();
        const bool frozen_weight_still_stable =
            std::fabs(final_graph.weights.at(1) - frozen_after) < 1.0e-6f;

        // Independently verify that phase gating changes transmission, not weight state.
        const std::vector<float> aligned_phases(N, 0.0f);
        const std::vector<float> edge_lags(2, 0.0f);
        runtime.configure_dynamic_phase_gate(aligned_phases, edge_lags, 0.2f);
        const bool phase_aligned_transmits = spike_seen(runtime, 1, 0, 24);
        auto misaligned_phases = aligned_phases;
        misaligned_phases[1] = 3.14159265358979323846f;
        runtime.update_dynamic_phase_state(misaligned_phases, edge_lags);
        const bool phase_misaligned_suppresses =
            !spike_seen(runtime, 1, 0, 24);
        runtime.disable_phase_gate();

        const auto csv_path = output_dir / "v48_mixed_plasticity.csv";
        std::ofstream csv(csv_path);
        if (!csv) throw std::runtime_error("cannot create V48 CSV");
        csv << "condition,edge,initial_weight,final_weight,passed\n";
        csv << "open_learning,0," << open_before << ',' << open_after << ','
            << (open_changed ? 1 : 0) << '\n';
        csv << "frozen_learning,1," << frozen_before << ',' << frozen_after << ','
            << (frozen_unchanged ? 1 : 0) << '\n';
        csv << "frozen_transmission,1," << frozen_after << ','
            << final_graph.weights.at(1) << ',' << (frozen_transmits ? 1 : 0) << '\n';
        const auto phase_final_weights = runtime.collect_connectivity();
        csv << "phase_aligned_transmission,0," << open_after << ','
            << phase_final_weights.weights.at(0) << ','
            << (phase_aligned_transmits ? 1 : 0) << '\n';
        csv << "phase_misaligned_suppression,0," << open_after << ','
            << phase_final_weights.weights.at(0) << ','
            << (phase_misaligned_suppresses ? 1 : 0) << '\n';
        csv.close();

        std::cout << "edge0_mode=open initial_weight=" << open_before
                  << " trained_weight=" << open_after << '\n'
                  << "edge1_mode=frozen initial_weight=" << frozen_before
                  << " trained_weight=" << frozen_after << '\n'
                  << "open_edge_weight_changed=" << (open_changed ? "PASS" : "FAIL") << '\n'
                  << "frozen_edge_weight_unchanged=" << (frozen_unchanged ? "PASS" : "FAIL") << '\n'
                  << "frozen_edge_transmission_after_detach=" << (frozen_transmits ? "PASS" : "FAIL") << '\n'
                  << "open_edge_transmission_after_detach=" << (open_transmits ? "PASS" : "FAIL") << '\n'
                  << "frozen_weight_persisted_after_detach=" << (frozen_weight_still_stable ? "PASS" : "FAIL") << '\n'
                  << "phase_aligned_transmission=" << (phase_aligned_transmits ? "PASS" : "FAIL") << '\n'
                  << "phase_misaligned_suppression=" << (phase_misaligned_suppresses ? "PASS" : "FAIL") << '\n'
                  << "completed_steps=" << runtime.completed_steps() << '\n'
                  << "artifacts=" << csv_path.string() << '\n'
                  << "LIMITATION=V48 proves dynamically loaded host module lifecycle and per-edge CUDA STDP enable/freeze policy on a persistent runtime. Module unload is invoked between synchronized run batches; this is not concurrent lock-free hot replacement or GPU-resident plugin execution.\n";

        return open_changed && frozen_unchanged && frozen_transmits &&
               open_transmits && frozen_weight_still_stable &&
               phase_aligned_transmits && phase_misaligned_suppresses ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "V48 failed: " << error.what() << '\n';
        return 1;
    }
}
