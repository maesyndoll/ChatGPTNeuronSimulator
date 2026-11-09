#include "core/parameters.h"
#include "learning/phase_structural_plasticity.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace spiking;
constexpr float kTau = 6.28318530717958647692f;

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = 2;
    p.simulation.inhibitory_count = 1;
    p.simulation.steps = 1;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 42042;
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
    g.row_offsets = {0, 1, 1, 2};
    g.target_indices = {1, 1};
    g.weights = {2.0f, -2.0f};
    g.delays = {1, 1};
    g.validate();
    return g;
}
}

int main(int argc, char** argv) {
    try {
        const std::filesystem::path dir = argc > 1 ? argv[1] :
            "experiments/logs/v42_dynamic_phase";
        std::filesystem::create_directories(dir);
        const auto topology_path = (dir / "dynamic_phase.csr").string();
        save_csr_topology(make_graph(), topology_path);

        auto model = make_model();
        CudaSimulationRuntime runtime(model, topology_path, 0, true, false);
        std::filesystem::remove(topology_path);

        AutonomousPhaseParameters pp;
        pp.dt_ms = model.simulation.dt;
        pp.intrinsic_frequency_hz = 8.0f;
        pp.activity_kick_rad = 0.10f;
        pp.coupling_learning_rate = 0.001f;
        pp.maximum_coupling = 1.0f;
        AutonomousPopulationPhase phase(3, pp);

        std::vector<float> neuron_phases(3, 0.0f);
        std::vector<float> edge_lags(2, 0.0f);
        runtime.configure_dynamic_phase_gate(neuron_phases, edge_lags, 0.75f);

        const auto start = runtime.completed_steps() + 1U;
        runtime.schedule_stimulus(0, start, 4, 500.0f);
        runtime.schedule_stimulus(2, start + 2U, 4, 500.0f);

        std::uint64_t e_source_spikes = 0;
        std::uint64_t i_source_spikes = 0;
        std::uint64_t target_spikes = 0;
        for (std::uint32_t step = 0; step < 80; ++step) {
            runtime.run_steps(1);
            const auto events = runtime.collect_spike_events();
            if (events.dropped_events != 0)
                throw std::runtime_error("Spike recorder dropped events");
            std::vector<std::uint32_t> population_spikes(3, 0U);
            for (const auto& event : events.events) {
                if (event.neuron_id == 0) { ++population_spikes[0]; ++e_source_spikes; }
                else if (event.neuron_id == 1) ++target_spikes;
                else if (event.neuron_id == 2) { ++population_spikes[2]; ++i_source_spikes; }
            }
            // The target is its own population; only observed spikes perturb it.
            for (const auto& event : events.events)
                if (event.neuron_id == 1) ++population_spikes[1];
            phase.step(population_spikes, true);
            for (std::uint32_t neuron = 0; neuron < neuron_phases.size(); ++neuron)
                neuron_phases[neuron] = phase.phase(neuron);
            edge_lags[0] = phase.learned_lag(0, 1);
            edge_lags[1] = phase.learned_lag(2, 1);
            runtime.update_dynamic_phase_state(neuron_phases, edge_lags);
        }

        const auto metrics = runtime.collect_metrics();
        const auto graph = runtime.collect_connectivity();
        graph.validate();
        const auto phase_delta = std::remainder(
            phase.phase(0) - phase.phase(1), kTau);
        std::cout << "V42 CUDA DYNAMIC-PHASE GATE INTEGRATION\n"
                  << "steps=" << runtime.completed_steps() << '\n'
                  << "E_source_spikes=" << e_source_spikes << '\n'
                  << "I_source_spikes=" << i_source_spikes << '\n'
                  << "target_spikes=" << target_spikes << '\n'
                  << "phase_A_rad=" << phase.phase(0) << '\n'
                  << "phase_target_rad=" << phase.phase(1) << '\n'
                  << "phase_I_rad=" << phase.phase(2) << '\n'
                  << "learned_A_to_target_lag=" << edge_lags[0] << '\n'
                  << "learned_I_to_target_lag=" << edge_lags[1] << '\n'
                  << "A_target_phase_difference=" << phase_delta << '\n'
                  << "finite_state=" << (metrics.finite_state ? "PASS" : "FAIL") << '\n'
                  << "CSR_validation=PASS\n"
                  << "NOTE: endogenous phase state is updated on host from observed CUDA spike events; "
                     "the learned relative-phase gate is applied in CUDA propagation. "
                     "This is a hybrid runtime integration, not yet fully device-resident phase dynamics.\n";
        return metrics.finite_state ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "V42 dynamic phase integration failed: " << error.what() << '\n';
        return 1;
    }
}
