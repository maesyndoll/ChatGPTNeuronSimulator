#include "learning/phase_structural_plasticity.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

spiking::CSRTopology make_graph() {
    spiking::CSRTopology g;
    g.excitatory_count = 6;
    g.inhibitory_count = 2;
    g.minimum_delay = 1;
    g.maximum_delay = 1;
    g.row_offsets = {0, 1, 2, 3, 4, 5, 6, 6, 6};
    g.target_indices = {1, 0, 3, 2, 5, 4};
    g.weights = {0.003f, 0.003f, 0.003f, 0.003f, 0.003f, 0.003f};
    g.delays.assign(6, 1U);
    g.validate();
    return g;
}
}

int main() {
    try {
        // Autonomous phase: no external phase schedule is supplied. Spike
        // activity is fed back to the phase state at every step.
        spiking::AutonomousPhaseParameters pp;
        pp.intrinsic_frequency_hz = 7.0f;
        pp.activity_kick_rad = 0.08f;
        pp.coupling_learning_rate = 0.02f;
        spiking::AutonomousPopulationPhase phase(3, pp);

        std::vector<std::uint32_t> activity{2, 2, 0};
        for (int step = 0; step < 2000; ++step) {
            phase.step(activity, true);
            if (step % 100 == 99) {
                activity[0] = 2;
                activity[1] = 2;
                activity[2] = (step / 100) % 2;
            }
        }
        const float phase_a = phase.phase(0);
        const float phase_b = phase.phase(1);
        const float learned_coupling = phase.coupling(0, 1);
        require(phase.steps() == 2000, "phase step counter incorrect");
        require(std::isfinite(phase_a) && std::isfinite(phase_b),
                "non-finite autonomous phase");
        require(learned_coupling > 0.0f,
                "co-active populations did not learn phase coupling");

        // Structural plasticity: high co-activity should grow edges, while
        // old inactive edges should be pruned. The inhibitory-source rows
        // remain inhibitory and are not randomly rewired.
        auto graph = make_graph();
        std::vector<float> local_activity{0.95f, 0.95f, 0.90f, 0.90f,
                                          0.02f, 0.02f, 0.5f, 0.5f};
        std::vector<std::uint32_t> ages(graph.synapse_count(), 5U);
        spiking::StructuralPlasticityParameters sp;
        sp.growth_threshold = 0.5f;
        sp.prune_threshold = 0.01f;
        sp.maximum_out_degree = 4;
        sp.resource_per_excitatory_neuron = 1.0f;
        sp.potentiation = 0.01f;
        sp.depression = 0.02f;
        spiking::ActivityStructuralPlasticity structural(sp);
        const auto report = structural.apply(graph, local_activity, ages, true);
        require(report.births > 0, "co-active populations did not grow edges");
        require(report.deaths > 0, "inactive pathways were not pruned");
        require(report.next_edge_ages.size() == graph.synapse_count(),
                "edge ages do not match updated CSR");
        graph.validate();

        // The structural-off control must be an exact no-op.
        auto control = make_graph();
        const auto before_targets = control.target_indices;
        const auto before_weights = control.weights;
        std::vector<std::uint32_t> control_ages(control.synapse_count(), 5U);
        const auto off = structural.apply(control, local_activity, control_ages, false);
        require(off.births == 0 && off.deaths == 0 &&
                control.target_indices == before_targets &&
                control.weights == before_weights,
                "structural-off control modified topology");

        std::cout << "V42 PHASE/STRUCTURE PRIMITIVE SMOKE TEST\n"
                  << "autonomous_phase_steps=" << phase.steps() << '\n'
                  << "phase_a_rad=" << phase_a << '\n'
                  << "phase_b_rad=" << phase_b << '\n'
                  << "learned_coupling_a_to_b=" << learned_coupling << '\n'
                  << "structural_births=" << report.births << '\n'
                  << "structural_deaths=" << report.deaths << '\n'
                  << "strengthened=" << report.strengthened << '\n'
                  << "weakened=" << report.weakened << '\n'
                  << "updated_edges=" << graph.synapse_count() << '\n'
                  << "CSR_validation=PASS\n"
                  << "structural_off_control=PASS\n"
                  << "NOTE: host-side phase/structural primitives only; this is not yet "
                     "the integrated CUDA memory experiment.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "V42 primitive smoke test failed: " << e.what() << '\n';
        return 1;
    }
}
