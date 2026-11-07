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
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr std::uint32_t kExcitatory = 3;
constexpr std::uint32_t kInhibitory = 1;
constexpr std::uint32_t kTotal = kExcitatory + kInhibitory;
constexpr std::uint32_t kTrainingPairs = 20;
constexpr std::uint32_t kPairIntervalSteps = 10;
constexpr std::uint32_t kRecallSteps = 12;
constexpr float kTau = 6.28318530717958647692f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kFrequencyHz = 10.0f;
constexpr float kGateHalfWidth = 20.0f * kPi / 180.0f;
constexpr float kInitialWeight = 0.05f;

ModelParameters make_parameters() {
    ModelParameters p;
    p.simulation.excitatory_count = kExcitatory;
    p.simulation.inhibitory_count = kInhibitory;
    p.simulation.steps = 256;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 12010;
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

StdpParameters make_stdp_parameters() {
    StdpParameters p;
    p.potentiation_rate = 50.0f;
    p.depression_rate = 0.00525f;
    p.presynaptic_tau_ms = 20.0f;
    p.postsynaptic_tau_ms = 20.0f;
    p.minimum_weight = 0.001f;
    p.maximum_weight = 250.0f;
    return p;
}

CSRTopology make_initial_topology() {
    CSRTopology graph;
    graph.excitatory_count = kExcitatory;
    graph.inhibitory_count = kInhibitory;
    graph.minimum_delay = 1;
    graph.maximum_delay = 1;
    graph.row_offsets = {0, 1, 1, 1, 1};
    graph.target_indices = {1};
    graph.weights = {kInitialWeight};
    graph.delays = {1};
    graph.validate();
    return graph;
}

std::uint64_t count_spikes(const SpikeEventBatch& batch, std::uint32_t neuron) {
    return static_cast<std::uint64_t>(std::count_if(
        batch.events.begin(), batch.events.end(),
        [neuron](const SpikeEvent& event) { return event.neuron_id == neuron; }));
}

struct RecallResult {
    std::uint64_t source_spikes = 0;
    std::uint64_t target_spikes = 0;
    std::uint64_t total_spikes = 0;
};

RecallResult run_recall(const ModelParameters& parameters,
                        const std::string& topology_file,
                        float desired_phase_rad) {
    CudaSimulationRuntime runtime(parameters, topology_file, 0, true, false);
    std::vector<float> tags(kTotal, kPi);
    tags[1] = 0.0f; // Only the learned postsynaptic target has the 0-rad address.
    const float phase_increment =
        kTau * kFrequencyHz * parameters.simulation.dt / 1000.0f;
    runtime.configure_phase_gate(
        tags, kFrequencyHz, kGateHalfWidth,
        desired_phase_rad - phase_increment);
    runtime.schedule_stimulus(0, 1, 1, 500.0f);
    runtime.run_steps(kRecallSteps);
    const auto events = runtime.collect_spike_events();
    if (events.dropped_events != 0)
        throw std::runtime_error("Recall spike-event recorder dropped events");
    return {
        count_spikes(events, 0),
        count_spikes(events, 1),
        static_cast<std::uint64_t>(events.events.size())
    };
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
    const std::string initial_file = "/tmp/neuromancer_v10_initial.csr";
    const std::string learned_file = "/tmp/neuromancer_v10_learned.csr";
    try {
        const auto parameters = make_parameters();
        const auto initial = make_initial_topology();
        save_csr_topology(initial, initial_file);

        const auto stdp = make_stdp_parameters();
        CudaSimulationRuntime trainer(
            parameters, initial_file, 0, true, true, stdp);
        require(trainer.plasticity_state_enabled(),
                "Training runtime did not enable plasticity state");

        // Repeated causal pairs: presynaptic neuron 0, then postsynaptic
        // neuron 1 one integration step later. No phase gate or recall cue
        // is active during learning; ordinary pair STDP is isolated here.
        for (std::uint32_t pair = 0; pair < kTrainingPairs; ++pair) {
            const std::uint64_t start =
                1ULL + static_cast<std::uint64_t>(pair) * kPairIntervalSteps;
            trainer.schedule_stimulus(0, start, 1, 500.0f);
            trainer.schedule_stimulus(1, start + 1ULL, 1, 500.0f);
        }
        trainer.run_steps(1U + kTrainingPairs * kPairIntervalSteps + 8U);
        const auto training_events = trainer.collect_spike_events();
        if (training_events.dropped_events != 0)
            throw std::runtime_error("Training spike-event recorder dropped events");

        const auto learned = trainer.collect_connectivity();
        require(learned.synapse_count() == 1,
                "Training unexpectedly changed the CSR edge layout");
        const float learned_weight = learned.weights.front();
        save_csr_topology(learned, learned_file);

        const auto baseline = run_recall(parameters, initial_file, 0.0f);
        const auto aligned = run_recall(parameters, learned_file, 0.0f);
        const auto mismatched = run_recall(parameters, learned_file, kPi);

        std::cout << "\nNEUROMANCER — STDP ASSOCIATION LEARNING V10\n";
        std::cout << "-------------------------------------------\n";
        std::cout << "network: one excitatory cue -> one excitatory target; fixed CSR topology\n";
        std::cout << "training: " << kTrainingPairs
                  << " causal pre/post pairs; " << kPairIntervalSteps * 0.5f
                  << " ms between pair starts\n";
        std::cout << "plasticity: conventional pair STDP; phase gate disabled during training\n";
        std::cout << "recall: frozen learned weights; no teacher spikes; one cue pulse\n";
        std::cout << "phase gate: target address 0 degrees; +/-20 degrees at 10 Hz\n\n";
        std::cout << "TRAINING\n";
        std::cout << "  cue spikes:             " << count_spikes(training_events, 0) << "\n";
        std::cout << "  teacher target spikes:  " << count_spikes(training_events, 1) << "\n";
        std::cout << "  initial cue->target weight: " << kInitialWeight << "\n";
        std::cout << "  learned cue->target weight: " << learned_weight << "\n";
        std::cout << "  weight change:          " << learned_weight - kInitialWeight << "\n\n";
        std::cout << "FROZEN RECALL\n";
        std::cout << "  untrained + aligned:    source=" << baseline.source_spikes
                  << " target=" << baseline.target_spikes << "\n";
        std::cout << "  learned + aligned:      source=" << aligned.source_spikes
                  << " target=" << aligned.target_spikes << "\n";
        std::cout << "  learned + mismatched:   source=" << mismatched.source_spikes
                  << " target=" << mismatched.target_spikes << "\n";

        require(count_spikes(training_events, 0) >= kTrainingPairs / 2,
                "Too few presynaptic training spikes were recorded");
        require(count_spikes(training_events, 1) >= kTrainingPairs / 2,
                "Too few teacher postsynaptic training spikes were recorded");
        require(learned_weight > kInitialWeight * 10.0f,
                "Causal pair STDP did not substantially potentiate the association");
        require(baseline.source_spikes > 0 && baseline.target_spikes == 0,
                "Untrained control unexpectedly recalled the target");
        require(aligned.source_spikes > 0 && aligned.target_spikes > 0,
                "Frozen learned weights failed to recall the target under aligned phase");
        require(mismatched.source_spikes > 0 && mismatched.target_spikes == 0,
                "Mismatched phase gate did not suppress learned synaptic recall");

        std::remove(initial_file.c_str());
        std::remove(learned_file.c_str());
        std::cout << "\nPASS: STDP learned a cue-target connection that survives frozen recall, "
                     "and phase gating controls its expression.\n";
        std::cout << "NOTE: Synaptic association is learned; the target's phase address "
                     "remains externally assigned.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::remove(initial_file.c_str());
        std::remove(learned_file.c_str());
        std::cerr << "V10 learning assay failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
