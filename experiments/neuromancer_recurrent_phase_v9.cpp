#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/csr_topology.h"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr std::uint32_t kAssemblies = 3;
constexpr std::uint32_t kPerAssembly = 8;
constexpr std::uint32_t kExcitatory = kAssemblies * kPerAssembly;
constexpr std::uint32_t kInhibitory = 6;
constexpr std::uint32_t kTotal = kExcitatory + kInhibitory;
constexpr std::uint32_t kSteps = 32;
constexpr float kTau = 6.28318530717958647692f;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kFrequencyHz = 10.0f;
constexpr float kGateHalfWidth = 20.0f * kPi / 180.0f;

enum class GateCondition { Ungated, Aligned, Mismatched, Shuffled };

struct Metrics {
    std::array<std::uint64_t, kAssemblies> excitatory{};
    std::uint64_t inhibitory = 0;
    std::uint64_t source_spikes = 0;
    std::uint64_t total_events = 0;
};

ModelParameters parameters() {
    ModelParameters p;
    p.simulation.excitatory_count = kExcitatory;
    p.simulation.inhibitory_count = kInhibitory;
    p.simulation.steps = kSteps;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 9001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;
    p.topology.minimum_delay = 1;
    p.topology.maximum_delay = 1;
    return p;
}

CSRTopology make_recurrent_topology() {
    CSRTopology graph;
    graph.excitatory_count = kExcitatory;
    graph.inhibitory_count = kInhibitory;
    graph.minimum_delay = 1;
    graph.maximum_delay = 1;
    graph.row_offsets.resize(kTotal + 1, 0);

    for (std::uint32_t source = 0; source < kTotal; ++source) {
        for (std::uint32_t target = 0; target < kTotal; ++target) {
            if (source == target) continue;
            graph.target_indices.push_back(target);
            graph.weights.push_back(source < kExcitatory
                ? (target < kExcitatory ? 200.0f : 180.0f)
                : (target < kExcitatory ? -80.0f : -12.0f));
            graph.delays.push_back(1);
        }
        graph.row_offsets[source + 1] =
            static_cast<std::uint32_t>(graph.target_indices.size());
    }
    graph.validate();
    return graph;
}

std::vector<float> phase_tags(GateCondition condition) {
    std::vector<float> tags(kTotal, 0.0f);
    for (std::uint32_t neuron = 0; neuron < kExcitatory; ++neuron)
        tags[neuron] = kTau * static_cast<float>(neuron / kPerAssembly) /
            static_cast<float>(kAssemblies);
    // Inhibitory neurons share assembly zero's phase address.
    for (std::uint32_t neuron = kExcitatory; neuron < kTotal; ++neuron)
        tags[neuron] = 0.0f;

    if (condition == GateCondition::Shuffled) {
        std::mt19937 rng(44221);
        std::shuffle(tags.begin(), tags.end(), rng);
    }
    return tags;
}

Metrics run_condition(const std::string& topology_file, GateCondition condition) {
    auto p = parameters();
    CudaSimulationRuntime runtime(p, topology_file, 0, true, false);

    if (condition != GateCondition::Ungated) {
        const auto tags = phase_tags(condition);
        const float increment = kTau * kFrequencyHz * p.simulation.dt / 1000.0f;
        const float desired_phase = condition == GateCondition::Mismatched ? kPi : 0.0f;
        // The first stimulated spike is emitted at step 1 and propagated when
        // completed_steps == 1. Align the oscillator at that exact boundary.
        runtime.configure_phase_gate(tags, kFrequencyHz, kGateHalfWidth,
                                      desired_phase - increment);
    }

    runtime.schedule_stimulus(0, 1, 1, 500.0f);
    runtime.run_steps(kSteps);
    const auto simulation = runtime.collect_metrics();
    if (!simulation.finite_state || simulation.completed_steps != kSteps)
        throw std::runtime_error("CUDA recurrent assay produced invalid simulation state");

    const auto events = runtime.collect_spike_events();
    if (events.dropped_events != 0)
        throw std::runtime_error("CUDA spike-event recorder dropped events");

    Metrics result;
    for (const auto& event : events.events) {
        ++result.total_events;
        if (event.neuron_id < kExcitatory) {
            ++result.excitatory[event.neuron_id / kPerAssembly];
            if (event.neuron_id == 0) ++result.source_spikes;
        } else {
            ++result.inhibitory;
        }
    }
    return result;
}

std::uint64_t recruited(const Metrics& m, std::uint32_t assembly) {
    // Exclude the externally stimulated seed from recruited assembly activity.
    return m.excitatory[assembly] -
        (assembly == 0 ? m.source_spikes : 0U);
}

void print_metrics(const char* label, const Metrics& m) {
    std::cout << label << "\n";
    std::cout << "  assembly 0 recruited: " << recruited(m, 0) << "\n";
    std::cout << "  assembly 1 spikes:    " << m.excitatory[1] << "\n";
    std::cout << "  assembly 2 spikes:    " << m.excitatory[2] << "\n";
    std::cout << "  inhibitory spikes:    " << m.inhibitory << "\n";
    std::cout << "  seed spikes:          " << m.source_spikes << "\n";
    std::cout << "  total recorded events:" << m.total_events << "\n";
}

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
} // namespace

int main() {
    const std::string topology_file = "/tmp/neuromancer_v9_recurrent_csr.bin";
    try {
        save_csr_topology(make_recurrent_topology(), topology_file);
        const auto ungated = run_condition(topology_file, GateCondition::Ungated);
        const auto aligned = run_condition(topology_file, GateCondition::Aligned);
        const auto mismatched = run_condition(topology_file, GateCondition::Mismatched);
        const auto shuffled = run_condition(topology_file, GateCondition::Shuffled);
        std::remove(topology_file.c_str());

        std::cout << "\nNEUROMANCER — RECURRENT PHASE-SELECTIVE RECRUITMENT V9\n";
        std::cout << "------------------------------------------------------\n";
        std::cout << "backend: Izhikevich CUDA runtime + recurrent excitatory/inhibitory CSR\n";
        std::cout << "network: 3 excitatory assemblies x 8 neurons + 6 inhibitory neurons\n";
        std::cout << "topology: all-to-all recurrent E/I graph, one-step synaptic delay\n";
        std::cout << "input: one externally stimulated seed neuron; all downstream spikes are synaptic\n";
        std::cout << "phase gate: GPU-side postsynaptic phase address, +/-20 degrees at 10 Hz\n";
        std::cout << "learning: disabled; weights and topology fixed across controls\n\n";
        print_metrics("UNGATED BASELINE", ungated);
        std::cout << "\n";
        print_metrics("PHASE-ALIGNED", aligned);
        std::cout << "\n";
        print_metrics("PHASE-MISMATCHED", mismatched);
        std::cout << "\n";
        print_metrics("SHUFFLED PHASE LABELS", shuffled);

        require(ungated.source_spikes > 0, "Ungated seed failed to spike");
        require(recruited(ungated, 0) + ungated.excitatory[1] +
                    ungated.excitatory[2] > 0,
                "Ungated network showed no recurrent recruitment");
        require(aligned.source_spikes > 0, "Aligned seed failed to spike");
        require(recruited(aligned, 0) > 0,
                "Phase-aligned condition failed to recruit its addressed assembly");
        require(aligned.excitatory[1] + aligned.excitatory[2] <
                    ungated.excitatory[1] + ungated.excitatory[2],
                "Aligned phase gate did not reduce off-target recruitment");
        require(mismatched.excitatory[1] + mismatched.excitatory[2] <
                    aligned.excitatory[1] + aligned.excitatory[2] + 1,
                "Mismatched phase unexpectedly recruited off-target assemblies");
        require(shuffled.excitatory[1] + shuffled.excitatory[2] <
                    ungated.excitatory[1] + ungated.excitatory[2],
                "Shuffled phase labels did not reduce off-target recruitment");

        std::cout << "\nPASS: CUDA phase gating controls recurrent synaptic delivery.\n";
        std::cout << "NOTE: This is a controlled mechanism assay, not yet learned phase addressing.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::remove(topology_file.c_str());
        std::cerr << "V9 recurrent phase assay failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
