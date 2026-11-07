#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace spiking;

constexpr std::uint32_t kDriverCount = 8;
constexpr std::uint32_t kTargetCount = 8;
constexpr std::uint32_t kExcitatoryCount = kDriverCount + 2 * kTargetCount;
constexpr std::uint32_t kInhibitoryCount = 4;
constexpr std::uint32_t kTotalNeurons = kExcitatoryCount + kInhibitoryCount;
constexpr std::uint32_t kSteps = 80;
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTau = 2.0f * kPi;
constexpr float kOscillatorPeriodMs = 100.0f;
constexpr float kGateHalfWidth = 30.0f * kPi / 180.0f;
constexpr float kSynapticWeight = 10.0f;
constexpr char kTopologyPath[] = "neuromancer_phase_recruitment_v9.spkcsr";

enum class Condition { Ungated, Aligned, Mismatched, Shuffled };

struct Result {
    std::uint64_t driver_spikes = 0;
    std::uint64_t aligned_target_spikes = 0;
    std::uint64_t mismatched_target_spikes = 0;
    std::uint64_t inhibitory_spikes = 0;
    std::uint64_t total_excitatory_spikes = 0;
};

ModelParameters parameters() {
    ModelParameters p;
    p.simulation.excitatory_count = kExcitatoryCount;
    p.simulation.inhibitory_count = kInhibitoryCount;
    p.simulation.steps = kSteps;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 91731;
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

CSRTopology make_topology() {
    CSRTopology topology;
    topology.excitatory_count = kExcitatoryCount;
    topology.inhibitory_count = kInhibitoryCount;
    topology.minimum_delay = 1;
    topology.maximum_delay = 1;
    std::vector<std::vector<std::pair<std::uint32_t, float>>> rows(kTotalNeurons);

    // Each externally stimulated driver neuron has a real CSR edge to every
    // neuron in both downstream target assemblies. Target cells are never
    // externally stimulated; any target spike must follow synaptic input.
    for (std::uint32_t source = 0; source < kDriverCount; ++source) {
        for (std::uint32_t target = kDriverCount;
             target < kExcitatoryCount; ++target) {
            rows[source].push_back({target, kSynapticWeight});
        }
    }

    topology.row_offsets.reserve(kTotalNeurons + 1);
    topology.row_offsets.push_back(0);
    for (auto& row : rows) {
        for (const auto& edge : row) {
            topology.target_indices.push_back(edge.first);
            topology.weights.push_back(edge.second);
            topology.delays.push_back(1);
        }
        topology.row_offsets.push_back(
            static_cast<std::uint32_t>(topology.target_indices.size()));
    }
    topology.validate();
    return topology;
}

std::vector<float> make_preferences(Condition condition) {
    std::vector<float> preferences(kTotalNeurons, 0.0f);
    for (std::uint32_t neuron = kDriverCount;
         neuron < kDriverCount + kTargetCount; ++neuron)
        preferences[neuron] = condition == Condition::Mismatched ? kPi : 0.0f;
    for (std::uint32_t neuron = kDriverCount + kTargetCount;
         neuron < kExcitatoryCount; ++neuron)
        preferences[neuron] = condition == Condition::Mismatched ? 0.0f : kPi;

    if (condition == Condition::Shuffled) {
        std::mt19937 rng(41041);
        std::uniform_real_distribution<float> phase(0.0f, kTau);
        for (std::uint32_t neuron = kDriverCount;
             neuron < kExcitatoryCount; ++neuron)
            preferences[neuron] = phase(rng);
    }
    return preferences;
}

Result run_condition(Condition condition) {
    auto p = parameters();
    CudaSimulationRuntime runtime(p, kTopologyPath, 0, true, false);
    if (condition != Condition::Ungated) {
        runtime.configure_phase_routing(
            make_preferences(condition), kOscillatorPeriodMs, 0.0f,
            kGateHalfWidth);
    }

    for (std::uint32_t neuron = 0; neuron < kDriverCount; ++neuron)
        runtime.schedule_stimulus(neuron, 1, 1, 250.0f);

    runtime.run_steps(kSteps);
    const auto metrics = runtime.collect_metrics();
    if (!metrics.finite_state || metrics.completed_steps != kSteps)
        throw std::runtime_error("CUDA runtime failed finite-step validation");
    const auto events = runtime.collect_spike_events();
    if (events.dropped_events != 0)
        throw std::runtime_error("Spike event recorder dropped events");

    Result result;
    for (const auto& event : events.events) {
        if (event.neuron_id < kDriverCount) ++result.driver_spikes;
        else if (event.neuron_id < kDriverCount + kTargetCount)
            ++result.aligned_target_spikes;
        else if (event.neuron_id < kExcitatoryCount)
            ++result.mismatched_target_spikes;
        else
            ++result.inhibitory_spikes;
        if (event.neuron_id < kExcitatoryCount)
            ++result.total_excitatory_spikes;
    }
    return result;
}

void print(const char* name, const Result& r) {
    std::cout << std::left << std::setw(13) << name
              << " | drivers " << std::setw(3) << r.driver_spikes
              << " | target-A " << std::setw(3) << r.aligned_target_spikes
              << " | target-B " << std::setw(3) << r.mismatched_target_spikes
              << " | inhibitory " << r.inhibitory_spikes << '\n';
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
    try {
        save_csr_topology(make_topology(), kTopologyPath);
        const auto ungated = run_condition(Condition::Ungated);
        const auto aligned = run_condition(Condition::Aligned);
        const auto mismatched = run_condition(Condition::Mismatched);
        const auto shuffled = run_condition(Condition::Shuffled);
        std::remove(kTopologyPath);

        std::cout << "\nNEUROMANCER — CUDA PHASE-SELECTIVE RECRUITMENT V9\n";
        std::cout << "------------------------------------------------\n";
        std::cout << "backend: Izhikevich CUDA runtime + deterministic CSR graph\n";
        std::cout << "source: 8 externally stimulated driver neurons, step 1 only\n";
        std::cout << "targets: 16 neurons, no external stimulation\n";
        std::cout << "routing: target phase preference checked inside CUDA CSR propagation\n";
        std::cout << "oscillator: 10 Hz; gate: +/-30 degrees; STDP: disabled\n\n";
        std::cout << "condition     | driver spikes | target-A | target-B | inhibitory\n";
        print("UNGATED", ungated);
        print("ALIGNED", aligned);
        print("MISMATCHED", mismatched);
        print("SHUFFLED", shuffled);

        require(ungated.driver_spikes >= kDriverCount,
                "Driver assembly did not spike");
        require(ungated.aligned_target_spikes > 0 &&
                ungated.mismatched_target_spikes > 0,
                "Ungated CSR propagation failed to recruit both target groups");
        require(aligned.aligned_target_spikes > 0,
                "Aligned phase address failed to recruit target A");
        require(aligned.mismatched_target_spikes == 0,
                "Aligned phase gate transmitted to mismatched target B");
        require(mismatched.mismatched_target_spikes > 0,
                "Mismatched phase address failed to recruit target B");
        require(mismatched.aligned_target_spikes == 0,
                "Mismatched phase gate transmitted to target A");

        std::cout << "\nPASS: phase preference changes actual postsynaptic recruitment "
                     "inside CUDA CSR propagation.\n";
        std::cout << "LIMIT: phase preferences and oscillator reference are externally "
                     "assigned; phase learning and autonomous oscillation are not tested.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::remove(kTopologyPath);
        std::cerr << "CUDA phase-recruitment assay failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
