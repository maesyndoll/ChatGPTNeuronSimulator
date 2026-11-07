#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
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

constexpr std::uint32_t kAssemblyCount = 3;
constexpr std::uint32_t kNeuronsPerAssembly = 8;
constexpr std::uint32_t kNoiseNeurons = 8;
constexpr std::uint32_t kExcitatoryCount =
    kAssemblyCount * kNeuronsPerAssembly + kNoiseNeurons;
constexpr std::uint32_t kInhibitoryCount = 8;
constexpr std::uint32_t kPeriodSteps = 200; // 100 ms at dt = 0.5 ms (10 Hz)
constexpr std::uint32_t kCycles = 4;
constexpr std::uint32_t kTotalSteps = kPeriodSteps * kCycles;
constexpr double kPi = 3.14159265358979323846;
constexpr double kTau = 2.0 * kPi;
constexpr double kGateHalfWidth = 20.0 * kPi / 180.0;
constexpr float kPulseAmplitude = 250.0f;

struct ScheduledPulse {
    std::uint32_t neuron = 0;
    std::uint32_t offset = 0;
};

struct PhaseMetrics {
    std::uint64_t spikes = 0;
    double plv = 0.0;
    double mean_phase = 0.0;
    double mean_error_degrees = 0.0;
};

struct AddressMetrics {
    std::uint64_t selected = 0;
    std::uint64_t true_positive = 0;
    std::uint64_t target_spikes = 0;
    std::uint64_t noise_spikes = 0;
    std::uint64_t selected_noise = 0;
    std::uint64_t selected_other_assemblies = 0;
    std::uint64_t other_assembly_spikes = 0;

    double precision() const {
        return selected ? static_cast<double>(true_positive) / selected : 0.0;
    }
    double recall() const {
        return target_spikes
            ? static_cast<double>(true_positive) / target_spikes : 0.0;
    }
    double noise_rejection() const {
        return noise_spikes
            ? 1.0 - static_cast<double>(selected_noise) / noise_spikes : 0.0;
    }
    double other_rejection() const {
        return other_assembly_spikes
            ? 1.0 - static_cast<double>(selected_other_assemblies) /
                other_assembly_spikes : 0.0;
    }
};

struct RunMetrics {
    SpikeEventBatch events;
    std::array<PhaseMetrics, kAssemblyCount> assemblies{};
    PhaseMetrics noise{};
    std::array<AddressMetrics, kAssemblyCount> addresses{};
    std::uint64_t total_excitatory_events = 0;
};

double wrap_phase(double phase) {
    phase = std::fmod(phase, kTau);
    return phase < 0.0 ? phase + kTau : phase;
}

double circular_distance(double a, double b) {
    return std::abs(std::remainder(a - b, kTau));
}

double target_phase(std::uint32_t assembly) {
    return kTau * static_cast<double>(assembly) / kAssemblyCount;
}

ModelParameters make_parameters() {
    ModelParameters p;
    p.simulation.excitatory_count = kExcitatoryCount;
    p.simulation.inhibitory_count = kInhibitoryCount;
    p.simulation.steps = kTotalSteps;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 81723;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;

    // Keep a real sparse E/I CSR graph active during the timing assay.
    p.topology.p_ee = 0.04;
    p.topology.p_ei = 0.08;
    p.topology.p_ie = 0.08;
    p.topology.p_ii = 0.04;
    p.topology.master_seed = 81723;
    p.topology.topology_seed = 194711;
    p.topology.delay_seed = 271828;
    p.topology.delay_distribution = DelayDistribution::Uniform;
    p.topology.minimum_delay = 1;
    p.topology.maximum_delay = 4;
    p.topology.fibonacci_delays = {1, 2, 3, 4};
    p.topology.fibonacci_weights = {1.0, 1.0, 1.0, 1.0};
    return p;
}

std::vector<std::vector<ScheduledPulse>> make_schedule(bool shuffle_assignment) {
    std::mt19937 rng(73021);
    std::normal_distribution<double> jitter_ms(0.0, 1.0);
    std::uniform_int_distribution<std::uint32_t> noise_phase(0, kPeriodSteps - 1);
    std::vector<std::vector<ScheduledPulse>> cycles(kCycles);

    for (std::uint32_t cycle = 0; cycle < kCycles; ++cycle) {
        auto& pulses = cycles[cycle];
        pulses.reserve(kExcitatoryCount);

        // The phase-coded condition assigns each assembly a distinct temporal
        // phase. The receiver never receives these labels; it sees spike times.
        for (std::uint32_t assembly = 0; assembly < kAssemblyCount; ++assembly) {
            const auto center = static_cast<double>(kPeriodSteps) *
                static_cast<double>(assembly) / kAssemblyCount;
            for (std::uint32_t local = 0; local < kNeuronsPerAssembly; ++local) {
                const auto jitter_steps = static_cast<int>(
                    std::lround(jitter_ms(rng) / 0.5));
                const auto offset = static_cast<int>(std::lround(center)) + jitter_steps;
                const auto wrapped = (offset % static_cast<int>(kPeriodSteps) +
                    static_cast<int>(kPeriodSteps)) % static_cast<int>(kPeriodSteps);
                pulses.push_back({
                    assembly * kNeuronsPerAssembly + local,
                    static_cast<std::uint32_t>(wrapped)
                });
            }
        }

        // Background neurons emit one asynchronous event per cycle.
        for (std::uint32_t local = 0; local < kNoiseNeurons; ++local) {
            pulses.push_back({
                kAssemblyCount * kNeuronsPerAssembly + local,
                noise_phase(rng)
            });
        }

        if (shuffle_assignment) {
            // Preserve the exact per-cycle event-time distribution while
            // scrambling which neuron receives each phase-locked event time.
            std::vector<std::uint32_t> offsets;
            offsets.reserve(pulses.size());
            for (const auto& pulse : pulses) offsets.push_back(pulse.offset);
            std::shuffle(offsets.begin(), offsets.end(), rng);
            for (std::size_t i = 0; i < pulses.size(); ++i)
                pulses[i].offset = offsets[i];
        }
    }
    return cycles;
}

RunMetrics run_condition(bool shuffle_assignment) {
    auto parameters = make_parameters();
    CudaSimulationRuntime runtime(parameters, 0, true, false);

    const auto schedule = make_schedule(shuffle_assignment);
    for (std::uint32_t cycle = 0; cycle < kCycles; ++cycle) {
        for (const auto& pulse : schedule[cycle]) {
            const std::uint64_t start =
                1ULL + static_cast<std::uint64_t>(cycle) * kPeriodSteps +
                pulse.offset;
            runtime.schedule_stimulus(
                pulse.neuron, start, 1, kPulseAmplitude
            );
        }
    }

    runtime.run_steps(kTotalSteps);
    const auto simulation = runtime.collect_metrics();
    if (!simulation.finite_state || simulation.completed_steps != kTotalSteps)
        throw std::runtime_error("CUDA runtime did not complete a finite simulation");

    RunMetrics result;
    result.events = runtime.collect_spike_events();
    if (result.events.dropped_events != 0)
        throw std::runtime_error("Spike-event ring dropped events");

    std::array<std::vector<double>, kAssemblyCount> phases;
    std::vector<double> noise_phases;

    for (const auto& event : result.events.events) {
        if (event.neuron_id >= kExcitatoryCount) continue;
        ++result.total_excitatory_events;
        const double phase = wrap_phase(
            kTau * static_cast<double>((event.step - 1) % kPeriodSteps) /
            static_cast<double>(kPeriodSteps)
        );

        if (event.neuron_id >= kAssemblyCount * kNeuronsPerAssembly) {
            noise_phases.push_back(phase);
        } else {
            const auto assembly = event.neuron_id / kNeuronsPerAssembly;
            phases[assembly].push_back(phase);
        }
    }

    const auto summarize = [](const std::vector<double>& values,
                              double expected_phase) {
        PhaseMetrics m;
        m.spikes = values.size();
        if (values.empty()) return m;
        std::complex<double> vector_sum{0.0, 0.0};
        for (double phase : values)
            vector_sum += std::polar(1.0, phase);
        m.plv = std::abs(vector_sum) / values.size();
        m.mean_phase = wrap_phase(std::arg(vector_sum));
        m.mean_error_degrees =
            circular_distance(m.mean_phase, expected_phase) * 180.0 / kPi;
        return m;
    };

    for (std::uint32_t assembly = 0; assembly < kAssemblyCount; ++assembly)
        result.assemblies[assembly] = summarize(phases[assembly], target_phase(assembly));
    result.noise = summarize(noise_phases, 0.0);

    for (std::uint32_t address = 0; address < kAssemblyCount; ++address) {
        auto& m = result.addresses[address];
        const double expected = target_phase(address);
        for (const auto& event : result.events.events) {
            if (event.neuron_id >= kExcitatoryCount) continue;
            const double phase = wrap_phase(
                kTau * static_cast<double>((event.step - 1) % kPeriodSteps) /
                static_cast<double>(kPeriodSteps)
            );
            const bool selected = circular_distance(phase, expected) <= kGateHalfWidth;
            const bool target = event.neuron_id <
                    kAssemblyCount * kNeuronsPerAssembly &&
                event.neuron_id / kNeuronsPerAssembly == address;
            const bool noise = event.neuron_id >=
                kAssemblyCount * kNeuronsPerAssembly;
            const bool other = !target && !noise;

            if (target) ++m.target_spikes;
            if (noise) ++m.noise_spikes;
            if (other) ++m.other_assembly_spikes;
            if (selected) {
                ++m.selected;
                if (target) ++m.true_positive;
                if (noise) ++m.selected_noise;
                if (other) ++m.selected_other_assemblies;
            }
        }
    }
    return result;
}

void print_condition(const std::string& label, const RunMetrics& metrics) {
    std::cout << "\n" << label << "\n";
    std::cout << "assembly | spikes | PLV | mean phase error\n";
    for (std::uint32_t i = 0; i < kAssemblyCount; ++i) {
        const auto& m = metrics.assemblies[i];
        std::cout << "    " << i << "    | " << std::setw(6) << m.spikes
                  << " | " << std::fixed << std::setprecision(3) << m.plv
                  << " | " << std::setprecision(2) << m.mean_error_degrees
                  << " deg\n";
    }
    std::cout << "noise    | " << std::setw(6) << metrics.noise.spikes
              << " | " << std::fixed << std::setprecision(3) << metrics.noise.plv
              << " | asynchronous reference\n";

    std::cout << "address | selected | precision | recall | noise rejection | other rejection\n";
    for (std::uint32_t i = 0; i < kAssemblyCount; ++i) {
        const auto& m = metrics.addresses[i];
        std::cout << "   " << i << "    | " << std::setw(8) << m.selected
                  << " | " << std::fixed << std::setprecision(3) << m.precision()
                  << "     | " << m.recall()
                  << "  | " << m.noise_rejection()
                  << "          | " << m.other_rejection() << "\n";
    }
    std::cout << "recorded excitatory spikes: " << metrics.total_excitatory_events
              << " (dropped " << metrics.events.dropped_events << ")\n";
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        const auto coherent = run_condition(false);
        const auto shuffled = run_condition(true);

        std::cout << "\nNEUROMANCER — CUDA PHASE-TIMING ASSAY V8\n";
        std::cout << "-----------------------------------------\n";
        std::cout << "backend: Izhikevich CUDA runtime + sparse E/I CSR propagation\n";
        std::cout << "oscillator: 10 Hz external phase reference\n";
        std::cout << "input: 3 phase-coded assemblies + asynchronous background\n";
        std::cout << "control: identical event-time histogram, shuffled across neuron IDs\n";
        std::cout << "learning: disabled to isolate temporal coding\n";
        std::cout << "phase gate: +/-20 degrees, computed from recorded spike times\n";

        print_condition("COHERENT ASSEMBLY TIMING", coherent);
        print_condition("SHUFFLED PHASE-TO-NEURON CONTROL", shuffled);

        for (std::uint32_t i = 0; i < kAssemblyCount; ++i) {
            require(coherent.assemblies[i].spikes >= kNeuronsPerAssembly * kCycles,
                    "Coherent assembly failed to produce the expected stimulus spikes");
            require(coherent.addresses[i].precision() > 0.70,
                    "Coherent phase address did not sufficiently enrich its assembly");
            require(coherent.addresses[i].recall() > 0.70,
                    "Coherent phase address failed to retain its assembly");
            require(shuffled.addresses[i].precision() <
                        coherent.addresses[i].precision(),
                    "Shuffled control did not reduce phase-address precision");
        }

        std::cout << "\nPASS: real CUDA spike timestamps preserve phase-addressable assemblies.\n";
        std::cout << "LIMIT: phase references and input timing are externally imposed; "
                     "no phase learning or autonomous oscillator is tested.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "CUDA phase-timing assay failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
