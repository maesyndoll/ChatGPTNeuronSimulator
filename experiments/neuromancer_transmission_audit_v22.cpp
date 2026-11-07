#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/csr_topology.h"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;

constexpr float kDtMs = 0.5f;
constexpr std::uint32_t kPulseStep = 50;
constexpr std::uint32_t kObservationSteps = 180;
constexpr double kEventTimeMs = 25.5;
constexpr double kAnalysisEndMs = 35.0;
constexpr std::array<float, 6> kWeights{{
    0.01f, 0.05f, 0.10f, 0.25f, 0.50f, 1.0f
}};

struct ConditionResult {
    float weight = 0.0f; // Zero denotes the disconnected-target control.
    std::vector<VoltageSample> target;
    std::uint64_t presynaptic_spikes = 0;
    std::uint64_t postsynaptic_spikes = 0;
    std::uint64_t presynaptic_step = 0;
    std::uint64_t dropped_events = 0;
    std::size_t edge_count = 0;
};

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = 2;
    p.simulation.inhibitory_count = 1;
    p.simulation.steps = kObservationSteps;
    p.simulation.dt = kDtMs;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 2022;
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

CSRTopology make_topology(float weight) {
    CSRTopology g;
    g.excitatory_count = 2;
    g.inhibitory_count = 1;
    g.minimum_delay = 1;
    g.maximum_delay = 1;
    g.row_offsets = {0, weight > 0.0f ? 1U : 0U, weight > 0.0f ? 1U : 0U, weight > 0.0f ? 1U : 0U};
    if (weight > 0.0f) {
        g.target_indices = {1};
        g.weights = {weight};
        g.delays = {1};
    }
    g.validate();
    return g;
}

ConditionResult run_condition(float weight, const ModelParameters& model,
                              const std::string& topology_path) {
    const auto topology = make_topology(weight);
    save_csr_topology(topology, topology_path);

    CudaSimulationRuntime runtime(model, topology_path, 0, true, false);
    runtime.schedule_stimulus(0, kPulseStep, 1, 500.0f);

    ConditionResult result;
    result.weight = weight;
    result.edge_count = topology.synapse_count();
    result.target.reserve(kObservationSteps);

    const std::vector<std::uint32_t> target_id{1};
    for (std::uint32_t step = 0; step < kObservationSteps; ++step) {
        runtime.run_steps(1);
        const auto samples = runtime.collect_voltage_samples(target_id);
        if (samples.size() != 1)
            throw std::runtime_error("Target voltage sample missing");
        result.target.push_back(samples.front());
    }

    const auto events = runtime.collect_spike_events();
    result.dropped_events = events.dropped_events;
    for (const auto& event : events.events) {
        if (event.neuron_id == 0) {
            ++result.presynaptic_spikes;
            result.presynaptic_step = event.step;
        } else if (event.neuron_id == 1) {
            ++result.postsynaptic_spikes;
        }
    }
    return result;
}

const VoltageSample& sample_at(const ConditionResult& result, double time_ms) {
    const auto it = std::find_if(result.target.begin(), result.target.end(),
        [time_ms](const VoltageSample& sample) {
            return std::abs(sample.time_ms - time_ms) < 1.0e-6;
        });
    if (it == result.target.end())
        throw std::runtime_error("Required event-aligned voltage sample is missing");
    return *it;
}

void write_csv(const std::string& path,
               const std::vector<ConditionResult>& results) {
    std::ofstream csv(path);
    if (!csv) throw std::runtime_error("Could not create V22 transmission CSV");
    csv << "condition,weight,step,time_ms,neuron_id,voltage_mv,recovery,spiked\n";
    csv << std::setprecision(10);
    for (const auto& result : results) {
        const char* condition = result.weight == 0.0f ? "disconnected_control" : "single_synapse";
        for (const auto& sample : result.target) {
            csv << condition << ',' << result.weight << ',' << sample.step << ','
                << sample.time_ms << ',' << sample.neuron_id << ','
                << sample.voltage_mv << ',' << sample.recovery << ','
                << (sample.spiked_this_step ? 1 : 0) << '\n';
        }
    }
    csv.close();
    if (!csv) throw std::runtime_error("Failed while writing V22 transmission CSV");
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
    const std::string topology_path = "/tmp/neuromancer_v22_transmission.csr";
    const std::string csv_path = "/tmp/neuromancer_v22_transmission.csv";

    try {
        const auto model = make_model();
        std::vector<ConditionResult> results;
        results.reserve(kWeights.size() + 1U);

        std::cout << "\nNEUROMANCER — SYNAPTIC TRANSMISSION AUDIT V22\n"
                  << "---------------------------------------------\n"
                  << "CUDA Izhikevich runtime; dt=" << kDtMs << " ms\n"
                  << "One presynaptic current pulse; identical initial state in every condition\n"
                  << "Control: same presynaptic neuron and schedule, but zero CSR edges\n"
                  << "Analysis window: " << kEventTimeMs << "–" << kAnalysisEndMs << " ms\n\n";

        results.push_back(run_condition(0.0f, model, topology_path));
        for (const float weight : kWeights)
            results.push_back(run_condition(weight, model, topology_path));

        const auto& control = results.front();
        require(control.presynaptic_spikes == 1,
                "Disconnected control did not produce exactly one presynaptic spike");
        require(control.postsynaptic_spikes == 0,
                "Disconnected control unexpectedly produced a postsynaptic spike");
        require(control.dropped_events == 0,
                "Disconnected control dropped spike events");

        for (const auto& result : results) {
            require(result.presynaptic_spikes == 1,
                    "A condition did not produce exactly one presynaptic spike");
            require(result.postsynaptic_spikes == 0,
                    "Postsynaptic neuron spiked during subthreshold transmission assay");
            require(result.dropped_events == 0,
                    "Spike-event buffer dropped events");
            require(result.target.size() == kObservationSteps,
                    "A condition has an incomplete target voltage trace");
            for (const auto& sample : result.target) {
                require(std::isfinite(sample.voltage_mv) &&
                        std::isfinite(sample.recovery),
                        "Non-finite postsynaptic state observed");
            }
        }

        std::cout << "weight,edges,pre_spike_step,V_at_25.5ms,delta_vs_disconnected_at_25.5ms,"
                     "peak_excess_25.5_to_35ms,peak_excess_time_ms,target_spikes\n";

        double previous_event_excess = -1.0e30;
        double lowest_event_excess = 0.0;
        double highest_event_excess = 0.0;
        for (std::size_t i = 1; i < results.size(); ++i) {
            const auto& result = results[i];
            const auto& at_event = sample_at(result, kEventTimeMs);
            const auto& control_at_event = sample_at(control, kEventTimeMs);
            const double event_excess =
                static_cast<double>(at_event.voltage_mv) -
                static_cast<double>(control_at_event.voltage_mv);

            double peak_excess = -1.0e30;
            double peak_time = 0.0;
            for (const auto& sample : result.target) {
                if (sample.time_ms < kEventTimeMs ||
                    sample.time_ms > kAnalysisEndMs)
                    continue;
                const auto& control_sample = sample_at(control, sample.time_ms);
                const double excess =
                    static_cast<double>(sample.voltage_mv) -
                    static_cast<double>(control_sample.voltage_mv);
                if (excess > peak_excess) {
                    peak_excess = excess;
                    peak_time = sample.time_ms;
                }
            }

            if (i == 1) lowest_event_excess = event_excess;
            if (i + 1 == results.size()) highest_event_excess = event_excess;

            std::cout << std::fixed << std::setprecision(5)
                      << result.weight << ',' << result.edge_count << ','
                      << result.presynaptic_step << ','
                      << at_event.voltage_mv << ',' << event_excess << ','
                      << peak_excess << ',' << peak_time << ','
                      << result.postsynaptic_spikes << '\n';

            require(event_excess + 1.0e-4 >= previous_event_excess,
                    "Event-aligned response was not monotonic with synaptic weight");
            previous_event_excess = event_excess;
        }

        require(highest_event_excess > lowest_event_excess + 0.10,
                "Largest synaptic weight did not exceed the smallest by the expected margin");

        write_csv(csv_path, results);
        std::remove(topology_path.c_str());

        std::cout << "\nPASS: presynaptic events verified, disconnected control recorded, "
                     "postsynaptic trajectories finite, and event-aligned response scales "
                     "with weight. No postsynaptic firing was observed.\n"
                  << "CSV: " << csv_path << "\n"
                  << "NEXT: use this measured transfer function to calibrate the first "
                     "two-class spiking inference benchmark.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::remove(topology_path.c_str());
        std::cerr << "V22 transmission audit failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
