#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/csr_topology.h"
#include "topology/topology_io.h"

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
constexpr std::uint32_t kPulseDuration = 4;
constexpr std::uint32_t kObservationSteps = 180;
constexpr std::array<float, 10> kCurrentAmplitudes{{
    1.0f, 2.0f, 5.0f, 10.0f, 20.0f,
    40.0f, 80.0f, 160.0f, 320.0f, 500.0f
}};
constexpr std::array<float, 6> kSynapticWeights{{
    0.01f, 0.05f, 0.10f, 0.25f, 0.50f, 1.0f
}};

ModelParameters make_model(std::uint32_t excitatory, std::uint32_t inhibitory) {
    ModelParameters p;
    p.simulation.excitatory_count = excitatory;
    p.simulation.inhibitory_count = inhibitory;
    p.simulation.steps = kObservationSteps;
    p.simulation.dt = kDtMs;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 2021;
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

CSRTopology make_empty_topology(std::uint32_t excitatory, std::uint32_t inhibitory) {
    CSRTopology g;
    g.excitatory_count = excitatory;
    g.inhibitory_count = inhibitory;
    g.minimum_delay = 1;
    g.maximum_delay = 1;
    g.row_offsets.assign(static_cast<std::size_t>(excitatory + inhibitory) + 1U, 0U);
    g.validate();
    return g;
}

CSRTopology make_single_edge_topology(float weight) {
    CSRTopology g;
    g.excitatory_count = 2;
    g.inhibitory_count = 1;
    g.minimum_delay = 1;
    g.maximum_delay = 1;
    g.row_offsets = {0, 1, 1, 1};
    g.target_indices = {1};
    g.weights = {weight};
    g.delays = {1};
    g.validate();
    return g;
}

void write_sample(std::ofstream& csv, const char* assay, float condition,
                  const VoltageSample& s) {
    csv << assay << ',' << condition << ',' << s.step << ','
        << s.time_ms << ',' << s.neuron_id << ',' << s.voltage_mv << ','
        << s.recovery << ',' << (s.spiked_this_step ? 1 : 0) << '\n';
}

void run_current_pulse(float amplitude, const ModelParameters& model,
                       const std::string& topology_path, std::ofstream& csv) {
    save_csr_topology(make_empty_topology(1, 1), topology_path);
    CudaSimulationRuntime runtime(model, topology_path, 0, true, false);
    runtime.schedule_stimulus(0, kPulseStep, kPulseDuration, amplitude);
    const std::vector<std::uint32_t> ids{0};
    for (std::uint32_t step = 0; step < kObservationSteps; ++step) {
        runtime.run_steps(1);
        for (const auto& sample : runtime.collect_voltage_samples(ids))
            write_sample(csv, "external_current", amplitude, sample);
    }
    const auto events = runtime.collect_spike_events();
    if (events.dropped_events != 0)
        throw std::runtime_error("Dropped spike events during current-pulse assay");
    std::cout << "External current " << amplitude << ": spikes="
              << events.events.size() << ", dropped=" << events.dropped_events << '\n';
}

void run_synaptic_pulse(float weight, const ModelParameters& model,
                        const std::string& topology_path, std::ofstream& csv) {
    save_csr_topology(make_single_edge_topology(weight), topology_path);
    CudaSimulationRuntime runtime(model, topology_path, 0, true, false);
    runtime.schedule_stimulus(0, kPulseStep, 1, 500.0f);
    const std::vector<std::uint32_t> ids{0, 1};
    for (std::uint32_t step = 0; step < kObservationSteps; ++step) {
        runtime.run_steps(1);
        for (const auto& sample : runtime.collect_voltage_samples(ids))
            write_sample(csv, "single_synapse", weight, sample);
    }
    const auto events = runtime.collect_spike_events();
    if (events.dropped_events != 0)
        throw std::runtime_error("Dropped spike events during synaptic-pulse assay");
    std::cout << "Synaptic weight " << weight << ": spikes="
              << events.events.size() << ", dropped=" << events.dropped_events << '\n';
}

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
} // namespace

int main() {
    const std::string topology_path = "/tmp/neuromancer_v21_assay.csr";
    const std::string csv_path = "/tmp/neuromancer_v21_neuron_dynamics.csv";
    try {
        const auto current_model = make_model(1, 1);
        const auto synaptic_model = make_model(2, 1);
        std::ofstream csv(csv_path);
        if (!csv) throw std::runtime_error("Could not create V21 dynamics CSV");
        csv << "assay,condition,step,time_ms,neuron_id,voltage_mv,recovery,spiked\n";
        csv << std::setprecision(10);

        std::cout << "\nNEUROMANCER — NEURON DYNAMICS IDENTIFICATION V21\n"
                  << "------------------------------------------------\n"
                  << "Integrator: actual CUDA Izhikevich midpoint RK2 implementation\n"
                  << "Excitatory model: a=0.02, b=0.20, c=-65 mV, d=8\n"
                  << "Inhibitory placeholder: a=0.10, b=0.20, c=-65 mV, d=2\n"
                  << "dt=" << kDtMs << " ms; threshold=30 mV; initial v=-65 mV\n"
                  << "Assay 1: isolated external-current pulse sweep; duration="
                  << kPulseDuration * kDtMs << " ms\n"
                  << "Assay 2: one presynaptic spike through one isolated CSR synapse\n"
                  << "CSV: " << csv_path << "\n\n";

        for (const float amplitude : kCurrentAmplitudes)
            run_current_pulse(amplitude, current_model, topology_path, csv);

        for (const float weight : kSynapticWeights)
            run_synaptic_pulse(weight, synaptic_model, topology_path, csv);

        csv.close();
        require(static_cast<bool>(csv), "Failed while writing V21 CSV");
        std::remove(topology_path.c_str());
        std::cout << "\nPASS: all current and single-synapse trajectories were recorded; "
                     "no event-buffer loss.\n";
        std::cout << "NEXT: fit the observed voltage/recovery trajectories to the "
                     "discrete RK2 state equations and quantify synaptic impulse response.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::remove(topology_path.c_str());
        std::cerr << "V21 dynamics identification failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
