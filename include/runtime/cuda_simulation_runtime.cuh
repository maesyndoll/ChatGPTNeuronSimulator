#pragma once

#include "core/parameter_manager.h"
#include "cuda/cuda_stream.cuh"
#include "cuda/device_buffer.cuh"
#include "cuda/neuron_izhikevich.cuh"
#include "topology/csr_topology.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace spiking {

struct SpikeEvent {
    std::uint64_t step = 0;
    std::uint32_t neuron_id = 0;
};

struct SpikeEventBatch {
    std::vector<SpikeEvent> events;
    std::uint64_t first_sequence = 0;
    std::uint64_t total_events = 0;
    std::uint64_t dropped_events = 0;
};

struct PhaseDynamicsConfig {
    bool enabled = false;
    // Coupling and natural frequencies are measured in radians per simulated ms.
    float coupling_strength = 0.01f;
    // [0, 1]: zero disables routing modulation; one gives full cosine gating.
    float modulation_strength = 1.0f;
};

struct SynapticPlasticityConfig {
    bool enabled = false;
    float trace_time_constant_ms = 20.0f;
    float potentiation_rate = 0.012f;
    float depression_rate = 0.010f;
    float minimum_weight_magnitude = 0.01f;
    float maximum_weight_magnitude = 5.0f;
};

struct SimulationMetrics {
    std::uint64_t completed_steps = 0;
    std::uint64_t excitatory_spikes = 0;
    std::uint64_t inhibitory_spikes = 0;
    std::uint64_t last_step_excitatory_spikes = 0;
    std::uint64_t last_step_inhibitory_spikes = 0;
    double mean_voltage_mv = 0.0;
    double excitatory_rate_hz = 0.0;
    double inhibitory_rate_hz = 0.0;
    // Kuramoto order parameter R in [0, 1], computed from GPU phase state.
    double phase_coherence = 0.0;
    // Mean absolute CSR weight; populated when plasticity is enabled.
    double mean_absolute_synaptic_weight = 0.0;
    bool finite_state = true;
};

class CudaSimulationRuntime {
public:
    explicit CudaSimulationRuntime(ModelParameters defaults = ModelParameters{},
                                   int device = 0);
    ~CudaSimulationRuntime() = default;

    CudaSimulationRuntime(const CudaSimulationRuntime&) = delete;
    CudaSimulationRuntime& operator=(const CudaSimulationRuntime&) = delete;

    ParameterUpdateResult apply_parameters(const ModelParameters& candidate);
    ParameterUpdateResult reset_parameters_to_defaults();
    void reinitialize_simulation();
    void configure_phase_dynamics(const PhaseDynamicsConfig& config);
    PhaseDynamicsConfig phase_dynamics() const noexcept { return phase_config_; }
    void configure_synaptic_plasticity(const SynapticPlasticityConfig& config);
    SynapticPlasticityConfig synaptic_plasticity() const noexcept {
        return plasticity_config_;
    }

    // Replace the per-neuron external input-current vector (model current units).
    void set_external_current(const std::vector<float>& current);
    void run();
    void run_steps(std::uint32_t steps);
    SimulationMetrics collect_metrics();
    SpikeEventBatch collect_spike_events();

    ModelParameters active_parameters() const;
    ModelParameters requested_parameters() const;
    bool has_pending_reinitialization() const;
    std::uint64_t simulation_generation() const;
    std::uint64_t parameter_version() const {
        return parameter_manager_.parameter_version();
    }
    std::uint64_t completed_steps() const noexcept { return completed_steps_; }
    int device_id() const noexcept { return device_id_; }

private:
    struct DeviceState {
        DeviceState(std::size_t count, const CSRTopology& topology);
        CudaStream stream;
        DeviceBuffer<float> a, b, c, d, current;
        DeviceBuffer<float> external_current;
        DeviceBuffer<float> voltage, recovery;
        DeviceBuffer<std::uint32_t> spike_counts;
        DeviceBuffer<std::uint8_t> spike_flags;

        // GPU-resident queue containing the neurons that spiked last step.
        DeviceBuffer<std::uint32_t> active_spike_sources;
        DeviceBuffer<std::uint32_t> active_spike_count;

        // Fixed-capacity GPU ring for recent spike events (12 bytes/event).
        DeviceBuffer<std::uint64_t> spike_event_steps;
        DeviceBuffer<std::uint32_t> spike_event_neurons;
        DeviceBuffer<unsigned long long> spike_event_count;

        // Persistent source-oriented CSR connectivity on the GPU.
        DeviceBuffer<std::uint32_t> row_offsets;
        DeviceBuffer<std::uint32_t> target_indices;
        DeviceBuffer<float> weights;
        DeviceBuffer<std::uint32_t> delays;

        // GPU-resident oscillator state and natural angular frequencies.
        DeviceBuffer<float> phases, next_phases, natural_frequencies;
        // Per-neuron exponentially decaying spike traces used by GPU STDP.
        DeviceBuffer<float> plasticity_traces;

        // Dense target-current ring: [ring_slot][neuron].
        DeviceBuffer<float> synaptic_current;
        std::size_t neuron_count;
        std::uint32_t edge_count;
        std::uint32_t ring_slots;

        NeuronParametersGPU neuron_parameters() noexcept;
        NeuronStateGPU neuron_state() noexcept;
    };

    std::unique_ptr<DeviceState> build_device_state(const ModelParameters& parameters);

    ParameterManager parameter_manager_;
    ModelParameters active_parameters_;
    std::unique_ptr<DeviceState> device_state_;
    PhaseDynamicsConfig phase_config_;
    SynapticPlasticityConfig plasticity_config_;
    std::uint64_t completed_steps_ = 0;
    std::uint64_t spike_event_read_cursor_ = 0;
    double elapsed_simulation_ms_ = 0.0;
    int device_id_ = 0;
};

} // namespace spiking
