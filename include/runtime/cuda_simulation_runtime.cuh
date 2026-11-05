#pragma once

#include "core/parameter_manager.h"
#include "cuda/cuda_stream.cuh"
#include "cuda/device_buffer.cuh"
#include "cuda/neuron_izhikevich.cuh"
#include "topology/csr_topology.h"

#include <cstdint>
#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

namespace spiking {

struct SpikeEvent {
    std::uint64_t step = 0;
    std::uint32_t neuron_id = 0;
};

struct VoltageSample {
    std::uint64_t step = 0;
    double time_ms = 0.0;
    std::uint32_t neuron_id = 0;
    float voltage_mv = 0.0f;
};

struct SpikeEventBatch {
    std::vector<SpikeEvent> events;
    std::uint64_t first_sequence = 0;
    std::uint64_t total_events = 0;
    std::uint64_t dropped_events = 0;
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
    bool finite_state = true;
};

class CudaSimulationRuntime {
public:
    explicit CudaSimulationRuntime(ModelParameters defaults = ModelParameters{},
                                   int device = 0,
                                   bool capture_spike_events = true);
    ~CudaSimulationRuntime() = default;

    CudaSimulationRuntime(const CudaSimulationRuntime&) = delete;
    CudaSimulationRuntime& operator=(const CudaSimulationRuntime&) = delete;

    ParameterUpdateResult apply_parameters(const ModelParameters& candidate);
    ParameterUpdateResult reset_parameters_to_defaults();
    void reinitialize_simulation();

    void run();
    void run_steps(std::uint32_t steps);
    SimulationMetrics collect_metrics();
    // Read post-integration membrane voltage for selected neurons.
    // Samples reflect threshold resets when a neuron spiked on this step.
    std::vector<VoltageSample> collect_voltage_samples(
        const std::vector<std::uint32_t>& neuron_ids);
    // Exact host-side fingerprint of current neuron parameters and state.
    std::uint64_t state_fingerprint();
    SpikeEventBatch collect_spike_events();

    // Conservative upper bound on batches that can run before the fixed GPU
    // event ring could fill, with additional headroom for bursty firing.
    std::uint32_t spike_event_collection_batch_interval(
        std::uint32_t batch_size) const noexcept;

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
        DeviceState(std::size_t count, const CSRTopology& topology,
                    bool capture_spike_events);
        CudaStream stream;
        DeviceBuffer<float> a, b, c, d, current;
        DeviceBuffer<float> voltage, recovery;
        DeviceBuffer<std::uint32_t> spike_counts;
        DeviceBuffer<std::uint8_t> spike_flags;

        // GPU-resident queue containing the neurons that spiked last step.
        DeviceBuffer<std::uint32_t> active_spike_sources;
        DeviceBuffer<std::uint32_t> active_spike_count;
        DeviceBuffer<unsigned long long> completed_steps_device;
        DeviceBuffer<std::uint32_t> completed_step_blocks;
        cudaGraph_t step_graph = nullptr;
        cudaGraphExec_t step_graph_exec = nullptr;

        ~DeviceState();

        // Fixed-capacity GPU ring for recent spike events (12 bytes/event).
        DeviceBuffer<std::uint64_t> spike_event_steps;
        DeviceBuffer<std::uint32_t> spike_event_neurons;
        DeviceBuffer<unsigned long long> spike_event_count;

        // Persistent source-oriented CSR connectivity on the GPU.
        DeviceBuffer<std::uint32_t> row_offsets;
        DeviceBuffer<std::uint32_t> target_indices;
        DeviceBuffer<float> weights;
        DeviceBuffer<std::uint32_t> delays;

        // Dense target-current ring: [ring_slot][neuron].
        DeviceBuffer<float> synaptic_current;
        std::size_t neuron_count;
        std::uint32_t edge_count;
        std::uint32_t ring_slots;
        bool capture_spike_events;

        NeuronParametersGPU neuron_parameters() noexcept;
        NeuronStateGPU neuron_state() noexcept;
    };

    std::unique_ptr<DeviceState> build_device_state(const ModelParameters& parameters,
                       bool capture_spike_events);

    ParameterManager parameter_manager_;
    ModelParameters active_parameters_;
    std::unique_ptr<DeviceState> device_state_;
    std::uint64_t completed_steps_ = 0;
    std::uint64_t spike_event_read_cursor_ = 0;
    double elapsed_simulation_ms_ = 0.0;
    int device_id_ = 0;
    bool capture_spike_events_ = true;
};

} // namespace spiking
