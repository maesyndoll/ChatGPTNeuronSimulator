#pragma once

#include "core/parameter_manager.h"
#include "cuda/cuda_stream.cuh"
#include "cuda/device_buffer.cuh"
#include "cuda/neuron_izhikevich.cuh"
#include "topology/csr_topology.h"
#include "plasticity/stdp.h"

#include <cstdint>
#include <memory>
#include <string>
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
    float recovery = 0.0f;
    bool spiked_this_step = false;
};

struct SpikeEventBatch {
    std::vector<SpikeEvent> events;
    std::uint64_t first_sequence = 0;
    std::uint64_t total_events = 0;
    std::uint64_t dropped_events = 0;
};

struct TopologyVisualEdge {
    std::uint32_t source = 0;
    std::uint32_t target = 0;
    float weight = 0.0f;
};

struct TopologyVisualSnapshot {
    std::uint32_t excitatory_count = 0;
    std::uint32_t inhibitory_count = 0;
    std::uint64_t synapse_count = 0;
    std::vector<std::uint32_t> neuron_ids;
    std::vector<TopologyVisualEdge> edges;
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
                                   bool capture_spike_events = true,
                                   bool enable_plasticity_state = false,
                                   StdpParameters stdp_parameters = {});

    // Initialize from a previously saved CSR topology instead of generating
    // a new random graph. Reinitialization reloads this same file.
    CudaSimulationRuntime(ModelParameters defaults,
                          const std::string& topology_file,
                          int device = 0,
                          bool capture_spike_events = true,
                          bool enable_plasticity_state = false,
                          StdpParameters stdp_parameters = {});
    ~CudaSimulationRuntime() = default;

    CudaSimulationRuntime(const CudaSimulationRuntime&) = delete;
    CudaSimulationRuntime& operator=(const CudaSimulationRuntime&) = delete;

    ParameterUpdateResult apply_parameters(const ModelParameters& candidate);
    ParameterUpdateResult reset_parameters_to_defaults();
    void reinitialize_simulation();

    void run();
    void run_steps(std::uint32_t steps);

    // Schedule an additive external-current pulse. Steps are zero-based;
    // the pulse is active on [start_step, start_step + duration_steps).
    void schedule_stimulus(std::uint32_t neuron_id,
                           std::uint64_t start_step,
                           std::uint32_t duration_steps,
                           float amplitude);
    void clear_stimuli();
    SimulationMetrics collect_metrics();
    std::vector<std::uint32_t> collect_spike_count_snapshot();
    // Read post-integration voltage and recovery state for selected neurons.
    // Samples reflect threshold resets and the recovery increment when spiking.
    std::vector<VoltageSample> collect_voltage_samples(
        const std::vector<std::uint32_t>& neuron_ids);
    // Return a bounded view of the GPU-resident CSR graph for visualization.
    TopologyVisualSnapshot collect_topology_visual_snapshot(
        std::uint32_t maximum_neurons = 256,
        std::uint32_t maximum_edges_per_neuron = 8);
    // Copy the exact CSR graph currently resident on the GPU to host memory.
    CSRTopology collect_connectivity();
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
    bool plasticity_state_enabled() const noexcept;

private:
    struct DeviceState {
        DeviceState(std::size_t count, const CSRTopology& topology,
                    bool capture_spike_events, bool enable_plasticity_state);
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

        // Optional reverse CSR index and zero-initialized learning traces.
        // Edge indices refer back to the single authoritative weights array.
        DeviceBuffer<std::uint32_t> incoming_row_offsets;
        DeviceBuffer<std::uint32_t> incoming_sources;
        DeviceBuffer<std::uint32_t> incoming_edge_indices;
        DeviceBuffer<float> presynaptic_traces;
        DeviceBuffer<float> postsynaptic_traces;
        bool plasticity_state_enabled = false;

        // Dense target-current ring: [ring_slot][neuron].
        DeviceBuffer<float> synaptic_current;

        // Persistent additive external current and fixed-address event schedule.
        DeviceBuffer<float> external_current;
        DeviceBuffer<ExternalStimulusGPU> external_stimuli;
        DeviceBuffer<std::uint32_t> external_stimulus_count;
        std::size_t neuron_count;
        std::uint32_t edge_count;
        std::uint32_t ring_slots;
        bool capture_spike_events;

        NeuronParametersGPU neuron_parameters() noexcept;
        NeuronStateGPU neuron_state() noexcept;
    };

    std::unique_ptr<DeviceState> build_device_state(const ModelParameters& parameters,
                       bool capture_spike_events, bool enable_plasticity_state);

    ParameterManager parameter_manager_;
    ModelParameters active_parameters_;
    std::string topology_file_;
    std::unique_ptr<DeviceState> device_state_;
    std::uint64_t completed_steps_ = 0;
    std::uint64_t spike_event_read_cursor_ = 0;
    double elapsed_simulation_ms_ = 0.0;
    int device_id_ = 0;
    bool capture_spike_events_ = true;
    bool enable_plasticity_state_ = false;
    StdpParameters stdp_parameters_{};
    std::vector<ExternalStimulusGPU> external_stimuli_;
    bool external_stimuli_dirty_ = false;
};

} // namespace spiking
