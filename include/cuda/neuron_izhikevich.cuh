#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace spiking {

struct NeuronParametersGPU {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    float* d = nullptr;
    float* current = nullptr;
};

struct NeuronStateGPU {
    float* voltage = nullptr;
    float* recovery = nullptr;
    std::uint32_t* spike_counts = nullptr;
    std::uint8_t* spike_flags = nullptr;

    // Compact list of neurons that spiked during the preceding integration.
    std::uint32_t* active_spike_sources = nullptr;
    std::uint32_t* active_spike_count = nullptr;
    unsigned long long* completed_steps = nullptr;
    std::uint32_t* completed_step_blocks = nullptr;

    // Bounded event recorder. Each event stores a one-based step and neuron ID.
    std::uint64_t* spike_event_steps = nullptr;
    std::uint32_t* spike_event_neurons = nullptr;
    unsigned long long* spike_event_count = nullptr;
    std::uint32_t spike_event_capacity = 0;
};

inline constexpr std::uint32_t kMaxExternalStimuli = 4096U;

struct ExternalStimulusGPU {
    std::uint32_t neuron_id = 0;
    std::uint64_t start_step = 0;
    std::uint64_t end_step = 0;
    float amplitude = 0.0f;
};

struct StdpParametersGPU {
    float potentiation_rate = 0.005f;
    float depression_rate = 0.00525f;
    float presynaptic_decay = 1.0f;
    float postsynaptic_decay = 1.0f;
    float minimum_weight = 0.001f;
    float maximum_weight = 1.0f;
};

struct SynapseParametersGPU {
    const std::uint32_t* row_offsets = nullptr;
    const std::uint32_t* target_indices = nullptr;
    float* weights = nullptr;
    const std::uint32_t* delays = nullptr;
    std::uint32_t edge_count = 0;
    float* pending_current = nullptr;
    std::uint32_t ring_slots = 0;
};

void initialize_neurons(
    NeuronParametersGPU parameters,
    NeuronStateGPU state,
    std::uint32_t excitatory_count,
    std::uint32_t inhibitory_count,
    std::uint32_t seed,
    float initial_voltage_min,
    float initial_voltage_max,
    float excitatory_current_min,
    float excitatory_current_max,
    float inhibitory_current_min,
    float inhibitory_current_max,
    cudaStream_t stream = nullptr
);

void propagate_spikes(
    NeuronStateGPU state,
    SynapseParametersGPU synapses,
    std::uint32_t neuron_count,
    cudaStream_t stream = nullptr
);

void apply_stdp(
    const std::uint32_t* active_spike_sources,
    const std::uint32_t* active_spike_count,
    const std::uint32_t* outgoing_row_offsets,
    const std::uint32_t* outgoing_targets,
    float* weights,
    const std::uint32_t* incoming_row_offsets,
    const std::uint32_t* incoming_sources,
    const std::uint32_t* incoming_edge_indices,
    float* presynaptic_traces,
    float* postsynaptic_traces,
    std::uint32_t neuron_count,
    std::uint32_t excitatory_count,
    float dt_ms,
    StdpParametersGPU parameters,
    cudaStream_t stream = nullptr
);

void simulate_neurons(
    NeuronParametersGPU parameters,
    NeuronStateGPU state,
    std::uint32_t neuron_count,
    float dt,
    float threshold,
    float* synaptic_current,
    std::uint32_t ring_slots,
    cudaStream_t stream = nullptr,
    float* external_current = nullptr
);

void apply_external_stimuli(
    const ExternalStimulusGPU* stimuli,
    const std::uint32_t* stimulus_count,
    const unsigned long long* completed_steps,
    float* external_current,
    cudaStream_t stream = nullptr
);

}
