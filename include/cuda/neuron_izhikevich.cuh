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

    // Bounded event recorder. Each event stores a one-based step and neuron ID.
    std::uint64_t* spike_event_steps = nullptr;
    std::uint32_t* spike_event_neurons = nullptr;
    unsigned long long* spike_event_count = nullptr;
    std::uint32_t spike_event_capacity = 0;
};

struct SynapseParametersGPU {
    const std::uint32_t* row_offsets = nullptr;
    const std::uint32_t* target_indices = nullptr;
    const float* weights = nullptr;
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

void simulate_neurons(
    NeuronParametersGPU parameters,
    NeuronStateGPU state,
    std::uint32_t neuron_count,
    float dt,
    float threshold,
    float* synaptic_current,
    std::uint32_t ring_slots,
    cudaStream_t stream = nullptr
);

}
