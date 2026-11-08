#include "cuda/neuron_izhikevich.cuh"

#include <cuda_runtime.h>
#include "cuda/cuda_check.cuh"
#include <cstddef>
#include <cstdint>\n#include <cmath>
#include <stdexcept>

namespace spiking {

__device__ unsigned int hash_rng(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ float uniform01(unsigned int x) {
    return static_cast<float>(hash_rng(x)) / 4294967295.0f;
}

__global__ void initialize_kernel(
    NeuronParametersGPU p,
    NeuronStateGPU s,
    std::uint32_t excitatory_count,
    std::uint32_t total_count,
    std::uint32_t seed,
    float v_min,
    float v_max,
    float exc_i_min,
    float exc_i_max,
    float inh_i_min,
    float inh_i_max
) {
    const std::uint32_t id =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= total_count) return;

    const bool excitatory = id < excitatory_count;

    const float r1 = uniform01(seed ^ (id * 2U + 1U));
    const float r2 = uniform01(seed ^ (id * 2U + 2U));

    // Regular-spiking excitatory population.
    if (excitatory) {
        p.a[id] = 0.02f;
        p.b[id] = 0.20f;
        p.c[id] = -65.0f;
        p.d[id] = 8.0f;

        s.voltage[id] = v_min + r1 * (v_max - v_min);
        p.current[id] =
            exc_i_min + r2 * (exc_i_max - exc_i_min);
    }
    // Fast-spiking inhibitory population.
    else {
        p.a[id] = 0.10f;
        p.b[id] = 0.20f;
        p.c[id] = -65.0f;
        p.d[id] = 2.0f;

        s.voltage[id] = v_min + r1 * (v_max - v_min);
        p.current[id] =
            inh_i_min + r2 * (inh_i_max - inh_i_min);
    }

    s.recovery[id] = p.b[id] * s.voltage[id];
    s.spike_counts[id] = 0;
    s.spike_flags[id] = 0;
}

constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kGoldenAngle = 2.39996322972865332f;

__global__ void initialize_phase_kernel(
    float* phases,
    float* natural_frequencies,
    std::uint32_t count,
    std::uint32_t seed
) {
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    const unsigned int phase_hash = hash_rng(seed ^ (id * 0x9e3779b9U + 0x85ebca6bU));
    const unsigned int frequency_hash = hash_rng(seed ^ (id * 0xc2b2ae35U + 0x27d4eb2fU));
    const float phase_offset = uniform01(hash_rng(seed ^ 0xa511e9b3U)) * kTwoPi;
    phases[id] = fmodf(phase_offset + static_cast<float>(id) * kGoldenAngle, kTwoPi);
    // Natural angular frequency in radians per simulated millisecond.
    natural_frequencies[id] = 0.002f + 0.006f * uniform01(frequency_hash ^ phase_hash);
}

__global__ void advance_phase_kernel(
    const float* phases,
    float* next_phases,
    const float* natural_frequencies,
    const std::uint32_t* row_offsets,
    const std::uint32_t* target_indices,
    const float* weights,
    std::uint32_t count,
    float dt,
    float coupling_strength
) {
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;

    float interaction = 0.0f;
    float normalization = 0.0f;
    const std::uint32_t begin = row_offsets[id];
    const std::uint32_t end = row_offsets[id + 1U];
    for (std::uint32_t edge = begin; edge < end; ++edge) {
        const float magnitude = fabsf(weights[edge]);
        if (magnitude > 0.0f) {
            const std::uint32_t target = target_indices[edge];
            interaction += magnitude * sinf(phases[target] - phases[id]);
            normalization += magnitude;
        }
    }
    if (normalization > 0.0f) interaction /= normalization;

    float next = phases[id] + dt *
        (natural_frequencies[id] + coupling_strength * interaction);
    next = fmodf(next, kTwoPi);
    if (next < 0.0f) next += kTwoPi;
    next_phases[id] = next;
}

__global__ void propagate_spikes_kernel(
    NeuronStateGPU state,
    SynapseParametersGPU synapses,
    std::uint32_t neuron_count,
    std::uint64_t step_index
) {
    // The count is produced by the preceding neuron kernel. Kernel launches
    // on the runtime stream are ordered, so the queue is complete here.
    const std::uint32_t active_count = *state.active_spike_count;

    // A fixed worker grid avoids reading the queue count back to the CPU.
    // Each block handles one or more active sources; its threads cooperate
    // over that source's outgoing CSR edges.
    for (std::uint32_t queue_index = blockIdx.x;
         queue_index < active_count;
         queue_index += gridDim.x) {
        const std::uint32_t source =
            state.active_spike_sources[queue_index];

        const std::uint32_t begin = synapses.row_offsets[source];
        const std::uint32_t end = synapses.row_offsets[source + 1];

        for (std::uint32_t edge = begin + threadIdx.x;
             edge < end;
             edge += blockDim.x) {
            const std::uint32_t target = synapses.target_indices[edge];
            const std::uint32_t delay = synapses.delays[edge];

            // Preserve the established one-based delivery timing:
            // a spike at step s with delay d is consumed at step s + d.
            const std::uint32_t slot = static_cast<std::uint32_t>(
                (step_index + static_cast<std::uint64_t>(delay) - 1ULL) %
                synapses.ring_slots);

            const std::size_t offset =
                static_cast<std::size_t>(slot) * neuron_count + target;

            float effective_weight = synapses.weights[edge];
            if (synapses.phases != nullptr &&
                synapses.phase_modulation_strength > 0.0f) {
                const float phase_delta =
                    synapses.phases[target] - synapses.phases[source];
                // Smoothly attenuate out-of-phase communication without
                // reversing excitatory/inhibitory synaptic sign.
                const float phase_gain = 1.0f -
                    0.5f * synapses.phase_modulation_strength *
                    (1.0f - cosf(phase_delta));
                effective_weight *= phase_gain;
            }

            // Fan-in from multiple sources is accumulated atomically.
            atomicAdd(&synapses.pending_current[offset], effective_weight);
        }
    }
}

__global__ void neuron_step_kernel(
    NeuronParametersGPU p,
    NeuronStateGPU s,
    std::uint32_t count,
    float dt,
    float threshold,
    float* synaptic_current,
    std::uint32_t ring_slot,
    std::uint64_t step_index
) {
    const std::uint32_t id =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= count) return;

    // This buffer represents spikes from this timestep only.
    s.spike_flags[id] = 0;

    float v = s.voltage[id];
    float u = s.recovery[id];

    const float a = p.a[id];
    const float b = p.b[id];
    const float c = p.c[id];
    const float d = p.d[id];
    const std::size_t current_offset =
        static_cast<std::size_t>(ring_slot) * count + id;
    const float current =
        p.current[id] + synaptic_current[current_offset];

    // Consume this slot exactly once. Propagation and integration are
    // ordered in the same CUDA stream.
    synaptic_current[current_offset] = 0.0f;

    // Izhikevich equations evaluated at the beginning of the step.
    const float dv1 =
        0.04f * v * v + 5.0f * v + 140.0f - u + current;
    const float du1 = a * (b * v - u);

    // Midpoint RK2.
    const float vm = v + 0.5f * dt * dv1;
    const float um = u + 0.5f * dt * du1;

    const float dv2 =
        0.04f * vm * vm + 5.0f * vm + 140.0f - um + current;
    const float du2 = a * (b * vm - um);

    v += dt * dv2;
    u += dt * du2;

    if (v >= threshold) {
        v = c;
        u += d;
        s.spike_counts[id] += 1U;
        s.spike_flags[id] = 1;

        // Append to a bounded GPU event ring. The monotonically increasing
        // counter is the event sequence; the host detects overwritten records.
        if (s.spike_event_count != nullptr &&
            s.spike_event_capacity != 0 &&
            s.spike_event_steps != nullptr &&
            s.spike_event_neurons != nullptr) {
            const unsigned long long event_index =
                atomicAdd(s.spike_event_count, 1ULL);
            const std::uint32_t event_slot = static_cast<std::uint32_t>(
                event_index % s.spike_event_capacity);
            s.spike_event_steps[event_slot] = step_index;
            s.spike_event_neurons[event_slot] = id;
        }

        // At most one queue entry per neuron per integration step.
        const std::uint32_t queue_index =
            atomicAdd(s.active_spike_count, 1U);
        if (queue_index < count) {
            s.active_spike_sources[queue_index] = id;
        }
    }

    s.voltage[id] = v;
    s.recovery[id] = u;
}

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
    cudaStream_t stream
) {
    constexpr int threads = 256;

    const std::uint32_t total =
        excitatory_count + inhibitory_count;

    const int blocks = (total + threads - 1) / threads;

    initialize_kernel<<<blocks, threads, 0, stream>>>(
        parameters,
        state,
        excitatory_count,
        total,
        seed,
        initial_voltage_min,
        initial_voltage_max,
        excitatory_current_min,
        excitatory_current_max,
        inhibitory_current_min,
        inhibitory_current_max
    );
    SPIKING_CUDA_CHECK(cudaGetLastError());
}

void initialize_phase_state(
    float* phases,
    float* natural_frequencies,
    std::uint32_t neuron_count,
    std::uint32_t seed,
    cudaStream_t stream
) {
    if (phases == nullptr || natural_frequencies == nullptr || neuron_count == 0)
        throw std::invalid_argument("Invalid phase-state buffers");
    constexpr int threads = 256;
    const int blocks = (neuron_count + threads - 1) / threads;
    initialize_phase_kernel<<<blocks, threads, 0, stream>>>(
        phases, natural_frequencies, neuron_count, seed);
    SPIKING_CUDA_CHECK(cudaGetLastError());
}

void advance_phase_state(
    const float* phases,
    float* next_phases,
    const float* natural_frequencies,
    const std::uint32_t* row_offsets,
    const std::uint32_t* target_indices,
    const float* weights,
    std::uint32_t neuron_count,
    float dt,
    float coupling_strength,
    cudaStream_t stream
) {
    if (phases == nullptr || next_phases == nullptr ||
        natural_frequencies == nullptr || row_offsets == nullptr ||
        (weights != nullptr && target_indices == nullptr))
        throw std::invalid_argument("Invalid phase-dynamics buffers");
    if (!std::isfinite(dt) || dt <= 0.0f ||
        !std::isfinite(coupling_strength) || coupling_strength < 0.0f)
        throw std::invalid_argument("Invalid phase-dynamics integration parameters");
    constexpr int threads = 256;
    const int blocks = (neuron_count + threads - 1) / threads;
    advance_phase_kernel<<<blocks, threads, 0, stream>>>(
        phases, next_phases, natural_frequencies, row_offsets,
        target_indices, weights, neuron_count, dt, coupling_strength);
    SPIKING_CUDA_CHECK(cudaGetLastError());
}

void propagate_spikes(
    NeuronStateGPU state,
    SynapseParametersGPU synapses,
    std::uint32_t neuron_count,
    std::uint64_t step_index,
    cudaStream_t stream
) {
    if (synapses.edge_count == 0) return;
    if (synapses.ring_slots == 0 || synapses.pending_current == nullptr)
        throw std::invalid_argument("Invalid synaptic delay ring");
    if (state.active_spike_count == nullptr ||
        state.active_spike_sources == nullptr)
        throw std::invalid_argument("Invalid active-spike queue");

    constexpr int threads = 128;
    constexpr std::uint32_t maximum_worker_blocks = 1024U;
    const std::uint32_t blocks =
        neuron_count < maximum_worker_blocks
            ? neuron_count
            : maximum_worker_blocks;

    propagate_spikes_kernel<<<blocks, threads, 0, stream>>>(
        state, synapses, neuron_count, step_index
    );
    SPIKING_CUDA_CHECK(cudaGetLastError());
}

void simulate_neurons(
    NeuronParametersGPU parameters,
    NeuronStateGPU state,
    std::uint32_t neuron_count,
    float dt,
    float threshold,
    float* synaptic_current,
    std::uint32_t ring_slot,
    std::uint64_t step_index,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    const int blocks = (neuron_count + threads - 1) / threads;

    neuron_step_kernel<<<blocks, threads, 0, stream>>>(
        parameters, state, neuron_count, dt, threshold,
        synaptic_current, ring_slot, step_index
    );
    SPIKING_CUDA_CHECK(cudaGetLastError());
}

}
