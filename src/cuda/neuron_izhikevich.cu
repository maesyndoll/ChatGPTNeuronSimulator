#include "cuda/neuron_izhikevich.cuh"

#include <cuda_runtime.h>
#include "cuda/cuda_check.cuh"
#include <cstddef>
#include <cstdint>
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

__global__ void propagate_spikes_kernel(
    NeuronStateGPU state,
    SynapseParametersGPU synapses,
    std::uint32_t neuron_count
) {
    // The count is produced by the preceding neuron kernel. Kernel launches
    // on the runtime stream are ordered, so the queue is complete here.
    __shared__ std::uint32_t active_count_shared;
    __shared__ unsigned long long completed_shared;
    if (threadIdx.x == 0) {
        active_count_shared = *state.active_spike_count;
        completed_shared = *state.completed_steps;
    }
    __syncthreads();
    const std::uint32_t active_count = active_count_shared;
    const unsigned long long completed = completed_shared;

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
            const bool source_is_inhibitory = source >= synapses.excitatory_count;
            const bool selected_source =
                (!synapses.phase_gate_inhibitory_only && !synapses.phase_gate_excitatory_only) ||
                (synapses.phase_gate_inhibitory_only && source_is_inhibitory) ||
                (synapses.phase_gate_excitatory_only && !source_is_inhibitory);
            const bool apply_phase_gate = synapses.phase_gate_enabled && selected_source;
            if (synapses.dynamic_phase_gate_enabled) {
                constexpr float tau = 6.28318530717958647692f;
                const float relative_phase = remainderf(
                    synapses.dynamic_neuron_phases[source] -
                    synapses.dynamic_neuron_phases[target] -
                    synapses.edge_phase_lags[edge], tau);
                if (fabsf(relative_phase) > synapses.gate_half_width_rad) continue;
            } else if (apply_phase_gate) {
                constexpr float tau = 6.28318530717958647692f;
                const float phase = fmodf(
                    synapses.phase_offset_rad +
                        static_cast<float>(completed) * synapses.phase_increment_rad,
                    tau);
                const float error = remainderf(
                    phase - synapses.preferred_phases[target], tau);
                if (fabsf(error) > synapses.gate_half_width_rad) continue;
            }
            const std::uint32_t delay = synapses.delays[edge];

            // Preserve the established one-based delivery timing:
            // a spike at step s with delay d is consumed at step s + d.
            const std::uint32_t slot = static_cast<std::uint32_t>(
                (completed + static_cast<unsigned long long>(delay) - 1ULL) %
                synapses.ring_slots);

            const std::size_t offset =
                static_cast<std::size_t>(slot) * neuron_count + target;

            // Fan-in from multiple sources is accumulated atomically.
            atomicAdd(&synapses.pending_current[offset], synapses.weights[edge]);
        }
    }
}

__global__ void decay_stdp_traces_kernel(
    float* pre_traces, float* post_traces, std::uint32_t count,
    float pre_decay, float post_decay
) {
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    pre_traces[id] *= pre_decay;
    post_traces[id] *= post_decay;
}

__global__ void stdp_depression_kernel(
    const std::uint32_t* active_sources,
    const std::uint32_t* active_count,
    const std::uint32_t* row_offsets,
    const std::uint32_t* targets,
    float* weights,
    const std::uint8_t* edge_plasticity_mask,
    const float* post_traces,
    std::uint32_t neuron_count,
    std::uint32_t excitatory_count,
    float depression_rate,
    float minimum_weight,
    float maximum_weight
) {
    const std::uint32_t queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *active_count;
    if (queue_index >= count) return;
    const std::uint32_t source = active_sources[queue_index];
    if (source >= excitatory_count || source >= neuron_count) return;
    for (std::uint32_t edge = row_offsets[source];
         edge < row_offsets[source + 1]; ++edge) {
        if (edge_plasticity_mask != nullptr && edge_plasticity_mask[edge] == 0) continue;
        const std::uint32_t target = targets[edge];
        weights[edge] -= depression_rate * post_traces[target];
    }
}

__global__ void stdp_potentiation_kernel(
    const std::uint32_t* active_sources,
    const std::uint32_t* active_count,
    const std::uint32_t* incoming_row_offsets,
    const std::uint32_t* incoming_sources,
    const std::uint32_t* incoming_edge_indices,
    float* weights,
    const std::uint8_t* edge_plasticity_mask,
    const float* pre_traces,
    std::uint32_t neuron_count,
    std::uint32_t excitatory_count,
    float potentiation_rate,
    float minimum_weight,
    float maximum_weight
) {
    const std::uint32_t queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *active_count;
    if (queue_index >= count) return;
    const std::uint32_t target = active_sources[queue_index];
    if (target >= neuron_count) return;
    for (std::uint32_t incoming = incoming_row_offsets[target];
         incoming < incoming_row_offsets[target + 1]; ++incoming) {
        const std::uint32_t source = incoming_sources[incoming];
        if (source >= excitatory_count) continue;
        const std::uint32_t edge = incoming_edge_indices[incoming];
        if (edge_plasticity_mask != nullptr && edge_plasticity_mask[edge] == 0) continue;
        weights[edge] += potentiation_rate * pre_traces[source];
    }
}

__global__ void stdp_clamp_weights_kernel(
    const std::uint32_t* row_offsets,
    float* weights,
    const std::uint8_t* edge_plasticity_mask,
    std::uint32_t neuron_count,
    std::uint32_t excitatory_count,
    float minimum_weight,
    float maximum_weight
) {
    const std::uint32_t source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= neuron_count || source >= excitatory_count) return;

    for (std::uint32_t edge = row_offsets[source];
         edge < row_offsets[source + 1]; ++edge) {
        if (edge_plasticity_mask != nullptr && edge_plasticity_mask[edge] == 0) continue;
        weights[edge] = fminf(
            maximum_weight, fmaxf(minimum_weight, weights[edge]));
    }
}

__global__ void increment_stdp_traces_kernel(
    const std::uint32_t* active_sources,
    const std::uint32_t* active_count,
    float* pre_traces,
    float* post_traces,
    std::uint32_t neuron_count
) {
    const std::uint32_t queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = *active_count;
    if (queue_index >= count) return;
    const std::uint32_t neuron = active_sources[queue_index];
    if (neuron >= neuron_count) return;
    pre_traces[neuron] += 1.0f;
    post_traces[neuron] += 1.0f;
}

__global__ void external_stimulus_kernel(
    const ExternalStimulusGPU* stimuli,
    const std::uint32_t* stimulus_count,
    const unsigned long long* completed_steps,
    float* external_current
) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= *stimulus_count || index >= kMaxExternalStimuli) return;

    const ExternalStimulusGPU stimulus = stimuli[index];
    const unsigned long long step = *completed_steps + 1ULL;

    if (step == stimulus.start_step)
        atomicAdd(&external_current[stimulus.neuron_id], stimulus.amplitude);

    if (step == stimulus.end_step)
        atomicAdd(&external_current[stimulus.neuron_id], -stimulus.amplitude);
}

__global__ void neuron_step_kernel(
    NeuronParametersGPU p,
    NeuronStateGPU s,
    std::uint32_t count,
    float dt,
    float threshold,
    float* synaptic_current,
    std::uint32_t ring_slots,
    float* external_current
) {
    const std::uint32_t id =
        blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned long long completed_shared;
    if (threadIdx.x == 0) completed_shared = *s.completed_steps;
    __syncthreads();

    if (id < count) {

    // This buffer represents spikes from this timestep only.
    s.spike_flags[id] = 0;

    float v = s.voltage[id];
    float u = s.recovery[id];

    const float a = p.a[id];
    const float b = p.b[id];
    const float c = p.c[id];
    const float d = p.d[id];
    const unsigned long long completed = completed_shared;
    const std::uint32_t ring_slot =
        static_cast<std::uint32_t>(completed % ring_slots);
    const std::size_t current_offset =
        static_cast<std::size_t>(ring_slot) * count + id;
    const float external =
        external_current != nullptr ? external_current[id] : 0.0f;
    const float current =
        p.current[id] + synaptic_current[current_offset] + external;

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
            s.spike_event_steps[event_slot] = completed + 1ULL;
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

    // All threads in this block have finished their neuron before the block
    // contributes to the grid completion count. The last block advances the
    // global step exactly once, without launching a separate kernel.
    __syncthreads();
    if (threadIdx.x == 0) {
        const std::uint32_t finished_blocks =
            atomicAdd(s.completed_step_blocks, 1U);
        if (finished_blocks + 1U == gridDim.x) {
            atomicAdd(s.completed_steps, 1ULL);
            atomicExch(s.completed_step_blocks, 0U);
        }
    }
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

void propagate_spikes(
    NeuronStateGPU state,
    SynapseParametersGPU synapses,
    std::uint32_t neuron_count,
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
        state, synapses, neuron_count
    );
    SPIKING_CUDA_CHECK(cudaGetLastError());
}

void apply_stdp(
    const std::uint32_t* active_spike_sources,
    const std::uint32_t* active_spike_count,
    const std::uint32_t* outgoing_row_offsets,
    const std::uint32_t* outgoing_targets,
    float* weights,
    const std::uint8_t* edge_plasticity_mask,
    const std::uint32_t* incoming_row_offsets,
    const std::uint32_t* incoming_sources,
    const std::uint32_t* incoming_edge_indices,
    float* presynaptic_traces,
    float* postsynaptic_traces,
    std::uint32_t neuron_count,
    std::uint32_t excitatory_count,
    float dt_ms,
    StdpParametersGPU parameters,
    cudaStream_t stream
) {
    if (neuron_count == 0 || active_spike_sources == nullptr ||
        active_spike_count == nullptr || outgoing_row_offsets == nullptr ||
        incoming_row_offsets == nullptr || presynaptic_traces == nullptr ||
        postsynaptic_traces == nullptr || excitatory_count > neuron_count ||
        !(dt_ms > 0.0f))
        throw std::invalid_argument("Invalid GPU STDP state");
    constexpr int threads = 256;
    const int neuron_blocks = static_cast<int>((neuron_count + threads - 1) / threads);
    decay_stdp_traces_kernel<<<neuron_blocks, threads, 0, stream>>>(
        presynaptic_traces, postsynaptic_traces, neuron_count,
        parameters.presynaptic_decay, parameters.postsynaptic_decay);
    SPIKING_CUDA_CHECK(cudaGetLastError());

    if (weights != nullptr && outgoing_targets != nullptr) {
        stdp_depression_kernel<<<neuron_blocks, threads, 0, stream>>>(
            active_spike_sources, active_spike_count, outgoing_row_offsets,
            outgoing_targets, weights, edge_plasticity_mask,
            postsynaptic_traces, neuron_count,
            excitatory_count, parameters.depression_rate, parameters.minimum_weight,
            parameters.maximum_weight);
        SPIKING_CUDA_CHECK(cudaGetLastError());
    }
    if (weights != nullptr && incoming_sources != nullptr &&
        incoming_edge_indices != nullptr) {
        stdp_potentiation_kernel<<<neuron_blocks, threads, 0, stream>>>(
            active_spike_sources, active_spike_count, incoming_row_offsets,
            incoming_sources, incoming_edge_indices, weights, edge_plasticity_mask,
            presynaptic_traces, neuron_count, excitatory_count,
            parameters.potentiation_rate, parameters.minimum_weight,
            parameters.maximum_weight);
        SPIKING_CUDA_CHECK(cudaGetLastError());
    }
    if (weights != nullptr) {
        stdp_clamp_weights_kernel<<<neuron_blocks, threads, 0, stream>>>(
            outgoing_row_offsets, weights, edge_plasticity_mask,
            neuron_count, excitatory_count,
            parameters.minimum_weight, parameters.maximum_weight);
        SPIKING_CUDA_CHECK(cudaGetLastError());
    }

    increment_stdp_traces_kernel<<<neuron_blocks, threads, 0, stream>>>(
        active_spike_sources, active_spike_count, presynaptic_traces,
        postsynaptic_traces, neuron_count);
    SPIKING_CUDA_CHECK(cudaGetLastError());
}

void apply_external_stimuli(
    const ExternalStimulusGPU* stimuli,
    const std::uint32_t* stimulus_count,
    const unsigned long long* completed_steps,
    float* external_current,
    cudaStream_t stream
) {
    if (stimuli == nullptr || stimulus_count == nullptr ||
        completed_steps == nullptr || external_current == nullptr)
        throw std::invalid_argument("Invalid external stimulation buffers");

    constexpr int threads = 256;
    constexpr int blocks =
        (kMaxExternalStimuli + threads - 1) / threads;

    external_stimulus_kernel<<<blocks, threads, 0, stream>>>(
        stimuli, stimulus_count, completed_steps, external_current);
    SPIKING_CUDA_CHECK(cudaGetLastError());
}

void simulate_neurons(
    NeuronParametersGPU parameters,
    NeuronStateGPU state,
    std::uint32_t neuron_count,
    float dt,
    float threshold,
    float* synaptic_current,
    std::uint32_t ring_slots,
    cudaStream_t stream,
    float* external_current
) {
    constexpr int threads = 256;
    const int blocks = (neuron_count + threads - 1) / threads;

    neuron_step_kernel<<<blocks, threads, 0, stream>>>(
        parameters, state, neuron_count, dt, threshold,
        synaptic_current, ring_slots, external_current
    );
    SPIKING_CUDA_CHECK(cudaGetLastError());

}

}
