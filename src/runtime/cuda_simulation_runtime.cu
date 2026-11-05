#include "runtime/cuda_simulation_runtime.cuh"

#include "cuda/cuda_check.cuh"
#include "topology/topology_generator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace spiking {

namespace {
// 4,194,304 records × 12 bytes = 48 MiB of bounded GPU event storage.
constexpr std::uint32_t kSpikeEventCapacity = 4U * 1024U * 1024U;
constexpr std::uint32_t kCudaGraphBatchSteps = 100U;
}

CudaSimulationRuntime::DeviceState::~DeviceState() {
    if (stream.get() != nullptr) (void)cudaStreamSynchronize(stream.get());
    if (step_graph_exec != nullptr) {
        (void)cudaGraphExecDestroy(step_graph_exec);
        step_graph_exec = nullptr;
    }
    if (step_graph != nullptr) {
        (void)cudaGraphDestroy(step_graph);
        step_graph = nullptr;
    }
}

CudaSimulationRuntime::DeviceState::DeviceState(
    std::size_t count, const CSRTopology& topology,
    bool capture_events)
    : a(count), b(count), c(count), d(count), current(count),
      voltage(count), recovery(count), spike_counts(count),
      spike_flags(count),
      active_spike_sources(count), active_spike_count(1),
      completed_steps_device(1),
      row_offsets(topology.row_offsets.size()),
      target_indices(topology.target_indices.size()),
      weights(topology.weights.size()),
      delays(topology.delays.size()),
      neuron_count(count),
      edge_count(static_cast<std::uint32_t>(topology.weights.size())),
      ring_slots(topology.maximum_delay + 1U),
      capture_spike_events(capture_events) {
    if (capture_spike_events) {
        spike_event_steps.allocate(kSpikeEventCapacity);
        spike_event_neurons.allocate(kSpikeEventCapacity);
        spike_event_count.allocate(1);
    }

    if (ring_slots == 0)
        throw std::length_error("Synaptic delay ring slot count overflow");

    if (count > std::numeric_limits<std::size_t>::max() / ring_slots)
        throw std::length_error("Synaptic delay ring element count overflow");

    const std::size_t ring_elements = count * ring_slots;
    if (ring_elements > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::length_error("Synaptic delay ring byte count overflow");

    const std::size_t ring_bytes = ring_elements * sizeof(float);
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    SPIKING_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    if (ring_bytes > free_bytes) {
        throw std::length_error(
            "Synaptic delay ring requires more GPU memory than is available");
    }

    synaptic_current.allocate(ring_elements);
}

NeuronParametersGPU CudaSimulationRuntime::DeviceState::neuron_parameters() noexcept {
    return {a.get(), b.get(), c.get(), d.get(), current.get()};
}

NeuronStateGPU CudaSimulationRuntime::DeviceState::neuron_state() noexcept {
    return {
        voltage.get(),
        recovery.get(),
        spike_counts.get(),
        spike_flags.get(),
        active_spike_sources.get(),
        active_spike_count.get(),
        capture_spike_events ? spike_event_steps.get() : nullptr,
        capture_spike_events ? spike_event_neurons.get() : nullptr,
        capture_spike_events ? spike_event_count.get() : nullptr,
        capture_spike_events ? kSpikeEventCapacity : 0U,
        completed_steps_device.get()
    };
}

CudaSimulationRuntime::CudaSimulationRuntime(
    ModelParameters defaults, int device, bool capture_spike_events)
    : parameter_manager_(std::move(defaults)),
      active_parameters_(parameter_manager_.active_parameters()),
      device_id_(device),
      capture_spike_events_(capture_spike_events) {
    SPIKING_CUDA_CHECK(cudaSetDevice(device_id_));
    device_state_ = build_device_state(active_parameters_, capture_spike_events_);
}

std::unique_ptr<CudaSimulationRuntime::DeviceState>
CudaSimulationRuntime::build_device_state(
    const ModelParameters& parameters, bool capture_spike_events) {
    const auto& p = parameters.simulation;
    const std::uint32_t total = p.neuron_count();
    if (total == 0) throw std::invalid_argument("Cannot initialize an empty network");

    const CSRTopology topology =
        generate_random_sparse_topology(p, parameters.topology);

    if (topology.weights.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::length_error("Topology has too many synapses for the CUDA runtime");
    }

    auto next = std::make_unique<DeviceState>(total, topology, capture_spike_events);

    const auto stream = next->stream.get();
    if (!topology.row_offsets.empty()) {
        SPIKING_CUDA_CHECK(cudaMemcpyAsync(
            next->row_offsets.get(), topology.row_offsets.data(),
            topology.row_offsets.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice, stream));
    }
    if (!topology.target_indices.empty()) {
        SPIKING_CUDA_CHECK(cudaMemcpyAsync(
            next->target_indices.get(), topology.target_indices.data(),
            topology.target_indices.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice, stream));
        SPIKING_CUDA_CHECK(cudaMemcpyAsync(
            next->weights.get(), topology.weights.data(),
            topology.weights.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream));
        SPIKING_CUDA_CHECK(cudaMemcpyAsync(
            next->delays.get(), topology.delays.data(),
            topology.delays.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice, stream));
    }

    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->synaptic_current.get(), 0,
        next->synaptic_current.size() * sizeof(float), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->active_spike_count.get(), 0,
        sizeof(std::uint32_t), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->completed_steps_device.get(), 0,
        sizeof(unsigned long long), stream));
    if (capture_spike_events) {
        SPIKING_CUDA_CHECK(cudaMemsetAsync(
            next->spike_event_count.get(), 0,
            sizeof(unsigned long long), stream));
    }

    initialize_neurons(
        next->neuron_parameters(), next->neuron_state(),
        p.excitatory_count, p.inhibitory_count, p.random_seed,
        p.initial_voltage_min, p.initial_voltage_max,
        p.excitatory_current_min, p.excitatory_current_max,
        p.inhibitory_current_min, p.inhibitory_current_max,
        stream);

    next->stream.synchronize();

    // Capture a reusable 100-step batch. Step-dependent values are read from
    // the device counter, so replay advances ring slots and event timestamps.
    const auto graph_state = next->neuron_state();
    const SynapseParametersGPU graph_synapses{
        next->row_offsets.get(), next->target_indices.get(),
        next->weights.get(), next->delays.get(), next->edge_count,
        next->synaptic_current.get(), next->ring_slots
    };
    SPIKING_CUDA_CHECK(cudaStreamBeginCapture(
        stream, cudaStreamCaptureModeThreadLocal));
    for (std::uint32_t i = 0; i < kCudaGraphBatchSteps; ++i) {
        propagate_spikes(graph_state, graph_synapses, total, stream);
        SPIKING_CUDA_CHECK(cudaMemsetAsync(
            next->active_spike_count.get(), 0,
            sizeof(std::uint32_t), stream));
        simulate_neurons(
            next->neuron_parameters(), graph_state, total,
            p.dt, p.threshold, next->synaptic_current.get(),
            next->ring_slots, stream);
    }
    SPIKING_CUDA_CHECK(cudaStreamEndCapture(stream, &next->step_graph));
    SPIKING_CUDA_CHECK(cudaGraphInstantiate(
        &next->step_graph_exec, next->step_graph, nullptr, nullptr, 0));
    return next;
}

ParameterUpdateResult CudaSimulationRuntime::apply_parameters(
    const ModelParameters& candidate) {
    auto result = parameter_manager_.apply_parameters(candidate);
    if (result.accepted) active_parameters_ = parameter_manager_.active_parameters();
    return result;
}

ParameterUpdateResult CudaSimulationRuntime::reset_parameters_to_defaults() {
    auto result = parameter_manager_.reset_parameters_to_defaults();
    if (result.accepted) active_parameters_ = parameter_manager_.active_parameters();
    return result;
}

void CudaSimulationRuntime::reinitialize_simulation() {
    // Prepare and initialize replacement GPU state before committing the new
    // active configuration. If allocation or initialization fails, the old
    // simulation and the manager's active snapshot remain intact.
    const auto requested = parameter_manager_.requested_parameters();
    auto replacement = build_device_state(requested, capture_spike_events_);
    const auto committed = parameter_manager_.reinitialize_simulation();
    device_state_.swap(replacement);
    active_parameters_ = committed;
    completed_steps_ = 0;
    elapsed_simulation_ms_ = 0.0;
    spike_event_read_cursor_ = 0;
}

void CudaSimulationRuntime::run() {
    run_steps(active_parameters_.simulation.steps);
}

void CudaSimulationRuntime::run_steps(std::uint32_t steps) {
    if (!device_state_) throw std::logic_error("CUDA simulation is not initialized");
    const auto& p = active_parameters_.simulation;
    const auto params = device_state_->neuron_parameters();
    const auto state = device_state_->neuron_state();
    const auto count = static_cast<std::uint32_t>(device_state_->neuron_count);
    const auto stream = device_state_->stream.get();

    const SynapseParametersGPU synapses{
        device_state_->row_offsets.get(), device_state_->target_indices.get(),
        device_state_->weights.get(), device_state_->delays.get(),
        device_state_->edge_count, device_state_->synaptic_current.get(),
        device_state_->ring_slots
    };

    while (steps >= kCudaGraphBatchSteps) {
        SPIKING_CUDA_CHECK(cudaGraphLaunch(device_state_->step_graph_exec, stream));
        steps -= kCudaGraphBatchSteps;
        completed_steps_ += kCudaGraphBatchSteps;
        elapsed_simulation_ms_ +=
            static_cast<double>(kCudaGraphBatchSteps) * p.dt;
    }

    for (std::uint32_t i = 0; i < steps; ++i) {
        propagate_spikes(state, synapses, count, stream);
        SPIKING_CUDA_CHECK(cudaMemsetAsync(
            device_state_->active_spike_count.get(), 0,
            sizeof(std::uint32_t), stream));
        simulate_neurons(
            params, state, count, p.dt, p.threshold,
            device_state_->synaptic_current.get(),
            device_state_->ring_slots, stream);
        ++completed_steps_;
        elapsed_simulation_ms_ += p.dt;
    }
}

SimulationMetrics CudaSimulationRuntime::collect_metrics() {
    if (!device_state_) throw std::logic_error("CUDA simulation is not initialized");
    // This runtime uses a non-blocking CUDA stream. The synchronous copies
    // below use the default stream and do not provide an ordering dependency
    // with this runtime's work, so explicitly finish queued simulation work
    // before reading device state.
    device_state_->stream.synchronize();
    const auto count = static_cast<std::uint32_t>(device_state_->neuron_count);
    std::vector<float> voltage(count);
    std::vector<float> recovery(count);
    std::vector<std::uint32_t> spikes(count);
    std::vector<std::uint8_t> spike_flags(count);
    SPIKING_CUDA_CHECK(cudaMemcpy(voltage.data(), device_state_->voltage.get(),
                                  count * sizeof(float), cudaMemcpyDeviceToHost));
    SPIKING_CUDA_CHECK(cudaMemcpy(recovery.data(), device_state_->recovery.get(),
                                  count * sizeof(float), cudaMemcpyDeviceToHost));
    SPIKING_CUDA_CHECK(cudaMemcpy(spikes.data(), device_state_->spike_counts.get(),
                                  count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    SPIKING_CUDA_CHECK(cudaMemcpy(spike_flags.data(), device_state_->spike_flags.get(),
                                  count * sizeof(std::uint8_t), cudaMemcpyDeviceToHost));

    SimulationMetrics metrics;
    metrics.completed_steps = completed_steps_;
    double voltage_sum = 0.0;
    for (std::uint32_t i = 0; i < count; ++i) {
        if (!std::isfinite(voltage[i]) || !std::isfinite(recovery[i])) {
            metrics.finite_state = false;
        }
        voltage_sum += voltage[i];
        if (i < active_parameters_.simulation.excitatory_count)
            metrics.excitatory_spikes += spikes[i];
        else
            metrics.inhibitory_spikes += spikes[i];

        if (spike_flags[i] != 0) {
            if (i < active_parameters_.simulation.excitatory_count)
                ++metrics.last_step_excitatory_spikes;
            else
                ++metrics.last_step_inhibitory_spikes;
        }
    }
    metrics.mean_voltage_mv = voltage_sum / static_cast<double>(count);

    const double elapsed_seconds = elapsed_simulation_ms_ / 1000.0;
    if (elapsed_seconds > 0.0) {
        metrics.excitatory_rate_hz = metrics.excitatory_spikes /
            (active_parameters_.simulation.excitatory_count * elapsed_seconds);
        metrics.inhibitory_rate_hz = metrics.inhibitory_spikes /
            (active_parameters_.simulation.inhibitory_count * elapsed_seconds);
    }
    return metrics;
}


std::uint32_t CudaSimulationRuntime::spike_event_collection_batch_interval(
    std::uint32_t batch_size) const noexcept {
    if (batch_size == 0 || !device_state_ || device_state_->neuron_count == 0)
        return 1;

    // In the absolute worst case every neuron spikes on every integration
    // step. Keep at least half of the ring free and cap the delay at eight
    // batches so live visualization remains responsive.
    std::uint64_t safe_batches = kSpikeEventCapacity / 2U;
    safe_batches /= static_cast<std::uint64_t>(device_state_->neuron_count);
    safe_batches /= static_cast<std::uint64_t>(batch_size);
    safe_batches = std::max<std::uint64_t>(1U,
        std::min<std::uint64_t>(safe_batches, 8U));
    return static_cast<std::uint32_t>(safe_batches);
}

SpikeEventBatch CudaSimulationRuntime::collect_spike_events() {
    if (!device_state_)
        throw std::logic_error("CUDA simulation is not initialized");
    if (!capture_spike_events_) return {};

    // The runtime stream is non-blocking, so default-stream copies below
    // are not a completion barrier for its queued kernels. Synchronize once
    // per event batch before reading the event counter and ring.
    device_state_->stream.synchronize();
    unsigned long long device_total = 0;
    SPIKING_CUDA_CHECK(cudaMemcpy(
        &device_total, device_state_->spike_event_count.get(),
        sizeof(device_total), cudaMemcpyDeviceToHost));

    SpikeEventBatch batch;
    batch.total_events = static_cast<std::uint64_t>(device_total);

    const std::uint64_t capacity = kSpikeEventCapacity;
    std::uint64_t begin = spike_event_read_cursor_;
    if (batch.total_events > begin + capacity) {
        const std::uint64_t earliest_available = batch.total_events - capacity;
        batch.dropped_events = earliest_available - begin;
        begin = earliest_available;
    }

    const std::uint64_t available = batch.total_events - begin;
    batch.first_sequence = begin;
    batch.events.resize(static_cast<std::size_t>(available));

    if (available != 0) {
        const std::uint32_t start_slot =
            static_cast<std::uint32_t>(begin % capacity);
        const std::uint32_t first_count = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(available, capacity - start_slot));
        const std::uint32_t second_count =
            static_cast<std::uint32_t>(available - first_count);

        std::vector<std::uint64_t> steps(static_cast<std::size_t>(available));
        std::vector<std::uint32_t> neurons(static_cast<std::size_t>(available));

        SPIKING_CUDA_CHECK(cudaMemcpy(
            steps.data(),
            device_state_->spike_event_steps.get() + start_slot,
            static_cast<std::size_t>(first_count) * sizeof(std::uint64_t),
            cudaMemcpyDeviceToHost));
        SPIKING_CUDA_CHECK(cudaMemcpy(
            neurons.data(),
            device_state_->spike_event_neurons.get() + start_slot,
            static_cast<std::size_t>(first_count) * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost));

        if (second_count != 0) {
            SPIKING_CUDA_CHECK(cudaMemcpy(
                steps.data() + first_count,
                device_state_->spike_event_steps.get(),
                static_cast<std::size_t>(second_count) * sizeof(std::uint64_t),
                cudaMemcpyDeviceToHost));
            SPIKING_CUDA_CHECK(cudaMemcpy(
                neurons.data() + first_count,
                device_state_->spike_event_neurons.get(),
                static_cast<std::size_t>(second_count) * sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost));
        }

        for (std::size_t i = 0; i < batch.events.size(); ++i) {
            batch.events[i] = {steps[i], neurons[i]};
        }
    }

    spike_event_read_cursor_ = batch.total_events;
    return batch;
}

ModelParameters CudaSimulationRuntime::active_parameters() const {
    return parameter_manager_.active_parameters();
}

ModelParameters CudaSimulationRuntime::requested_parameters() const {
    return parameter_manager_.requested_parameters();
}

bool CudaSimulationRuntime::has_pending_reinitialization() const {
    return parameter_manager_.has_pending_reinitialization();
}

std::uint64_t CudaSimulationRuntime::simulation_generation() const {
    return parameter_manager_.simulation_generation();
}

} // namespace spiking
