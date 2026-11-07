#include "runtime/cuda_simulation_runtime.cuh"

#include "cuda/cuda_check.cuh"
#include "topology/topology_generator.h"
#include "topology/topology_io.h"
#include "plasticity/stdp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace spiking {

namespace {
// 4,194,304 records × 12 bytes = 48 MiB of bounded GPU event storage.
constexpr std::uint32_t kSpikeEventCapacity = 4U * 1024U * 1024U;
constexpr std::uint32_t kCudaGraphBatchSteps = 1000U;
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
    bool capture_events, bool enable_learning_state)
    : a(count), b(count), c(count), d(count), current(count),
      voltage(count), recovery(count), spike_counts(count),
      spike_flags(count),
      active_spike_sources(count), active_spike_count(1),
      completed_steps_device(1), completed_step_blocks(1),
      row_offsets(topology.row_offsets.size()),
      target_indices(topology.target_indices.size()),
      weights(topology.weights.size()),
      delays(topology.delays.size()), preferred_phases(count),
      dynamic_phases(count), edge_phase_lags(topology.weights.size()),
      plasticity_state_enabled(enable_learning_state),
      external_current(count),
      external_stimuli(kMaxExternalStimuli),
      external_stimulus_count(1),
      neuron_count(count),
      edge_count(static_cast<std::uint32_t>(topology.weights.size())),
      ring_slots(topology.maximum_delay + 1U),
      capture_spike_events(capture_events) {
    if (plasticity_state_enabled) {
        const auto incoming = build_incoming_csr_index(topology);
        incoming_row_offsets.allocate(incoming.row_offsets.size());
        incoming_sources.allocate(incoming.source_indices.size());
        incoming_edge_indices.allocate(incoming.edge_indices.size());
        presynaptic_traces.allocate(count);
        postsynaptic_traces.allocate(count);
        const auto stream = this->stream.get();
        SPIKING_CUDA_CHECK(cudaMemcpyAsync(
            incoming_row_offsets.get(), incoming.row_offsets.data(),
            incoming.row_offsets.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice, stream));
        if (!incoming.source_indices.empty()) {
            SPIKING_CUDA_CHECK(cudaMemcpyAsync(
                incoming_sources.get(), incoming.source_indices.data(),
                incoming.source_indices.size() * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice, stream));
            SPIKING_CUDA_CHECK(cudaMemcpyAsync(
                incoming_edge_indices.get(), incoming.edge_indices.data(),
                incoming.edge_indices.size() * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice, stream));
        }
        SPIKING_CUDA_CHECK(cudaMemsetAsync(
            presynaptic_traces.get(), 0, count * sizeof(float), stream));
        SPIKING_CUDA_CHECK(cudaMemsetAsync(
            postsynaptic_traces.get(), 0, count * sizeof(float), stream));
        // The incoming index is a constructor-local host object. Complete its
        // uploads before its vectors are destroyed; this sync is opt-in only.
        SPIKING_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
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
        completed_steps_device.get(),
        completed_step_blocks.get(),
        capture_spike_events ? spike_event_steps.get() : nullptr,
        capture_spike_events ? spike_event_neurons.get() : nullptr,
        capture_spike_events ? spike_event_count.get() : nullptr,
        capture_spike_events ? kSpikeEventCapacity : 0U
    };
}

CudaSimulationRuntime::CudaSimulationRuntime(
    ModelParameters defaults, int device, bool capture_spike_events,
    bool enable_plasticity_state, StdpParameters stdp_parameters)
    : parameter_manager_(std::move(defaults)),
      active_parameters_(parameter_manager_.active_parameters()),
      device_id_(device),
      capture_spike_events_(capture_spike_events),
      enable_plasticity_state_(enable_plasticity_state),
      stdp_parameters_(stdp_parameters) {
    SPIKING_CUDA_CHECK(cudaSetDevice(device_id_));
    device_state_ = build_device_state(active_parameters_, capture_spike_events_,
                                       enable_plasticity_state_);
}

CudaSimulationRuntime::CudaSimulationRuntime(
    ModelParameters defaults, const std::string& topology_file, int device,
    bool capture_spike_events, bool enable_plasticity_state,
    StdpParameters stdp_parameters)
    : parameter_manager_(std::move(defaults)),
      active_parameters_(parameter_manager_.active_parameters()),
      topology_file_(topology_file),
      device_id_(device),
      capture_spike_events_(capture_spike_events),
      enable_plasticity_state_(enable_plasticity_state),
      stdp_parameters_(stdp_parameters) {
    if (topology_file_.empty()) {
        throw std::invalid_argument("Topology filename must not be empty");
    }

    SPIKING_CUDA_CHECK(cudaSetDevice(device_id_));
    device_state_ = build_device_state(active_parameters_, capture_spike_events_,
                                       enable_plasticity_state_);
}

std::unique_ptr<CudaSimulationRuntime::DeviceState>
CudaSimulationRuntime::build_device_state(
    const ModelParameters& parameters, bool capture_spike_events,
    bool enable_plasticity_state) {
    const auto& p = parameters.simulation;
    const std::uint32_t total = p.neuron_count();
    if (total == 0) throw std::invalid_argument("Cannot initialize an empty network");

    const CSRTopology topology = topology_file_.empty()
        ? generate_random_sparse_topology(p, parameters.topology)
        : load_csr_topology(topology_file_);

    if (topology.excitatory_count != p.excitatory_count ||
        topology.inhibitory_count != p.inhibitory_count) {
        throw std::invalid_argument(
            "Saved topology population counts do not match simulation parameters");
    }

    if (topology.weights.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::length_error("Topology has too many synapses for the CUDA runtime");
    }

    auto next = std::make_unique<DeviceState>(
        total, topology, capture_spike_events, enable_plasticity_state);

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
        next->preferred_phases.get(), 0,
        next->preferred_phases.size() * sizeof(float), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->synaptic_current.get(), 0,
        next->synaptic_current.size() * sizeof(float), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->external_current.get(), 0,
        next->external_current.size() * sizeof(float), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->external_stimulus_count.get(), 0,
        sizeof(std::uint32_t), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->active_spike_count.get(), 0,
        sizeof(std::uint32_t), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->completed_steps_device.get(), 0,
        sizeof(unsigned long long), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->completed_step_blocks.get(), 0,
        sizeof(std::uint32_t), stream));
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
    auto replacement = build_device_state(
        requested, capture_spike_events_, enable_plasticity_state_);
    const auto committed = parameter_manager_.reinitialize_simulation();
    device_state_.swap(replacement);
    active_parameters_ = committed;
    completed_steps_ = 0;
    elapsed_simulation_ms_ = 0.0;
    spike_event_read_cursor_ = 0;
    external_stimuli_.clear();
    external_stimuli_dirty_ = false;
}

void CudaSimulationRuntime::schedule_stimulus(
    std::uint32_t neuron_id,
    std::uint64_t start_step,
    std::uint32_t duration_steps,
    float amplitude) {
    if (!device_state_)
        throw std::logic_error("CUDA simulation is not initialized");
    if (neuron_id >= device_state_->neuron_count)
        throw std::out_of_range("External stimulus neuron ID is out of range");
    if (duration_steps == 0)
        throw std::invalid_argument("External stimulus duration must be positive");
    if (!std::isfinite(amplitude))
        throw std::invalid_argument("External stimulus amplitude must be finite");
    if (start_step <= completed_steps_)
        throw std::invalid_argument("External stimulus must start in a future step");
    if (start_step > std::numeric_limits<std::uint64_t>::max() - duration_steps)
        throw std::overflow_error("External stimulus end step overflows");
    if (external_stimuli_.size() >= kMaxExternalStimuli)
        throw std::length_error("External stimulus schedule is full");

    external_stimuli_.push_back({
        neuron_id, start_step, start_step + duration_steps, amplitude
    });
    external_stimuli_dirty_ = true;
}

void CudaSimulationRuntime::configure_phase_gate(
    const std::vector<float>& preferred_phases_rad,
    float oscillator_frequency_hz,
    float gate_half_width_rad,
    float phase_offset_rad) {
    if (!device_state_)
        throw std::logic_error("CUDA simulation is not initialized");
    if (preferred_phases_rad.size() != device_state_->neuron_count)
        throw std::invalid_argument("Phase-tag count must equal neuron count");
    constexpr float tau = 6.28318530717958647692f;
    if (!std::isfinite(oscillator_frequency_hz) || oscillator_frequency_hz <= 0.0f ||
        !std::isfinite(gate_half_width_rad) || gate_half_width_rad < 0.0f ||
        gate_half_width_rad > 3.14159265358979323846f ||
        !std::isfinite(phase_offset_rad))
        throw std::invalid_argument("Invalid phase-gate parameters");
    for (const float phase : preferred_phases_rad) {
        if (!std::isfinite(phase))
            throw std::invalid_argument("Neuron phase tags must be finite");
    }
    device_state_->stream.synchronize();
    SPIKING_CUDA_CHECK(cudaMemcpy(
        device_state_->preferred_phases.get(), preferred_phases_rad.data(),
        preferred_phases_rad.size() * sizeof(float), cudaMemcpyHostToDevice));
    phase_frequency_hz_ = oscillator_frequency_hz;
    phase_gate_half_width_rad_ = gate_half_width_rad;
    phase_offset_rad_ = std::fmod(phase_offset_rad, tau);
    phase_gate_inhibitory_only_ = false;
    phase_gate_excitatory_only_ = false;
    phase_gate_enabled_ = true;
}

void CudaSimulationRuntime::configure_inhibitory_phase_gate(
    const std::vector<float>& preferred_phases_rad,
    float oscillator_frequency_hz,
    float gate_half_width_rad,
    float phase_offset_rad) {
    configure_phase_gate(preferred_phases_rad, oscillator_frequency_hz,
                         gate_half_width_rad, phase_offset_rad);
    phase_gate_inhibitory_only_ = true;
    phase_gate_excitatory_only_ = false;
}

void CudaSimulationRuntime::configure_excitatory_phase_gate(
    const std::vector<float>& preferred_phases_rad,
    float oscillator_frequency_hz,
    float gate_half_width_rad,
    float phase_offset_rad) {
    configure_phase_gate(preferred_phases_rad, oscillator_frequency_hz,
                         gate_half_width_rad, phase_offset_rad);
    phase_gate_inhibitory_only_ = false;
    phase_gate_excitatory_only_ = true;
}

void CudaSimulationRuntime::disable_phase_gate() noexcept {
    phase_gate_enabled_ = false;
    phase_gate_inhibitory_only_ = false;
    phase_gate_excitatory_only_ = false;
}

void CudaSimulationRuntime::clear_stimuli() {
    if (!device_state_)
        throw std::logic_error("CUDA simulation is not initialized");

    device_state_->stream.synchronize();
    SPIKING_CUDA_CHECK(cudaMemset(
        device_state_->external_current.get(), 0,
        device_state_->external_current.size() * sizeof(float)));
    SPIKING_CUDA_CHECK(cudaMemset(
        device_state_->external_stimulus_count.get(), 0,
        sizeof(std::uint32_t)));
    external_stimuli_.clear();
    external_stimuli_dirty_ = false;
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

    if (external_stimuli_dirty_) {
        device_state_->stream.synchronize();
        if (!external_stimuli_.empty()) {
            SPIKING_CUDA_CHECK(cudaMemcpy(
                device_state_->external_stimuli.get(),
                external_stimuli_.data(),
                external_stimuli_.size() * sizeof(ExternalStimulusGPU),
                cudaMemcpyHostToDevice));
        }
        const auto stimulus_count =
            static_cast<std::uint32_t>(external_stimuli_.size());
        SPIKING_CUDA_CHECK(cudaMemcpy(
            device_state_->external_stimulus_count.get(),
            &stimulus_count, sizeof(stimulus_count),
            cudaMemcpyHostToDevice));
        external_stimuli_dirty_ = false;
    }

    if (!std::isfinite(stdp_parameters_.potentiation_rate) ||
        stdp_parameters_.potentiation_rate < 0.0f ||
        !std::isfinite(stdp_parameters_.depression_rate) ||
        stdp_parameters_.depression_rate < 0.0f ||
        !std::isfinite(stdp_parameters_.presynaptic_tau_ms) ||
        stdp_parameters_.presynaptic_tau_ms <= 0.0f ||
        !std::isfinite(stdp_parameters_.postsynaptic_tau_ms) ||
        stdp_parameters_.postsynaptic_tau_ms <= 0.0f ||
        !std::isfinite(stdp_parameters_.minimum_weight) ||
        stdp_parameters_.minimum_weight <= 0.0f ||
        !std::isfinite(stdp_parameters_.maximum_weight) ||
        stdp_parameters_.maximum_weight < stdp_parameters_.minimum_weight)
        throw std::invalid_argument("Invalid STDP parameters");
    const StdpParametersGPU gpu_stdp{
        stdp_parameters_.potentiation_rate,
        stdp_parameters_.depression_rate,
        std::exp(-p.dt / stdp_parameters_.presynaptic_tau_ms),
        std::exp(-p.dt / stdp_parameters_.postsynaptic_tau_ms),
        stdp_parameters_.minimum_weight,
        stdp_parameters_.maximum_weight
    };

    const SynapseParametersGPU synapses{
        device_state_->row_offsets.get(), device_state_->target_indices.get(),
        device_state_->weights.get(), device_state_->delays.get(),
        device_state_->edge_count, device_state_->synaptic_current.get(),
        device_state_->ring_slots,
        phase_gate_enabled_ ? device_state_->preferred_phases.get() : nullptr,
        phase_gate_enabled_
            ? static_cast<float>(6.28318530717958647692 * phase_frequency_hz_ * p.dt / 1000.0)
            : 0.0f,
        phase_offset_rad_, phase_gate_half_width_rad_, phase_gate_enabled_,
        p.excitatory_count, phase_gate_inhibitory_only_,
        phase_gate_excitatory_only_
    };

    const auto apply_learning = [&]() {
        if (!device_state_->plasticity_state_enabled) return;
        apply_stdp(
            state.active_spike_sources, state.active_spike_count,
            device_state_->row_offsets.get(), device_state_->target_indices.get(),
            device_state_->weights.get(),
            device_state_->incoming_row_offsets.get(),
            device_state_->incoming_sources.get(),
            device_state_->incoming_edge_indices.get(),
            device_state_->presynaptic_traces.get(),
            device_state_->postsynaptic_traces.get(),
            count, p.excitatory_count, p.dt, gpu_stdp, stream);
    };

    // Build the reusable graph only when a full batch is first requested.
    // This avoids graph setup overhead for tests and interactive single-step use.
    if (steps >= kCudaGraphBatchSteps &&
        device_state_->step_graph_exec == nullptr) {
        device_state_->stream.synchronize();
        const auto graph_state = device_state_->neuron_state();
        SPIKING_CUDA_CHECK(cudaStreamBeginCapture(
            stream, cudaStreamCaptureModeThreadLocal));
        for (std::uint32_t i = 0; i < kCudaGraphBatchSteps; ++i) {
            apply_external_stimuli(
                device_state_->external_stimuli.get(),
                device_state_->external_stimulus_count.get(),
                graph_state.completed_steps,
                device_state_->external_current.get(), stream);
            if (device_state_->plasticity_state_enabled) {
                apply_stdp(
                    graph_state.active_spike_sources, graph_state.active_spike_count,
                    device_state_->row_offsets.get(), device_state_->target_indices.get(),
                    device_state_->weights.get(),
                    device_state_->incoming_row_offsets.get(),
                    device_state_->incoming_sources.get(),
                    device_state_->incoming_edge_indices.get(),
                    device_state_->presynaptic_traces.get(),
                    device_state_->postsynaptic_traces.get(),
                    count, p.excitatory_count, p.dt, gpu_stdp, stream);
            }
            propagate_spikes(graph_state, synapses, count, stream);
            SPIKING_CUDA_CHECK(cudaMemsetAsync(
                device_state_->active_spike_count.get(), 0,
                sizeof(std::uint32_t), stream));
            simulate_neurons(
                params, graph_state, count, p.dt, p.threshold,
                device_state_->synaptic_current.get(),
                device_state_->ring_slots, stream,
                device_state_->external_current.get());
        }
        SPIKING_CUDA_CHECK(cudaStreamEndCapture(
            stream, &device_state_->step_graph));
        SPIKING_CUDA_CHECK(cudaGraphInstantiate(
            &device_state_->step_graph_exec,
            device_state_->step_graph, nullptr, nullptr, 0));
    }

    while (steps >= kCudaGraphBatchSteps) {
        SPIKING_CUDA_CHECK(cudaGraphLaunch(device_state_->step_graph_exec, stream));
        steps -= kCudaGraphBatchSteps;
        completed_steps_ += kCudaGraphBatchSteps;
        elapsed_simulation_ms_ +=
            static_cast<double>(kCudaGraphBatchSteps) * p.dt;
    }

    for (std::uint32_t i = 0; i < steps; ++i) {
        apply_external_stimuli(
            device_state_->external_stimuli.get(),
            device_state_->external_stimulus_count.get(),
            state.completed_steps,
            device_state_->external_current.get(), stream);
        apply_learning();
        propagate_spikes(state, synapses, count, stream);
        SPIKING_CUDA_CHECK(cudaMemsetAsync(
            device_state_->active_spike_count.get(), 0,
            sizeof(std::uint32_t), stream));
        simulate_neurons(
            params, state, count, p.dt, p.threshold,
            device_state_->synaptic_current.get(),
            device_state_->ring_slots, stream,
            device_state_->external_current.get());
        ++completed_steps_;
        elapsed_simulation_ms_ += p.dt;
    }
}

std::vector<VoltageSample> CudaSimulationRuntime::collect_voltage_samples(
    const std::vector<std::uint32_t>& neuron_ids) {
    if (!device_state_)
        throw std::logic_error("CUDA simulation is not initialized");

    const auto count = device_state_->neuron_count;
    for (const auto neuron_id : neuron_ids) {
        if (neuron_id >= count)
            throw std::out_of_range("Voltage trace neuron ID is out of range");
    }

    std::vector<VoltageSample> samples;
    samples.reserve(neuron_ids.size());
    if (neuron_ids.empty()) return samples;

    // The runtime stream is non-blocking. Finish queued kernels before the
    // synchronous device-to-host read, just as collect_metrics() does.
    device_state_->stream.synchronize();

    const double time_ms = elapsed_simulation_ms_;
    for (const auto neuron_id : neuron_ids) {
        float voltage = 0.0f;
        float recovery = 0.0f;
        std::uint8_t spike_flag = 0;
        SPIKING_CUDA_CHECK(cudaMemcpy(
            &voltage, device_state_->voltage.get() + neuron_id,
            sizeof(float), cudaMemcpyDeviceToHost));
        SPIKING_CUDA_CHECK(cudaMemcpy(
            &recovery, device_state_->recovery.get() + neuron_id,
            sizeof(float), cudaMemcpyDeviceToHost));
        SPIKING_CUDA_CHECK(cudaMemcpy(
            &spike_flag, device_state_->spike_flags.get() + neuron_id,
            sizeof(std::uint8_t), cudaMemcpyDeviceToHost));
        samples.push_back({
            completed_steps_,
            time_ms,
            neuron_id,
            voltage,
            recovery,
            spike_flag != 0
        });
    }
    return samples;
}

TopologyVisualSnapshot CudaSimulationRuntime::collect_topology_visual_snapshot(
    std::uint32_t maximum_neurons,
    std::uint32_t maximum_edges_per_neuron) {
    if (!device_state_)
        throw std::logic_error("CUDA simulation is not initialized");
    if (maximum_neurons == 0 || maximum_edges_per_neuron == 0)
        throw std::invalid_argument("Topology visualization limits must be non-zero");

    // Transfer the compact CSR row index and selected contiguous row ranges.
    // Never copy the full multi-million-edge graph for visualization.
    device_state_->stream.synchronize();

    TopologyVisualSnapshot snapshot;
    snapshot.excitatory_count = active_parameters_.simulation.excitatory_count;
    snapshot.inhibitory_count = active_parameters_.simulation.inhibitory_count;
    snapshot.synapse_count = device_state_->edge_count;

    std::vector<std::uint32_t> row_offsets(device_state_->neuron_count + 1U);
    SPIKING_CUDA_CHECK(cudaMemcpy(
        row_offsets.data(), device_state_->row_offsets.get(),
        row_offsets.size() * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost));

    const auto append_population = [&](std::uint32_t first,
                                       std::uint32_t count,
                                       std::uint32_t budget) {
        if (count == 0 || budget == 0) return;
        const std::uint32_t selected = std::min(count, budget);
        const std::uint32_t block_width = std::min<std::uint32_t>(8U, selected);
        const std::uint32_t blocks = (selected + block_width - 1U) / block_width;
        for (std::uint32_t block = 0; block < blocks; ++block) {
            const std::uint32_t block_index = block * block_width;
            const std::uint32_t block_count =
                std::min(block_width, selected - block_index);
            const std::uint32_t maximum_start = count - block_count;
            const std::uint32_t relative_start = blocks <= 1
                ? maximum_start / 2U
                : static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(block) * maximum_start) /
                    (blocks - 1U));
            const std::uint32_t source_start = first + relative_start;
            const std::uint32_t source_end = source_start + block_count;
            const std::uint32_t edge_begin = row_offsets[source_start];
            const std::uint32_t edge_end = row_offsets[source_end];

            for (std::uint32_t source = source_start;
                 source < source_end; ++source)
                snapshot.neuron_ids.push_back(source);

            if (edge_end == edge_begin) continue;
            std::vector<std::uint32_t> targets(edge_end - edge_begin);
            std::vector<float> weights(edge_end - edge_begin);
            SPIKING_CUDA_CHECK(cudaMemcpy(
                targets.data(), device_state_->target_indices.get() + edge_begin,
                targets.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
            SPIKING_CUDA_CHECK(cudaMemcpy(
                weights.data(), device_state_->weights.get() + edge_begin,
                weights.size() * sizeof(float), cudaMemcpyDeviceToHost));

            for (std::uint32_t source = source_start;
                 source < source_end; ++source) {
                const std::uint32_t local_begin = row_offsets[source] - edge_begin;
                const std::uint32_t degree =
                    row_offsets[source + 1U] - row_offsets[source];
                const std::uint32_t take =
                    std::min(degree, maximum_edges_per_neuron);
                for (std::uint32_t sample = 0; sample < take; ++sample) {
                    const std::uint32_t local_edge = local_begin +
                        (take <= 1 ? 0U :
                         static_cast<std::uint32_t>(
                            (static_cast<std::uint64_t>(sample) * (degree - 1U)) /
                            (take - 1U)));
                    const std::uint32_t target = targets[local_edge];
                    snapshot.edges.push_back({source, target, weights[local_edge]});
                    snapshot.neuron_ids.push_back(target);
                }
            }
        }
    };

    const std::uint32_t excitatory = snapshot.excitatory_count;
    const std::uint32_t inhibitory = snapshot.inhibitory_count;
    const std::uint32_t excitatory_budget = inhibitory == 0
        ? maximum_neurons : std::max<std::uint32_t>(1U, maximum_neurons / 2U);
    const std::uint32_t inhibitory_budget = inhibitory == 0
        ? 0U : maximum_neurons - excitatory_budget;
    append_population(0U, excitatory, excitatory_budget);
    append_population(excitatory, inhibitory, inhibitory_budget);

    std::sort(snapshot.neuron_ids.begin(), snapshot.neuron_ids.end());
    snapshot.neuron_ids.erase(
        std::unique(snapshot.neuron_ids.begin(), snapshot.neuron_ids.end()),
        snapshot.neuron_ids.end());
    return snapshot;
}

CSRTopology CudaSimulationRuntime::collect_connectivity() {
    if (!device_state_)
        throw std::logic_error("CUDA simulation is not initialized");

    // All graph arrays are immutable after initialization. Synchronize the
    // owning stream before copying the exact device-resident graph to host.
    device_state_->stream.synchronize();

    CSRTopology topology;
    topology.excitatory_count = active_parameters_.simulation.excitatory_count;
    topology.inhibitory_count = active_parameters_.simulation.inhibitory_count;
    topology.minimum_delay = active_parameters_.topology.minimum_delay;
    topology.maximum_delay = active_parameters_.topology.maximum_delay;

    topology.row_offsets.resize(device_state_->neuron_count + 1U);
    topology.target_indices.resize(device_state_->edge_count);
    topology.weights.resize(device_state_->edge_count);
    topology.delays.resize(device_state_->edge_count);

    SPIKING_CUDA_CHECK(cudaMemcpy(
        topology.row_offsets.data(), device_state_->row_offsets.get(),
        topology.row_offsets.size() * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost));

    if (device_state_->edge_count != 0) {
        SPIKING_CUDA_CHECK(cudaMemcpy(
            topology.target_indices.data(), device_state_->target_indices.get(),
            topology.target_indices.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost));
        SPIKING_CUDA_CHECK(cudaMemcpy(
            topology.weights.data(), device_state_->weights.get(),
            topology.weights.size() * sizeof(float),
            cudaMemcpyDeviceToHost));
        SPIKING_CUDA_CHECK(cudaMemcpy(
            topology.delays.data(), device_state_->delays.get(),
            topology.delays.size() * sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost));
    }

    topology.validate();
    return topology;
}

std::vector<std::uint32_t> CudaSimulationRuntime::collect_spike_count_snapshot() {
    if (!device_state_)
        throw std::logic_error("CUDA simulation is not initialized");
    // The runtime stream is non-blocking; finish queued kernels before copying.
    device_state_->stream.synchronize();
    const auto count = device_state_->neuron_count;
    std::vector<std::uint32_t> snapshot(count);
    SPIKING_CUDA_CHECK(cudaMemcpy(
        snapshot.data(), device_state_->spike_counts.get(),
        count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    return snapshot;
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


std::uint64_t CudaSimulationRuntime::state_fingerprint() {
    if (!device_state_) throw std::logic_error("CUDA simulation is not initialized");
    device_state_->stream.synchronize();

    const auto count = static_cast<std::size_t>(device_state_->neuron_count);
    std::vector<float> a(count), b(count), c(count), d(count);
    std::vector<float> current(count), voltage(count), recovery(count);
    const auto copy = [count](float* destination, const DeviceBuffer<float>& source) {
        SPIKING_CUDA_CHECK(cudaMemcpy(
            destination, source.get(), count * sizeof(float),
            cudaMemcpyDeviceToHost));
    };
    copy(a.data(), device_state_->a);
    copy(b.data(), device_state_->b);
    copy(c.data(), device_state_->c);
    copy(d.data(), device_state_->d);
    copy(current.data(), device_state_->current);
    copy(voltage.data(), device_state_->voltage);
    copy(recovery.data(), device_state_->recovery);

    // FNV-1a over the exact float bit patterns, in a stable field order.
    std::uint64_t hash = 14695981039346656037ULL;
    const auto add = [&hash](const std::vector<float>& values) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
        const std::size_t size = values.size() * sizeof(float);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    };
    add(a);
    add(b);
    add(c);
    add(d);
    add(current);
    add(voltage);
    add(recovery);
    return hash;
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

bool CudaSimulationRuntime::plasticity_state_enabled() const noexcept {
    return device_state_ != nullptr && device_state_->plasticity_state_enabled;
}

bool CudaSimulationRuntime::has_pending_reinitialization() const {
    return parameter_manager_.has_pending_reinitialization();
}

std::uint64_t CudaSimulationRuntime::simulation_generation() const {
    return parameter_manager_.simulation_generation();
}

} // namespace spiking
