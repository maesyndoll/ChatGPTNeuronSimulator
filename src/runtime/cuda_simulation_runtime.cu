#include "runtime/cuda_simulation_runtime.cuh"

#include "cuda/cuda_check.cuh"
#include "topology/topology_generator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace spiking {

namespace {
// 4,194,304 records × 12 bytes = 48 MiB of bounded GPU event storage.
constexpr std::uint32_t kSpikeEventCapacity = 4U * 1024U * 1024U;

constexpr std::uint64_t kSpikeEventCapacity64 = 4ULL * 1024ULL * 1024ULL;

// Estimate every persistent device allocation before DeviceState's constructor
// starts issuing cudaMalloc calls. Keep headroom for the display driver and
// temporary CUDA/runtime allocations instead of trusting cudaMalloc to fail.
std::uint64_t estimate_device_bytes(
    std::uint64_t neurons, std::uint64_t edges, std::uint64_t ring_slots) {
    if (ring_slots != 0 &&
        neurons > std::numeric_limits<std::uint64_t>::max() / ring_slots) {
        throw std::length_error("Estimated synaptic ring element count overflows");
    }
    std::uint64_t bytes = 0;
    const auto add = [&bytes](std::uint64_t count, std::uint64_t element_size) {
        if (element_size != 0 &&
            count > (std::numeric_limits<std::uint64_t>::max() - bytes) /
                        element_size) {
            throw std::length_error("Estimated CUDA allocation size overflows");
        }
        bytes += count * element_size;
    };

    // Neuron parameters (a,b,c,d,current), external input, and state (voltage,recovery),
    // spike counts/flags, and active-spike source list.
    add(neurons, 6ULL * sizeof(float));
    add(neurons, 2ULL * sizeof(float));
    add(neurons, sizeof(std::uint32_t));
    add(neurons, sizeof(std::uint8_t));
    add(neurons, sizeof(std::uint32_t));
    add(neurons, 3ULL * sizeof(float));      // phase, next phase, natural frequency
    add(neurons, sizeof(float));              // STDP spike traces

    add(1, sizeof(std::uint32_t));             // active spike count
    add(kSpikeEventCapacity64, sizeof(std::uint64_t));
    add(kSpikeEventCapacity64, sizeof(std::uint32_t));
    add(1, sizeof(unsigned long long));        // event count

    add(neurons + 1, sizeof(std::uint32_t));   // CSR row offsets
    add(edges, sizeof(std::uint32_t));         // CSR targets
    add(edges, sizeof(float));                 // CSR weights
    add(edges, sizeof(std::uint32_t));         // CSR delays
    add(neurons * ring_slots, sizeof(float));  // dense delayed-current ring
    return bytes;
}

std::uint32_t adaptive_row_capacity(const CSRTopology& topology) {
    std::uint32_t maximum_degree = 0;
    for (std::size_t row = 0; row + 1 < topology.row_offsets.size(); ++row)
        maximum_degree = std::max(maximum_degree,
            topology.row_offsets[row + 1] - topology.row_offsets[row]);
    const std::uint64_t proposed = std::max<std::uint64_t>(
        32ULL, static_cast<std::uint64_t>(maximum_degree) * 2ULL + 8ULL);
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(
        proposed, topology.neuron_count() > 0 ? topology.neuron_count() - 1ULL : 0ULL));
}

std::uint64_t mib(std::uint64_t bytes) {
    return (bytes + 1024ULL * 1024ULL - 1ULL) / (1024ULL * 1024ULL);
}
}

CudaSimulationRuntime::DeviceState::DeviceState(
    std::size_t count, const CSRTopology& topology)
    : a(count), b(count), c(count), d(count), current(count),
      external_current(count),
      voltage(count), recovery(count), spike_counts(count),
      spike_flags(count),
      active_spike_sources(count), active_spike_count(1),
      spike_event_steps(kSpikeEventCapacity),
      spike_event_neurons(kSpikeEventCapacity), spike_event_count(1),
      row_offsets(count),
      row_counts(count),
      target_indices(count * adaptive_row_capacity(topology)),
      weights(count * adaptive_row_capacity(topology)),
      delays(count * adaptive_row_capacity(topology)),
      phases(count), next_phases(count), natural_frequencies(count),
      plasticity_traces(count),
      neuron_count(count),
      edge_count(static_cast<std::uint32_t>(topology.weights.size())),
      row_capacity(adaptive_row_capacity(topology)),
      ring_slots(topology.maximum_delay + 1U) {
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
        spike_event_steps.get(),
        spike_event_neurons.get(),
        spike_event_count.get(),
        kSpikeEventCapacity
    };
}

CudaSimulationRuntime::CudaSimulationRuntime(ModelParameters defaults, int device)
    : parameter_manager_(std::move(defaults)),
      active_parameters_(parameter_manager_.active_parameters()),
      device_id_(device) {
    SPIKING_CUDA_CHECK(cudaSetDevice(device_id_));
    device_state_ = build_device_state(active_parameters_);
}

CudaSimulationRuntime::CudaSimulationRuntime(
    ModelParameters defaults, CSRTopology topology, int device)
    : parameter_manager_(std::move(defaults)),
      active_parameters_(parameter_manager_.active_parameters()),
      device_id_(device) {
    SPIKING_CUDA_CHECK(cudaSetDevice(device_id_));
    topology.validate();
    if (topology.excitatory_count != active_parameters_.simulation.excitatory_count ||
        topology.inhibitory_count != active_parameters_.simulation.inhibitory_count)
        throw std::invalid_argument("Custom CSR population sizes must match model parameters");
    device_state_ = build_device_state(active_parameters_, &topology);
}

std::unique_ptr<CudaSimulationRuntime::DeviceState>
CudaSimulationRuntime::build_device_state(
    const ModelParameters& parameters, const CSRTopology* topology_override) {
    const auto& p = parameters.simulation;
    const std::uint32_t total = p.neuron_count();
    if (total == 0) throw std::invalid_argument("Cannot initialize an empty network");

    // Reject obviously oversized configurations before the host-side CSR
    // generator allocates potentially hundreds of MiB of edge arrays.
    const auto& tp = parameters.topology;
    const long double e = p.excitatory_count;
    const long double i = p.inhibitory_count;
    const long double probabilities[] = {tp.p_ee, tp.p_ei, tp.p_ie, tp.p_ii};
    for (const auto probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0.0L || probability > 1.0L)
            throw std::invalid_argument("Connection probabilities must be finite and in [0, 1]");
    }
    const long double means[] = {
        e * (e - 1.0L) * tp.p_ee,
        e * i * tp.p_ei,
        i * e * tp.p_ie,
        i * (i - 1.0L) * tp.p_ii
    };
    long double expected_edges = 0.0L;
    long double edge_variance = 0.0L;
    for (int k = 0; k < 4; ++k) {
        expected_edges += means[k];
        edge_variance += means[k] * (1.0L - probabilities[k]);
    }
    const long double upper_edges_ld = std::ceil(
        expected_edges + 6.0L * std::sqrt(edge_variance) + 64.0L);
    if (!std::isfinite(static_cast<double>(upper_edges_ld)) ||
        upper_edges_ld > static_cast<long double>(
            std::numeric_limits<std::uint32_t>::max())) {
        throw std::length_error("Estimated topology exceeds the 32-bit CSR edge limit");
    }
    const auto upper_edges = static_cast<std::uint64_t>(upper_edges_ld);
    const std::uint64_t row_bytes =
        (static_cast<std::uint64_t>(total) + 1ULL) * sizeof(std::uint32_t);
    const std::uint64_t csr_bytes_per_edge =
        sizeof(std::uint32_t) + sizeof(float) + sizeof(std::uint32_t);
    if (row_bytes > tp.memory_budget_bytes ||
        upper_edges > (tp.memory_budget_bytes - row_bytes) / csr_bytes_per_edge) {
        throw std::length_error(
            "Estimated network topology exceeds the configured host topology memory budget. "
            "Reduce neuron counts or connection probabilities.");
    }

    const std::uint64_t ring_slots_estimate =
        static_cast<std::uint64_t>(tp.maximum_delay) + 1ULL;
    const std::uint64_t estimated_device_bytes = estimate_device_bytes(
        total, upper_edges, ring_slots_estimate);
    std::size_t preflight_free_bytes = 0;
    std::size_t preflight_total_bytes = 0;
    SPIKING_CUDA_CHECK(cudaMemGetInfo(
        &preflight_free_bytes, &preflight_total_bytes));
    const std::uint64_t preflight_reserve = std::max<std::uint64_t>(
        256ULL * 1024ULL * 1024ULL,
        static_cast<std::uint64_t>(preflight_total_bytes) / 10ULL);
    const std::uint64_t preflight_usable =
        preflight_free_bytes > preflight_reserve
            ? preflight_free_bytes - preflight_reserve : 0ULL;
    if (estimated_device_bytes > preflight_usable) {
        throw std::length_error(
            "Estimated network needs approximately " +
            std::to_string(mib(estimated_device_bytes)) +
            " MiB of GPU memory; only " +
            std::to_string(mib(preflight_usable)) +
            " MiB is safely available. Reduce neuron counts, connection "
            "probabilities, or maximum synaptic delay.");
    }

    const CSRTopology topology = topology_override
        ? *topology_override
        : generate_random_sparse_topology(p, parameters.topology);

    if (topology.weights.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::length_error("Topology has too many synapses for the CUDA runtime");
    }

    const std::uint64_t ring_slots =
        static_cast<std::uint64_t>(topology.maximum_delay) + 1ULL;
    const std::uint64_t required_bytes = estimate_device_bytes(
        total, topology.weights.size(), ring_slots);

    std::size_t free_bytes = 0;
    std::size_t total_device_bytes = 0;
    SPIKING_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_device_bytes));

    // Preserve at least 256 MiB or 10% of total VRAM, whichever is larger.
    const std::uint64_t reserve_bytes = std::max<std::uint64_t>(
        256ULL * 1024ULL * 1024ULL,
        static_cast<std::uint64_t>(total_device_bytes) / 10ULL);
    const std::uint64_t usable_bytes =
        free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0ULL;
    if (required_bytes > usable_bytes) {
        throw std::length_error(
            "Network requires approximately " + std::to_string(mib(required_bytes)) +
            " MiB of GPU memory, but only " +
            std::to_string(mib(usable_bytes)) +
            " MiB is available after the safety reserve. Reduce neuron counts, "
            "connection probabilities, or maximum synaptic delay.");
    }

    auto next = std::make_unique<DeviceState>(total, topology);

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
        next->external_current.get(), 0,
        next->external_current.size() * sizeof(float), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->plasticity_traces.get(), 0,
        next->plasticity_traces.size() * sizeof(float), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->synaptic_current.get(), 0,
        next->synaptic_current.size() * sizeof(float), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->active_spike_count.get(), 0,
        sizeof(std::uint32_t), stream));
    SPIKING_CUDA_CHECK(cudaMemsetAsync(
        next->spike_event_count.get(), 0,
        sizeof(unsigned long long), stream));

    initialize_neurons(
        next->neuron_parameters(), next->neuron_state(),
        p.excitatory_count, p.inhibitory_count, p.random_seed,
        p.initial_voltage_min, p.initial_voltage_max,
        p.excitatory_current_min, p.excitatory_current_max,
        p.inhibitory_current_min, p.inhibitory_current_max,
        stream);

    initialize_phase_state(
        next->phases.get(), next->natural_frequencies.get(), total,
        p.random_seed ^ 0x6d2b79f5U, stream);

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
    auto replacement = build_device_state(requested);
    const auto committed = parameter_manager_.reinitialize_simulation();
    device_state_.swap(replacement);
    active_parameters_ = committed;
    completed_steps_ = 0;
    elapsed_simulation_ms_ = 0.0;
    spike_event_read_cursor_ = 0;
}

void CudaSimulationRuntime::configure_synaptic_plasticity(
    const SynapticPlasticityConfig& config) {
    if (!std::isfinite(config.trace_time_constant_ms) ||
        config.trace_time_constant_ms <= 0.0f ||
        !std::isfinite(config.potentiation_rate) ||
        config.potentiation_rate < 0.0f ||
        !std::isfinite(config.depression_rate) ||
        config.depression_rate < 0.0f ||
        !std::isfinite(config.minimum_weight_magnitude) ||
        config.minimum_weight_magnitude <= 0.0f ||
        !std::isfinite(config.maximum_weight_magnitude) ||
        config.maximum_weight_magnitude < config.minimum_weight_magnitude) {
        throw std::invalid_argument("Invalid synaptic plasticity configuration");
    }
    if (device_state_) device_state_->stream.synchronize();
    plasticity_config_ = config;
}

void CudaSimulationRuntime::configure_phase_dynamics(
    const PhaseDynamicsConfig& config) {
    if (!std::isfinite(config.coupling_strength) ||
        config.coupling_strength < 0.0f ||
        !std::isfinite(config.modulation_strength) ||
        config.modulation_strength < 0.0f ||
        config.modulation_strength > 1.0f) {
        throw std::invalid_argument(
            "Phase coupling must be finite and non-negative; modulation must be in [0, 1]");
    }
    if (device_state_) device_state_->stream.synchronize();
    phase_config_ = config;
}

void CudaSimulationRuntime::set_neuron_phases(
    const std::vector<float>& phases) {
    if (!device_state_ || phases.size() != device_state_->neuron_count)
        throw std::invalid_argument("Phase vector must match neuron count");
    constexpr float two_pi = 6.28318530717958647692f;
    std::vector<float> wrapped(phases.size());
    for (std::size_t i = 0; i < phases.size(); ++i) {
        if (!std::isfinite(phases[i]))
            throw std::invalid_argument("Neuron phases must be finite");
        wrapped[i] = std::fmod(phases[i], two_pi);
        if (wrapped[i] < 0.0f) wrapped[i] += two_pi;
    }
    device_state_->stream.synchronize();
    SPIKING_CUDA_CHECK(cudaMemcpy(
        device_state_->phases.get(), wrapped.data(),
        wrapped.size() * sizeof(float), cudaMemcpyHostToDevice));
    SPIKING_CUDA_CHECK(cudaMemcpy(
        device_state_->next_phases.get(), wrapped.data(),
        wrapped.size() * sizeof(float), cudaMemcpyHostToDevice));
}

void CudaSimulationRuntime::set_external_current(
    const std::vector<float>& current) {
    if (!device_state_ || current.size() != device_state_->neuron_count)
        throw std::invalid_argument("External current vector must match neuron count");
    for (const float value : current) {
        if (!std::isfinite(value))
            throw std::invalid_argument("External current values must be finite");
    }
    device_state_->stream.synchronize();
    SPIKING_CUDA_CHECK(cudaMemcpy(
        device_state_->external_current.get(), current.data(),
        current.size() * sizeof(float), cudaMemcpyHostToDevice));
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

    for (std::uint32_t i = 0; i < steps; ++i) {
        const std::uint32_t ring_slot = static_cast<std::uint32_t>(
            completed_steps_ % device_state_->ring_slots);

        if (phase_config_.enabled && phase_config_.evolve_phases) {
            advance_phase_state(
                device_state_->phases.get(),
                device_state_->next_phases.get(),
                device_state_->natural_frequencies.get(),
                device_state_->row_offsets.get(),
                device_state_->row_counts.get(),
                device_state_->target_indices.get(),
                device_state_->weights.get(),
                count, p.dt, phase_config_.coupling_strength,
                device_state_->stream.get());
            device_state_->phases.swap(device_state_->next_phases);
        }

        // Refresh the phase pointer after the ping-pong buffer swap.
        // The remaining CSR and current-ring pointers stay stable.
        const SynapseParametersGPU synapses{
            device_state_->row_offsets.get(),
            device_state_->row_counts.get(),
            device_state_->target_indices.get(),
            device_state_->weights.get(),
            device_state_->delays.get(),
            device_state_->edge_count,
            device_state_->synaptic_current.get(),
            device_state_->ring_slots,
            phase_config_.enabled ? device_state_->phases.get() : nullptr,
            phase_config_.enabled ? phase_config_.modulation_strength : 0.0f
        };

        // The flags still describe spikes from the preceding integration.
        // Both kernels execute in-order on the same stream.
        propagate_spikes(
            state, synapses, count, completed_steps_,
            device_state_->stream.get());

        // Propagation has consumed the previous queue. Reset its count
        // before integration appends this step's newly generated spikes.
        SPIKING_CUDA_CHECK(cudaMemsetAsync(
            device_state_->active_spike_count.get(), 0,
            sizeof(std::uint32_t), device_state_->stream.get()));

        simulate_neurons(
            params, state, count, p.dt, p.threshold,
            device_state_->synaptic_current.get(),
            device_state_->external_current.get(), ring_slot,
            completed_steps_ + 1ULL,
            device_state_->stream.get());

        if (plasticity_config_.enabled && device_state_->edge_count != 0) {
            const float trace_decay = std::exp(
                -p.dt / plasticity_config_.trace_time_constant_ms);
            update_synaptic_plasticity(
                device_state_->weights.get(),
                device_state_->row_offsets.get(),
                device_state_->row_counts.get(),
                device_state_->target_indices.get(),
                state.spike_flags,
                device_state_->plasticity_traces.get(),
                count, p.dt, trace_decay,
                plasticity_config_.potentiation_rate,
                plasticity_config_.depression_rate,
                plasticity_config_.minimum_weight_magnitude,
                plasticity_config_.maximum_weight_magnitude,
                device_state_->stream.get());
        }

        ++completed_steps_;
        elapsed_simulation_ms_ += p.dt;
    }
}

SimulationMetrics CudaSimulationRuntime::collect_metrics() {
    if (!device_state_) throw std::logic_error("CUDA simulation is not initialized");
    device_state_->stream.synchronize();

    const auto count = static_cast<std::uint32_t>(device_state_->neuron_count);
    std::vector<float> voltage(count);
    std::vector<float> recovery(count);
    std::vector<std::uint32_t> spikes(count);
    std::vector<std::uint8_t> spike_flags(count);
    std::vector<float> phases;
    std::vector<float> weights;
    if (phase_config_.enabled) phases.resize(count);
    // Report the persistent CSR weights even after plasticity is frozen.\n    // Disabling updates must not make the learned network appear to have zero weights.\n    weights.resize(device_state_->edge_count);
    SPIKING_CUDA_CHECK(cudaMemcpy(voltage.data(), device_state_->voltage.get(),
                                  count * sizeof(float), cudaMemcpyDeviceToHost));
    SPIKING_CUDA_CHECK(cudaMemcpy(recovery.data(), device_state_->recovery.get(),
                                  count * sizeof(float), cudaMemcpyDeviceToHost));
    SPIKING_CUDA_CHECK(cudaMemcpy(spikes.data(), device_state_->spike_counts.get(),
                                  count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    SPIKING_CUDA_CHECK(cudaMemcpy(spike_flags.data(), device_state_->spike_flags.get(),
                                  count * sizeof(std::uint8_t), cudaMemcpyDeviceToHost));
    if (phase_config_.enabled) {
        SPIKING_CUDA_CHECK(cudaMemcpy(phases.data(), device_state_->phases.get(),
                                      count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (!weights.empty()) {
        SPIKING_CUDA_CHECK(cudaMemcpy(weights.data(), device_state_->weights.get(),
                                      weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    SimulationMetrics metrics;
    metrics.completed_steps = completed_steps_;
    double voltage_sum = 0.0;
    double phase_cosine_sum = 0.0;
    double phase_sine_sum = 0.0;
    for (std::uint32_t i = 0; i < count; ++i) {
        if (!std::isfinite(voltage[i]) || !std::isfinite(recovery[i])) {
            metrics.finite_state = false;
        }
        voltage_sum += voltage[i];
        if (phase_config_.enabled) {
            if (!std::isfinite(phases[i])) {
                metrics.finite_state = false;
            } else {
                phase_cosine_sum += std::cos(static_cast<double>(phases[i]));
                phase_sine_sum += std::sin(static_cast<double>(phases[i]));
            }
        }
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
    if (!weights.empty()) {
        double magnitude_sum = 0.0;
        for (const float weight : weights) {
            if (!std::isfinite(weight)) metrics.finite_state = false;
            magnitude_sum += std::abs(static_cast<double>(weight));
        }
        metrics.mean_absolute_synaptic_weight =
            magnitude_sum / static_cast<double>(weights.size());
    }
    if (phase_config_.enabled) {
        const double phase_count = static_cast<double>(count);
        metrics.phase_coherence = std::sqrt(
            phase_cosine_sum * phase_cosine_sum +
            phase_sine_sum * phase_sine_sum) / phase_count;
    }

    const double elapsed_seconds = elapsed_simulation_ms_ / 1000.0;
    if (elapsed_seconds > 0.0) {
        metrics.excitatory_rate_hz = metrics.excitatory_spikes /
            (active_parameters_.simulation.excitatory_count * elapsed_seconds);
        metrics.inhibitory_rate_hz = metrics.inhibitory_spikes /
            (active_parameters_.simulation.inhibitory_count * elapsed_seconds);
    }
    return metrics;
}


std::vector<float> CudaSimulationRuntime::collect_neuron_voltages() {
    if (!device_state_) throw std::logic_error("CUDA simulation is not initialized");
    device_state_->stream.synchronize();
    const auto count = static_cast<std::uint32_t>(device_state_->neuron_count);
    std::vector<float> voltage(count);
    SPIKING_CUDA_CHECK(cudaMemcpy(voltage.data(), device_state_->voltage.get(),
                                  count * sizeof(float), cudaMemcpyDeviceToHost));
    return voltage;
}

SpikeEventBatch CudaSimulationRuntime::collect_spike_events() {
    if (!device_state_)
        throw std::logic_error("CUDA simulation is not initialized");

    // Synchronize once at the batch boundary. The worker thread exclusively
    // owns this runtime, so no kernel can append while the ring is copied.
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
