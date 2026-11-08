#include "runtime/simulation_controller.h"

#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace spiking {

SimulationController::SimulationController(
    ModelParameters defaults,
    int device,
    std::uint32_t batch_size,
    std::uint64_t telemetry_interval_steps)
    : batch_size_(batch_size),
      telemetry_interval_steps_(telemetry_interval_steps) {
    if (batch_size_ == 0) {
        throw std::invalid_argument(
            "Simulation controller batch_size must be positive");
    }
    if (telemetry_interval_steps_ == 0) {
        throw std::invalid_argument(
            "Simulation controller telemetry interval must be positive");
    }

    std::promise<void> initialized;
    auto ready = initialized.get_future();

    worker_ = std::thread(
        &SimulationController::worker_main,
        this,
        std::move(defaults),
        device,
        std::move(initialized));

    try {
        ready.get();
    } catch (...) {
        if (worker_.joinable()) worker_.join();
        throw;
    }
}

SimulationController::~SimulationController() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        status_ = ControllerStatus::ShuttingDown;
    }

    condition_.notify_all();

    if (worker_.joinable()) worker_.join();
}

void SimulationController::start() {
    submit([this](CudaSimulationRuntime&) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (status_ == ControllerStatus::Faulted) {
            throw std::logic_error(
                "Cannot start a faulted simulation; reinitialize first");
        }
        if (status_ == ControllerStatus::ShuttingDown) {
            throw std::logic_error("Simulation controller is shutting down");
        }

        last_execution_error_ = nullptr;
        status_ = ControllerStatus::Running;
    });

    condition_.notify_one();
}

void SimulationController::pause() {
    submit([this](CudaSimulationRuntime& runtime) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (status_ != ControllerStatus::Faulted) {
                status_ = ControllerStatus::Paused;
            }
        }

        // Publish the final state at the pause boundary.
        const auto metrics = runtime.collect_metrics();
        const auto events = runtime.collect_spike_events();
        publish_spike_events(events, runtime.simulation_generation());
        publish_metrics(metrics);
    });
}

void SimulationController::step(std::uint32_t steps) {
    submit([this, steps](CudaSimulationRuntime& runtime) {
        run_steps_with_stimuli(runtime, steps);
        const auto metrics = runtime.collect_metrics();
        const auto events = runtime.collect_spike_events();
        publish_spike_events(events, runtime.simulation_generation());
        publish_metrics(metrics);

        std::lock_guard<std::mutex> lock(mutex_);
        if (status_ == ControllerStatus::Ready) {
            status_ = ControllerStatus::Paused;
        }
    });
}

void SimulationController::stimulate_neurons(
    const std::vector<std::uint32_t>& neuron_ids,
    float amplitude,
    std::uint32_t duration_steps) {
    if (neuron_ids.empty())
        throw std::invalid_argument("stimulate_neurons requires at least one neuron ID");
    if (!std::isfinite(amplitude) || amplitude == 0.0f)
        throw std::invalid_argument("stimulus amplitude must be finite and non-zero");
    if (duration_steps == 0 || duration_steps > 1000000U)
        throw std::invalid_argument("stimulus duration must be between 1 and 1000000 steps");

    submit([this, neuron_ids, amplitude, duration_steps](CudaSimulationRuntime& runtime) {
        const auto parameters = runtime.active_parameters();
        const std::uint64_t count = static_cast<std::uint64_t>(
            parameters.simulation.excitatory_count) +
            parameters.simulation.inhibitory_count;
        for (const auto id : neuron_ids) {
            if (id >= count)
                throw std::out_of_range("stimulus neuron ID is outside the active population");
        }
        active_stimuli_.push_back({neuron_ids, amplitude, duration_steps});
        external_current_dirty_ = true;
    });
}

void SimulationController::run_steps_with_stimuli(
    CudaSimulationRuntime& runtime, std::uint32_t steps) {
    std::uint32_t remaining = steps;
    while (remaining > 0) {
        if (!active_stimuli_.empty() || external_current_dirty_) {
            const auto parameters = runtime.active_parameters();
            const std::size_t count = static_cast<std::size_t>(
                parameters.simulation.excitatory_count) +
                parameters.simulation.inhibitory_count;
            std::vector<float> current(count, 0.0f);
            for (const auto& pulse : active_stimuli_) {
                for (const auto id : pulse.neuron_ids)
                    current[id] += pulse.amplitude;
            }
            runtime.set_external_current(current);
            external_current_dirty_ = false;
        }

        std::uint32_t chunk = remaining;
        for (const auto& pulse : active_stimuli_)
            chunk = std::min(chunk, pulse.remaining_steps);
        if (chunk == 0) break;

        runtime.run_steps(chunk);
        remaining -= chunk;
        if (!active_stimuli_.empty()) {
            for (auto& pulse : active_stimuli_)
                pulse.remaining_steps -= chunk;
            active_stimuli_.erase(
                std::remove_if(active_stimuli_.begin(), active_stimuli_.end(),
                    [](const StimulusPulse& pulse) { return pulse.remaining_steps == 0; }),
                active_stimuli_.end());
            // Rebuild the external-current vector at the exact pulse expiry
            // boundary, so shorter pulses are not stretched to batch length.
            external_current_dirty_ = true;
        }
    }
}

ParameterUpdateResult SimulationController::apply_parameters(
    const ModelParameters& candidate) {
    return submit([candidate](CudaSimulationRuntime& runtime) {
        return runtime.apply_parameters(candidate);
    });
}

ParameterUpdateResult SimulationController::reset_parameters_to_defaults() {
    return submit([](CudaSimulationRuntime& runtime) {
        return runtime.reset_parameters_to_defaults();
    });
}

void SimulationController::reinitialize_simulation() {
    submit([this](CudaSimulationRuntime& runtime) {
        runtime.reinitialize_simulation();
        const auto metrics = runtime.collect_metrics();
        const auto events = runtime.collect_spike_events();
        publish_spike_events(events, runtime.simulation_generation());
        publish_metrics(metrics);

        std::lock_guard<std::mutex> lock(mutex_);
        last_execution_error_ = nullptr;
        status_ = ControllerStatus::Paused;
    });
}

SimulationMetrics SimulationController::telemetry() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return telemetry_.metrics;
}

TelemetrySnapshot SimulationController::telemetry_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return telemetry_;
}

SpikeEventSnapshot SimulationController::spike_events_snapshot(
    std::uint64_t after_sequence,
    std::size_t maximum_events) const {
    std::lock_guard<std::mutex> lock(mutex_);

    SpikeEventSnapshot snapshot;
    snapshot.total_events = spike_event_total_events_;
    snapshot.dropped_events = spike_event_dropped_events_;
    snapshot.simulation_generation = spike_event_generation_;
    snapshot.first_sequence = spike_events_.empty()
        ? spike_event_total_events_
        : spike_events_.front().sequence;

    if (maximum_events == 0 || spike_events_.empty()) {
        return snapshot;
    }

    for (const auto& event : spike_events_) {
        if (event.sequence < after_sequence) {
            continue;
        }
        snapshot.events.push_back(event);
        if (snapshot.events.size() >= maximum_events) {
            break;
        }
    }
    return snapshot;
}

ControllerConfigurationSnapshot
SimulationController::configuration_snapshot() const {
    return const_cast<SimulationController*>(this)->submit(
        [](CudaSimulationRuntime& runtime) {
            ControllerConfigurationSnapshot snapshot;
            snapshot.requested_parameters = runtime.requested_parameters();
            snapshot.active_parameters = runtime.active_parameters();
            snapshot.has_pending_reinitialization =
                runtime.has_pending_reinitialization();
            snapshot.parameter_version = runtime.parameter_version();
            snapshot.simulation_generation = runtime.simulation_generation();
            return snapshot;
        }
    );
}

std::uint64_t SimulationController::transact_tissue(
    std::function<void(tissue::TissueModel&)> mutation) {
    if (!mutation) {
        throw std::invalid_argument("Tissue transaction callback is empty");
    }
    return submit([this, mutation = std::move(mutation)](CudaSimulationRuntime&) mutable {
        // Validate and mutate an isolated candidate. Any exception leaves the
        // currently published tissue and revision untouched.
        tissue::TissueModel candidate = tissue_model_;
        mutation(candidate);
        candidate.validate();
        tissue_model_ = std::move(candidate);
        return ++tissue_revision_;
    });
}

TissueSnapshot SimulationController::tissue_snapshot() const {
    return const_cast<SimulationController*>(this)->submit(
        [this](CudaSimulationRuntime&) {
            return TissueSnapshot{tissue_model_, tissue_revision_};
        });
}

ControllerStatus SimulationController::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

bool SimulationController::is_running() const {
    return status() == ControllerStatus::Running;
}

bool SimulationController::wait_until_steps(
    std::uint64_t target,
    std::chrono::milliseconds timeout) const {
    std::unique_lock<std::mutex> lock(mutex_);

    const bool reached = condition_.wait_for(lock, timeout, [this, target] {
        return observed_completed_steps_ >= target ||
               status_ == ControllerStatus::Faulted ||
               status_ == ControllerStatus::ShuttingDown;
    });

    return reached && observed_completed_steps_ >= target;
}

std::exception_ptr SimulationController::last_execution_error() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_execution_error_;
}

void SimulationController::publish_spike_events(
    const SpikeEventBatch& batch, std::uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);

    // A lower cumulative counter indicates that the runtime was reinitialized.
    if (generation != spike_event_generation_) {
        spike_events_.clear();
        spike_event_dropped_events_ = 0;
        spike_event_generation_ = generation;
    } else if (batch.total_events < spike_event_total_events_) {
        spike_events_.clear();
        spike_event_dropped_events_ = 0;
    }

    spike_event_total_events_ = batch.total_events;
    spike_event_dropped_events_ += batch.dropped_events;

    for (std::size_t index = 0; index < batch.events.size(); ++index) {
        const auto& event = batch.events[index];
        spike_events_.push_back({
            batch.first_sequence + static_cast<std::uint64_t>(index),
            event.step,
            event.neuron_id
        });

        if (spike_events_.size() > kMaximumRetainedSpikeEvents) {
            spike_events_.pop_front();
            ++spike_event_dropped_events_;
        }
    }
}

void SimulationController::publish_metrics(const SimulationMetrics& metrics) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        telemetry_.metrics = metrics;
        ++telemetry_.sequence;
        if (metrics.completed_steps > observed_completed_steps_) {
            observed_completed_steps_ = metrics.completed_steps;
        }
    }
    condition_.notify_all();
}

void SimulationController::worker_main(
    ModelParameters defaults,
    int device,
    std::promise<void> initialized) {
    std::unique_ptr<CudaSimulationRuntime> runtime;

    try {
        runtime = std::make_unique<CudaSimulationRuntime>(
            std::move(defaults), device);
        const auto initial_metrics = runtime->collect_metrics();
        const auto initial_events = runtime->collect_spike_events();
        publish_spike_events(initial_events, runtime->simulation_generation());
        publish_metrics(initial_metrics);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            spike_event_generation_ = runtime->simulation_generation();
        }
        initialized.set_value();
    } catch (...) {
        initialized.set_exception(std::current_exception());
        return;
    }

    std::uint64_t next_telemetry_step = telemetry_interval_steps_;
    std::uint64_t observed_generation = runtime->simulation_generation();

    for (;;) {
        std::function<void(CudaSimulationRuntime&)> command;
        bool execute_batch = false;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] {
                return status_ == ControllerStatus::ShuttingDown ||
                       !commands_.empty() ||
                       status_ == ControllerStatus::Running;
            });

            // Finish already-submitted commands before leaving the worker.
            if (!commands_.empty()) {
                command = std::move(commands_.front());
                commands_.pop_front();
            } else if (status_ == ControllerStatus::ShuttingDown) {
                break;
            } else {
                execute_batch = status_ == ControllerStatus::Running;
            }
        }

        if (command) {
            command(*runtime);

            // A reinitialization starts a new completed-step epoch.
            const auto generation = runtime->simulation_generation();
            if (generation != observed_generation) {
                observed_generation = generation;
                const auto completed = runtime->completed_steps();
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    observed_completed_steps_ = completed;
                }
                next_telemetry_step = completed + telemetry_interval_steps_;
                condition_.notify_all();
            }
            continue;
        }

        if (!execute_batch) continue;

        try {
            run_steps_with_stimuli(*runtime, batch_size_);

            const auto generation = runtime->simulation_generation();
            if (generation != observed_generation) {
                observed_generation = generation;
                next_telemetry_step =
                    runtime->completed_steps() + telemetry_interval_steps_;
            }

            const auto completed = runtime->completed_steps();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                observed_completed_steps_ = completed;
            }
            condition_.notify_all();

            // Drain the GPU event ring after every batch so that a busy
            // simulation cannot overwrite events while waiting for telemetry.
            const auto events = runtime->collect_spike_events();
            publish_spike_events(events, runtime->simulation_generation());

            // Keep telemetry collection at its existing lower frequency.
            if (completed >= next_telemetry_step) {
                const auto metrics = runtime->collect_metrics();
                publish_metrics(metrics);
                next_telemetry_step = completed + telemetry_interval_steps_;
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                last_execution_error_ = std::current_exception();
                status_ = ControllerStatus::Faulted;
            }
            condition_.notify_all();
        }
    }

    // CUDA runtime destruction remains on its owning worker thread.
    runtime.reset();
}

} // namespace spiking
