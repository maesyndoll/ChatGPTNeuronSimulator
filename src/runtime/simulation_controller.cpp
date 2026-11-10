#include "runtime/simulation_controller.h"
#include "topology/topology_io.h"
#include "runtime/graft_package.h"
#include "runtime/neural_self_observer.h"

#include <stdexcept>

namespace spiking {

SimulationController::SimulationController(
    ModelParameters defaults,
    int device,
    std::uint32_t batch_size,
    std::uint64_t telemetry_interval_steps,
    bool collect_spike_events,
    bool enable_plasticity_state,
    StdpParameters stdp_parameters)
    : batch_size_(batch_size),
      telemetry_interval_steps_(telemetry_interval_steps),
      collect_spike_events_(collect_spike_events) {
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
        std::move(initialized),
        collect_spike_events_,
        enable_plasticity_state,
        stdp_parameters);

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
        if (collect_spike_events_) {
            const auto events = runtime.collect_spike_events();
            publish_spike_events(events, runtime.simulation_generation());
        }
        publish_metrics(metrics);
    });
}

void SimulationController::step(std::uint32_t steps) {
    submit([this, steps](CudaSimulationRuntime& runtime) {
        runtime.run_steps(steps);
        const auto metrics = runtime.collect_metrics();
        if (collect_spike_events_) {
            const auto events = runtime.collect_spike_events();
            publish_spike_events(events, runtime.simulation_generation());
        }
        publish_metrics(metrics);

        std::lock_guard<std::mutex> lock(mutex_);
        if (status_ == ControllerStatus::Ready) {
            status_ = ControllerStatus::Paused;
        }
    });
}


GraftPackageInfo SimulationController::attach_graft_package(
    const std::string& package_path, bool plasticity_enabled) {
    return submit([this, package_path, plasticity_enabled](CudaSimulationRuntime& runtime) {
        const auto info = GraftPackageInspector::inspect(package_path);
        const auto topology = load_csr_topology(info.topology_path.string());
        runtime.attach_graft_topology(topology, plasticity_enabled);
        const auto metrics = runtime.collect_metrics();
        if (collect_spike_events_) {
            const auto events = runtime.collect_spike_events();
            publish_spike_events(events, runtime.simulation_generation());
        }
        publish_metrics(metrics);
        return info;
    });
}

void SimulationController::detach_graft_package() {
    submit([this](CudaSimulationRuntime& runtime) {
        runtime.detach_graft_topology();
        const auto metrics = runtime.collect_metrics();
        if (collect_spike_events_) {
            const auto events = runtime.collect_spike_events();
            publish_spike_events(events, runtime.simulation_generation());
        }
        publish_metrics(metrics);
    });
}

bool SimulationController::graft_attached() const {
    return const_cast<SimulationController*>(this)->submit(
        [](CudaSimulationRuntime& runtime) { return runtime.graft_attached(); });
}

void SimulationController::schedule_stimulus(
    std::uint32_t neuron_id,
    std::uint64_t start_step,
    std::uint32_t duration_steps,
    float amplitude) {
    submit([=](CudaSimulationRuntime& runtime) {
        runtime.schedule_stimulus(
            neuron_id, start_step, duration_steps, amplitude);
    });
}

void SimulationController::schedule_stimuli(
    const std::vector<ExternalStimulus>& stimuli) {
    if (stimuli.empty()) return;
    submit([stimuli](CudaSimulationRuntime& runtime) {
        for (const auto& stimulus : stimuli) {
            runtime.schedule_stimulus(
                stimulus.neuron_id, stimulus.start_step,
                stimulus.duration_steps, stimulus.amplitude);
        }
    });
}

void SimulationController::clear_stimuli() {
    submit([](CudaSimulationRuntime& runtime) {
        runtime.clear_stimuli();
    });
}

CSRTopology SimulationController::connectivity_snapshot() const {
    return const_cast<SimulationController*>(this)->submit(
        [](CudaSimulationRuntime& runtime) {
            return runtime.collect_connectivity();
        }
    );
}

std::vector<VoltageSample> SimulationController::collect_voltage_samples(
    const std::vector<std::uint32_t>& neuron_ids) const {
    return const_cast<SimulationController*>(this)->submit(
        [neuron_ids](CudaSimulationRuntime& runtime) {
            return runtime.collect_voltage_samples(neuron_ids);
        }
    );
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
        if (collect_spike_events_) {
            const auto events = runtime.collect_spike_events();
            publish_spike_events(events, runtime.simulation_generation());
        }
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

TopologyVisualSnapshot SimulationController::topology_visual_snapshot(
    std::uint32_t maximum_neurons,
    std::uint32_t maximum_edges_per_neuron) const {
    return const_cast<SimulationController*>(this)->submit(
        [maximum_neurons, maximum_edges_per_neuron]
        (CudaSimulationRuntime& runtime) {
            return runtime.collect_topology_visual_snapshot(
                maximum_neurons, maximum_edges_per_neuron);
        }
    );
}

TopologyStatistics SimulationController::connectivity_statistics() const {
    return const_cast<SimulationController*>(this)->submit(
        [](CudaSimulationRuntime& runtime) {
            return analyze_csr_topology(runtime.collect_connectivity());
        }
    );
}

SelfObservationReport SimulationController::self_observation_snapshot(
    float weak_weight_threshold,
    std::size_t maximum_ranked_neurons) const {
    return const_cast<SimulationController*>(this)->submit(
        [weak_weight_threshold, maximum_ranked_neurons]
        (CudaSimulationRuntime& runtime) {
            const auto topology = runtime.collect_connectivity();
            const auto counts = runtime.collect_spike_count_snapshot();
            std::vector<std::uint64_t> activity;
            activity.reserve(counts.size());
            for (const auto count : counts) {
                activity.push_back(count);
            }
            return observe_connectome(
                topology, activity, weak_weight_threshold,
                maximum_ranked_neurons);
        }
    );
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
        return observed_completed_steps_.load(std::memory_order_acquire) >= target ||
               status_ == ControllerStatus::Faulted ||
               status_ == ControllerStatus::ShuttingDown;
    });

    return reached && observed_completed_steps_.load(std::memory_order_acquire) >= target;
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
        const auto previous = observed_completed_steps_.load(std::memory_order_relaxed);
        if (metrics.completed_steps > previous) {
            observed_completed_steps_.store(metrics.completed_steps, std::memory_order_release);
        }
    }
    condition_.notify_all();
}

void SimulationController::worker_main(
    ModelParameters defaults,
    int device,
    std::promise<void> initialized,
    bool collect_spike_events,
    bool enable_plasticity_state,
    StdpParameters stdp_parameters) {
    std::unique_ptr<CudaSimulationRuntime> runtime;

    try {
        runtime = std::make_unique<CudaSimulationRuntime>(
            std::move(defaults), device, collect_spike_events,
            enable_plasticity_state, stdp_parameters);
        const auto initial_metrics = runtime->collect_metrics();
        if (collect_spike_events) {
            const auto initial_events = runtime->collect_spike_events();
            publish_spike_events(initial_events, runtime->simulation_generation());
        }
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
    std::uint32_t event_collection_interval = collect_spike_events
        ? runtime->spike_event_collection_batch_interval(batch_size_)
        : 0U;
    std::uint32_t batches_since_event_collection = 0;
    std::uint32_t batches_since_progress_notification = 0;

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
                    observed_completed_steps_.store(completed, std::memory_order_release);
                }
                next_telemetry_step = completed + telemetry_interval_steps_;
                event_collection_interval = collect_spike_events
                    ? runtime->spike_event_collection_batch_interval(batch_size_)
                    : 0U;
                batches_since_event_collection = 0;
                batches_since_progress_notification = 0;
                condition_.notify_all();
            }
            continue;
        }

        if (!execute_batch) continue;

        try {
            runtime->run_steps(batch_size_);

            const auto generation = runtime->simulation_generation();
            if (generation != observed_generation) {
                observed_generation = generation;
                next_telemetry_step =
                    runtime->completed_steps() + telemetry_interval_steps_;
            }

            const auto completed = runtime->completed_steps();
            observed_completed_steps_.store(completed, std::memory_order_release);
            if (++batches_since_progress_notification >=
                kProgressNotificationBatches) {
                batches_since_progress_notification = 0;
                condition_.notify_all();
            }

            // Avoid forcing a GPU/CPU synchronization after every small
            // simulation batch. The interval is bounded from the worst-case
            // spike production rate and leaves half of the GPU ring unused.
            if (collect_spike_events) ++batches_since_event_collection;
            if (collect_spike_events &&
                batches_since_event_collection >= event_collection_interval) {
                const auto events = runtime->collect_spike_events();
                publish_spike_events(events, runtime->simulation_generation());
                batches_since_event_collection = 0;
            }

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
