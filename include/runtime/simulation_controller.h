#pragma once

#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace spiking {

enum class ControllerStatus {
    Ready,
    Running,
    Paused,
    Faulted,
    ShuttingDown
};

struct TelemetrySnapshot {
    SimulationMetrics metrics;
    std::uint64_t sequence = 0;
};

struct SequencedSpikeEvent {
    std::uint64_t sequence = 0;
    std::uint64_t step = 0;
    std::uint32_t neuron_id = 0;
};

struct SpikeEventSnapshot {
    std::vector<SequencedSpikeEvent> events;
    std::uint64_t first_sequence = 0;
    std::uint64_t total_events = 0;
    std::uint64_t dropped_events = 0;
    std::uint64_t simulation_generation = 0;
};

struct ControllerConfigurationSnapshot {
    ModelParameters requested_parameters;
    ModelParameters active_parameters;
    bool has_pending_reinitialization = false;
    std::uint64_t parameter_version = 0;
    std::uint64_t simulation_generation = 0;
};

// The worker thread exclusively owns CudaSimulationRuntime.
// Other threads interact only through commands and cached snapshots.
class SimulationController {
public:
    explicit SimulationController(
        ModelParameters defaults = ModelParameters{},
        int device = 0,
        std::uint32_t batch_size = 100,
        std::uint64_t telemetry_interval_steps = 1000,
        bool collect_spike_events = true,
        bool enable_plasticity_state = false,
        StdpParameters stdp_parameters = {});

    ~SimulationController();

    SimulationController(const SimulationController&) = delete;
    SimulationController& operator=(const SimulationController&) = delete;
    SimulationController(SimulationController&&) = delete;
    SimulationController& operator=(SimulationController&&) = delete;

    void start();
    void pause();
    void step(std::uint32_t steps);

    // Runtime operations are serialized through the owning worker thread.
    void schedule_stimulus(std::uint32_t neuron_id,
                           std::uint64_t start_step,
                           std::uint32_t duration_steps,
                           float amplitude);
    void clear_stimuli();
    CSRTopology connectivity_snapshot() const;

    ParameterUpdateResult apply_parameters(const ModelParameters& candidate);
    ParameterUpdateResult reset_parameters_to_defaults();
    void reinitialize_simulation();

    SimulationMetrics telemetry() const;
    TelemetrySnapshot telemetry_snapshot() const;
    SpikeEventSnapshot spike_events_snapshot(
        std::uint64_t after_sequence = 0,
        std::size_t maximum_events = 4096) const;
    ControllerConfigurationSnapshot configuration_snapshot() const;
    // Analyze the exact CSR graph currently resident in the simulation runtime.
    TopologyStatistics connectivity_statistics() const;

    ControllerStatus status() const;
    bool is_running() const;

    // Waits for completed simulation progress without forcing a CUDA
    // synchronization or telemetry copy. Returns false on timeout or fault.
    bool wait_until_steps(
        std::uint64_t target,
        std::chrono::milliseconds timeout) const;

    std::exception_ptr last_execution_error() const;

private:
    template <typename Function>
    auto submit(Function&& function)
        -> std::invoke_result_t<std::decay_t<Function>,
                                CudaSimulationRuntime&> {
        using Result = std::invoke_result_t<std::decay_t<Function>,
                                            CudaSimulationRuntime&>;

        auto promise = std::make_shared<std::promise<Result>>();
        auto future = promise->get_future();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (status_ == ControllerStatus::ShuttingDown) {
                throw std::logic_error(
                    "Simulation controller is shutting down");
            }

            commands_.emplace_back(
                [promise, fn = std::forward<Function>(function)]
                (CudaSimulationRuntime& runtime) mutable {
                    try {
                        if constexpr (std::is_void_v<Result>) {
                            fn(runtime);
                            promise->set_value();
                        } else {
                            promise->set_value(fn(runtime));
                        }
                    } catch (...) {
                        promise->set_exception(std::current_exception());
                    }
                });
        }

        condition_.notify_one();

        if constexpr (std::is_void_v<Result>) {
            future.get();
        } else {
            return future.get();
        }
    }

    void worker_main(
        ModelParameters defaults,
        int device,
        std::promise<void> initialized,
        bool collect_spike_events,
        bool enable_plasticity_state,
        StdpParameters stdp_parameters);

    void publish_metrics(const SimulationMetrics& metrics);
    void publish_spike_events(
        const SpikeEventBatch& batch, std::uint64_t generation);

    const std::uint32_t batch_size_;
    const std::uint64_t telemetry_interval_steps_;
    const bool collect_spike_events_;

    mutable std::mutex mutex_;
    mutable std::condition_variable condition_;
    std::deque<std::function<void(CudaSimulationRuntime&)>> commands_;
    std::thread worker_;

    ControllerStatus status_ = ControllerStatus::Ready;
    std::uint64_t observed_completed_steps_ = 0;
    TelemetrySnapshot telemetry_;
    std::deque<SequencedSpikeEvent> spike_events_;
    std::uint64_t spike_event_total_events_ = 0;
    std::uint64_t spike_event_dropped_events_ = 0;
    std::uint64_t spike_event_generation_ = 0;
    static constexpr std::size_t kMaximumRetainedSpikeEvents = 1048576;
    std::exception_ptr last_execution_error_;
};

} // namespace spiking
