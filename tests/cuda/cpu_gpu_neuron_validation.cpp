#include "analysis/izhikevich_analysis.h"
#include "cuda/neuron_izhikevich.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

template <typename T>
class DeviceValue {
public:
    DeviceValue() {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer_), sizeof(T)),
                   "cudaMalloc");
    }

    ~DeviceValue() {
        if (pointer_) cudaFree(pointer_);
    }

    DeviceValue(const DeviceValue&) = delete;
    DeviceValue& operator=(const DeviceValue&) = delete;

    T* get() { return pointer_; }

    void upload(const T& value) {
        cuda_check(cudaMemcpy(pointer_, &value, sizeof(T),
                              cudaMemcpyHostToDevice), "upload");
    }

    T download() const {
        T value{};
        cuda_check(cudaMemcpy(&value, pointer_, sizeof(T),
                              cudaMemcpyDeviceToHost), "download");
        return value;
    }

private:
    T* pointer_ = nullptr;
};

struct TestResult {
    double max_voltage_error = 0.0;
    double max_recovery_error = 0.0;
    std::uint32_t cpu_spikes = 0;
    std::uint32_t gpu_spikes = 0;
};

TestResult compare_trajectory(
    const spiking::analysis::NeuronParameters& neuron,
    double initial_voltage,
    double initial_recovery,
    double current,
    double dt,
    std::uint32_t steps,
    double threshold,
    const std::string& label) {

    using namespace spiking;

    const auto cpu = analysis::simulate_trajectory(
        neuron, initial_voltage, initial_recovery,
        current, dt, steps, threshold);

    DeviceValue<float> d_a, d_b, d_c, d_d, d_current;
    DeviceValue<float> d_voltage, d_recovery, d_synaptic_current;
    DeviceValue<std::uint32_t> d_spike_count, d_active_source;
    DeviceValue<std::uint32_t> d_active_count;
    DeviceValue<std::uint32_t> d_completed_blocks;
    DeviceValue<unsigned long long> d_completed_steps;
    DeviceValue<std::uint8_t> d_spike_flag;

    d_a.upload(static_cast<float>(neuron.a));
    d_b.upload(static_cast<float>(neuron.b));
    d_c.upload(static_cast<float>(neuron.c));
    d_d.upload(static_cast<float>(neuron.d));
    d_current.upload(static_cast<float>(current));

    d_voltage.upload(static_cast<float>(initial_voltage));
    d_recovery.upload(static_cast<float>(initial_recovery));
    d_synaptic_current.upload(0.0f);
    d_spike_count.upload(0U);
    d_active_count.upload(0U);
    d_completed_blocks.upload(0U);
    d_completed_steps.upload(0ULL);
    d_spike_flag.upload(0U);
    d_active_source.upload(0U);

    NeuronParametersGPU parameters;
    parameters.a = d_a.get();
    parameters.b = d_b.get();
    parameters.c = d_c.get();
    parameters.d = d_d.get();
    parameters.current = d_current.get();

    NeuronStateGPU state;
    state.voltage = d_voltage.get();
    state.recovery = d_recovery.get();
    state.spike_counts = d_spike_count.get();
    state.spike_flags = d_spike_flag.get();
    state.active_spike_sources = d_active_source.get();
    state.active_spike_count = d_active_count.get();
    state.completed_steps = d_completed_steps.get();
    state.completed_step_blocks = d_completed_blocks.get();

    TestResult result;

    if (cpu.samples.size() != static_cast<std::size_t>(steps) + 1) {
        throw std::runtime_error(label + ": unexpected CPU sample count");
    }

    for (std::uint32_t step = 0; step < steps; ++step) {
        // The isolated-neuron test has no incoming synaptic events.
        cuda_check(cudaMemset(d_active_count.get(), 0, sizeof(std::uint32_t)),
                   "reset active spike count");

        simulate_neurons(
            parameters, state, 1U, static_cast<float>(dt),
            static_cast<float>(threshold),
            d_synaptic_current.get(), 1U);

        const double gpu_voltage = d_voltage.download();
        const double gpu_recovery = d_recovery.download();
        const bool gpu_spiked = d_spike_flag.download() != 0;

        const auto& cpu_sample = cpu.samples[step + 1];

        const double voltage_error =
            std::abs(gpu_voltage - cpu_sample.voltage);
        const double recovery_error =
            std::abs(gpu_recovery - cpu_sample.recovery);

        result.max_voltage_error =
            std::max(result.max_voltage_error, voltage_error);
        result.max_recovery_error =
            std::max(result.max_recovery_error, recovery_error);

        if (cpu_sample.spiked) ++result.cpu_spikes;
        if (gpu_spiked) ++result.gpu_spikes;
    }

    const double voltage_tolerance = 5.0e-3;
    const double recovery_tolerance = 5.0e-3;

    std::cout << label
              << ": max voltage error=" << result.max_voltage_error
              << ", max recovery error=" << result.max_recovery_error
              << ", CPU spikes=" << result.cpu_spikes
              << ", GPU spikes=" << result.gpu_spikes << '\n';

    if (result.max_voltage_error > voltage_tolerance ||
        result.max_recovery_error > recovery_tolerance ||
        result.cpu_spikes != result.gpu_spikes) {
        throw std::runtime_error(label + ": CPU/GPU trajectory mismatch");
    }

    if (d_completed_steps.download() != steps) {
        throw std::runtime_error(label + ": GPU step counter mismatch");
    }

    return result;
}

} // namespace

int main() {
    try {
        int device_count = 0;
        cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) {
            throw std::runtime_error("No CUDA device available");
        }

        spiking::analysis::NeuronParameters regular;
        regular.a = 0.02;
        regular.b = 0.20;
        regular.c = -65.0;
        regular.d = 8.0;

        spiking::analysis::NeuronParameters fast;
        fast.a = 0.10;
        fast.b = 0.20;
        fast.c = -65.0;
        fast.d = 2.0;

        constexpr double initial_voltage = -65.0;
        constexpr double current = 10.0;
        constexpr double dt = 0.01;
        constexpr std::uint32_t steps = 1000;
        constexpr double threshold = 30.0;

        compare_trajectory(
            regular, initial_voltage, regular.b * initial_voltage,
            current, dt, steps, threshold, "Regular-spiking");

        compare_trajectory(
            fast, initial_voltage, fast.b * initial_voltage,
            current, dt, steps, threshold, "Fast-spiking");

        // Force a spike in one step and verify the post-spike reset path.
        const auto reset_result = compare_trajectory(
            regular, 29.0, 0.0, 1000.0, 0.01, 1, 30.0,
            "Spike/reset");

        if (reset_result.cpu_spikes != 1 ||
            reset_result.gpu_spikes != 1) {
            throw std::runtime_error("Expected exactly one reset-test spike");
        }

        std::cout << "CPU/GPU neuron validation passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "CPU/GPU neuron validation failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
