#include "cuda/neuron_izhikevich.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

template <typename T>
class DeviceArray {
public:
    explicit DeviceArray(std::size_t count) : count_(count) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceArray() { if (data_) (void)cudaFree(data_); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    T* get() { return data_; }
    void upload(const std::vector<T>& values) {
        if (values.size() != count_) throw std::runtime_error("upload size mismatch");
        cuda_check(cudaMemcpy(data_, values.data(), count_ * sizeof(T),
                              cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
    std::vector<T> download() const {
        std::vector<T> values(count_);
        cuda_check(cudaMemcpy(values.data(), data_, count_ * sizeof(T),
                              cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        return values;
    }
private:
    T* data_ = nullptr;
    std::size_t count_;
};

bool near(float a, float b, float tolerance = 2.0e-6f) {
    return std::abs(a - b) <= tolerance;
}
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    using namespace spiking;
    try {
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        DeviceArray<std::uint32_t> outgoing_offsets(4), outgoing_targets(3);
        DeviceArray<std::uint32_t> incoming_offsets(4), incoming_sources(3);
        DeviceArray<std::uint32_t> incoming_edges(3), active_sources(3);
        DeviceArray<std::uint32_t> active_count(1);
        DeviceArray<float> weights(3), pre_traces(3), post_traces(3);

        outgoing_offsets.upload({0, 2, 2, 3});
        outgoing_targets.upload({1, 2, 0});
        incoming_offsets.upload({0, 1, 2, 3});
        incoming_sources.upload({2, 0, 0});
        incoming_edges.upload({2, 0, 1});
        active_sources.upload({0, 1, 2});
        active_count.upload({1});
        weights.upload({0.5f, 0.5f, -0.5f});
        pre_traces.upload({0.0f, 0.0f, 0.0f});
        post_traces.upload({0.0f, 0.0f, 0.0f});

        const float decay = std::exp(-0.1f);
        const StdpParametersGPU parameters{
            0.1f, 0.2f, decay, decay, 0.05f, 0.9f
        };
        auto step = [&](std::uint32_t neuron) {
            active_sources.upload({neuron, 1, 2});
            apply_stdp(active_sources.get(), active_count.get(),
                       outgoing_offsets.get(), outgoing_targets.get(),
                       weights.get(), incoming_offsets.get(),
                       incoming_sources.get(), incoming_edges.get(),
                       pre_traces.get(), post_traces.get(),
                       3, 2, 1.0f, parameters);
            cuda_check(cudaDeviceSynchronize(), "STDP kernel synchronization");
        };

        // A presynaptic event followed by its postsynaptic event potentiates.
        step(0);
        step(1);
        auto w = weights.download();
        check(near(w[0], 0.5f + 0.1f * decay),
              "GPU STDP did not potentiate a pre-before-post pair");
        check(near(w[2], -0.5f),
              "GPU STDP modified an inhibitory-source weight");

        // Reset state, then reverse the event order to test depression.
        weights.upload({0.5f, 0.5f, -0.5f});
        pre_traces.upload({0.0f, 0.0f, 0.0f});
        post_traces.upload({0.0f, 0.0f, 0.0f});
        step(1);
        step(0);
        w = weights.download();
        check(near(w[0], 0.5f - 0.2f * decay),
              "GPU STDP did not depress a post-before-pre pair");

        // Simultaneous pre/post activity has no zero-lag contribution.
        weights.upload({0.5f, 0.5f, -0.5f});
        pre_traces.upload({0.0f, 0.0f, 0.0f});
        post_traces.upload({0.0f, 0.0f, 0.0f});
        active_sources.upload({0, 1, 2});
        active_count.upload({2});
        apply_stdp(active_sources.get(), active_count.get(),
                   outgoing_offsets.get(), outgoing_targets.get(),
                   weights.get(), incoming_offsets.get(),
                   incoming_sources.get(), incoming_edges.get(),
                   pre_traces.get(), post_traces.get(),
                   3, 2, 1.0f, parameters);
        cuda_check(cudaDeviceSynchronize(), "simultaneous STDP synchronization");
        w = weights.download();
        check(near(w[0], 0.5f),
              "simultaneous GPU spikes caused a zero-lag weight change");

        // Weight bounds are enforced on GPU, and inhibitory weights remain signed
        // and untouched even when an inhibitory neuron emits a presynaptic event.
        weights.upload({0.899f, 0.5f, -0.5f});
        pre_traces.upload({10.0f, 0.0f, 0.0f});
        post_traces.upload({0.0f, 0.0f, 0.0f});
        step(1);
        w = weights.download();
        check(near(w[0], parameters.maximum_weight),
              "GPU STDP failed to clamp potentiation to its maximum");
        check(near(w[2], -0.5f),
              "GPU STDP changed an inhibitory-source weight");

        std::cout << "All GPU STDP kernel tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "GPU STDP test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
