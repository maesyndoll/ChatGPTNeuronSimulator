#include "cuda/neuron_izhikevich.cuh"
#include "plasticity/stdp.h"

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


        // Compare the CUDA implementation against the CPU reference over
        // a multi-step sequence, including repeated and simultaneous spikes.
        const std::vector<std::uint32_t> host_sources{0, 0, 2};
        const std::vector<std::uint32_t> host_targets{1, 2, 0};
        std::vector<float> host_weights{0.5f, 0.5f, -0.5f};
        StdpTraces host_traces{
            std::vector<float>(3, 0.0f),
            std::vector<float>(3, 0.0f)
        };

        const StdpParameters host_parameters{
            0.1f, 0.2f, 10.0f, 10.0f, 0.05f, 0.9f
        };

        weights.upload(host_weights);
        pre_traces.upload({0.0f, 0.0f, 0.0f});
        post_traces.upload({0.0f, 0.0f, 0.0f});

        const std::vector<std::vector<std::uint32_t>> event_sequence{
            {0}, {1}, {1, 2}, {0}, {2}, {0, 1}, {}, {1}
        };

        for (std::size_t step_index = 0;
             step_index < event_sequence.size(); ++step_index) {
            std::vector<std::uint8_t> host_events(3, 0);
            std::vector<std::uint32_t> gpu_events(3, 0);

            for (std::size_t i = 0;
                 i < event_sequence[step_index].size(); ++i) {
                const auto neuron = event_sequence[step_index][i];
                host_events[neuron] = 1;
                gpu_events[i] = neuron;
            }

            active_sources.upload(gpu_events);
            const auto active_count_value =
                static_cast<std::uint32_t>(
                    event_sequence[step_index].size());
            active_count.upload({active_count_value});

            apply_pair_stdp_step(
                host_weights, host_sources, host_targets,
                3, 2, host_events, host_traces, 1.0f,
                host_parameters);

            apply_stdp(
                active_sources.get(), active_count.get(),
                outgoing_offsets.get(), outgoing_targets.get(),
                weights.get(), incoming_offsets.get(),
                incoming_sources.get(), incoming_edges.get(),
                pre_traces.get(), post_traces.get(),
                3, 2, 1.0f, parameters);

            cuda_check(cudaDeviceSynchronize(),
                       "CPU/GPU STDP sequence synchronization");

            const auto gpu_weights = weights.download();
            const auto gpu_pre = pre_traces.download();
            const auto gpu_post = post_traces.download();

            if (step_index == 5) {
                std::cerr << "Step 5 trace comparison:\\n";
                for (std::size_t neuron = 0; neuron < 3; ++neuron) {
                    std::cerr << "neuron " << neuron
                              << " pre CPU/GPU="
                              << host_traces.presynaptic[neuron] << "/"
                              << gpu_pre[neuron]
                              << " post CPU/GPU="
                              << host_traces.postsynaptic[neuron] << "/"
                              << gpu_post[neuron] << '\n';
                }
            }

            for (std::size_t edge = 0; edge < host_weights.size(); ++edge) {
                if (!near(gpu_weights[edge], host_weights[edge], 5.0e-6f)) {
                    std::cerr << "STDP mismatch: step=" << step_index
                              << ", edge=" << edge
                              << ", CPU=" << host_weights[edge]
                              << ", GPU=" << gpu_weights[edge] << '\n';
                    throw std::runtime_error(
                        "CPU/GPU STDP weight mismatch at sequence step");
                }
            }
            for (std::size_t neuron = 0; neuron < 3; ++neuron) {
                check(near(gpu_pre[neuron],
                           host_traces.presynaptic[neuron], 5.0e-6f),
                      "CPU/GPU presynaptic trace mismatch");
                check(near(gpu_post[neuron],
                           host_traces.postsynaptic[neuron], 5.0e-6f),
                      "CPU/GPU postsynaptic trace mismatch");
            }
        }


        // Validate the CUDA timing window against the analytical
        // pair-based STDP response at several pre/post intervals.
        {
            const float a_plus = 0.01f;
            const float a_minus = 0.012f;
            const float tau_pre = 20.0f;
            const float tau_post = 25.0f;
            const float dt_ms = 1.0f;

            const StdpParametersGPU curve_parameters{
                a_plus,
                a_minus,
                std::exp(-dt_ms / tau_pre),
                std::exp(-dt_ms / tau_post),
                0.05f,
                0.9f
            };

            auto gpu_event = [&](const std::vector<std::uint32_t>& neurons) {
                std::vector<std::uint32_t> padded(3, 0);
                for (std::size_t i = 0; i < neurons.size(); ++i)
                    padded[i] = neurons[i];

                active_sources.upload(padded);
                active_count.upload({
                    static_cast<std::uint32_t>(neurons.size())
                });

                apply_stdp(
                    active_sources.get(), active_count.get(),
                    outgoing_offsets.get(), outgoing_targets.get(),
                    weights.get(), incoming_offsets.get(),
                    incoming_sources.get(), incoming_edges.get(),
                    pre_traces.get(), post_traces.get(),
                    3, 2, dt_ms, curve_parameters);

                cuda_check(cudaDeviceSynchronize(),
                           "CUDA timing-window synchronization");
            };

            auto measure_pair = [&](bool pre_first, int lag) {
                weights.upload({0.5f, 0.5f, -0.5f});
                pre_traces.upload({0.0f, 0.0f, 0.0f});
                post_traces.upload({0.0f, 0.0f, 0.0f});

                gpu_event({pre_first ? 0U : 1U});

                for (int elapsed = 1; elapsed < lag; ++elapsed)
                    gpu_event({});

                gpu_event({pre_first ? 1U : 0U});
                return weights.download()[0] - 0.5f;
            };

            for (const int lag : {1, 5, 10, 20, 40, 100}) {
                const float ltp = measure_pair(true, lag);
                const float ltd = measure_pair(false, lag);

                const float expected_ltp =
                    a_plus * std::exp(-static_cast<float>(lag) / tau_pre);
                const float expected_ltd =
                    -a_minus * std::exp(-static_cast<float>(lag) / tau_post);

                check(near(ltp, expected_ltp, 5.0e-6f),
                      "CUDA potentiation timing curve differs from analytical response");
                check(near(ltd, expected_ltd, 5.0e-6f),
                      "CUDA depression timing curve differs from analytical response");
            }
        }

        std::cout << "All GPU STDP kernel tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "GPU STDP test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
