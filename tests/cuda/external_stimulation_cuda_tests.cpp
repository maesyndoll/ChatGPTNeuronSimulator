#include "cuda/neuron_izhikevich.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check_cuda(cudaError_t status, const char* where) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " +
                                 cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename T>
struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(std::size_t n) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)),
                   "cudaMalloc");
    }
    ~Buffer() { if (ptr) (void)cudaFree(ptr); }
};
bool near(float a, float b) { return std::abs(a - b) < 1.0e-6f; }
}

int main() {
    using namespace spiking;
    try {
        check_cuda(cudaSetDevice(0), "cudaSetDevice");

        Buffer<ExternalStimulusGPU> events(kMaxExternalStimuli);
        Buffer<std::uint32_t> event_count(1);
        Buffer<unsigned long long> completed_steps(1);
        Buffer<float> current(3);

        const std::vector<ExternalStimulusGPU> schedule{
            {1, 2, 5, 3.0f},   // neuron 1: steps 2, 3, 4
            {1, 3, 4, 2.0f},   // overlaps: neuron 1 receives 5 at step 3
            {2, 3, 5, -1.0f}   // neuron 2: steps 3, 4
        };

        check_cuda(cudaMemcpy(events.ptr, schedule.data(),
                              schedule.size() * sizeof(ExternalStimulusGPU),
                              cudaMemcpyHostToDevice), "upload events");
        const std::uint32_t count =
            static_cast<std::uint32_t>(schedule.size());
        check_cuda(cudaMemcpy(event_count.ptr, &count, sizeof(count),
                              cudaMemcpyHostToDevice), "upload count");
        check_cuda(cudaMemset(current.ptr, 0, 3 * sizeof(float)),
                   "clear current");

        const float expected[5][3] = {
            {0.0f, 0.0f, 0.0f},
            {0.0f, 3.0f, 0.0f},
            {0.0f, 5.0f, -1.0f},
            {0.0f, 3.0f, -1.0f},
            {0.0f, 0.0f, 0.0f}
        };

        for (unsigned long long step = 1; step <= 5; ++step) {
            const unsigned long long completed = step - 1;
            check_cuda(cudaMemcpy(completed_steps.ptr, &completed,
                                  sizeof(completed),
                                  cudaMemcpyHostToDevice), "set step");
            apply_external_stimuli(events.ptr, event_count.ptr,
                                   completed_steps.ptr, current.ptr);
            check_cuda(cudaDeviceSynchronize(), "event kernel sync");

            float actual[3]{};
            check_cuda(cudaMemcpy(actual, current.ptr, sizeof(actual),
                                  cudaMemcpyDeviceToHost), "read current");
            for (int neuron = 0; neuron < 3; ++neuron)
                require(near(actual[neuron], expected[step - 1][neuron]),
                        "External stimulus timing or additive overlap is incorrect");
        }

        std::cout << "External stimulation CUDA tests passed.\\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "External stimulation CUDA test failed: "
                  << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
