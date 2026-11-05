#pragma once

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace spiking {

inline void cuda_check(cudaError_t status, const char* expression,
                       const char* file, int line) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << "CUDA call failed: " << expression << " ("
            << cudaGetErrorString(status) << ") at " << file << ':' << line;
    throw std::runtime_error(message.str());
}

} // namespace spiking

#define SPIKING_CUDA_CHECK(expression) \
    ::spiking::cuda_check((expression), #expression, __FILE__, __LINE__)
