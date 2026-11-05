#pragma once

#include "cuda/cuda_check.cuh"

namespace spiking {

class CudaStream {
public:
    CudaStream() { SPIKING_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~CudaStream() { if (stream_ != nullptr) (void)cudaStreamDestroy(stream_); }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const { SPIKING_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    cudaStream_t stream_ = nullptr;
};

} // namespace spiking
