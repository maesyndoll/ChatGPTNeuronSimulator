#pragma once

#include "cuda/cuda_check.cuh"

#include <cstddef>
#include <utility>

namespace spiking {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        reset();
        if (count == 0) return;
        T* allocation = nullptr;
        SPIKING_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&allocation),
                                      count * sizeof(T)));
        data_ = allocation;
        size_ = count;
    }

    void reset() noexcept {
        if (data_ != nullptr) {
            // Destructors must not throw. Runtime operations check their own
            // synchronization and launch errors before resources are released.
            (void)cudaFree(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }

    void swap(DeviceBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace spiking
