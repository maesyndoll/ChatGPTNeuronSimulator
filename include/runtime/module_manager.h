#pragma once

#include "runtime/module_api_v1.h"

#include <cstdint>
#include <mutex>
#include <string>

namespace spiking {

class RuntimeModuleManager {
public:
    explicit RuntimeModuleManager(NeuromancerHostApiV1 host);
    ~RuntimeModuleManager();

    RuntimeModuleManager(const RuntimeModuleManager&) = delete;
    RuntimeModuleManager& operator=(const RuntimeModuleManager&) = delete;

    void load(const std::string& shared_library_path);
    void configure(const std::string& key, const std::string& value);
    void dispatch_spikes(const NeuromancerSpikeEventV1* events,
                         std::uint32_t event_count);
    void unload();
    bool loaded() const;
    std::string module_name() const;

private:
    mutable std::mutex mutex_;
    NeuromancerHostApiV1 host_{};
    void* library_ = nullptr;
    void* instance_ = nullptr;
    const NeuromancerModuleV1* module_ = nullptr;
};

} // namespace spiking
