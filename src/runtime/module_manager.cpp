#include "runtime/module_manager.h"

#include <dlfcn.h>
#include <stdexcept>

namespace spiking {

RuntimeModuleManager::RuntimeModuleManager(NeuromancerHostApiV1 host)
    : host_(host) {
    if (host_.abi_version != NEUROMANCER_MODULE_ABI_V1 ||
        host_.struct_size < sizeof(NeuromancerHostApiV1) ||
        host_.set_edge_plasticity == nullptr) {
        throw std::invalid_argument("Invalid Neuromancer host API V1");
    }
}

RuntimeModuleManager::~RuntimeModuleManager() {
    try { unload(); } catch (...) {}
}

void RuntimeModuleManager::load(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (library_ != nullptr) throw std::logic_error("A runtime module is already loaded");
    void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error(std::string("dlopen failed: ") +
                                 (error ? error : "unknown error"));
    }
    dlerror();
    auto query = reinterpret_cast<NeuromancerModuleQueryV1>(
        dlsym(library, "neuromancer_module_query_v1"));
    const char* symbol_error = dlerror();
    if (symbol_error != nullptr || query == nullptr) {
        dlclose(library);
        throw std::runtime_error("Module is missing neuromancer_module_query_v1");
    }
    const NeuromancerModuleV1* module = query();
    if (module == nullptr || module->abi_version != NEUROMANCER_MODULE_ABI_V1 ||
        module->struct_size < sizeof(NeuromancerModuleV1) ||
        module->name == nullptr || module->on_load == nullptr ||
        module->configure == nullptr || module->on_spikes == nullptr ||
        module->on_unload == nullptr) {
        dlclose(library);
        throw std::runtime_error("Module ABI or lifecycle table is invalid");
    }
    void* instance = nullptr;
    if (module->on_load(&host_, &instance) != 0) {
        dlclose(library);
        throw std::runtime_error("Module on_load rejected attachment");
    }
    library_ = library;
    instance_ = instance;
    module_ = module;
}

void RuntimeModuleManager::configure(const std::string& key,
                                     const std::string& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (module_ == nullptr) throw std::logic_error("No runtime module is loaded");
    if (module_->configure(instance_, key.c_str(), value.c_str()) != 0)
        throw std::runtime_error("Runtime module rejected configuration");
}

void RuntimeModuleManager::dispatch_spikes(
    const NeuromancerSpikeEventV1* events, std::uint32_t event_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (module_ == nullptr) return;
    if (event_count != 0 && events == nullptr)
        throw std::invalid_argument("Spike event pointer is null");
    module_->on_spikes(instance_, events, event_count);
}

void RuntimeModuleManager::unload() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (module_ == nullptr) return;
    // Caller must invoke unload at a simulation synchronization boundary.
    module_->on_unload(instance_);
    module_ = nullptr;
    instance_ = nullptr;
    if (library_ != nullptr) {
        dlclose(library_);
        library_ = nullptr;
    }
}

bool RuntimeModuleManager::loaded() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return module_ != nullptr;
}

std::string RuntimeModuleManager::module_name() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return module_ != nullptr ? module_->name : std::string{};
}

} // namespace spiking
