#pragma once

#include "topology/csr_topology.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace spiking {

// Optional state arrays are stored alongside a module's immutable CSR graph.
// Empty arrays mean that the corresponding state is not available.
struct NeuralModuleState {
    std::vector<float> membrane_voltage;
    std::vector<float> recovery;
    std::vector<float> adaptation;
    std::vector<float> phase;
    std::vector<float> presynaptic_trace;
    std::vector<float> postsynaptic_trace;
    std::vector<float> synapse_phase_lag;
    std::vector<std::uint8_t> edge_plasticity_mask;

    void validate(const CSRTopology& topology) const;
};

struct NeuralModuleInfo {
    std::uint32_t schema_version = 1;
    std::string id;
    std::string name;
    std::string description;
    std::filesystem::path package_directory;
    std::filesystem::path manifest_path;
    std::filesystem::path topology_path;
    std::filesystem::path state_path;
    std::uint32_t excitatory_count = 0;
    std::uint32_t inhibitory_count = 0;
    std::uint64_t synapse_count = 0;
};

struct NeuralModuleData {
    NeuralModuleInfo info;
    CSRTopology topology;
    NeuralModuleState state;
};

// Persistent, data-only neural modules. Each module is an independently
// addressable CSR population bundle with optional resumable neural state.
// No executable code is loaded by this store.
class NeuralMemoryStore {
public:
    explicit NeuralMemoryStore(std::filesystem::path root_directory);

    const std::filesystem::path& root_directory() const noexcept {
        return root_directory_;
    }

    void save(const std::string& id,
              const std::string& name,
              const CSRTopology& topology,
              const NeuralModuleState& state = {},
              const std::string& description = {});

    NeuralModuleData load(const std::string& id) const;
    NeuralModuleInfo inspect(const std::string& id) const;
    std::vector<NeuralModuleInfo> list() const;
    bool contains(const std::string& id) const;
    void erase(const std::string& id) const;

private:
    static void validate_id(const std::string& id);
    std::filesystem::path module_directory(const std::string& id) const;

    std::filesystem::path root_directory_;
};

} // namespace spiking
