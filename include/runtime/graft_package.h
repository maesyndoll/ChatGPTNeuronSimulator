#pragma once

#include "topology/csr_topology.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace spiking {

struct GraftPackageInfo {
    std::uint32_t schema_version = 1;
    std::string id;
    std::string name;
    std::string description;
    std::filesystem::path package_directory;
    std::filesystem::path manifest_path;
    std::filesystem::path topology_path;
    std::filesystem::path module_binary_path;
    std::uint32_t excitatory_count = 0;
    std::uint32_t inhibitory_count = 0;
    std::uint64_t synapse_count = 0;
    std::vector<std::string> input_adapters;
    std::vector<std::string> output_adapters;
    bool has_module_binary = false;
};

// A graft package is an immutable on-disk bundle. Inspection validates its
// manifest, path containment, optional shared object, and exact SPKCSR graph.
// It never loads executable code or mutates a running simulation.
class GraftPackageInspector {
public:
    static GraftPackageInfo inspect(const std::filesystem::path& package);
};

} // namespace spiking
