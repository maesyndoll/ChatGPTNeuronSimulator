#include "runtime/graft_package.h"

#include "topology/topology_io.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

namespace spiking {
namespace {

using Json = nlohmann::json;

std::string required_string(const Json& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_string() ||
        it->get_ref<const std::string&>().empty()) {
        throw std::invalid_argument(std::string("Graft manifest requires non-empty string: ") + key);
    }
    return it->get<std::string>();
}

std::uint32_t required_u32(const Json& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end() || (!it->is_number_unsigned() &&
        !(it->is_number_integer() && it->get<std::int64_t>() >= 0))) {
        throw std::invalid_argument(std::string("Graft manifest requires non-negative integer: ") + key);
    }
    const auto value = it->get<std::uint64_t>();
    if (value > UINT32_MAX)
        throw std::invalid_argument(std::string("Graft manifest integer exceeds uint32: ") + key);
    return static_cast<std::uint32_t>(value);
}

std::filesystem::path contained_file(const std::filesystem::path& root,
                                     const std::string& relative,
                                     const char* field) {
    const std::filesystem::path rel(relative);
    if (rel.empty() || rel.is_absolute())
        throw std::invalid_argument(std::string(field) + " must be a relative package path");
    for (const auto& part : rel) {
        if (part == "..")
            throw std::invalid_argument(std::string(field) + " must not escape the package directory");
    }

    const auto candidate = std::filesystem::canonical(root / rel);
    const auto canonical_root = std::filesystem::canonical(root);
    auto root_it = canonical_root.begin();
    auto candidate_it = candidate.begin();
    for (; root_it != canonical_root.end() && candidate_it != candidate.end();
         ++root_it, ++candidate_it) {
        if (*root_it != *candidate_it)
            throw std::invalid_argument(std::string(field) + " resolves outside the package directory");
    }
    if (root_it != canonical_root.end() || !std::filesystem::is_regular_file(candidate))
        throw std::invalid_argument(std::string(field) + " must resolve to a regular file inside the package");
    return candidate;
}

std::vector<std::string> optional_names(const Json& manifest, const char* key) {
    std::vector<std::string> names;
    const auto it = manifest.find(key);
    if (it == manifest.end()) return names;
    if (!it->is_array())
        throw std::invalid_argument(std::string(key) + " must be an array of adapter names");
    for (const auto& value : *it) {
        if (!value.is_string() || value.get_ref<const std::string&>().empty())
            throw std::invalid_argument(std::string(key) + " contains an invalid adapter name");
        names.push_back(value.get<std::string>());
    }
    return names;
}

} // namespace

GraftPackageInfo GraftPackageInspector::inspect(
    const std::filesystem::path& package) {
    std::error_code error;
    const auto input = std::filesystem::absolute(package, error);
    if (error) throw std::runtime_error("Could not resolve graft package path: " + error.message());

    const auto manifest = std::filesystem::is_directory(input)
        ? input / "neuromancer.graft.json" : input;
    if (!std::filesystem::is_regular_file(manifest))
        throw std::invalid_argument("Graft package must contain neuromancer.graft.json");

    const auto manifest_path = std::filesystem::canonical(manifest);
    const auto root = manifest_path.parent_path();

    Json document;
    try {
        std::ifstream stream(manifest_path);
        if (!stream) throw std::runtime_error("Could not open graft manifest");
        stream >> document;
    } catch (const Json::exception& exception) {
        throw std::invalid_argument(std::string("Invalid graft manifest JSON: ") + exception.what());
    }

    if (!document.is_object())
        throw std::invalid_argument("Graft manifest root must be a JSON object");

    GraftPackageInfo info;
    info.schema_version = required_u32(document, "schema_version");
    if (info.schema_version != 1)
        throw std::invalid_argument("Unsupported graft manifest schema_version");
    info.id = required_string(document, "id");
    info.name = required_string(document, "name");
    const auto description = document.find("description");
    if (description != document.end()) {
        if (!description->is_string())
            throw std::invalid_argument("description must be a string");
        info.description = description->get<std::string>();
    }
    info.package_directory = root;
    info.manifest_path = manifest_path;
    info.excitatory_count = required_u32(document, "excitatory_count");
    info.inhibitory_count = required_u32(document, "inhibitory_count");
    if (static_cast<std::uint64_t>(info.excitatory_count) +
        info.inhibitory_count == 0)
        throw std::invalid_argument("Graft must declare at least one neuron");

    info.topology_path = contained_file(
        root, required_string(document, "topology_file"), "topology_file");
    const auto topology = load_csr_topology(info.topology_path.string());
    if (topology.excitatory_count != info.excitatory_count ||
        topology.inhibitory_count != info.inhibitory_count) {
        throw std::invalid_argument("Graft manifest population dimensions do not match its CSR topology");
    }
    info.synapse_count = topology.synapse_count();

    const auto module = document.find("module_binary");
    if (module != document.end() && !module->is_null()) {
        if (!module->is_string() || module->get_ref<const std::string&>().empty())
            throw std::invalid_argument("module_binary must be a relative path string or null");
        info.module_binary_path = contained_file(
            root, module->get<std::string>(), "module_binary");
        if (info.module_binary_path.extension() != ".so")
            throw std::invalid_argument("module_binary must reference a Linux .so file");
        info.has_module_binary = true;
    }

    info.input_adapters = optional_names(document, "input_adapters");
    info.output_adapters = optional_names(document, "output_adapters");
    return info;
}

} // namespace spiking
