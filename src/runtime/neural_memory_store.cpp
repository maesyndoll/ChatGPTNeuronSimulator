#include "runtime/neural_memory_store.h"

#include "topology/topology_io.h"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace spiking {
namespace {

using Json = nlohmann::json;
constexpr std::array<char, 8> kStateMagic = {
    'N', 'S', 'T', 'A', 'T', 'E', '0', '1'
};
constexpr std::uint32_t kStateVersion = 1;

void write_bytes(std::ofstream& out, const char* data, std::size_t size) {
    out.write(data, static_cast<std::streamsize>(size));
    if (!out) throw std::runtime_error("Failed writing neural module state");
}

void read_bytes(std::ifstream& in, char* data, std::size_t size) {
    in.read(data, static_cast<std::streamsize>(size));
    if (!in) throw std::runtime_error("Truncated or unreadable neural module state");
}

void write_u32(std::ofstream& out, std::uint32_t value) {
    char bytes[4];
    for (unsigned i = 0; i < 4; ++i)
        bytes[i] = static_cast<char>((value >> (8U * i)) & 0xffU);
    write_bytes(out, bytes, sizeof(bytes));
}

std::uint32_t read_u32(std::ifstream& in) {
    unsigned char bytes[4];
    read_bytes(in, reinterpret_cast<char*>(bytes), sizeof(bytes));
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

void write_u64(std::ofstream& out, std::uint64_t value) {
    char bytes[8];
    for (unsigned i = 0; i < 8; ++i)
        bytes[i] = static_cast<char>((value >> (8U * i)) & 0xffU);
    write_bytes(out, bytes, sizeof(bytes));
}

std::uint64_t read_u64(std::ifstream& in) {
    unsigned char bytes[8];
    read_bytes(in, reinterpret_cast<char*>(bytes), sizeof(bytes));
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i)
        value |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
    return value;
}

void write_float(std::ofstream& out, float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t),
                  "Neural state format requires 32-bit floats");
    static_assert(std::numeric_limits<float>::is_iec559,
                  "Neural state format requires IEEE-754 floats");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(out, bits);
}

float read_float(std::ifstream& in) {
    const auto bits = read_u32(in);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void write_float_vector(std::ofstream& out, const std::vector<float>& values) {
    write_u64(out, values.size());
    for (const auto value : values) write_float(out, value);
}

std::vector<float> read_float_vector(std::ifstream& in, std::size_t expected,
                                     const char* field) {
    const auto count = read_u64(in);
    if (count != 0 && count != expected)
        throw std::invalid_argument(std::string("Invalid neural state array length: ") + field);
    std::vector<float> values(static_cast<std::size_t>(count));
    for (auto& value : values) value = read_float(in);
    return values;
}

void write_byte_vector(std::ofstream& out,
                       const std::vector<std::uint8_t>& values) {
    write_u64(out, values.size());
    if (!values.empty())
        write_bytes(out, reinterpret_cast<const char*>(values.data()), values.size());
}

std::vector<std::uint8_t> read_byte_vector(std::ifstream& in,
                                           std::size_t expected,
                                           const char* field) {
    const auto count = read_u64(in);
    if (count != 0 && count != expected)
        throw std::invalid_argument(std::string("Invalid neural state array length: ") + field);
    std::vector<std::uint8_t> values(static_cast<std::size_t>(count));
    if (!values.empty())
        read_bytes(in, reinterpret_cast<char*>(values.data()), values.size());
    return values;
}

void write_state_file(const std::filesystem::path& path,
                      const CSRTopology& topology,
                      const NeuralModuleState& state) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Could not create neural module state file");
    write_bytes(out, kStateMagic.data(), kStateMagic.size());
    write_u32(out, kStateVersion);
    write_u32(out, topology.neuron_count());
    write_u64(out, topology.synapse_count());
    write_float_vector(out, state.membrane_voltage);
    write_float_vector(out, state.recovery);
    write_float_vector(out, state.adaptation);
    write_float_vector(out, state.phase);
    write_float_vector(out, state.presynaptic_trace);
    write_float_vector(out, state.postsynaptic_trace);
    write_float_vector(out, state.synapse_phase_lag);
    write_byte_vector(out, state.edge_plasticity_mask);
    out.flush();
    if (!out) throw std::runtime_error("Failed flushing neural module state file");
}

NeuralModuleState read_state_file(const std::filesystem::path& path,
                                  const CSRTopology& topology) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open neural module state file");
    std::array<char, 8> magic{};
    read_bytes(in, magic.data(), magic.size());
    if (magic != kStateMagic)
        throw std::invalid_argument("Invalid neural module state signature");
    if (read_u32(in) != kStateVersion)
        throw std::invalid_argument("Unsupported neural module state version");
    if (read_u32(in) != topology.neuron_count() ||
        read_u64(in) != topology.synapse_count())
        throw std::invalid_argument("Neural module state dimensions do not match CSR topology");

    NeuralModuleState state;
    const auto neurons = static_cast<std::size_t>(topology.neuron_count());
    const auto edges = topology.synapse_count();
    state.membrane_voltage = read_float_vector(in, neurons, "membrane_voltage");
    state.recovery = read_float_vector(in, neurons, "recovery");
    state.adaptation = read_float_vector(in, neurons, "adaptation");
    state.phase = read_float_vector(in, neurons, "phase");
    state.presynaptic_trace = read_float_vector(in, neurons, "presynaptic_trace");
    state.postsynaptic_trace = read_float_vector(in, neurons, "postsynaptic_trace");
    state.synapse_phase_lag = read_float_vector(in, edges, "synapse_phase_lag");
    state.edge_plasticity_mask = read_byte_vector(in, edges, "edge_plasticity_mask");

    char extra = 0;
    if (in.read(&extra, 1))
        throw std::invalid_argument("Neural module state contains trailing data");
    if (!in.eof())
        throw std::runtime_error("Error checking neural module state length");
    state.validate(topology);
    return state;
}

std::string required_string(const Json& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end() || !it->is_string() ||
        it->get_ref<const std::string&>().empty())
        throw std::invalid_argument(std::string("Neural module manifest requires: ") + key);
    return it->get<std::string>();
}

std::uint32_t required_u32(const Json& object, const char* key) {
    const auto it = object.find(key);
    if (it == object.end() || (!it->is_number_unsigned() &&
        !(it->is_number_integer() && it->get<std::int64_t>() >= 0)))
        throw std::invalid_argument(std::string("Neural module manifest requires integer: ") + key);
    const auto value = it->get<std::uint64_t>();
    if (value > UINT32_MAX)
        throw std::invalid_argument(std::string("Neural module manifest integer overflow: ") + key);
    return static_cast<std::uint32_t>(value);
}

std::filesystem::path contained_file(const std::filesystem::path& root,
                                     const std::string& relative,
                                     const char* field) {
    const std::filesystem::path rel(relative);
    if (rel.empty() || rel.is_absolute())
        throw std::invalid_argument(std::string(field) + " must be a relative path");
    for (const auto& part : rel)
        if (part == "..")
            throw std::invalid_argument(std::string(field) + " must not contain parent traversal");

    const auto canonical_root = std::filesystem::canonical(root);
    const auto candidate = std::filesystem::canonical(root / rel);
    auto root_it = canonical_root.begin();
    auto candidate_it = candidate.begin();
    for (; root_it != canonical_root.end() && candidate_it != candidate.end();
         ++root_it, ++candidate_it) {
        if (*root_it != *candidate_it)
            throw std::invalid_argument(std::string(field) + " resolves outside its module directory");
    }
    if (root_it != canonical_root.end() || !std::filesystem::is_regular_file(candidate))
        throw std::invalid_argument(std::string(field) + " must be a regular file inside its module directory");
    return candidate;
}

void write_manifest(const std::filesystem::path& path,
                    const std::string& id, const std::string& name,
                    const std::string& description,
                    const CSRTopology& topology) {
    Json manifest = {
        {"schema_version", 1},
        {"id", id},
        {"name", name},
        {"description", description},
        {"excitatory_count", topology.excitatory_count},
        {"inhibitory_count", topology.inhibitory_count},
        {"topology_file", "topology.spkcsr"},
        {"state_file", "state.nstate"}
    };
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Could not create neural module manifest");
    out << manifest.dump(2) << '\n';
    out.flush();
    if (!out) throw std::runtime_error("Failed flushing neural module manifest");
}

} // namespace

void NeuralModuleState::validate(const CSRTopology& topology) const {
    topology.validate();
    const auto neurons = static_cast<std::size_t>(topology.neuron_count());
    const auto edges = topology.synapse_count();
    const auto check_neuron = [neurons](const std::vector<float>& values,
                                         const char* field) {
        if (!values.empty() && values.size() != neurons)
            throw std::invalid_argument(std::string(field) + " must be empty or match neuron count");
        for (const auto value : values)
            if (!std::isfinite(value))
                throw std::invalid_argument(std::string(field) + " contains a non-finite value");
    };
    const auto check_edge = [edges](const std::vector<float>& values,
                                    const char* field) {
        if (!values.empty() && values.size() != edges)
            throw std::invalid_argument(std::string(field) + " must be empty or match synapse count");
        for (const auto value : values)
            if (!std::isfinite(value))
                throw std::invalid_argument(std::string(field) + " contains a non-finite value");
    };
    check_neuron(membrane_voltage, "membrane_voltage");
    check_neuron(recovery, "recovery");
    check_neuron(adaptation, "adaptation");
    check_neuron(phase, "phase");
    check_neuron(presynaptic_trace, "presynaptic_trace");
    check_neuron(postsynaptic_trace, "postsynaptic_trace");
    check_edge(synapse_phase_lag, "synapse_phase_lag");
    if (!edge_plasticity_mask.empty() && edge_plasticity_mask.size() != edges)
        throw std::invalid_argument("edge_plasticity_mask must be empty or match synapse count");
    for (const auto enabled : edge_plasticity_mask)
        if (enabled > 1)
            throw std::invalid_argument("edge_plasticity_mask values must be zero or one");
}

NeuralMemoryStore::NeuralMemoryStore(std::filesystem::path root_directory)
    : root_directory_(std::filesystem::absolute(std::move(root_directory)).lexically_normal()) {
    if (root_directory_.empty())
        throw std::invalid_argument("Neural memory root directory cannot be empty");
}

void NeuralMemoryStore::validate_id(const std::string& id) {
    if (id.empty() || id == "." || id == "..")
        throw std::invalid_argument("Neural module ID is empty or reserved");
    for (const unsigned char c : id) {
        const bool valid = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                           (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.';
        if (!valid)
            throw std::invalid_argument("Neural module ID may contain only letters, digits, dot, dash, and underscore");
    }
}

std::filesystem::path NeuralMemoryStore::module_directory(const std::string& id) const {
    validate_id(id);
    return root_directory_ / id;
}

void NeuralMemoryStore::save(const std::string& id,
                             const std::string& name,
                             const CSRTopology& topology,
                             const NeuralModuleState& state,
                             const std::string& description) {
    validate_id(id);
    if (name.empty()) throw std::invalid_argument("Neural module name cannot be empty");
    topology.validate();
    state.validate(topology);
    std::filesystem::create_directories(root_directory_);

    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto staging = root_directory_ / ("." + id + ".staging-" + std::to_string(tick));
    const auto backup = root_directory_ / ("." + id + ".backup-" + std::to_string(tick));
    const auto destination = module_directory(id);
    std::filesystem::create_directory(staging);
    bool moved_old = false;
    try {
        save_csr_topology(topology, (staging / "topology.spkcsr").string());
        write_state_file(staging / "state.nstate", topology, state);
        write_manifest(staging / "neuromancer.module.json",
                       id, name, description, topology);

        if (std::filesystem::exists(destination)) {
            std::filesystem::rename(destination, backup);
            moved_old = true;
        }
        try {
            std::filesystem::rename(staging, destination);
        } catch (...) {
            if (moved_old) {
                std::error_code ignored;
                std::filesystem::rename(backup, destination, ignored);
            }
            throw;
        }
        if (moved_old) {
            std::error_code ignored;
            std::filesystem::remove_all(backup, ignored);
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove_all(staging, ignored);
        throw;
    }
}

NeuralModuleInfo NeuralMemoryStore::inspect(const std::string& id) const {
    const auto directory = module_directory(id);
    const auto manifest_path = std::filesystem::canonical(directory / "neuromancer.module.json");
    Json manifest;
    try {
        std::ifstream in(manifest_path);
        if (!in) throw std::runtime_error("Could not open neural module manifest");
        in >> manifest;
    } catch (const Json::exception& error) {
        throw std::invalid_argument(std::string("Invalid neural module manifest JSON: ") + error.what());
    }
    if (!manifest.is_object())
        throw std::invalid_argument("Neural module manifest root must be an object");
    const auto schema = required_u32(manifest, "schema_version");
    if (schema != 1) throw std::invalid_argument("Unsupported neural module schema version");

    NeuralModuleInfo info;
    info.schema_version = schema;
    info.id = required_string(manifest, "id");
    info.name = required_string(manifest, "name");
    const auto description = manifest.find("description");
    if (description != manifest.end()) {
        if (!description->is_string())
            throw std::invalid_argument("Neural module description must be a string");
        info.description = description->get<std::string>();
    }
    info.package_directory = std::filesystem::canonical(directory);
    info.manifest_path = manifest_path;
    info.excitatory_count = required_u32(manifest, "excitatory_count");
    info.inhibitory_count = required_u32(manifest, "inhibitory_count");
    if (info.id != id)
        throw std::invalid_argument("Neural module directory ID does not match its manifest ID");

    info.topology_path = contained_file(
        info.package_directory, required_string(manifest, "topology_file"), "topology_file");
    info.state_path = contained_file(
        info.package_directory, required_string(manifest, "state_file"), "state_file");
    const auto topology = load_csr_topology(info.topology_path.string());
    if (topology.excitatory_count != info.excitatory_count ||
        topology.inhibitory_count != info.inhibitory_count)
        throw std::invalid_argument("Neural module manifest dimensions do not match CSR topology");
    info.synapse_count = topology.synapse_count();
    (void)read_state_file(info.state_path, topology);
    return info;
}

NeuralModuleData NeuralMemoryStore::load(const std::string& id) const {
    NeuralModuleData result;
    result.info = inspect(id);
    result.topology = load_csr_topology(result.info.topology_path.string());
    result.state = read_state_file(result.info.state_path, result.topology);
    return result;
}

std::vector<NeuralModuleInfo> NeuralMemoryStore::list() const {
    std::vector<NeuralModuleInfo> modules;
    if (!std::filesystem::exists(root_directory_)) return modules;
    for (const auto& entry : std::filesystem::directory_iterator(root_directory_)) {
        if (!entry.is_directory()) continue;
        if (!std::filesystem::exists(entry.path() / "neuromancer.module.json")) continue;
        modules.push_back(inspect(entry.path().filename().string()));
    }
    return modules;
}

bool NeuralMemoryStore::contains(const std::string& id) const {
    validate_id(id);
    return std::filesystem::is_regular_file(
        module_directory(id) / "neuromancer.module.json");
}

void NeuralMemoryStore::erase(const std::string& id) const {
    const auto directory = module_directory(id);
    if (!std::filesystem::exists(directory)) return;
    const auto canonical_root = std::filesystem::canonical(root_directory_);
    const auto canonical_directory = std::filesystem::canonical(directory);
    auto root_it = canonical_root.begin();
    auto directory_it = canonical_directory.begin();
    for (; root_it != canonical_root.end() && directory_it != canonical_directory.end();
         ++root_it, ++directory_it) {
        if (*root_it != *directory_it)
            throw std::invalid_argument("Neural module directory resolves outside memory root");
    }
    if (root_it != canonical_root.end() || directory_it == canonical_directory.end())
        throw std::invalid_argument("Invalid neural module directory");
    std::filesystem::remove_all(canonical_directory);
}

} // namespace spiking
