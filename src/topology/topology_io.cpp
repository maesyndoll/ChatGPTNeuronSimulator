#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace spiking {
namespace {

constexpr std::array<char, 8> kMagic = {
    'S', 'P', 'K', 'C', 'S', 'R', '0', '1'
};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint64_t kMaximumEdges = UINT32_MAX;

void write_bytes(std::ofstream& out, const char* data, std::size_t size) {
    out.write(data, static_cast<std::streamsize>(size));
    if (!out) {
        throw std::runtime_error("Failed writing CSR topology file");
    }
}

void read_bytes(std::ifstream& in, char* data, std::size_t size) {
    in.read(data, static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("Truncated or unreadable CSR topology file");
    }
}

void write_u32(std::ofstream& out, std::uint32_t value) {
    char bytes[4];
    for (unsigned i = 0; i < 4; ++i) {
        bytes[i] = static_cast<char>((value >> (8U * i)) & 0xffU);
    }
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
    for (unsigned i = 0; i < 8; ++i) {
        bytes[i] = static_cast<char>((value >> (8U * i)) & 0xffU);
    }
    write_bytes(out, bytes, sizeof(bytes));
}

std::uint64_t read_u64(std::ifstream& in) {
    unsigned char bytes[8];
    read_bytes(in, reinterpret_cast<char*>(bytes), sizeof(bytes));
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
    }
    return value;
}

void write_float(std::ofstream& out, float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t),
                  "CSR format requires 32-bit float");
    static_assert(std::numeric_limits<float>::is_iec559,
                  "CSR format requires IEEE-754 float");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(out, bits);
}

float read_float(std::ifstream& in) {
    const std::uint32_t bits = read_u32(in);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

TopologyDistributionStats summarize(const std::vector<double>& values) {
    TopologyDistributionStats result;
    result.count = values.size();

    if (values.empty()) {
        return result;
    }

    result.minimum = values.front();
    result.maximum = values.front();
    double total = 0.0;

    for (const double value : values) {
        result.minimum = std::min(result.minimum, value);
        result.maximum = std::max(result.maximum, value);
        total += value;
    }

    result.mean = total / static_cast<double>(values.size());
    return result;
}

} // namespace

void save_csr_topology(const CSRTopology& topology,
                       const std::string& filename) {
    topology.validate();

    if (topology.synapse_count() > kMaximumEdges) {
        throw std::length_error("CSR file format supports at most UINT32_MAX edges");
    }

    std::ofstream out(filename, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("Could not open CSR topology file for writing: " +
                                 filename);
    }

    write_bytes(out, kMagic.data(), kMagic.size());
    write_u32(out, kVersion);
    write_u32(out, topology.excitatory_count);
    write_u32(out, topology.inhibitory_count);
    write_u32(out, topology.minimum_delay);
    write_u32(out, topology.maximum_delay);
    write_u64(out, static_cast<std::uint64_t>(topology.synapse_count()));

    for (const auto value : topology.row_offsets) write_u32(out, value);
    for (const auto value : topology.target_indices) write_u32(out, value);
    for (const auto value : topology.weights) write_float(out, value);
    for (const auto value : topology.delays) write_u32(out, value);

    out.flush();
    if (!out) {
        throw std::runtime_error("Failed flushing CSR topology file");
    }
}

CSRTopology load_csr_topology(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open CSR topology file: " + filename);
    }

    std::array<char, 8> magic{};
    read_bytes(in, magic.data(), magic.size());
    if (magic != kMagic) {
        throw std::invalid_argument("Invalid CSR topology file signature");
    }

    const auto version = read_u32(in);
    if (version != kVersion) {
        throw std::invalid_argument("Unsupported CSR topology file version");
    }

    CSRTopology topology;
    topology.excitatory_count = read_u32(in);
    topology.inhibitory_count = read_u32(in);
    topology.minimum_delay = read_u32(in);
    topology.maximum_delay = read_u32(in);

    const std::uint64_t edge_count = read_u64(in);
    const std::uint64_t neuron_count =
        static_cast<std::uint64_t>(topology.excitatory_count) +
        topology.inhibitory_count;

    if (neuron_count == 0 || neuron_count > UINT32_MAX ||
        edge_count > kMaximumEdges) {
        throw std::invalid_argument("CSR topology file declares invalid dimensions");
    }

    // Validate the exact file length before allocating from file-controlled
    // dimensions. The version-1 header occupies 36 bytes.
    constexpr std::uint64_t header_size = 36;
    const std::uint64_t offset_count = neuron_count + 1;

    if (edge_count >
        (std::numeric_limits<std::uint64_t>::max() - header_size -
         offset_count * 4ULL) / 12ULL) {
        throw std::length_error("CSR topology file dimensions overflow");
    }

    const std::uint64_t expected_size =
        header_size + offset_count * 4ULL + edge_count * 12ULL;

    in.seekg(0, std::ios::end);
    const std::streampos file_end = in.tellg();
    if (file_end < std::streampos(0)) {
        throw std::runtime_error("Could not determine CSR topology file size");
    }

    const auto actual_size = static_cast<std::uint64_t>(
        static_cast<std::streamoff>(file_end));

    if (actual_size != expected_size) {
        throw std::invalid_argument(
            "CSR topology file size does not match its declared dimensions");
    }

    in.seekg(static_cast<std::streamoff>(header_size), std::ios::beg);
    if (!in) {
        throw std::runtime_error("Could not seek to CSR topology payload");
    }

    topology.row_offsets.resize(static_cast<std::size_t>(offset_count));
    topology.target_indices.resize(static_cast<std::size_t>(edge_count));
    topology.weights.resize(static_cast<std::size_t>(edge_count));
    topology.delays.resize(static_cast<std::size_t>(edge_count));

    for (auto& value : topology.row_offsets) value = read_u32(in);
    for (auto& value : topology.target_indices) value = read_u32(in);
    for (auto& value : topology.weights) value = read_float(in);
    for (auto& value : topology.delays) value = read_u32(in);

    char extra = 0;
    if (in.read(&extra, 1)) {
        throw std::invalid_argument("CSR topology file contains trailing data");
    }
    if (!in.eof()) {
        throw std::runtime_error("Error while checking CSR topology file length");
    }

    topology.validate();
    return topology;
}

TopologyStatistics analyze_csr_topology(const CSRTopology& topology) {
    topology.validate();

    TopologyStatistics result;
    result.excitatory_count = topology.excitatory_count;
    result.inhibitory_count = topology.inhibitory_count;
    result.synapse_count = topology.synapse_count();

    const std::uint64_t neurons = topology.neuron_count();
    const std::uint64_t possible_edges = neurons * (neurons - 1ULL);
    result.density = possible_edges == 0 ? 0.0 :
        static_cast<double>(result.synapse_count) /
        static_cast<double>(possible_edges);

    std::vector<double> outgoing;
    std::vector<double> incoming(neurons, 0.0);
    std::vector<double> weights;
    std::vector<double> delays;

    outgoing.reserve(neurons);
    weights.reserve(topology.synapse_count());
    delays.reserve(topology.synapse_count());

    for (std::uint32_t source = 0; source < neurons; ++source) {
        const auto begin = topology.row_offsets[source];
        const auto end = topology.row_offsets[source + 1];
        outgoing.push_back(static_cast<double>(end - begin));

        const bool source_exc = source < topology.excitatory_count;

        for (std::uint32_t edge = begin; edge < end; ++edge) {
            const auto target = topology.target_indices[edge];
            const bool target_exc = target < topology.excitatory_count;

            if (source_exc && target_exc) ++result.excitatory_to_excitatory;
            else if (source_exc && !target_exc) ++result.excitatory_to_inhibitory;
            else if (!source_exc && target_exc) ++result.inhibitory_to_excitatory;
            else ++result.inhibitory_to_inhibitory;

            incoming[target] += 1.0;
            weights.push_back(static_cast<double>(topology.weights[edge]));
            delays.push_back(static_cast<double>(topology.delays[edge]));
        }
    }

    result.outgoing_degree = summarize(outgoing);
    result.incoming_degree = summarize(incoming);
    result.weights = summarize(weights);
    result.delays = summarize(delays);
    return result;
}

} // namespace spiking
