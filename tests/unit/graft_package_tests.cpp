#include "runtime/graft_package.h"
#include "topology/topology_io.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct TemporaryDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("neuromancer-graft-test-" + std::to_string(
            std::filesystem::file_time_type::clock::now().time_since_epoch().count()));
    TemporaryDirectory() { std::filesystem::create_directories(path); }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};
void write_manifest(const std::filesystem::path& path, const std::string& topology,
                    std::uint32_t excitatory = 2, std::uint32_t inhibitory = 0) {
    std::ofstream out(path / "neuromancer.graft.json");
    out << "{\n"
        << "  \"schema_version\": 1,\n"
        << "  \"id\": \"test.adapter\",\n"
        << "  \"name\": \"Test Adapter\",\n"
        << "  \"description\": \"fixture\",\n"
        << "  \"excitatory_count\": " << excitatory << ",\n"
        << "  \"inhibitory_count\": " << inhibitory << ",\n"
        << "  \"topology_file\": \"" << topology << "\",\n"
        << "  \"module_binary\": null,\n"
        << "  \"input_adapters\": [\"codec.in\"],\n"
        << "  \"output_adapters\": [\"codec.out\"]\n"
        << "}\n";
}
spiking::CSRTopology fixture_topology() {
    spiking::CSRTopology graph;
    graph.excitatory_count = 2;
    graph.inhibitory_count = 0;
    graph.minimum_delay = 1;
    graph.maximum_delay = 1;
    graph.row_offsets = {0, 1, 1};
    graph.target_indices = {1};
    graph.weights = {1.0f};
    graph.delays = {1};
    graph.validate();
    return graph;
}
}
int main() {
    try {
        TemporaryDirectory temp;
        const auto package = temp.path / "valid";
        std::filesystem::create_directories(package);
        spiking::save_csr_topology(fixture_topology(),
                                   (package / "topology.spkcsr").string());
        write_manifest(package, "topology.spkcsr");
        const auto info = spiking::GraftPackageInspector::inspect(package);
        require(info.id == "test.adapter", "package ID was not parsed");
        require(info.name == "Test Adapter", "package name was not parsed");
        require(info.excitatory_count == 2 && info.inhibitory_count == 0,
                "population dimensions were not parsed");
        require(info.synapse_count == 1, "CSR edge count was not inspected");
        require(info.input_adapters.size() == 1 && info.output_adapters.size() == 1,
                "adapter interfaces were not parsed");
        require(!info.has_module_binary, "non-executable package marked executable");

        const auto mismatch = temp.path / "mismatch";
        std::filesystem::create_directories(mismatch);
        spiking::save_csr_topology(fixture_topology(),
                                   (mismatch / "topology.spkcsr").string());
        write_manifest(mismatch, "topology.spkcsr", 3, 0);
        bool mismatch_rejected = false;
        try { (void)spiking::GraftPackageInspector::inspect(mismatch); }
        catch (const std::invalid_argument&) { mismatch_rejected = true; }
        require(mismatch_rejected, "manifest/CSR dimension mismatch accepted");

        const auto traversal = temp.path / "traversal";
        std::filesystem::create_directories(traversal);
        write_manifest(traversal, "../outside.spkcsr");
        bool traversal_rejected = false;
        try { (void)spiking::GraftPackageInspector::inspect(traversal); }
        catch (const std::exception&) { traversal_rejected = true; }
        require(traversal_rejected, "package path traversal accepted");

        std::cout << "GRAFT PACKAGE TESTS: PASS\n"
                  << "manifest_parse=PASS csr_validation=PASS dimension_match=PASS "
                  << "adapter_interfaces=PASS path_traversal_rejected=PASS "
                  << "executable_not_loaded=PASS\n"
                  << "LIMITATION=Package inspection only; no live topology attachment "
                     "or dlopen is performed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GRAFT PACKAGE TESTS: FAIL: " << error.what() << '\n';
        return 1;
    }
}
