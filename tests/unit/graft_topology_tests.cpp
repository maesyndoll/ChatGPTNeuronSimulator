#include "runtime/graft_topology.h"

#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
spiking::CSRTopology graph(std::uint32_t excitatory,
                           std::uint32_t inhibitory,
                           std::vector<std::vector<std::tuple<std::uint32_t,float,std::uint32_t>>> rows) {
    spiking::CSRTopology result;
    result.excitatory_count = excitatory;
    result.inhibitory_count = inhibitory;
    result.minimum_delay = 1;
    result.maximum_delay = 3;
    result.row_offsets.push_back(0);
    for (const auto& row : rows) {
        for (const auto& edge : row) {
            result.target_indices.push_back(std::get<0>(edge));
            result.weights.push_back(std::get<1>(edge));
            result.delays.push_back(std::get<2>(edge));
        }
        result.row_offsets.push_back(static_cast<std::uint32_t>(result.target_indices.size()));
    }
    result.validate();
    return result;
}
}
int main() {
    try {
        const auto base = graph(1, 1, {
            {{1, 0.5f, 1}}, {{0, -0.4f, 2}}
        });
        const auto graft = graph(1, 1, {
            {{1, 0.7f, 3}}, {{0, -0.6f, 1}}
        });
        const auto merged = spiking::merge_graft_topology(base, graft);
        require(merged.topology.excitatory_count == 2 &&
                merged.topology.inhibitory_count == 2,
                "merged E/I population counts are wrong");
        require(merged.topology.neuron_count() == 4 &&
                merged.topology.synapse_count() == 4,
                "merged graph dimensions are wrong");
        require(merged.base_neuron_map == std::vector<std::uint32_t>({0, 2}),
                "base inhibitory neuron was not remapped");
        require(merged.graft_neuron_map == std::vector<std::uint32_t>({1, 3}),
                "graft neurons were not mapped into E/I blocks");
        require(merged.topology.target_indices ==
                    std::vector<std::uint32_t>({2, 3, 0, 1}),
                "merged CSR targets are not correctly remapped");
        require(merged.plasticity_mask ==
                    std::vector<std::uint8_t>({1, 0, 1, 0}),
                "graft edges are not frozen independently of transmission");
        require(merged.topology.weights[1] == 0.7f &&
                merged.topology.weights[3] == -0.6f,
                "frozen graft weights were not retained");
        require(merged.topology.delays[1] == 3,
                "graft delay metadata was not retained");
        require(merged.base_edge_map == std::vector<std::uint32_t>({0, 2}) &&
                merged.graft_edge_map == std::vector<std::uint32_t>({1, 3}),
                "edge remapping was not returned for state migration");
        const auto open = spiking::merge_graft_topology(base, graft, {0, 1}, true);
        require(open.plasticity_mask == std::vector<std::uint8_t>({0, 1, 1, 1}),
                "mixed plasticity policy was not retained");
        bool rejected = false;
        try { (void)spiking::merge_graft_topology(base, graft, {1}); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "wrong-sized base plasticity mask accepted");
        std::cout << "GRAFT TOPOLOGY MERGE TESTS: PASS\n"
                  << "population_ordering=PASS neuron_remap=PASS csr_merge=PASS\n"
                  << "frozen_transmission_policy=PASS mixed_plasticity=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GRAFT TOPOLOGY MERGE TESTS: FAIL: " << error.what() << '\n';
        return 1;
    }
}
