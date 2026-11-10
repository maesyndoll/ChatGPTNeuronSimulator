#include "runtime/neural_memory_store.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct TemporaryDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("neuromancer-memory-test-" + std::to_string(
            std::filesystem::file_time_type::clock::now().time_since_epoch().count()));
    TemporaryDirectory() { std::filesystem::create_directories(path); }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

spiking::CSRTopology fixture_topology() {
    spiking::CSRTopology graph;
    graph.excitatory_count = 2;
    graph.inhibitory_count = 1;
    graph.minimum_delay = 1;
    graph.maximum_delay = 2;
    graph.row_offsets = {0, 1, 2, 3};
    graph.target_indices = {1, 2, 0};
    graph.weights = {0.5f, 0.7f, -0.4f};
    graph.delays = {1, 2, 1};
    graph.validate();
    return graph;
}
}

int main() {
    try {
        TemporaryDirectory temp;
        spiking::NeuralMemoryStore store(temp.path / "memory");
        const auto topology = fixture_topology();

        spiking::NeuralModuleState state;
        state.membrane_voltage = {-64.0f, -62.5f, -67.0f};
        state.recovery = {1.0f, 1.5f, -2.0f};
        state.adaptation = {0.0f, 0.1f, 0.2f};
        state.phase = {0.2f, 1.1f, 2.4f};
        state.presynaptic_trace = {0.1f, 0.2f, 0.3f};
        state.postsynaptic_trace = {0.4f, 0.5f, 0.6f};
        state.synapse_phase_lag = {0.0f, 0.25f, -0.25f};
        state.edge_plasticity_mask = {1, 0, 1};

        store.save("visual.features", "Visual Features", topology, state,
                   "Persisted sensory feature population");
        require(store.contains("visual.features"), "saved module was not indexed");
        const auto info = store.inspect("visual.features");
        require(info.excitatory_count == 2 && info.inhibitory_count == 1,
                "module population metadata did not round-trip");
        require(info.synapse_count == 3, "module synapse count did not round-trip");

        const auto loaded = store.load("visual.features");
        require(loaded.topology.row_offsets == topology.row_offsets &&
                loaded.topology.target_indices == topology.target_indices &&
                loaded.topology.weights == topology.weights &&
                loaded.topology.delays == topology.delays,
                "CSR topology did not round-trip");
        require(loaded.state.membrane_voltage == state.membrane_voltage &&
                loaded.state.recovery == state.recovery &&
                loaded.state.adaptation == state.adaptation &&
                loaded.state.phase == state.phase &&
                loaded.state.presynaptic_trace == state.presynaptic_trace &&
                loaded.state.postsynaptic_trace == state.postsynaptic_trace &&
                loaded.state.synapse_phase_lag == state.synapse_phase_lag &&
                loaded.state.edge_plasticity_mask == state.edge_plasticity_mask,
                "neural state did not round-trip");
        require(store.list().size() == 1, "module listing returned the wrong count");

        auto replacement = state;
        replacement.membrane_voltage[0] = -55.0f;
        store.save("visual.features", "Visual Features v2", topology, replacement,
                   "Updated module state");
        const auto updated = store.load("visual.features");
        require(updated.info.name == "Visual Features v2" &&
                updated.state.membrane_voltage[0] == -55.0f,
                "module replacement did not publish updated state");

        bool invalid_id_rejected = false;
        try { store.save("../escape", "Invalid", topology); }
        catch (const std::invalid_argument&) { invalid_id_rejected = true; }
        require(invalid_id_rejected, "path traversal module ID was accepted");

        auto invalid_state = state;
        invalid_state.phase.pop_back();
        bool invalid_dimensions_rejected = false;
        try { store.save("bad.state", "Bad State", topology, invalid_state); }
        catch (const std::invalid_argument&) { invalid_dimensions_rejected = true; }
        require(invalid_dimensions_rejected, "invalid neural state dimensions were accepted");

        store.erase("visual.features");
        require(!store.contains("visual.features"), "module erase did not remove its index entry");
        require(store.list().empty(), "module list was not empty after erase");

        std::cout << "NEURAL MEMORY STORE TESTS: PASS\n"
                  << "csr_round_trip=PASS neural_state_round_trip=PASS module_index=PASS\n"
                  << "atomic_replacement=PASS path_traversal_rejected=PASS "
                     "dimension_validation=PASS erase=PASS\n"
                  << "LIMITATION=Persistent storage only; no live GPU attach/detach yet.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NEURAL MEMORY STORE TESTS: FAIL: " << error.what() << '\n';
        return 1;
    }
}
