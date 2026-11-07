#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr std::uint32_t K = 8;
constexpr std::uint32_t E = 3 * K;
constexpr std::uint32_t N = E + 1;
constexpr float PI = 3.14159265358979323846f;

struct Edge {
    std::uint32_t source;
    std::uint32_t target;
    float weight;
    std::uint32_t delay = 1;
};

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = E;
    p.simulation.inhibitory_count = 1;
    p.simulation.steps = 1;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 46001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;
    p.topology.minimum_delay = 1;
    p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology make_graph(const std::vector<Edge>& edges) {
    CSRTopology g;
    g.excitatory_count = E;
    g.inhibitory_count = 1;
    g.minimum_delay = 1;
    g.maximum_delay = 1;
    g.row_offsets.assign(N + 1, 0);
    std::vector<std::vector<Edge>> rows(N);
    for (const auto& edge : edges) {
        if (edge.source >= E || edge.target >= E || edge.source == edge.target)
            throw std::runtime_error("V46 edge has invalid neuron identity");
        rows[edge.source].push_back(edge);
    }
    for (std::uint32_t source = 0; source < N; ++source) {
        for (const auto& edge : rows[source]) {
            g.target_indices.push_back(edge.target);
            g.weights.push_back(edge.weight);
            g.delays.push_back(edge.delay);
        }
        g.row_offsets[source + 1] =
            static_cast<std::uint32_t>(g.target_indices.size());
    }
    g.validate();
    return g;
}

struct Result {
    std::string name;
    std::uint64_t source_spikes = 0;
    std::uint64_t b_spikes = 0;
    std::uint64_t c_spikes = 0;
    std::uint32_t b_unique = 0;
    std::uint32_t c_unique = 0;
    float b_vmax = -1000.0f;
    float c_vmax = -1000.0f;
    std::uint64_t dropped = 0;
    bool finite = true;
};

Result run_trial(const std::filesystem::path& dir, const std::string& name,
                 const std::vector<Edge>& edges, std::uint32_t cue_count,
                 std::uint32_t steps) {
    const auto path = dir / (name + ".csr");
    const auto graph = make_graph(edges);
    save_csr_topology(graph, path.string());
    CudaSimulationRuntime runtime(make_model(), path.string(), 0, true, false);
    std::filesystem::remove(path);

    Result out;
    out.name = name;
    std::array<bool, K> b_seen{}, c_seen{};
    std::vector<std::uint32_t> sampled;
    for (std::uint32_t id = K; id < E; ++id) sampled.push_back(id);
    const auto start = runtime.completed_steps() + 1U;
    for (std::uint32_t id = 0; id < cue_count; ++id)
        runtime.schedule_stimulus(id, start, 4, 500.0f);

    for (std::uint32_t step = 0; step < steps; ++step) {
        runtime.run_steps(1);
        const auto voltages = runtime.collect_voltage_samples(sampled);
        for (const auto& v : voltages) {
            if (v.neuron_id < 2 * K)
                out.b_vmax = std::max(out.b_vmax, v.voltage_mv);
            else
                out.c_vmax = std::max(out.c_vmax, v.voltage_mv);
            out.finite = out.finite && std::isfinite(v.voltage_mv);
        }
        const auto batch = runtime.collect_spike_events();
        out.dropped += batch.dropped_events;
        for (const auto& event : batch.events) {
            if (event.neuron_id < K) ++out.source_spikes;
            else if (event.neuron_id < 2 * K) {
                ++out.b_spikes;
                b_seen[event.neuron_id - K] = true;
            } else if (event.neuron_id < 3 * K) {
                ++out.c_spikes;
                c_seen[event.neuron_id - 2 * K] = true;
            }
        }
    }
    out.b_unique = static_cast<std::uint32_t>(
        std::count(b_seen.begin(), b_seen.end(), true));
    out.c_unique = static_cast<std::uint32_t>(
        std::count(c_seen.begin(), c_seen.end(), true));
    out.finite = out.finite && runtime.collect_metrics().finite_state;
    runtime.collect_connectivity().validate();
    return out;
}

std::vector<Edge> chain_edges(float weight) {
    std::vector<Edge> edges;
    for (std::uint32_t i = 0; i < K; ++i) {
        edges.push_back({i, K + i, weight, 1});
        edges.push_back({K + i, 2 * K + i, weight, 1});
    }
    return edges;
}
} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path dir = argc > 1 ? argv[1] :
        "experiments/logs/v46_cuda_transmission_diagnostics";
    try {
        std::filesystem::create_directories(dir);

        // Stage 1: reproduce the known-good V42 single-synapse control and
        // compare it with an intentionally weak connection.
        const auto single_strong = run_trial(
            dir, "single_strong", {{0, K, 30.0f, 1}}, 1, 40);
        const auto single_weak = run_trial(
            dir, "single_weak", {{0, K, 0.01f, 1}}, 1, 40);

        // Stage 2: an engineered one-to-one A->B positive control. Only four
        // of eight A neurons receive external current.
        std::vector<Edge> ab;
        for (std::uint32_t i = 0; i < K; ++i)
            ab.push_back({i, K + i, 30.0f, 1});
        const auto assembly = run_trial(dir, "assembly_AB_positive", ab, 4, 48);

        // Stage 3: engineered A->B->C chain. The only external cue is A[0..3].
        const auto chain = run_trial(
            dir, "chain_ABC_positive", chain_edges(30.0f), 4, 64);

        // Stage 4: matched no-path negative control.
        const auto no_path = run_trial(dir, "chain_no_path_negative", {}, 4, 64);

        const std::array<const Result*, 5> results{
            &single_strong, &single_weak, &assembly, &chain, &no_path
        };
        std::ofstream csv(dir / "v46_diagnostics.csv");
        if (!csv) throw std::runtime_error("Cannot create V46 diagnostics CSV");
        csv << "condition,source_spikes,B_spikes,B_unique,C_spikes,C_unique,"
               "B_vmax_mv,C_vmax_mv,dropped,finite\n";
        for (const auto* r : results) {
            csv << r->name << ',' << r->source_spikes << ',' << r->b_spikes
                << ',' << r->b_unique << ',' << r->c_spikes << ','
                << r->c_unique << ',' << r->b_vmax << ',' << r->c_vmax
                << ',' << r->dropped << ',' << (r->finite ? 1 : 0) << '\n';
        }

        const bool single_pass = single_strong.source_spikes > 0 &&
                                 single_strong.b_spikes > 0;
        const bool weak_suppressed = single_weak.b_spikes < single_strong.b_spikes;
        const bool assembly_pass = assembly.source_spikes > 0 &&
                                   assembly.b_unique >= 1;
        const bool chain_pass = chain.b_unique >= 1 && chain.c_unique >= 1;
        const bool negative_clean = no_path.b_unique == 0 && no_path.c_unique == 0;
        bool finite = true, no_drops = true;
        for (const auto* r : results) {
            finite = finite && r->finite;
            no_drops = no_drops && r->dropped == 0;
        }

        std::cout << "NEUROMANCER V46 CUDA TRANSMISSION/RECRUITMENT DIAGNOSTICS\n"
                  << "scope=engineered positive controls; not learned associative memory\n";
        for (const auto* r : results) {
            std::cout << "condition=" << r->name
                      << " source_spikes=" << r->source_spikes
                      << " B_spikes=" << r->b_spikes
                      << " B_unique=" << r->b_unique << "/8"
                      << " C_spikes=" << r->c_spikes
                      << " C_unique=" << r->c_unique << "/8"
                      << " B_vmax_mv=" << r->b_vmax
                      << " C_vmax_mv=" << r->c_vmax
                      << " dropped=" << r->dropped
                      << " finite=" << (r->finite ? "PASS" : "FAIL") << '\n';
        }
        std::cout << "stage1_single_synapse_transmission=" << (single_pass ? "PASS" : "FAIL") << '\n'
                  << "stage1_weak_weight_control=" << (weak_suppressed ? "PASS" : "FAIL") << '\n'
                  << "stage2_assembly_recruitment=" << (assembly_pass ? "PASS" : "FAIL") << '\n'
                  << "stage3_A_to_B_to_C_chain=" << (chain_pass ? "PASS" : "FAIL") << '\n'
                  << "stage4_no_path_negative_control=" << (negative_clean ? "PASS" : "FAIL") << '\n'
                  << "finite_state=" << (finite ? "PASS" : "FAIL") << '\n'
                  << "no_dropped_events=" << (no_drops ? "PASS" : "FAIL") << '\n'
                  << "artifacts=" << (dir / "v46_diagnostics.csv").string() << '\n'
                  << "LIMITATION=V46 uses deliberately engineered synaptic pathways to isolate CUDA transmission and recruitment. It does not test learning, structural plasticity, phase learning, or associative-memory acquisition.\n";

        return single_pass && weak_suppressed && assembly_pass && chain_pass &&
               negative_clean && finite && no_drops ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "V46 diagnostic failed: " << error.what() << '\n';
        return 1;
    }
}
