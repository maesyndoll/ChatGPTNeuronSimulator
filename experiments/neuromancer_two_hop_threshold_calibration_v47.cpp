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

struct Edge { std::uint32_t source, target; float weight; };

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = E;
    p.simulation.inhibitory_count = 1;
    p.simulation.steps = 1;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 47001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = p.simulation.inhibitory_current_max = 0.0f;
    p.topology.minimum_delay = p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology make_graph(const std::vector<Edge>& edges) {
    CSRTopology g;
    g.excitatory_count = E;
    g.inhibitory_count = 1;
    g.minimum_delay = g.maximum_delay = 1;
    g.row_offsets.assign(N + 1, 0);
    std::vector<std::vector<Edge>> rows(N);
    for (const auto& e : edges) {
        if (e.source >= E || e.target >= E || e.source == e.target)
            throw std::runtime_error("V47 invalid edge");
        rows[e.source].push_back(e);
    }
    for (std::uint32_t s = 0; s < N; ++s) {
        for (const auto& e : rows[s]) {
            g.target_indices.push_back(e.target);
            g.weights.push_back(e.weight);
            g.delays.push_back(1);
        }
        g.row_offsets[s + 1] = static_cast<std::uint32_t>(g.target_indices.size());
    }
    g.validate();
    return g;
}

struct Result {
    std::string name;
    std::uint64_t a_spikes=0, b_spikes=0, c_spikes=0, dropped=0;
    std::uint32_t b_unique=0, c_unique=0;
    float b_vmax=-1000.0f, c_vmax=-1000.0f;
    bool finite=true;
};

Result run_trial(const std::filesystem::path& dir, const std::string& name,
                 const std::vector<Edge>& edges, std::uint32_t cue_population,
                 std::uint32_t cue_count, std::uint32_t steps) {
    const auto file = dir / (name + ".csr");
    save_csr_topology(make_graph(edges), file.string());
    CudaSimulationRuntime rt(make_model(), file.string(), 0, true, false);
    std::filesystem::remove(file);

    Result r; r.name=name;
    std::array<bool,K> b_seen{}, c_seen{};
    std::vector<std::uint32_t> sampled;
    for (std::uint32_t id=K; id<E; ++id) sampled.push_back(id);
    const auto start=rt.completed_steps()+1U;
    for (std::uint32_t i=0; i<cue_count; ++i)
        rt.schedule_stimulus(cue_population*K+i, start, 4, 500.0f);

    for (std::uint32_t step=0; step<steps; ++step) {
        rt.run_steps(1);
        for (const auto& v : rt.collect_voltage_samples(sampled)) {
            if (v.neuron_id < 2*K) r.b_vmax=std::max(r.b_vmax,v.voltage_mv);
            else r.c_vmax=std::max(r.c_vmax,v.voltage_mv);
            r.finite=r.finite && std::isfinite(v.voltage_mv);
        }
        const auto batch=rt.collect_spike_events();
        r.dropped += batch.dropped_events;
        for (const auto& ev : batch.events) {
            if (ev.neuron_id < K) ++r.a_spikes;
            else if (ev.neuron_id < 2*K) {
                ++r.b_spikes; b_seen[ev.neuron_id-K]=true;
            } else if (ev.neuron_id < 3*K) {
                ++r.c_spikes; c_seen[ev.neuron_id-2*K]=true;
            }
        }
    }
    r.b_unique=static_cast<std::uint32_t>(std::count(b_seen.begin(),b_seen.end(),true));
    r.c_unique=static_cast<std::uint32_t>(std::count(c_seen.begin(),c_seen.end(),true));
    r.finite=r.finite && rt.collect_metrics().finite_state;
    rt.collect_connectivity().validate();
    return r;
}

std::vector<Edge> make_chain(float ab, float bc) {
    std::vector<Edge> e;
    for (std::uint32_t i=0;i<K;++i) {
        e.push_back({i,K+i,ab});
        e.push_back({K+i,2*K+i,bc});
    }
    return e;
}
}

int main(int argc,char** argv) {
    const std::filesystem::path dir=argc>1?argv[1]:"experiments/logs/v47_two_hop_threshold";
    try {
        std::filesystem::create_directories(dir);
        std::vector<Result> results;
        // Isolate the second hop: externally drive B, never C.
        for (float w : {30.0f,50.0f,75.0f,100.0f,150.0f,200.0f}) {
            const auto label="direct_BC_w"+std::to_string(static_cast<int>(w));
            std::vector<Edge> e;
            for (std::uint32_t i=0;i<K;++i) e.push_back({K+i,2*K+i,w});
            results.push_back(run_trial(dir,label,e,1,4,48));
        }
        // Then test the complete chain with the second-hop weight calibrated
        // independently above. A is the only externally stimulated population.
        results.push_back(run_trial(dir,"chain_AB30_BC100",make_chain(30.0f,100.0f),0,4,64));
        results.push_back(run_trial(dir,"chain_AB100_BC100",make_chain(100.0f,100.0f),0,4,64));
        results.push_back(run_trial(dir,"chain_AB100_BC200",make_chain(100.0f,200.0f),0,4,64));
        results.push_back(run_trial(dir,"no_path_negative",{},0,4,64));

        std::ofstream csv(dir/"v47_two_hop_threshold.csv");
        if(!csv) throw std::runtime_error("cannot create V47 CSV");
        csv<<"condition,A_spikes,B_spikes,B_unique,C_spikes,C_unique,B_vmax_mv,C_vmax_mv,dropped,finite\n";
        for(const auto& r:results)
            csv<<r.name<<','<<r.a_spikes<<','<<r.b_spikes<<','<<r.b_unique<<','
               <<r.c_spikes<<','<<r.c_unique<<','<<r.b_vmax<<','<<r.c_vmax<<','
               <<r.dropped<<','<<(r.finite?1:0)<<'\n';

        bool finite=true, no_drops=true;
        for(const auto& r:results) {
            finite=finite&&r.finite; no_drops=no_drops&&(r.dropped==0);
            std::cout<<"condition="<<r.name<<" A_spikes="<<r.a_spikes
                     <<" B_spikes="<<r.b_spikes<<" B_unique="<<r.b_unique<<"/8"
                     <<" C_spikes="<<r.c_spikes<<" C_unique="<<r.c_unique<<"/8"
                     <<" B_vmax_mv="<<r.b_vmax<<" C_vmax_mv="<<r.c_vmax
                     <<" dropped="<<r.dropped<<" finite="<<(r.finite?"PASS":"FAIL")<<'\n';
        }
        const auto& negative=results.back();
        bool direct_bc_recruits=false;
        for(std::size_t i=0;i<6;++i) direct_bc_recruits=direct_bc_recruits||(results[i].c_unique>0);
        bool chain_recruits=false;
        for(std::size_t i=6;i<9;++i)
            chain_recruits=chain_recruits||(results[i].b_unique>0&&results[i].c_unique>0);
        const bool negative_clean=negative.b_unique==0&&negative.c_unique==0;
        std::cout<<"direct_B_to_C_recruitment="<<(direct_bc_recruits?"PASS":"FAIL")<<'\n'
                 <<"A_to_B_to_C_chain_recruitment="<<(chain_recruits?"PASS":"FAIL")<<'\n'
                 <<"no_path_negative_control="<<(negative_clean?"PASS":"FAIL")<<'\n'
                 <<"finite_state="<<(finite?"PASS":"FAIL")<<'\n'
                 <<"no_dropped_events="<<(no_drops?"PASS":"FAIL")<<'\n'
                 <<"artifacts="<<(dir/"v47_two_hop_threshold.csv").string()<<'\n'
                 <<"LIMITATION=V47 calibrates engineered two-hop transmission and does not test synaptic learning, phase learning, or structural plasticity.\n";
        return direct_bc_recruits&&negative_clean&&finite&&no_drops?0:2;
    } catch(const std::exception& e) {
        std::cerr<<"V47 diagnostic failed: "<<e.what()<<'\n';
        return 1;
    }
}
