#include "core/parameters.h"
#include "learning/phase_structural_plasticity.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr float kPi = 3.14159265358979323846f;

float train_and_retrieve_lag() {
    AutonomousPhaseParameters p;
    p.dt_ms = 0.5f;
    p.intrinsic_frequency_hz = 8.0f;
    p.activity_kick_rad = 0.04f;
    p.coupling_learning_rate = 0.001f;
    p.maximum_coupling = 0.5f;
    p.coupling_decay = 0.0001f;
    p.temporal_association_window_steps = 299;
    p.temporal_lag_learning_rate = 0.10f;
    p.minimum_temporal_confidence = 0.20f;
    AutonomousPopulationPhase phase(3, p);
    constexpr int period = 300;
    constexpr std::array<int, 8> control_offsets{20,55,90,125,160,195,230,265};
    for (int t=0; t<6000; ++t) {
        std::vector<std::uint32_t> spikes(3,0U);
        const int within=t%period;
        const int cycle=t/period;
        if (within==0) spikes[0]=1;
        if (within==5) spikes[1]=1;
        if (within==control_offsets[static_cast<std::size_t>(cycle)%control_offsets.size()])
            spikes[2]=1;
        phase.step(spikes,true);
    }
    const float lag=phase.learned_lag(0,1);
    const float confidence=phase.temporal_confidence(0,1);
    const float control=phase.temporal_confidence(0,2);
    if (!std::isfinite(lag) || confidence < 0.80f || confidence < control+0.20f)
        throw std::runtime_error("Temporal phase learner did not pass discrimination gate");
    for(int i=0;i<2000;++i) phase.step({0U,0U,0U},false);
    if(std::fabs(std::remainder(phase.learned_lag(0,1)-lag,2.0f*kPi))>1.0e-6f)
        throw std::runtime_error("Learned lag changed during frozen interval");
    std::cout << "learned_lag_rad=" << lag
              << " learned_confidence=" << confidence
              << " control_confidence=" << control << '\n';
    return lag;
}

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count=2;
    p.simulation.inhibitory_count=1;
    p.simulation.steps=1;
    p.simulation.dt=0.5f;
    p.simulation.threshold=30.0f;
    p.simulation.random_seed=42420;
    p.simulation.initial_voltage_min=-65.0f;
    p.simulation.initial_voltage_max=-65.0f;
    p.simulation.excitatory_current_min=0.0f;
    p.simulation.excitatory_current_max=0.0f;
    p.simulation.inhibitory_current_min=0.0f;
    p.simulation.inhibitory_current_max=0.0f;
    p.topology.p_ee=p.topology.p_ei=p.topology.p_ie=p.topology.p_ii=0.0;
    p.topology.minimum_delay=p.topology.maximum_delay=1;
    p.topology.fibonacci_delays={1};
    p.topology.fibonacci_weights={1.0};
    return p;
}
CSRTopology make_graph() {
    CSRTopology g;
    g.excitatory_count=2; g.inhibitory_count=1;
    g.minimum_delay=g.maximum_delay=1;
    g.row_offsets={0,1,1,1};
    g.target_indices={1}; g.weights={30.0f}; g.delays={1};
    g.validate(); return g;
}
struct Result { std::string name; std::uint64_t source=0,target=0; float vmax=-1000.0f; bool finite=true; };

Result run(const std::string& name,const std::filesystem::path& path,
           int mode,float learned_lag) {
    save_csr_topology(make_graph(),path.string());
    CudaSimulationRuntime runtime(make_model(),path.string(),0,true,false);
    std::filesystem::remove(path);
    if(mode!=0) {
        // The edge's learned preferred lag and the imposed source-target
        // phase difference must use the same reference. The CUDA gate tests
        // phase[source] - phase[target] - edge_phase_lag.
        const float source_phase = mode==1 ? learned_lag : learned_lag+kPi;
        runtime.configure_dynamic_phase_gate(
            {source_phase,0.0f,0.0f},{learned_lag},0.03f);
    }
    const auto start=runtime.completed_steps()+1U;
    runtime.schedule_stimulus(0,start,4,500.0f);
    Result r; r.name=name;
    for(std::uint32_t i=0;i<40;++i) {
        runtime.run_steps(1);
        const auto events=runtime.collect_spike_events();
        if(events.dropped_events) throw std::runtime_error("Spike events dropped");
        for(const auto& e:events.events) {
            if(e.neuron_id==0) ++r.source;
            if(e.neuron_id==1) ++r.target;
        }
        const auto v=runtime.collect_voltage_samples({1});
        if(v.size()!=1) throw std::runtime_error("Missing target voltage sample");
        r.finite=r.finite && std::isfinite(v[0].voltage_mv);
        r.vmax=std::max(r.vmax,v[0].voltage_mv);
    }
    r.finite=r.finite && runtime.collect_metrics().finite_state;
    runtime.collect_connectivity().validate();
    return r;
}
}

int main(int argc,char** argv) {
    try {
        const std::filesystem::path dir=argc>1?argv[1]:"experiments/logs/v42_learned_phase_cuda";
        std::filesystem::create_directories(dir);
        const float lag=train_and_retrieve_lag();
        const auto open=run("ungated",dir/"ungated.csr",0,lag);
        const auto learned=run("learned_lag_aligned",dir/"learned.csr",1,lag);
        const auto wrong=run("learned_lag_misaligned",dir/"misaligned.csr",2,lag);
        std::cout << "V42 LEARNED PHASE TO CUDA TRANSFER TEST\n";
        for(const auto* r:{&open,&learned,&wrong})
            std::cout << "condition="<<r->name<<" source_spikes="<<r->source
                      <<" target_spikes="<<r->target<<" target_v_max_mv="<<r->vmax
                      <<" finite="<<(r->finite?"PASS":"FAIL")<<'\n';
        const bool matched=open.source==learned.source && learned.source==wrong.source;
        const bool learned_pass=learned.target==open.target &&
            std::fabs(learned.vmax-open.vmax)<1.0e-4f;
        const bool wrong_suppressed=wrong.target<learned.target ||
            wrong.vmax<learned.vmax-1.0f;
        std::cout<<"matched_source_spikes="<<(matched?"PASS":"FAIL")<<'\n'
                 <<"learned_lag_restores_transmission="<<(learned_pass?"PASS":"FAIL")<<'\n'
                 <<"wrong_lag_suppresses_transmission="<<(wrong_suppressed?"PASS":"FAIL")<<'\n'
                 <<"interpretation=learned host-controller lag is applied to CUDA phase gate; "
                   "phase state is configured for transfer test, not autonomously evolved on GPU\n";
        return matched&&learned_pass&&wrong_suppressed&&open.finite&&learned.finite&&wrong.finite?0:2;
    } catch(const std::exception& e) {
        std::cerr<<"V42 learned-phase CUDA transfer failed: "<<e.what()<<'\n';
        return 1;
    }
}
