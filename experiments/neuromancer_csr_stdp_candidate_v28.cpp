#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_generator.h"
#include "topology/topology_io.h"
#include "plasticity/stdp.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
using json = nlohmann::json;
constexpr std::uint32_t E=1024, I=256, N=1280, INPUTS=64, OA=64, OB=65;
constexpr std::uint32_t STEPS=100, PULSE=4, TEACHER_LAG=3;
constexpr float CURRENT=500.0f, DT=0.5f;

ModelParameters model_for(const json& c, std::uint32_t seed) {
    ModelParameters p;
    p.simulation.excitatory_count=E; p.simulation.inhibitory_count=I;
    p.simulation.steps=STEPS; p.simulation.dt=DT; p.simulation.threshold=30.0f;
    p.simulation.random_seed=seed; p.simulation.initial_voltage_min=-65.0f;
    p.simulation.initial_voltage_max=-65.0f;
    p.simulation.excitatory_current_min=0; p.simulation.excitatory_current_max=0;
    p.simulation.inhibitory_current_min=0; p.simulation.inhibitory_current_max=0;
    auto& t=p.topology;
    t.p_ee=c.at("p_ee"); t.p_ei=c.at("p_ei"); t.p_ie=c.at("p_ie"); t.p_ii=c.at("p_ii");
    t.topology_seed=c.at("seed"); t.master_seed=c.at("seed"); t.delay_seed=c.at("seed").get<std::uint64_t>()+1;
    t.minimum_delay=1; t.maximum_delay=c.at("max_delay");
    t.delay_distribution=DelayDistribution::Uniform; t.memory_budget_bytes=536870912ULL;
    t.fibonacci_delays={1}; t.fibonacci_weights={1.0};
    return p;
}
CSRTopology make_topology(const ModelParameters& p, const json& c) {
    TopologyGeneratorOptions o;
    o.minimum_weight=c.at("weight_min"); o.maximum_weight=c.at("weight_max");
    auto g=generate_random_sparse_topology(p.simulation,p.topology,o);
    // Ensure the benchmark's 64 input features have explicit candidate edges
    // to both readout neurons, as in the V26 assay.
    std::vector<std::vector<std::pair<std::uint32_t,std::pair<float,std::uint32_t>>>> rows(N);
    for(std::uint32_t s=0;s<N;++s)
      for(std::uint32_t e=g.row_offsets[s];e<g.row_offsets[s+1];++e)
        rows[s].push_back({g.target_indices[e],{g.weights[e],g.delays[e]}});
    for(std::uint32_t s=0;s<INPUTS;++s) for(auto target:{OA,OB}) {
      auto& row=rows[s]; auto it=std::lower_bound(row.begin(),row.end(),target,
        [](const auto& x,std::uint32_t y){return x.first<y;});
      if(it!=row.end() && it->first==target) it->second.first=0.02f;
      else row.insert(it,{target,{0.02f,1U}});
    }
    g.row_offsets.assign(N+1,0); g.target_indices.clear(); g.weights.clear(); g.delays.clear();
    for(std::uint32_t s=0;s<N;++s) {
      for(const auto& x:rows[s]) {g.target_indices.push_back(x.first);g.weights.push_back(x.second.first);g.delays.push_back(x.second.second);}
      g.row_offsets[s+1]=static_cast<std::uint32_t>(g.weights.size());
    }
    g.validate(); return g;
}
struct Example { std::uint32_t label; std::array<std::uint8_t,INPUTS> x{}; };
Example example(std::uint32_t label,std::uint32_t sample,std::uint32_t seed) {
    Example a; a.label=label;
    for(std::uint32_t j=0;j<32;++j) a.x[(label?32:0)+j]=1;
    std::mt19937 r(seed+label*1009U+sample*9176U);
    std::uniform_int_distribution<std::uint32_t>d(0,31);
    auto d1=d(r),d2=d(r); while(d2==d1)d2=d(r);
    auto n1=d(r),n2=d(r); while(n2==n1)n2=d(r);
    a.x[(label?32:0)+d1]=0; a.x[(label?32:0)+d2]=0;
    a.x[(label?0:32)+n1]=1; a.x[(label?0:32)+n2]=1;
    return a;
}
void stimulate(CudaSimulationRuntime& r,const Example& ex,bool teacher) {
    auto base=r.completed_steps();
    for(std::uint32_t i=0;i<INPUTS;++i) if(ex.x[i])
      r.schedule_stimulus(i,base+1,PULSE,CURRENT);
    if(teacher) r.schedule_stimulus(ex.label?OB:OA,base+1+TEACHER_LAG,PULSE,CURRENT);
    r.run_steps(STEPS); r.clear_stimuli();
}
struct Score { std::uint32_t correct=0,total=0,c0=0,c1=0; std::uint64_t e=0,i=0,out=0,dropped=0; };
Score evaluate(const std::string& path,const ModelParameters& p,std::uint32_t seed) {
    Score s;
    for(std::uint32_t label=0;label<2;++label) for(std::uint32_t j=16;j<32;++j) {
      CudaSimulationRuntime r(p,path,0,true,false);
      auto ex=example(label,j,seed); stimulate(r,ex,false);
      auto ev=r.collect_spike_events(); std::uint64_t a=0,b=0;
      for(const auto& x:ev.events) { if(x.neuron_id<INPUTS)++s.e; else if(x.neuron_id==OA){++a;++s.out;} else if(x.neuron_id==OB){++b;++s.out;} else if(x.neuron_id>=E)++s.i; }
      s.dropped+=ev.dropped_events;
      if(a!=b && ((a>b?0U:1U)==label)){++s.correct;if(label==0)++s.c0;else ++s.c1;}
      ++s.total;
    }
    return s;
}
}
int main(int argc,char**argv) {
 try {
  if(argc!=3) throw std::runtime_error("usage: neuromancer_csr_stdp_candidate_v28 config.json result.json");
  std::ifstream in(argv[1]); json c; in>>c;
  const auto seed=c.at("seed").get<std::uint32_t>();
  auto p=model_for(c,seed); auto g=make_topology(p,c);
  std::filesystem::path result=argv[2]; std::filesystem::create_directories(result.parent_path());
  auto topology_path=(result.parent_path()/(result.stem().string()+".csr")).string();
  save_csr_topology(g,topology_path);
  StdpParameters stdp; stdp.potentiation_rate=c.at("a_plus"); stdp.depression_rate=c.at("a_minus");
  stdp.presynaptic_tau_ms=c.at("tau_pre"); stdp.postsynaptic_tau_ms=c.at("tau_post");
  stdp.minimum_weight=0.001f; stdp.maximum_weight=1.0f;
  auto before=evaluate(topology_path,p,seed);
  CudaSimulationRuntime train(p,topology_path,0,false,true,stdp);
  const std::string control=c.value("control", "candidate");\n  for(std::uint32_t epoch=0;epoch<(control=="untrained"?0U:3U);++epoch) {
    std::vector<Example> examples;
    for(std::uint32_t label=0;label<2;++label) for(std::uint32_t j=0;j<16;++j) examples.push_back(example(label,j,seed+17));
    std::mt19937 order(seed+epoch); std::shuffle(examples.begin(),examples.end(),order);
    for(const auto& ex:examples) { auto taught=ex; if(control=="shuffled_labels") taught.label=1U-taught.label; stimulate(train,taught,true); }
  }
  auto m=train.collect_metrics(); auto learned=train.collect_connectivity();
  save_csr_topology(learned,topology_path);
  auto after=evaluate(topology_path,p,seed);
  double meanw=0; for(float w:learned.weights)meanw+=std::abs(w);
  if(!learned.weights.empty())meanw/=learned.weights.size();
  json out={{"control",control},{"seed",seed},{"synapses",learned.synapse_count()},{"e_spikes",m.excitatory_spikes},
    {"i_spikes",m.inhibitory_spikes},{"e_rate_hz",m.excitatory_rate_hz},{"i_rate_hz",m.inhibitory_rate_hz},
    {"mean_voltage_mv",m.mean_voltage_mv},{"finite_state",m.finite_state},
    {"baseline_accuracy",before.total?double(before.correct)/before.total:0.0},
    {"accuracy",after.total?double(after.correct)/after.total:0.0},
    {"class0_correct",after.c0},{"class1_correct",after.c1},{"output_spikes",after.out},
    {"dropped_events",after.dropped},{"mean_abs_weight",meanw},{"topology",topology_path},
    {"a_plus",stdp.potentiation_rate},{"a_minus",stdp.depression_rate},
    {"tau_pre",stdp.presynaptic_tau_ms},{"tau_post",stdp.postsynaptic_tau_ms},
    {"p_ee",p.topology.p_ee},{"p_ei",p.topology.p_ei},{"p_ie",p.topology.p_ie},{"p_ii",p.topology.p_ii}};
  std::ofstream o(argv[2]); o<<out.dump(2)<<'\n';
  std::cout<<out.dump()<<'\n';
  return (!m.finite_state||after.dropped)?2:0;
 } catch(const std::exception&e){std::cerr<<"V28 candidate error: "<<e.what()<<'\n';return 1;}
}