#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_generator.h"
#include "topology/topology_io.h"
#include "plasticity/stdp.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
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
    g.validate();
    const float ei_scale=c.value("ei_weight_scale",1.0f);
    if(ei_scale!=1.0f) for(std::uint32_t source=0;source<E;++source)
      for(std::uint32_t edge=g.row_offsets[source];edge<g.row_offsets[source+1];++edge)
        if(g.target_indices[edge]>=E) g.weights[edge]=std::min(1.0f,g.weights[edge]*ei_scale);
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

std::uint32_t edge_state(float w) {
    const float a=std::abs(w);
    return a<0.01f?0U:(a<0.10f?1U:2U);
}
std::uint64_t edge_key(std::uint32_t s,std::uint32_t t) {
    return (static_cast<std::uint64_t>(s)<<32U)|t;
}
json transition_matrix(const CSRTopology& before,const CSRTopology& after) {
    std::map<std::uint64_t,float> old_edges,new_edges;
    for(std::uint32_t s=0;s<N;++s) {
      for(std::uint32_t e=before.row_offsets[s];e<before.row_offsets[s+1];++e)
        old_edges[edge_key(s,before.target_indices[e])]=before.weights[e];
      for(std::uint32_t e=after.row_offsets[s];e<after.row_offsets[s+1];++e)
        new_edges[edge_key(s,after.target_indices[e])]=after.weights[e];
    }
    std::uint64_t m[3][3]={{0,0,0},{0,0,0},{0,0,0}};
    std::uint64_t born=0,dead=0;
    for(const auto& kv:old_edges) {
      auto it=new_edges.find(kv.first);
      if(it==new_edges.end()){++dead;continue;}
      ++m[edge_state(kv.second)][edge_state(it->second)];
    }
    for(const auto& kv:new_edges) if(old_edges.find(kv.first)==old_edges.end()) ++born;
    json matrix=json::array();
    for(int i=0;i<3;++i) matrix.push_back({m[i][0],m[i][1],m[i][2]});
    return {{"state_labels",{"weak","medium","strong"}},
            {"transition_counts",matrix},{"born",born},{"dead",dead}};
}
json structural_update(CSRTopology& g,float prune_below,float growth_fraction,
                       bool inhibitory_regulated,std::uint64_t inhibitory_spikes,
                       std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::vector<std::pair<std::uint32_t,std::pair<float,std::uint32_t>>>> rows(N);
    std::uint64_t pruned=0;
    std::set<std::uint64_t> occupied;
    for(std::uint32_t s=0;s<N;++s) for(std::uint32_t e=g.row_offsets[s];e<g.row_offsets[s+1];++e) {
      const auto t=g.target_indices[e]; const float w=g.weights[e];
      if(std::abs(w)<prune_below){++pruned;continue;}
      rows[s].push_back({t,{w,g.delays[e]}}); occupied.insert(edge_key(s,t));
    }
    std::uint64_t requested=static_cast<std::uint64_t>(std::llround(g.synapse_count()*growth_fraction));
    std::uint64_t grown=0;
    std::uniform_int_distribution<std::uint32_t> any_source(0,N-1), any_target(0,N-1);
    std::uniform_int_distribution<std::uint32_t> e_source(0,E-1), i_target(E,N-1);
    const bool recruit=inhibitory_regulated && inhibitory_spikes==0;
    if(recruit) requested=std::max<std::uint64_t>(requested,std::max<std::uint64_t>(8,I/8));
    const std::uint64_t cap=static_cast<std::uint64_t>(N)*N/4;
    for(std::uint64_t tries=0;grown<requested && tries<requested*100+1000 && occupied.size()<cap;++tries) {
      const std::uint32_t source=recruit?e_source(rng):any_source(rng);
      const std::uint32_t target=recruit?i_target(rng):any_target(rng);
      if(source==target)continue;
      const auto key=edge_key(source,target);
      if(!occupied.insert(key).second)continue;
      const float magnitude=0.01f;
      const float weight=source<E?magnitude:-magnitude;
      rows[source].push_back({target,{weight,1U}}); ++grown;
    }
    g.row_offsets.assign(N+1,0);g.target_indices.clear();g.weights.clear();g.delays.clear();
    for(std::uint32_t s=0;s<N;++s) {
      std::sort(rows[s].begin(),rows[s].end(),[](const auto&a,const auto&b){return a.first<b.first;});
      for(const auto& x:rows[s]){g.target_indices.push_back(x.first);g.weights.push_back(x.second.first);g.delays.push_back(x.second.second);}
      g.row_offsets[s+1]=static_cast<std::uint32_t>(g.weights.size());
    }
    g.validate();
    return {{"pruned",pruned},{"grown",grown},{"synapses_after",g.synapse_count()},
            {"inhibitory_recruitment_triggered",recruit}};
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
  const std::string control=c.value("control", "candidate");
  const std::string architecture=c.value("architecture","fixed_baseline");
  const bool structural=(architecture=="structural_growth" || architecture=="inhibitory_regulated");
  const float prune_below=c.value("prune_below",0.0025f);
  const float growth_fraction=c.value("growth_fraction",0.01f);
  json epoch_history=json::array();
  SimulationMetrics aggregate{};
  CSRTopology previous=load_csr_topology(topology_path);
  std::uint64_t total_pruned=0,total_grown=0;
  for(std::uint32_t epoch=0;epoch<(control=="untrained"?0U:3U);++epoch) {
    CudaSimulationRuntime train(p,topology_path,0,false,true,stdp);
    std::vector<Example> examples;
    for(std::uint32_t label=0;label<2;++label) for(std::uint32_t j=0;j<16;++j) examples.push_back(example(label,j,seed+17));
    std::mt19937 order(seed+epoch); std::shuffle(examples.begin(),examples.end(),order);
    for(const auto& ex:examples) { auto taught=ex; if(control=="shuffled_labels") taught.label=1U-taught.label; stimulate(train,taught,true); }
    auto m_epoch=train.collect_metrics(); auto learned=train.collect_connectivity();
    aggregate.excitatory_spikes+=m_epoch.excitatory_spikes;
    aggregate.inhibitory_spikes+=m_epoch.inhibitory_spikes;
    aggregate.mean_voltage_mv=m_epoch.mean_voltage_mv;
    aggregate.excitatory_rate_hz=m_epoch.excitatory_rate_hz;
    aggregate.inhibitory_rate_hz=m_epoch.inhibitory_rate_hz;
    aggregate.finite_state=aggregate.finite_state&&m_epoch.finite_state;
    auto transitions=transition_matrix(previous,learned);
    json update={{"pruned",0},{"grown",0},{"synapses_after",learned.synapse_count()},
                {"inhibitory_recruitment_triggered",false}};
    if(structural && epoch<2) {
      update=structural_update(learned,prune_below,growth_fraction,
          architecture=="inhibitory_regulated",m_epoch.inhibitory_spikes,seed+epoch*7919U);
      total_pruned+=update["pruned"].get<std::uint64_t>();
      total_grown+=update["grown"].get<std::uint64_t>();
    }
    save_csr_topology(learned,topology_path);
    epoch_history.push_back({{"epoch",epoch+1},{"e_spikes",m_epoch.excitatory_spikes},
      {"i_spikes",m_epoch.inhibitory_spikes},{"synapses_before_update",previous.synapse_count()},
      {"synapses_after_stdp",transitions["born"].get<std::uint64_t>()+
        (previous.synapse_count()-transitions["dead"].get<std::uint64_t>())},
      {"edge_markov",transitions},{"structural_update",update}});
    previous=load_csr_topology(topology_path);
  }
  auto m=aggregate; auto learned=load_csr_topology(topology_path);
  auto after=evaluate(topology_path,p,seed);
  double meanw=0; for(float w:learned.weights)meanw+=std::abs(w);
  if(!learned.weights.empty())meanw/=learned.weights.size();
  const auto topology_stats=analyze_csr_topology(learned);
  json out={{"control",control},{"architecture",architecture},{"seed",seed},{"synapses",learned.synapse_count()},
    {"ee_synapses",topology_stats.excitatory_to_excitatory},
    {"ei_synapses",topology_stats.excitatory_to_inhibitory},
    {"ie_synapses",topology_stats.inhibitory_to_excitatory},
    {"ii_synapses",topology_stats.inhibitory_to_inhibitory},
    {"topology_density",topology_stats.density},{"e_spikes",m.excitatory_spikes},
    {"i_spikes",m.inhibitory_spikes},{"e_rate_hz",m.excitatory_rate_hz},{"i_rate_hz",m.inhibitory_rate_hz},
    {"mean_voltage_mv",m.mean_voltage_mv},{"finite_state",m.finite_state},
    {"baseline_accuracy",before.total?double(before.correct)/before.total:0.0},
    {"accuracy",after.total?double(after.correct)/after.total:0.0},
    {"class0_correct",after.c0},{"class1_correct",after.c1},{"output_spikes",after.out},
    {"dropped_events",after.dropped},{"mean_abs_weight",meanw},{"epoch_history",epoch_history},{"total_pruned",total_pruned},{"total_grown",total_grown},{"ei_weight_scale",c.value("ei_weight_scale",1.0f)},{"topology",topology_path},
    {"a_plus",stdp.potentiation_rate},{"a_minus",stdp.depression_rate},
    {"tau_pre",stdp.presynaptic_tau_ms},{"tau_post",stdp.postsynaptic_tau_ms},
    {"p_ee",p.topology.p_ee},{"p_ei",p.topology.p_ei},{"p_ie",p.topology.p_ie},{"p_ii",p.topology.p_ii}};
  std::ofstream o(argv[2]); o<<out.dump(2)<<'\n';
  std::cout<<out.dump()<<'\n';
  return (!m.finite_state||after.dropped)?2:0;
 } catch(const std::exception&e){std::cerr<<"V28 candidate error: "<<e.what()<<'\n';return 1;}
}