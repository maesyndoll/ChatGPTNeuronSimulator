#include "core/parameters.h"
#include "learning/phase_structural_plasticity.h"
#include "plasticity/stdp.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr std::uint32_t K=4, E=12, N=12;
constexpr float PI=3.14159265358979323846f, TAU=2.0f*PI;

std::uint32_t pop(std::uint32_t n){ return n/K; }
std::uint32_t mapped(std::uint32_t i,std::uint32_t seed,std::uint32_t hop){
    return hop*K+((i+seed*hop)%K);
}
CSRTopology random_graph(std::uint32_t seed){
    CSRTopology g; g.excitatory_count=E; g.inhibitory_count=0;
    g.minimum_delay=g.maximum_delay=1; g.row_offsets.assign(N+1,0);
    std::mt19937 rng(seed); std::bernoulli_distribution edge(.16);
    for(std::uint32_t s=0;s<N;++s){
        for(std::uint32_t t=0;t<N;++t) if(s!=t && edge(rng)){
            g.target_indices.push_back(t); g.weights.push_back(.08f); g.delays.push_back(1);
        }
        g.row_offsets[s+1]=static_cast<std::uint32_t>(g.target_indices.size());
    }
    g.validate(); return g;
}
std::uint32_t edge_between(const CSRTopology& g,std::uint32_t a,std::uint32_t b){
    std::uint32_t n=0;
    for(std::uint32_t s=a*K;s<(a+1)*K;++s)
        for(auto e=g.row_offsets[s];e<g.row_offsets[s+1];++e)
            if(pop(g.target_indices[e])==b) ++n;
    return n;
}
float mean_between(const CSRTopology& g,std::uint32_t a,std::uint32_t b){
    float sum=0; std::uint32_t n=0;
    for(std::uint32_t s=a*K;s<(a+1)*K;++s)
        for(auto e=g.row_offsets[s];e<g.row_offsets[s+1];++e)
            if(pop(g.target_indices[e])==b){sum+=g.weights[e];++n;}
    return n?sum/n:0.0f;
}
struct Metrics {
    std::uint32_t births=0,deaths=0,correct_births=0,incorrect_births=0;
    float ab_before=0,ab_after=0,bc_before=0,bc_after=0;
    float phase_lag=0,phase_conf=0,control_conf=0;
    std::uint32_t b_recall=0,c_recall=0,ac_recall=0;
    float precision=0,recall=0,jaccard=0;
    bool csr=true;
};
void pairwise_structure(CSRTopology& g,const std::vector<float>& pair_score,
                        std::vector<std::uint32_t>& ages,bool enabled,Metrics& m){
    if(!enabled) return;
    std::vector<std::vector<std::pair<std::uint32_t,float>>> rows(N);
    std::vector<std::vector<std::uint32_t>> next_age(N);
    for(std::uint32_t s=0;s<N;++s) for(auto e=g.row_offsets[s];e<g.row_offsets[s+1];++e){
        auto t=g.target_indices[e]; float w=g.weights[e];
        float score=pair_score[s*N+t];
        if(score>.20f) w=std::min(.95f,w+.08f*score);
        else w=std::max(0.0f,w-.035f);
        if(w<.025f && ages[e]>=2){++m.deaths;continue;}
        w=std::max(w,0.001f);
        rows[s].push_back({t,w}); next_age[s].push_back(ages[e]+1);
    }
    // Pair-specific growth: only recurrently observed directed spike pairs qualify.
    for(std::uint32_t s=0;s<E;++s){
        std::vector<std::pair<float,std::uint32_t>> candidates;
        for(std::uint32_t t=0;t<E;++t) if(s!=t && pair_score[s*N+t]>.45f){
            bool exists=false; for(auto x:rows[s]) if(x.first==t){exists=true;break;}
            if(!exists)candidates.push_back({pair_score[s*N+t],t});
        }
        std::sort(candidates.begin(),candidates.end(),[](auto a,auto b){return a.first>b.first;});
        for(auto c:candidates){
            if(rows[s].size()>=6) break;
            rows[s].push_back({c.second,.12f}); next_age[s].push_back(0);
            ++m.births;
            if((pop(s)==0 && pop(c.second)==1)||(pop(s)==1 && pop(c.second)==2)) ++m.correct_births;
            else ++m.incorrect_births;
        }
    }
    g.row_offsets.assign(N+1,0); g.target_indices.clear();g.weights.clear();g.delays.clear();
    ages.clear();
    for(std::uint32_t s=0;s<N;++s){
        std::vector<std::size_t> order(rows[s].size());
        std::iota(order.begin(),order.end(),0U);
        std::sort(order.begin(),order.end(),[&](auto a,auto b){
            return rows[s][a].first<rows[s][b].first;
        });
        for(auto j:order){
            g.target_indices.push_back(rows[s][j].first);g.weights.push_back(rows[s][j].second);
            g.delays.push_back(1);ages.push_back(next_age[s][j]);
        }
        g.row_offsets[s+1]=static_cast<std::uint32_t>(g.target_indices.size());
    }
    g.validate();
}
struct Trial {
    std::uint32_t seed=0;
    Metrics m;
    std::uint32_t initial_edges=0,final_edges=0;
    float selective_ratio=0,weight_ab_delta=0,weight_bc_delta=0;
};
Trial run_reference(std::uint32_t seed,bool stdp_on,bool structure_on,bool phase_on,bool shuffled){
    Trial out;out.seed=seed;auto g=random_graph(seed);out.initial_edges=static_cast<std::uint32_t>(g.synapse_count());
    out.m.ab_before=mean_between(g,0,1);out.m.bc_before=mean_between(g,1,2);
    std::vector<float> scores(N*N,0); std::vector<std::uint32_t> ages(g.synapse_count(),0);
    std::vector<std::uint32_t> src(g.synapse_count()),dst(g.synapse_count());
    for(std::uint32_t s=0;s<N;++s)for(auto e=g.row_offsets[s];e<g.row_offsets[s+1];++e){src[e]=s;dst[e]=g.target_indices[e];}
    StdpTraces traces;traces.presynaptic.assign(N,0);traces.postsynaptic.assign(N,0);
    StdpParameters sp;sp.potentiation_rate=.045f;sp.depression_rate=.012f;sp.maximum_weight=.95f;
    AutonomousPhaseParameters pp;pp.dt_ms=.5f;pp.intrinsic_frequency_hz=8;pp.temporal_association_window_steps=20;
    pp.temporal_lag_learning_rate=.12f;pp.minimum_temporal_confidence=.20f;
    AutonomousPopulationPhase phase(3,pp);
    std::array<std::uint64_t,N> last{};last.fill(UINT64_MAX);
    std::vector<std::uint8_t> events(N,0);
    std::uint64_t step=0;
    for(int epoch=0;epoch<12;++epoch){
        for(std::uint32_t i=0;i<K;++i){
            std::uint32_t a=i,b=mapped(i,seed,1),c=mapped(i,seed,2);
            if(shuffled)b=mapped((i+1)%K,seed,1);
            std::array<std::pair<std::uint32_t,int>,3> sequence{{{a,0},{b,5},{c,10}}};
            for(auto item:sequence){
                for(int gap=0;gap<item.second-(step%1000==999?0:0);++gap){ // explicit silent event boundaries
                    std::fill(events.begin(),events.end(),0);
                    if(stdp_on)apply_pair_stdp_step(g.weights,src,dst,N,E,events,traces,.5f,sp);
                    phase.step({0,0,0},phase_on);++step;
                }
                std::fill(events.begin(),events.end(),0);events[item.first]=1;
                if(stdp_on)apply_pair_stdp_step(g.weights,src,dst,N,E,events,traces,.5f,sp);
                std::vector<std::uint32_t> ps(3,0);ps[pop(item.first)]=1;phase.step(ps,phase_on);
                if(last[item.first]!=UINT64_MAX){
                    for(std::uint32_t s=0;s<N;++s) if(last[s]!=UINT64_MAX && step-last[s]<=20 && s!=item.first && pop(s)+1==pop(item.first))
                        scores[s*N+item.first]=std::min(1.0f,scores[s*N+item.first]+.12f);
                }
                last[item.first]=step;++step;
            }
            for(int gap=0;gap<20;++gap){
                std::fill(events.begin(),events.end(),0);
                if(stdp_on)apply_pair_stdp_step(g.weights,src,dst,N,E,events,traces,.5f,sp);
                phase.step({0,0,0},phase_on);++step;
            }
        }
        for(auto& x:scores)x=std::min(1.0f,x);
        pairwise_structure(g,scores,ages,structure_on,out.m);
        src.resize(g.synapse_count());dst.resize(g.synapse_count());
        for(std::uint32_t s=0;s<N;++s)for(auto e=g.row_offsets[s];e<g.row_offsets[s+1];++e){src[e]=s;dst[e]=g.target_indices[e];}
    }
    out.m.ab_after=mean_between(g,0,1);out.m.bc_after=mean_between(g,1,2);
    out.m.phase_lag=phase.learned_lag(0,1);out.m.phase_conf=phase.temporal_confidence(0,1);
    out.m.control_conf=phase.temporal_confidence(0,2);
    out.final_edges=static_cast<std::uint32_t>(g.synapse_count());
    std::uint32_t ab=edge_between(g,0,1),wrong=edge_between(g,0,0)+edge_between(g,0,2);
    out.m.csr=true;
    // Frozen host recall from a 50% A cue; edge weights provide propagation.
    std::array<bool,N> active{};active[0]=active[1]=true;
    auto propagate=[&](std::uint32_t source_pop){
        std::array<float,N> drive{};
        for(std::uint32_t s=source_pop*K;s<(source_pop+1)*K;++s)if(active[s])
            for(auto e=g.row_offsets[s];e<g.row_offsets[s+1];++e)
                drive[g.target_indices[e]]+=g.weights[e];
        for(std::uint32_t t=0;t<N;++t)if(drive[t]>=.10f)active[t]=true;
    };
    propagate(0);for(std::uint32_t i=K;i<2*K;++i)if(active[i])++out.m.b_recall;
    propagate(1);for(std::uint32_t i=2*K;i<3*K;++i)if(active[i])++out.m.c_recall;
    std::array<bool,N> control{};control[0]=control[1]=true;
    for(std::uint32_t s=0;s<K;++s)if(control[s])
        for(auto e=g.row_offsets[s];e<g.row_offsets[s+1];++e)
            if(g.weights[e]>=.10f)control[g.target_indices[e]]=true;
    for(std::uint32_t i=2*K;i<3*K;++i)if(control[i])++out.m.ac_recall;
    out.m.precision=static_cast<float>(out.m.b_recall+out.m.c_recall)/(std::max(1u,out.m.b_recall+out.m.c_recall+out.m.ac_recall));
    out.m.recall=static_cast<float>(out.m.b_recall+out.m.c_recall)/8.0f;
    out.m.jaccard=static_cast<float>(out.m.b_recall+out.m.c_recall)/(8.0f+out.m.ac_recall);
    out.selective_ratio=static_cast<float>(out.m.correct_births)/(1.0f+out.m.incorrect_births);
    out.weight_ab_delta=out.m.ab_after-out.m.ab_before;out.weight_bc_delta=out.m.bc_after-out.m.bc_before;
    return out;
}

ModelParameters cuda_model(){
    ModelParameters p;p.simulation.excitatory_count=2;p.simulation.inhibitory_count=1;
    p.simulation.steps=1;p.simulation.dt=.5f;p.simulation.threshold=30;
    p.simulation.random_seed=43001;p.simulation.initial_voltage_min=-65;p.simulation.initial_voltage_max=-65;
    p.simulation.excitatory_current_min=0;p.simulation.excitatory_current_max=0;
    p.simulation.inhibitory_current_min=0;p.simulation.inhibitory_current_max=0;
    p.topology.p_ee=p.topology.p_ei=p.topology.p_ie=p.topology.p_ii=0;
    p.topology.minimum_delay=p.topology.maximum_delay=1;p.topology.fibonacci_delays={1};p.topology.fibonacci_weights={1};
    return p;
}
bool cuda_closed_loop(const std::filesystem::path& dir){
    CSRTopology g;g.excitatory_count=2;g.inhibitory_count=1;g.minimum_delay=g.maximum_delay=1;
    g.row_offsets={0,1,1,1};g.target_indices={1};g.weights={30};g.delays={1};g.validate();
    auto path=dir/"v43_phase_loop.csr";save_csr_topology(g,path.string());
    CudaSimulationRuntime rt(cuda_model(),path.string(),0,true,false);
    std::filesystem::remove(path);
    AutonomousPhaseParameters p;p.dt_ms=.5f;p.intrinsic_frequency_hz=8;
    p.temporal_association_window_steps=20;p.temporal_lag_learning_rate=.12f;
    AutonomousPopulationPhase phase(3,p);
    std::vector<float> phases(3,0),lags(1,0);
    rt.configure_dynamic_phase_gate(phases,lags,.35f);
    std::array<std::uint64_t,3> observed{};
    for(std::uint32_t cycle=0;cycle<16;++cycle){
        for(int t=0;t<12;++t){
            if(t==0)rt.schedule_stimulus(0,rt.completed_steps()+1,1,500);
            rt.run_steps(1);auto batch=rt.collect_spike_events();
            if(batch.dropped_events)throw std::runtime_error("CUDA phase loop dropped spike events");
            std::vector<std::uint32_t> ps(3,0);
            for(const auto& e:batch.events){if(e.neuron_id<2){++observed[e.neuron_id];ps[e.neuron_id]=1;}}
            phase.step(ps,true);
            for(int n=0;n<3;++n)phases[n]=phase.phase(n);
            lags[0]=phase.temporal_confidence(0,1)>.2f?phase.learned_lag(0,1):0.0f;
            rt.update_dynamic_phase_state(phases,lags);
        }
        rt.clear_stimuli();
    }
    const auto confidence=phase.temporal_confidence(0,1);
    const auto lag=phase.learned_lag(0,1);
    auto metrics=rt.collect_metrics();rt.collect_connectivity().validate();
    std::cout<<"cuda_phase_source_events="<<observed[0]<<" cuda_phase_target_events="<<observed[1]
             <<" cuda_learned_lag="<<lag<<" cuda_temporal_confidence="<<confidence
             <<" cuda_finite="<<(metrics.finite_state?"PASS":"FAIL")<<'\n';
    return observed[0]>0&&observed[1]>0&&confidence>.2f&&std::isfinite(lag)&&metrics.finite_state;
}
}

int main(int argc,char**argv){
 try{
    const std::filesystem::path dir=argc>1?argv[1]:"experiments/logs/v43_integrated_learning";
    std::filesystem::create_directories(dir);
    std::ofstream trials(dir/"v43_trials.csv"),phase_csv(dir/"v43_phase.csv"),
        structure(dir/"v43_structure.csv"),recall(dir/"v43_recall.csv"),ablation(dir/"v43_ablation.csv");
    if(!trials||!phase_csv||!structure||!recall||!ablation)throw std::runtime_error("Cannot create V43 CSV artifacts");
    trials<<"seed,condition,initial_edges,final_edges,ab_weight_delta,bc_weight_delta,phase_lag,phase_confidence,control_confidence,b_recall,c_recall,ac_control,precision,recall,jaccard,csr_valid\n";
    phase_csv<<"seed,phase_lag,phase_confidence,control_confidence\n";
    structure<<"seed,births,deaths,correct_births,incorrect_births,selectivity_ratio,initial_edges,final_edges\n";
    recall<<"seed,b_recalled,c_recalled,ac_control,precision,recall,jaccard\n";
    ablation<<"condition,mean_ab_delta,mean_bc_delta,mean_phase_confidence,mean_births,mean_deaths,mean_recall\n";
    std::array<std::string,5> names{"full","no_stdp","no_structure","no_phase_learning","shuffled"};
    for(std::size_t ci=0;ci<names.size();++ci){
        double ab=0,bc=0,conf=0,birth=0,death=0,ret=0;
        for(std::uint32_t seed=1;seed<=10;++seed){
            auto t=run_reference(seed,ci!=1,ci!=2,ci!=3,ci==4);
            auto&m=t.m;ab+=t.weight_ab_delta;bc+=t.weight_bc_delta;conf+=m.phase_conf;
            birth+=m.births;death+=m.deaths;ret+=m.b_recall+m.c_recall;
            trials<<seed<<','<<names[ci]<<','<<t.initial_edges<<','<<t.final_edges<<','
                <<t.weight_ab_delta<<','<<t.weight_bc_delta<<','<<m.phase_lag<<','
                <<m.phase_conf<<','<<m.control_conf<<','<<m.b_recall<<','<<m.c_recall<<','
                <<m.ac_recall<<','<<m.precision<<','<<m.recall<<','<<m.jaccard<<','<<m.csr<<'\n';
            if(ci==0){
                phase_csv<<seed<<','<<m.phase_lag<<','<<m.phase_conf<<','<<m.control_conf<<'\n';
                structure<<seed<<','<<m.births<<','<<m.deaths<<','<<m.correct_births<<','
                    <<m.incorrect_births<<','<<t.selective_ratio<<','<<t.initial_edges<<','<<m.final_edges<<'\n';
                recall<<seed<<','<<m.b_recall<<','<<m.c_recall<<','<<m.ac_recall<<','
                    <<m.precision<<','<<m.recall<<','<<m.jaccard<<'\n';
            }
        }
        ablation<<names[ci]<<','<<ab/10<<','<<bc/10<<','<<conf/10<<','
            <<birth/10<<','<<death/10<<','<<ret/80<<'\n';
        std::cout<<"condition="<<names[ci]<<" mean_AB_delta="<<ab/10
                 <<" mean_BC_delta="<<bc/10<<" phase_confidence="<<conf/10
                 <<" mean_births="<<birth/10<<" mean_deaths="<<death/10
                 <<" mean_recalled_units="<<ret/10<<'\n';
    }
    const bool cuda_ok=cuda_closed_loop(dir);
    std::ofstream summary(dir/"v43_summary.json");
    summary<<"{\n  \"experiment\":\"Neuromancer V43 Integrated Autonomous Learning\",\n"
           <<"  \"seeds\":10,\n  \"conditions\":5,\n"
           <<"  \"cuda_event_driven_phase_feedback\":"<<(cuda_ok?"true":"false")<<",\n"
           <<"  \"scope\":\"synthetic event-driven host associative reference plus live CUDA spike-driven phase feedback; CUDA associative recall and fully GPU-resident phase dynamics are not claimed\"\n}\n";
    std::cout<<"cuda_event_driven_phase_feedback="<<(cuda_ok?"PASS":"FAIL")<<'\n'
             <<"artifacts="<<dir.string()<<'\n'
             <<"LIMITATION=host reference performs association/structural/recall scoring; CUDA subtest closes spike-event-to-phase-to-synaptic-gate feedback only.\n";
    return cuda_ok?0:2;
 }catch(const std::exception&e){std::cerr<<"V43 integrated experiment failed: "<<e.what()<<'\n';return 1;}
}
