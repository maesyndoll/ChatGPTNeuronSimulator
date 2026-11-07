#include "learning/phase_structural_plasticity.h"
#include "plasticity/stdp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace spiking;
constexpr std::uint32_t kAssembly = 16;
constexpr std::uint32_t kE = 3 * kAssembly;
constexpr std::uint32_t kN = kE + 1;
constexpr float kTau = 6.2831853071795864769f;

CSRTopology make_random_graph(std::uint32_t seed) {
    CSRTopology g;
    g.excitatory_count = kE;
    g.inhibitory_count = 1;
    g.minimum_delay = g.maximum_delay = 1;
    g.row_offsets.assign(kN + 1, 0);
    std::mt19937 rng(seed);
    std::bernoulli_distribution edge(0.12);
    for (std::uint32_t s = 0; s < kE; ++s) {
        for (std::uint32_t t = 0; t < kE; ++t) {
            if (s != t && edge(rng)) {
                g.target_indices.push_back(t);
                g.weights.push_back(0.005f);
                g.delays.push_back(1);
            }
        }
        g.row_offsets[s + 1] = static_cast<std::uint32_t>(g.target_indices.size());
    }
    g.row_offsets[kN] = static_cast<std::uint32_t>(g.target_indices.size());
    g.validate();
    return g;
}

std::uint32_t mapped(std::uint32_t i, std::uint32_t seed, std::uint32_t hop) {
    return hop * kAssembly + ((i + seed * hop) % kAssembly);
}

bool has_edge(const CSRTopology& g, std::uint32_t s, std::uint32_t t) {
    for (auto e = g.row_offsets[s]; e < g.row_offsets[s + 1]; ++e)
        if (g.target_indices[e] == t) return true;
    return false;
}

std::uint32_t edge_count_between(const CSRTopology& g, std::uint32_t sb,
                                 std::uint32_t tb) {
    std::uint32_t n = 0;
    for (std::uint32_t s = sb; s < sb + kAssembly; ++s)
        for (auto e = g.row_offsets[s]; e < g.row_offsets[s + 1]; ++e)
            if (g.target_indices[e] >= tb &&
                g.target_indices[e] < tb + kAssembly) ++n;
    return n;
}

float mean_between(const CSRTopology& g, std::uint32_t sb, std::uint32_t tb) {
    double sum = 0.0;
    std::uint32_t n = 0;
    for (std::uint32_t s = sb; s < sb + kAssembly; ++s)
        for (auto e = g.row_offsets[s]; e < g.row_offsets[s + 1]; ++e)
            if (g.target_indices[e] >= tb &&
                g.target_indices[e] < tb + kAssembly) {
                sum += g.weights[e]; ++n;
            }
    return n ? static_cast<float>(sum / n) : 0.0f;
}

void update_stdp(CSRTopology& g, StdpTraces& traces,
                 std::uint32_t neuron, float dt, const StdpParameters& p) {
    std::vector<std::uint32_t> src, dst;
    src.reserve(g.synapse_count()); dst.reserve(g.synapse_count());
    for (std::uint32_t s = 0; s < kN; ++s)
        for (auto e = g.row_offsets[s]; e < g.row_offsets[s + 1]; ++e) {
            src.push_back(s); dst.push_back(g.target_indices[e]);
        }
    std::vector<std::uint8_t> spikes(kN, 0);
    spikes[neuron] = 1;
    apply_pair_stdp_step(g.weights, src, dst, kN, kE, spikes, traces, dt, p);
}

struct Result {
    float ab_before=0, ab_after=0, bc_before=0, bc_after=0, ac_after=0;
    std::uint32_t ab_candidates=0, bc_candidates=0;
    std::uint32_t births=0, deaths=0;
    float ab_lag=0, ac_lag=0, ab_conf=0, ac_conf=0;
    std::uint32_t ab_recalled=0, bc_recalled=0, ac_control=0;
    std::uint32_t initial_edges=0, final_edges=0;
    bool csr=false;
};

std::vector<std::uint32_t> recall(const CSRTopology& g,
                                  const std::vector<std::uint32_t>& cue,
                                  std::uint32_t target_begin, float threshold) {
    std::vector<std::uint32_t> out;
    for (std::uint32_t t=target_begin; t<target_begin+kAssembly; ++t) {
        float input=0;
        for (auto s:cue)
            for (auto e=g.row_offsets[s]; e<g.row_offsets[s+1]; ++e)
                if (g.target_indices[e]==t) input += g.weights[e];
        if (input >= threshold) out.push_back(t);
    }
    return out;
}

Result run_seed(std::uint32_t seed) {
    auto g=make_random_graph(seed);
    Result r;
    r.initial_edges=static_cast<std::uint32_t>(g.synapse_count());
    r.ab_candidates=edge_count_between(g,0,kAssembly);
    r.bc_candidates=edge_count_between(g,kAssembly,2*kAssembly);
    r.ab_before=mean_between(g,0,kAssembly);
    r.bc_before=mean_between(g,kAssembly,2*kAssembly);
    const auto ac_before=mean_between(g,0,2*kAssembly);

    std::vector<std::uint32_t> src,dst;
    for (std::uint32_t s=0;s<kN;++s)
        for (auto e=g.row_offsets[s];e<g.row_offsets[s+1];++e) {
            src.push_back(s); dst.push_back(g.target_indices[e]);
        }
    StdpTraces traces;
    traces.presynaptic.assign(kN,0.0f);
    traces.postsynaptic.assign(kN,0.0f);
    StdpParameters sp;
    sp.potentiation_rate=0.035f;
    sp.depression_rate=0.002f;
    sp.presynaptic_tau_ms=20.0f;
    sp.postsynaptic_tau_ms=20.0f;
    sp.minimum_weight=0.001f;
    sp.maximum_weight=0.25f;

    AutonomousPhaseParameters pp;
    pp.intrinsic_frequency_hz=8.0f;
    pp.temporal_association_window_steps=39;
    pp.temporal_lag_learning_rate=0.10f;
    pp.minimum_temporal_confidence=0.20f;
    AutonomousPopulationPhase phase(3,pp);

    // Interleave paired events. Association labels determine stimuli only;
    // the initial graph is a global random graph and contains no designed blocks.
    for (int epoch=0;epoch<12;++epoch) {
        for (std::uint32_t i=0;i<kAssembly;++i) {
            const auto a=i;
            const auto b=mapped(i,seed,1);
            const auto c=mapped(i,seed,2);
            std::vector<std::uint32_t> ps(3,0);
            ps[0]=1; phase.step(ps,true);
            update_stdp(g,traces,a,0.5f,sp);
            for(int gap=0;gap<5;++gap) {
                std::fill(ps.begin(),ps.end(),0); phase.step(ps,true);
                update_stdp(g,traces,kN-1,0.5f,sp);
            }
            std::fill(ps.begin(),ps.end(),0); ps[1]=1; phase.step(ps,true);
            update_stdp(g,traces,b,0.5f,sp);
            for(int gap=0;gap<80;++gap) {
                std::fill(ps.begin(),ps.end(),0); phase.step(ps,true);
                update_stdp(g,traces,kN-1,0.5f,sp);
            }
            std::fill(ps.begin(),ps.end(),0); ps[1]=1; phase.step(ps,true);
            update_stdp(g,traces,b,0.5f,sp);
            for(int gap=0;gap<5;++gap) {
                std::fill(ps.begin(),ps.end(),0); phase.step(ps,true);
                update_stdp(g,traces,kN-1,0.5f,sp);
            }
            std::fill(ps.begin(),ps.end(),0); ps[2]=1; phase.step(ps,true);
            update_stdp(g,traces,c,0.5f,sp);
            for(int gap=0;gap<80;++gap) {
                std::fill(ps.begin(),ps.end(),0); phase.step(ps,true);
                update_stdp(g,traces,kN-1,0.5f,sp);
            }
        }
    }

    r.ab_after=mean_between(g,0,kAssembly);
    r.bc_after=mean_between(g,kAssembly,2*kAssembly);
    r.ac_after=mean_between(g,0,2*kAssembly);
    r.ab_lag=phase.learned_lag(0,1);
    r.ac_lag=phase.learned_lag(0,2);
    r.ab_conf=phase.temporal_confidence(0,1);
    r.ac_conf=phase.temporal_confidence(0,2);

    StructuralPlasticityParameters st;
    st.potentiation=0.015f;
    st.depression=0.004f;
    st.growth_threshold=0.80f;
    st.prune_threshold=0.002f;
    st.minimum_epochs_before_prune=3;
    st.maximum_out_degree=24;
    st.maximum_weight=0.25f;
    st.resource_per_excitatory_neuron=1.0f;
    ActivityStructuralPlasticity structural(st);
    std::vector<std::uint32_t> ages(g.synapse_count(),0);
    // Repeated pair-specific activity epochs permit growth of correlated
    // source->target edges and pruning of inactive candidates.
    for(int epoch=0;epoch<6;++epoch) {
        for(std::uint32_t i=0;i<kAssembly;++i) {
            std::vector<float> activity(kN,0.0f);
            activity[i]=1.0f;
            activity[mapped(i,seed,1)]=1.0f;
            auto rep=structural.apply(g,activity,ages,true);
            r.births+=rep.births; r.deaths+=rep.deaths;
            ages=std::move(rep.next_edge_ages);
            std::fill(activity.begin(),activity.end(),0.0f);
            activity[mapped(i,seed,1)]=1.0f;
            activity[mapped(i,seed,2)]=1.0f;
            rep=structural.apply(g,activity,ages,true);
            r.births+=rep.births; r.deaths+=rep.deaths;
            ages=std::move(rep.next_edge_ages);
        }
    }

    // Frozen graph recall: cue two of sixteen A units; only graph weights
    // are used. This is a recruitment diagnostic, not yet CUDA recall.
    const std::vector<std::uint32_t> cue{0U,1U};
    const auto b=recall(g,cue,kAssembly,0.015f);
    const auto c=recall(g,b,2*kAssembly,0.015f);
    const auto ac=recall(g,cue,2*kAssembly,0.015f);
    r.ab_recalled=static_cast<std::uint32_t>(b.size());
    r.bc_recalled=static_cast<std::uint32_t>(c.size());
    r.ac_control=static_cast<std::uint32_t>(ac.size());
    r.final_edges=static_cast<std::uint32_t>(g.synapse_count());
    g.validate(); r.csr=true;
    (void)ac_before;
    return r;
}
}

int main(int argc,char** argv) {
    try {
        const std::filesystem::path out=argc>1?argv[1]:
            "experiments/logs/v42_unprewired_discovery";
        std::filesystem::create_directories(out);
        std::ofstream csv(out/"v42_unprewired_discovery.csv");
        if(!csv) throw std::runtime_error("Cannot create discovery CSV");
        csv<<"seed,initial_edges,final_edges,ab_initial_edges,bc_initial_edges,"
              "ab_weight_before,ab_weight_after,bc_weight_before,bc_weight_after,"
              "ac_control_weight,ab_lag,ac_lag,ab_confidence,ac_confidence,"
              "structural_births,structural_deaths,b_recalled_units,"
              "c_recalled_units,ac_control_units,csr_valid\n";
        double abw=0,bcw=0,abl=0,acl=0,abc=0,acc=0;
        std::uint64_t births=0,deaths=0;
        bool valid=true;
        constexpr int seeds=10;
        std::cout<<"V42 UNPREWIRED ASSOCIATION + PHASE + STRUCTURE DISCOVERY\n"
                 <<"scope=host-side random-CSR reference experiment; not CUDA-runtime recall\n"
                 <<"seeds="<<seeds<<" assemblies=3x16 cue=2/16\n";
        for(std::uint32_t seed=1;seed<=seeds;++seed) {
            const auto r=run_seed(seed);
            abw+=r.ab_after-r.ab_before; bcw+=r.bc_after-r.bc_before;
            abl+=r.ab_lag; acl+=r.ac_lag; abc+=r.ab_conf; acc+=r.ac_conf;
            births+=r.births; deaths+=r.deaths; valid=valid&&r.csr;
            csv<<seed<<','<<r.initial_edges<<','<<r.final_edges<<','
               <<r.ab_candidates<<','<<r.bc_candidates<<','
               <<r.ab_before<<','<<r.ab_after<<','<<r.bc_before<<','<<r.bc_after<<','
               <<r.ac_after<<','<<r.ab_lag<<','<<r.ac_lag<<','
               <<r.ab_conf<<','<<r.ac_conf<<','<<r.births<<','<<r.deaths<<','
               <<r.ab_recalled<<','<<r.bc_recalled<<','<<r.ac_control<<','
               <<(r.csr?1:0)<<'\n';
            std::cout<<"seed="<<seed<<" AB_delta="<<(r.ab_after-r.ab_before)
                     <<" BC_delta="<<(r.bc_after-r.bc_before)
                     <<" AB_conf="<<r.ab_conf<<" AC_conf="<<r.ac_conf
                     <<" births="<<r.births<<" deaths="<<r.deaths
                     <<" B_units="<<r.ab_recalled<<" C_units="<<r.bc_recalled
                     <<" AC_control_units="<<r.ac_control<<'\n';
        }
        csv.close();
        std::cout<<"mean_AB_weight_delta="<<abw/seeds
                 <<" mean_BC_weight_delta="<<bcw/seeds
                 <<" mean_AB_lag="<<abl/seeds<<" mean_AC_lag="<<acl/seeds
                 <<" mean_AB_confidence="<<abc/seeds
                 <<" mean_AC_confidence="<<acc/seeds
                 <<" total_births="<<births<<" total_deaths="<<deaths<<'\n'
                 <<"CSR_validation="<<(valid?"PASS":"FAIL")<<'\n'
                 <<"artifact="<<(out/"v42_unprewired_discovery.csv").string()<<'\n'
                 <<"LIMITATION=host-side reference only; phase lag is measured but not yet applied to CUDA propagation; structural changes are synaptic, not dendritic morphology.\n";
        return valid?0:2;
    } catch(const std::exception& e) {
        std::cerr<<"V42 unprewired discovery failed: "<<e.what()<<'\n';
        return 1;
    }
}
