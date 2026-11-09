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
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;
constexpr float PI = 3.14159265358979323846f;

struct Check {
    std::string name;
    bool pass = false;
    std::string detail;
};

void emit(const Check& c) {
    std::cout << c.name << '=' << (c.pass ? "PASS" : "FAIL")
              << " detail=" << c.detail << '\n';
}

// V51: identical pre/post events are delivered to the STDP reference in all
// conditions. A separate phase-compatibility decision commits or rejects the
// candidate weight update, leaving traces and event opportunities identical.
struct PhaseLearningResult {
    float weight = 0.0f;
    std::uint32_t pre_events = 0;
    std::uint32_t post_events = 0;
    std::uint32_t pair_opportunities = 0;
};

PhaseLearningResult run_phase_learning(bool phase_gate, bool aligned, bool plastic) {
    std::vector<float> weights{0.20f};
    const std::vector<std::uint32_t> sources{0}, targets{1};
    StdpTraces traces{{0.0f, 0.0f}, {0.0f, 0.0f}};
    StdpParameters p;
    p.potentiation_rate = 0.08f;
    p.depression_rate = 0.0f;
    p.presynaptic_tau_ms = 20.0f;
    p.postsynaptic_tau_ms = 20.0f;
    p.minimum_weight = 0.001f;
    p.maximum_weight = 1.0f;
    PhaseLearningResult out;
    constexpr std::uint32_t trials = 32;
    for (std::uint32_t trial = 0; trial < trials; ++trial) {
        for (std::uint32_t step = 0; step < 12; ++step) {
            std::vector<std::uint8_t> spikes(2, 0U);
            if (step == 0) { spikes[0] = 1U; ++out.pre_events; }
            if (step == 4) { spikes[1] = 1U; ++out.post_events; ++out.pair_opportunities; }
            auto candidate = weights;
            apply_pair_stdp_step(candidate, sources, targets, 2, 2,
                                 spikes, traces, 0.5f, p);
            const bool phase_compatible = !phase_gate || aligned;
            if (plastic && phase_compatible) weights.swap(candidate);
        }
    }
    out.weight = weights.front();
    return out;
}

Check run_v51() {
    const auto aligned = run_phase_learning(true, true, true);
    const auto misaligned = run_phase_learning(true, false, true);
    const auto ungated = run_phase_learning(false, false, true);
    const auto frozen = run_phase_learning(true, true, false);
    const bool matched = aligned.pre_events == misaligned.pre_events &&
        aligned.post_events == misaligned.post_events &&
        aligned.pair_opportunities == misaligned.pair_opportunities &&
        aligned.pair_opportunities == 32;
    const bool phase_selective = aligned.weight > 0.20f &&
        std::fabs(misaligned.weight - 0.20f) < 1.0e-7f;
    const bool ungated_control = ungated.weight > 0.20f;
    const bool frozen_control = std::fabs(frozen.weight - 0.20f) < 1.0e-7f;
    std::cout << "V51_DIRECT_PHASE_GATED_STDP_REFERENCE\n"
              << "aligned_weight=0.2->" << aligned.weight
              << " misaligned_weight=0.2->" << misaligned.weight
              << " ungated_weight=0.2->" << ungated.weight
              << " frozen_weight=0.2->" << frozen.weight
              << " matched_pair_opportunities=" << aligned.pair_opportunities
              << "/" << misaligned.pair_opportunities << '\n';
    return {"v51_direct_phase_gated_stdp",
            matched && phase_selective && ungated_control && frozen_control,
            "matched_pre_post_events=" + std::to_string(matched) +
            "; phase_selective=" + std::to_string(phase_selective) +
            "; ungated_control=" + std::to_string(ungated_control) +
            "; frozen_control=" + std::to_string(frozen_control) +
            "; host_reference_only"};
}

// V52: learn stable directed lag from live-like event sequences, reject a
// variable-delay control, then verify the learned lag remains frozen.
Check run_v52() {
    AutonomousPhaseParameters p;
    p.dt_ms = 0.5f;
    p.intrinsic_frequency_hz = 8.0f;
    p.activity_kick_rad = 0.04f;
    p.coupling_learning_rate = 0.001f;
    p.maximum_coupling = 0.5f;
    p.temporal_association_window_steps = 299;
    p.temporal_lag_learning_rate = 0.10f;
    p.minimum_temporal_confidence = 0.20f;
    AutonomousPopulationPhase phase(3, p);
    constexpr int period = 300;
    constexpr int train_steps = 6000;
    constexpr std::array<int, 8> variable{20,55,90,125,160,195,230,265};
    for (int t = 0; t < train_steps; ++t) {
        std::vector<std::uint32_t> spikes(3, 0U);
        const int within = t % period;
        const int cycle = t / period;
        if (within == 0) spikes[0] = 1U;
        if (within == 5) spikes[1] = 1U;
        if (within == variable[static_cast<std::size_t>(cycle) % variable.size()])
            spikes[2] = 1U;
        phase.step(spikes, true);
    }
    const float paired_confidence = phase.temporal_confidence(0, 1);
    const float control_confidence = phase.temporal_confidence(0, 2);
    const float learned_lag = phase.learned_lag(0, 1);
    const float learned_coupling = phase.coupling(0, 1);
    for (int t = 0; t < 2000; ++t) phase.step({0U,0U,0U}, false);
    const bool retained = std::fabs(std::remainder(
        phase.learned_lag(0,1) - learned_lag, 2.0f * PI)) < 1.0e-7f;
    const bool finite = std::isfinite(paired_confidence) &&
        std::isfinite(control_confidence) && std::isfinite(learned_coupling);
    const bool discriminates = paired_confidence > control_confidence + 0.20f;
    std::cout << "V52_AUTONOMOUS_PHASE_ADAPTATION\n"
              << "training_steps=" << train_steps
              << " learned_lag_rad=" << learned_lag
              << " paired_confidence=" << paired_confidence
              << " variable_control_confidence=" << control_confidence
              << " learned_coupling=" << learned_coupling
              << " lag_retained=" << (retained ? "YES" : "NO") << '\n';
    return {"v52_autonomous_phase_adaptation",
            finite && discriminates && retained,
            "fixed_delay_vs_variable_delay_temporal_association; host_controller"};
}

// V53: co-active groups should form/strengthen edges; inactive old weak edges
// should prune after the configured age. A disabled control must be identical.
CSRTopology structural_graph() {
    CSRTopology g;
    g.excitatory_count = 6;
    g.inhibitory_count = 1;
    g.minimum_delay = g.maximum_delay = 1;
    g.row_offsets = {0,1,2,3,4,5,6,6};
    g.target_indices = {1,0,3,2,5,4};
    g.weights = {0.08f,0.08f,0.08f,0.08f,0.001f,0.001f};
    g.delays.assign(6,1U);
    g.validate();
    return g;
}

Check run_v53() {
    auto graph = structural_graph();
    auto disabled_graph = graph;
    std::vector<std::uint32_t> ages(graph.synapse_count(), 4U);
    StructuralPlasticityParameters p;
    p.potentiation = 0.08f;
    p.depression = 0.02f;
    p.growth_threshold = 0.50f;
    p.prune_threshold = 0.005f;
    p.minimum_epochs_before_prune = 3;
    p.maximum_out_degree = 4;
    p.maximum_weight = 0.5f;
    p.resource_per_excitatory_neuron = 1.0f;
    ActivityStructuralPlasticity structural(p);
    std::vector<float> activity{0.95f,0.95f,0.95f,0.95f,0.0f,0.0f,0.0f};
    auto report = structural.apply(graph, activity, ages, true);
    const auto disabled = structural.apply(disabled_graph, activity, ages, false);
    const bool valid = graph.validate() == void() ;
    (void)valid;
    const bool births = report.births > 0;
    const bool deaths = report.deaths > 0;
    const bool strengthened = report.strengthened > 0;
    const bool disabled_unchanged = disabled.births == 0 && disabled.deaths == 0 &&
        disabled_graph.target_indices == structural_graph().target_indices &&
        disabled_graph.weights == structural_graph().weights;
    std::cout << "V53_ACTIVITY_STRUCTURAL_PLASTICITY\n"
              << "births=" << report.births << " deaths=" << report.deaths
              << " strengthened=" << report.strengthened
              << " weakened=" << report.weakened
              << " edges_after=" << graph.synapse_count()
              << " disabled_control_unchanged=" << (disabled_unchanged ? "YES" : "NO") << '\n';
    return {"v53_activity_structural_plasticity",
            births && deaths && strengthened && disabled_unchanged,
            "CSR_topology_remodeling; local_synaptic_structure_not_dendrite_morphology"};
}

ModelParameters inhibition_model() {
    ModelParameters p;
    p.simulation.excitatory_count = 16;
    p.simulation.inhibitory_count = 1;
    p.simulation.steps = 1;
    p.simulation.dt = 0.5f;
    p.simulation.threshold = 30.0f;
    p.simulation.random_seed = 54001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = p.simulation.inhibitory_current_max = 0.0f;
    p.topology.minimum_delay = p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

CSRTopology inhibition_graph(bool add_inhibition) {
    constexpr std::uint32_t E = 16, N = 17;
    CSRTopology g;
    g.excitatory_count = E;
    g.inhibitory_count = 1;
    g.minimum_delay = g.maximum_delay = 1;
    g.row_offsets.assign(N + 1,0U);
    for (std::uint32_t s=0;s<N;++s) {
        if (s<4) {
            g.target_indices.push_back(8+s);
            g.weights.push_back(150.0f);
            g.delays.push_back(1U);
        }
        if (add_inhibition && s==E) {
            for (std::uint32_t t=8;t<12;++t) {
                g.target_indices.push_back(t);
                g.weights.push_back(-220.0f);
                g.delays.push_back(1U);
            }
        }
        g.row_offsets[s+1]=static_cast<std::uint32_t>(g.target_indices.size());
    }
    g.validate();
    return g;
}

struct InhibitionResult {
    std::string name;
    std::uint64_t excitatory_spikes=0, inhibitory_spikes=0, target_spikes=0, dropped=0;
    std::uint32_t target_unique=0;
    bool finite=true;
};

InhibitionResult run_inhibition(const std::filesystem::path& dir,
                                const std::string& name,
                                bool include_excitation, bool include_inhibition) {
    const auto path=dir/(name+".csr");
    auto graph=inhibition_graph(include_inhibition);
    if (!include_excitation) {
        graph.target_indices.clear(); graph.weights.clear(); graph.delays.clear();
        std::fill(graph.row_offsets.begin(),graph.row_offsets.end(),0U);
    }
    save_csr_topology(graph,path.string());
    CudaSimulationRuntime rt(inhibition_model(),path.string(),0,true,false);
    std::filesystem::remove(path);
    InhibitionResult r; r.name=name;
    const auto start=rt.completed_steps()+1U;
    if (include_excitation)
        for (std::uint32_t i=0;i<4;++i) rt.schedule_stimulus(i,start,4,500.0f);
    if (include_inhibition)
        rt.schedule_stimulus(16,start,4,500.0f);
    std::array<bool,4> seen{};
    for (std::uint32_t step=0;step<48;++step) {
        rt.run_steps(1);
        auto events=rt.collect_spike_events();
        r.dropped+=events.dropped_events;
        for (const auto& e:events.events) {
            if (e.neuron_id<16) ++r.excitatory_spikes;
            else ++r.inhibitory_spikes;
            if (e.neuron_id>=8 && e.neuron_id<12) {
                ++r.target_spikes; seen[e.neuron_id-8]=true;
            }
        }
    }
    r.target_unique=static_cast<std::uint32_t>(std::count(seen.begin(),seen.end(),true));
    r.finite=rt.collect_metrics().finite_state;
    rt.collect_connectivity().validate();
    return r;
}

Check run_v54(const std::filesystem::path& dir) {
    const auto excitation=run_inhibition(dir,"excitation_only",true,false);
    const auto competition=run_inhibition(dir,"excitation_plus_inhibition",true,true);
    const auto inhibition_only=run_inhibition(dir,"inhibition_only",false,true);
    auto print=[](const InhibitionResult& r) {
        std::cout<<"condition="<<r.name
                 <<" excitatory_spikes="<<r.excitatory_spikes
                 <<" inhibitory_spikes="<<r.inhibitory_spikes
                 <<" target_spikes="<<r.target_spikes
                 <<" target_unique="<<r.target_unique<<"/4"
                 <<" dropped="<<r.dropped
                 <<" finite="<<(r.finite?"PASS":"FAIL")<<'\n';
    };
    std::cout<<"V54_INHIBITORY_COMPETITION\n";
    print(excitation); print(competition); print(inhibition_only);
    const bool excitation_recruits=excitation.target_spikes>0;
    const bool inhibition_reduces=competition.inhibitory_spikes>0 &&
        competition.target_spikes<excitation.target_spikes;
    const bool inhibition_negative=inhibition_only.target_spikes==0;
    const bool safe=excitation.finite&&competition.finite&&inhibition_only.finite&&
        excitation.dropped==0&&competition.dropped==0&&inhibition_only.dropped==0;
    return {"v54_inhibitory_competition",
            excitation_recruits&&inhibition_reduces&&inhibition_negative&&safe,
            "CUDA_inhibitory_transmission_and_recruitment_competition"};
}
}

int main(int argc,char** argv) {
    const std::filesystem::path dir=argc>1?argv[1]:"experiments/logs/v51_v54_batch";
    try {
        std::filesystem::create_directories(dir);
        std::cout<<"NEUROMANCER MULTI-EXPERIMENT VALIDATION BATCH V51-V54\n";
        std::vector<Check> checks;
        checks.push_back(run_v51());
        checks.push_back(run_v52());
        checks.push_back(run_v53());
        checks.push_back(run_v54(dir));
        std::ofstream csv(dir/"v51_v54_validation.csv");
        if(!csv) throw std::runtime_error("cannot create validation CSV");
        csv<<"experiment,pass,detail\n";
        bool all=true;
        for(const auto& c:checks) {
            emit(c); all=all&&c.pass;
            csv<<c.name<<','<<(c.pass?1:0)<<",\""<<c.detail<<"\"\n";
        }
        std::cout<<"batch_acceptance="<<(all?"PASS":"FAIL")<<'\n'
                 <<"artifacts="<<(dir/"v51_v54_validation.csv").string()<<'\n'
                 <<"LIMITATION=V51-V53 are host-side reference/controller tests; V54 exercises CUDA inhibitory transmission. This batch does not yet establish CUDA-resident autonomous phase learning, dendritic morphology, or associative recall.\n";
        return all?0:3;
    } catch(const std::exception& e) {
        std::cerr<<"V51-V54 batch failed: "<<e.what()<<'\n';
        return 1;
    }
}
