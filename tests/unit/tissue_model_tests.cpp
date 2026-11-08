#include "tissue/tissue_model.h"
#include <iostream>
#include <stdexcept>
#include <string>
using namespace spiking;using namespace spiking::tissue;
namespace {
void check(bool v,const std::string& m){if(!v)throw std::runtime_error(m);}
NeuralModule sample(const std::string& name,bool output){
 NeuralModule m;m.name=name;m.topology.excitatory_count=2;m.topology.inhibitory_count=1;
 m.topology.row_offsets={0,1,1,1};m.topology.target_indices={1};m.topology.weights={0.1f};m.topology.delays={1};
 m.populations={{1,CellKind::ExcitatoryNeuron,2,12,0.1f,0.8f,0.2f},
 {2,CellKind::InhibitoryNeuron,1,40,0,1,0},{3,CellKind::Astrocyte,4,0,0,0.9f,0.1f},
 {4,CellKind::Microglia,2,0,0,1,0}};
 m.ports={{1,1,output?PortDirection::Output:PortDirection::Input,2}};return m;
}
}
int main(){try{
 TissueModel tissue;auto a=tissue.add_module(sample("sensory",true));auto b=tissue.add_module(sample("association",false));
 tissue.connect(BindingAdapter{0,a,1,b,1,0.75f,2,true});tissue.validate();
 auto structural=tissue.clone_module(a,CloneMode::StructuralOnly);
 check(structural.id!=a,"clone ID");check(structural.populations[0].mean_rate_hz==0,"structural clone resets activity");
 check(structural.populations[0].metabolic_reserve==1,"structural clone resets reserve");
 check(tissue.adapters().size()==1,"clone must not copy external adapters");
 auto stateful=tissue.clone_module(a,CloneMode::Stateful);check(stateful.populations[0].mean_rate_hz==12,"stateful clone preserves state");
 auto detached=tissue.excise_module(b);check(tissue.adapters().empty(),"excision removes dangling adapters");
 check(!tissue.modules().count(b),"excised module removed");auto grafted=tissue.graft_module(detached);
 check(grafted==b,"graft preserves module identity");tissue.connect(BindingAdapter{0,a,1,grafted,1,0.5f,1,true});tissue.validate();
 CandidateMetrics base{0.30,0.98,0.98,0.50,0.40,0.01},good{0.24,0.97,0.97,0.45,0.35,0.01},bad{0.18,0.70,0.60,0.40,0.30,0.02};
 check(candidate_is_promotable(base,good),"valid improvement promoted");check(!candidate_is_promotable(base,bad),"unstable candidate rejected");
 bool rejected=false;try{tissue.connect(BindingAdapter{0,a,1,grafted,1,1,1,true});}catch(const std::invalid_argument&){rejected=true;}
 check(rejected,"port capacity enforced");
 std::cout<<"Tissue model tests passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<"Tissue model test failure: "<<e.what()<<"\n";return 1;}}
