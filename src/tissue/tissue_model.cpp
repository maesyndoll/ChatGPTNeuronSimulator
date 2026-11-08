#include "tissue/tissue_model.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
namespace spiking::tissue {
namespace {
bool unit(float v){return std::isfinite(v)&&v>=0.0f&&v<=1.0f;}
bool unit(double v){return std::isfinite(v)&&v>=0.0&&v<=1.0;}
const ModulePort& port(const NeuralModule& m,std::uint64_t id){
 auto i=std::find_if(m.ports.begin(),m.ports.end(),[id](const auto& p){return p.id==id;});
 if(i==m.ports.end()) throw std::invalid_argument("Adapter references unknown port");
 return *i;
}
}
std::uint64_t TissueModel::add_module(NeuralModule m){
 if(m.id==0)m.id=next_module_id_++;
 if(modules_.count(m.id))throw std::invalid_argument("Module ID already exists");
 validate_module(m); next_module_id_=std::max(next_module_id_,m.id+1);
 auto id=m.id; modules_.emplace(id,std::move(m)); return id;
}
NeuralModule TissueModel::clone_module(std::uint64_t id,CloneMode mode){
 NeuralModule c=module(id); c.id=next_module_id_++; c.name+=" copy"; c.generation=0;
 if(mode==CloneMode::StructuralOnly)for(auto& p:c.populations){
  p.mean_rate_hz=0.0f;p.damage_fraction=0.0f;p.metabolic_reserve=1.0f;p.immune_activation=0.0f;
 }
 validate_module(c);modules_.emplace(c.id,c);return c;
}
NeuralModule TissueModel::excise_module(std::uint64_t id){
 auto i=modules_.find(id);if(i==modules_.end())throw std::out_of_range("Unknown module");
 NeuralModule detached=i->second;modules_.erase(i);
 adapters_.erase(std::remove_if(adapters_.begin(),adapters_.end(),[id](const auto& a){
  return a.source_module_id==id||a.target_module_id==id;
 }),adapters_.end());return detached;
}
std::uint64_t TissueModel::graft_module(NeuralModule m){return add_module(std::move(m));}
std::uint64_t TissueModel::connect(BindingAdapter a){
 if(a.id==0)a.id=next_adapter_id_++;
 if(std::any_of(adapters_.begin(),adapters_.end(),[&](const auto& x){return x.id==a.id;}))
  throw std::invalid_argument("Adapter ID already exists");
 validate_adapter(a);next_adapter_id_=std::max(next_adapter_id_,a.id+1);
 auto id=a.id;adapters_.push_back(std::move(a));return id;
}
bool TissueModel::disconnect(std::uint64_t id){
 auto n=adapters_.size();adapters_.erase(std::remove_if(adapters_.begin(),adapters_.end(),
 [id](const auto& a){return a.id==id;}),adapters_.end());return n!=adapters_.size();
}
const NeuralModule& TissueModel::module(std::uint64_t id)const{
 auto i=modules_.find(id);if(i==modules_.end())throw std::out_of_range("Unknown module");return i->second;
}
NeuralModule& TissueModel::edit_module(std::uint64_t id){
 auto i=modules_.find(id);if(i==modules_.end())throw std::out_of_range("Unknown module");return i->second;
}
const std::map<std::uint64_t,NeuralModule>& TissueModel::modules()const noexcept{return modules_;}
const std::vector<BindingAdapter>& TissueModel::adapters()const noexcept{return adapters_;}
void TissueModel::validate_module(const NeuralModule& m)const{
 if(m.id==0||m.name.empty())throw std::invalid_argument("Module requires ID and name");
 m.topology.validate();std::unordered_set<std::uint64_t> pids;std::uint64_t e=0,i=0;
 for(const auto& p:m.populations){
  if(p.id==0||!pids.insert(p.id).second||p.size==0)throw std::invalid_argument("Invalid population identity or size");
  if(!std::isfinite(p.mean_rate_hz)||p.mean_rate_hz<0||!unit(p.damage_fraction)||!unit(p.metabolic_reserve)||!unit(p.immune_activation))
   throw std::invalid_argument("Population state outside valid bounds");
  if(p.kind==CellKind::ExcitatoryNeuron)e+=p.size;
  if(p.kind==CellKind::InhibitoryNeuron)i+=p.size;
 }
 if(e!=m.topology.excitatory_count||i!=m.topology.inhibitory_count)
  throw std::invalid_argument("Neuron population sizes must match CSR E/I counts");
 std::unordered_set<std::uint64_t> ports;
 for(const auto& p:m.ports)if(p.id==0||!ports.insert(p.id).second||!pids.count(p.population_id)||p.connection_capacity==0)
  throw std::invalid_argument("Invalid module port");
}
void TissueModel::validate_adapter(const BindingAdapter& a)const{
 if(a.id==0||a.source_module_id==a.target_module_id||!std::isfinite(a.gain)||a.gain<0||a.delay_steps==0)
  throw std::invalid_argument("Invalid adapter identity or parameters");
 auto s=modules_.find(a.source_module_id),t=modules_.find(a.target_module_id);
 if(s==modules_.end()||t==modules_.end())throw std::invalid_argument("Adapter endpoints must exist");
 const auto& sp=port(s->second,a.source_port_id);const auto& tp=port(t->second,a.target_port_id);
 if(sp.direction!=PortDirection::Output||tp.direction!=PortDirection::Input)
  throw std::invalid_argument("Adapter must connect output to input");
 auto sc=std::count_if(adapters_.begin(),adapters_.end(),[&](const auto& x){return x.source_module_id==a.source_module_id&&x.source_port_id==a.source_port_id;});
 auto tc=std::count_if(adapters_.begin(),adapters_.end(),[&](const auto& x){return x.target_module_id==a.target_module_id&&x.target_port_id==a.target_port_id;});
 if(sc>=sp.connection_capacity||tc>=tp.connection_capacity)throw std::invalid_argument("Adapter exceeds port capacity");
}
void TissueModel::validate()const{
 for(const auto& x:modules_){if(x.first!=x.second.id)throw std::invalid_argument("Module key/ID mismatch");validate_module(x.second);}
 std::unordered_set<std::uint64_t> ids;
 std::map<std::pair<std::uint64_t,std::uint64_t>,std::uint32_t> out_count,in_count;
 for(const auto& a:adapters_){
  if(!ids.insert(a.id).second)throw std::invalid_argument("Duplicate adapter ID");
  auto s=modules_.find(a.source_module_id),t=modules_.find(a.target_module_id);
  if(s==modules_.end()||t==modules_.end())throw std::invalid_argument("Dangling adapter");
  const auto& sp=port(s->second,a.source_port_id);const auto& tp=port(t->second,a.target_port_id);
  if(sp.direction!=PortDirection::Output||tp.direction!=PortDirection::Input||!std::isfinite(a.gain)||a.gain<0||a.delay_steps==0)
   throw std::invalid_argument("Invalid adapter");
  if(++out_count[{a.source_module_id,a.source_port_id}]>sp.connection_capacity||
     ++in_count[{a.target_module_id,a.target_port_id}]>tp.connection_capacity)
   throw std::invalid_argument("Adapter port capacity exceeded");
 }
}
StructuralPlasticityReport apply_structural_plasticity(
    CSRTopology& topology,
    const std::vector<float>& pair_coactivity,
    const StructuralPlasticityPolicy& p) {
    topology.validate();
    const std::uint64_t n = topology.neuron_count();
    if (n != 0 && n > std::numeric_limits<std::size_t>::max() / n) {
        throw std::length_error("Coactivity matrix dimensions overflow");
    }
    if (pair_coactivity.size() != static_cast<std::size_t>(n * n)) {
        throw std::invalid_argument("Coactivity matrix must be N x N");
    }
    const float values[] = {
        p.learning_rate, p.coactivity_baseline, p.prune_coactivity_below,
        p.prune_weight_below, p.sprout_coactivity_at_or_above,
        p.initial_sprout_weight, p.minimum_weight_magnitude,
        p.maximum_weight_magnitude
    };
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Non-finite structural plasticity policy");
        }
    }
    if (p.learning_rate < 0.0f ||
        p.coactivity_baseline < 0.0f || p.coactivity_baseline > 1.0f ||
        p.prune_coactivity_below < 0.0f || p.prune_coactivity_below > 1.0f ||
        p.sprout_coactivity_at_or_above < 0.0f ||
        p.sprout_coactivity_at_or_above > 1.0f ||
        p.prune_weight_below < 0.0f ||
        p.minimum_weight_magnitude <= 0.0f ||
        p.maximum_weight_magnitude < p.minimum_weight_magnitude ||
        p.initial_sprout_weight < p.minimum_weight_magnitude ||
        p.initial_sprout_weight > p.maximum_weight_magnitude ||
        p.maximum_out_degree == 0 ||
        p.sprout_delay_steps < topology.minimum_delay ||
        p.sprout_delay_steps > topology.maximum_delay) {
        throw std::invalid_argument("Invalid structural plasticity policy");
    }
    for (float value : pair_coactivity) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            throw std::invalid_argument("Pairwise coactivity must be finite and in [0,1]");
        }
    }

    struct Edge { std::uint32_t target; float weight; std::uint32_t delay; };
    std::vector<std::vector<Edge>> rows(n);
    StructuralPlasticityReport report;
    for (std::uint32_t source = 0; source < n; ++source) {
        const bool excitatory = source < topology.excitatory_count;
        const auto begin = topology.row_offsets[source];
        const auto end = topology.row_offsets[source + 1];
        rows[source].reserve(std::min<std::uint64_t>(
            p.maximum_out_degree, static_cast<std::uint64_t>(end - begin) + 8));
        for (std::uint32_t edge = begin; edge < end; ++edge) {
            const auto target = topology.target_indices[edge];
            const float coactivity = pair_coactivity[
                static_cast<std::size_t>(source) * n + target];
            const float old_magnitude = std::abs(topology.weights[edge]);
            const float updated = std::clamp(
                old_magnitude + p.learning_rate *
                    (coactivity - p.coactivity_baseline),
                p.minimum_weight_magnitude, p.maximum_weight_magnitude);
            if (coactivity < p.prune_coactivity_below ||
                updated < p.prune_weight_below) {
                ++report.pruned;
                continue;
            }
            if (coactivity > p.coactivity_baseline) ++report.potentiated;
            else if (coactivity < p.coactivity_baseline) ++report.depressed;
            rows[source].push_back({
                target, excitatory ? updated : -updated, topology.delays[edge]});
        }

        if (rows[source].size() < p.maximum_out_degree) {
            for (std::uint32_t target = 0; target < n; ++target) {
                if (target == source ||
                    pair_coactivity[static_cast<std::size_t>(source) * n + target]
                        < p.sprout_coactivity_at_or_above) {
                    continue;
                }
                const auto position = std::lower_bound(
                    rows[source].begin(), rows[source].end(), target,
                    [](const Edge& edge, std::uint32_t value) {
                        return edge.target < value;
                    });
                if (position != rows[source].end() && position->target == target) {
                    continue;
                }
                if (rows[source].size() >= p.maximum_out_degree) break;
                rows[source].insert(position, Edge{
                    target,
                    excitatory ? p.initial_sprout_weight : -p.initial_sprout_weight,
                    p.sprout_delay_steps});
                ++report.sprouted;
            }
        }
    }

    CSRTopology candidate = topology;
    candidate.row_offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    candidate.target_indices.clear();
    candidate.weights.clear();
    candidate.delays.clear();
    for (std::uint32_t source = 0; source < n; ++source) {
        for (const auto& edge : rows[source]) {
            candidate.target_indices.push_back(edge.target);
            candidate.weights.push_back(edge.weight);
            candidate.delays.push_back(edge.delay);
        }
        if (candidate.target_indices.size() >
            std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("Remodeled CSR exceeds 32-bit row offsets");
        }
        candidate.row_offsets[source + 1] =
            static_cast<std::uint32_t>(candidate.target_indices.size());
    }
    candidate.validate();
    topology = std::move(candidate);
    return report;
}

double objective_score(const CandidateMetrics& m,const OptimizationPolicy& p){
 const double vals[]={m.prediction_error,m.retention_score,m.stability_score,m.energy_cost,m.structural_cost,m.damaged_fraction,
  p.prediction_weight,p.retention_weight,p.stability_weight,p.energy_weight,p.structural_weight};
 for(double v:vals)if(!std::isfinite(v))throw std::invalid_argument("Non-finite optimization metric");
 if(m.prediction_error<0||!unit(m.retention_score)||!unit(m.stability_score)||!unit(m.energy_cost)||!unit(m.structural_cost)||!unit(m.damaged_fraction))
  throw std::invalid_argument("Optimization metric outside valid range");
 if(p.prediction_weight<0||p.retention_weight<0||p.stability_weight<0||p.energy_weight<0||p.structural_weight<0||
  !unit(p.minimum_retention)||!unit(p.minimum_stability)||!unit(p.maximum_damage)||p.maximum_prediction_error_regression<0||p.minimum_objective_improvement<0)
  throw std::invalid_argument("Invalid optimization policy");
 double w=p.prediction_weight+p.retention_weight+p.stability_weight+p.energy_weight+p.structural_weight;
 if(!(w>0))throw std::invalid_argument("Objective weights must have positive sum");
 return (p.prediction_weight*m.prediction_error+p.retention_weight*(1-m.retention_score)+
  p.stability_weight*(1-m.stability_score)+p.energy_weight*m.energy_cost+p.structural_weight*m.structural_cost)/w;
}
bool candidate_is_promotable(const CandidateMetrics& b,const CandidateMetrics& c,const OptimizationPolicy& p){
 const double bs=objective_score(b,p),cs=objective_score(c,p);
 if(c.retention_score<p.minimum_retention||c.stability_score<p.minimum_stability||
  c.damaged_fraction>p.maximum_damage||c.prediction_error>b.prediction_error+p.maximum_prediction_error_regression)return false;
 return cs+p.minimum_objective_improvement<bs;
}
} // namespace spiking::tissue
