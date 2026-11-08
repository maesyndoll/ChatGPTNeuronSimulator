#pragma once
#include "topology/csr_topology.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>
namespace spiking::tissue {
enum class CellKind { ExcitatoryNeuron, InhibitoryNeuron, Astrocyte, Microglia, Oligodendrocyte };
enum class PortDirection { Input, Output };
enum class CloneMode { StructuralOnly, Stateful };
struct Population {
    std::uint64_t id=0; CellKind kind=CellKind::ExcitatoryNeuron; std::uint32_t size=0;
    float mean_rate_hz=0.0f, damage_fraction=0.0f, metabolic_reserve=1.0f, immune_activation=0.0f;
};
struct ModulePort { std::uint64_t id=0, population_id=0; PortDirection direction=PortDirection::Input; std::uint32_t connection_capacity=1; };
struct NeuralModule {
    std::uint64_t id=0; std::string name; std::uint64_t generation=0;
    CSRTopology topology; std::vector<Population> populations; std::vector<ModulePort> ports;
};
struct BindingAdapter {
    std::uint64_t id=0, source_module_id=0, source_port_id=0, target_module_id=0, target_port_id=0;
    float gain=1.0f; std::uint32_t delay_steps=1; bool enabled=true;
};
// Clones never inherit external adapters implicitly. Rebinding is explicit.
class TissueModel {
public:
    std::uint64_t add_module(NeuralModule module);
    NeuralModule clone_module(std::uint64_t module_id, CloneMode mode);
    NeuralModule excise_module(std::uint64_t module_id);
    std::uint64_t graft_module(NeuralModule module);
    std::uint64_t connect(BindingAdapter adapter);
    bool disconnect(std::uint64_t adapter_id);
    const NeuralModule& module(std::uint64_t module_id) const;
    NeuralModule& edit_module(std::uint64_t module_id);
    const std::map<std::uint64_t,NeuralModule>& modules() const noexcept;
    const std::vector<BindingAdapter>& adapters() const noexcept;
    void validate() const;
private:
    void validate_module(const NeuralModule& module) const;
    void validate_adapter(const BindingAdapter& adapter) const;
    std::uint64_t next_module_id_=1, next_adapter_id_=1;
    std::map<std::uint64_t,NeuralModule> modules_;
    std::vector<BindingAdapter> adapters_;
};
// Pairwise coactivity is a row-major N x N matrix with values in [0, 1].
// It is supplied by the runtime/experiment layer; this routine does not
// fabricate correlations from mean firing rates.
struct StructuralPlasticityPolicy {
    float learning_rate = 0.05f;
    float coactivity_baseline = 0.25f;
    float prune_coactivity_below = 0.05f;
    float prune_weight_below = 0.02f;
    float sprout_coactivity_at_or_above = 0.75f;
    float initial_sprout_weight = 0.05f;
    float minimum_weight_magnitude = 0.001f;
    float maximum_weight_magnitude = 3.0f;
    std::uint32_t maximum_out_degree = 64;
    std::uint32_t sprout_delay_steps = 1;
};

struct StructuralPlasticityReport {
    std::uint64_t potentiated = 0;
    std::uint64_t depressed = 0;
    std::uint64_t pruned = 0;
    std::uint64_t sprouted = 0;
};

// Deterministic, bounded CSR remodeling. The input coactivity matrix must
// contain N*N finite values in [0,1]. Invalid input leaves topology unchanged.
StructuralPlasticityReport apply_structural_plasticity(
    CSRTopology& topology,
    const std::vector<float>& pair_coactivity,
    const StructuralPlasticityPolicy& policy = {});

struct CandidateMetrics {
    double prediction_error=1.0, retention_score=1.0, stability_score=1.0;
    double energy_cost=0.0, structural_cost=0.0, damaged_fraction=0.0;
};
struct OptimizationPolicy {
    double prediction_weight=0.50, retention_weight=0.20, stability_weight=0.15;
    double energy_weight=0.10, structural_weight=0.05;
    double minimum_retention=0.90, minimum_stability=0.90, maximum_damage=0.05;
    double maximum_prediction_error_regression=0.02, minimum_objective_improvement=0.001;
};
double objective_score(const CandidateMetrics& metrics,const OptimizationPolicy& policy={});
bool candidate_is_promotable(const CandidateMetrics& baseline,const CandidateMetrics& candidate,const OptimizationPolicy& policy={});
} // namespace spiking::tissue
