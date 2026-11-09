#pragma once

#include "topology/csr_topology.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spiking {

// Small deterministic population-level phase model. It is deliberately an
// explicit host-side controller, not yet a CUDA neuron-state variable.
// Phase receives autonomous angular drift, spike-triggered phase response,
// and learned Kuramoto-style coupling from other populations.
struct AutonomousPhaseParameters {
    float dt_ms = 0.5f;
    float intrinsic_frequency_hz = 8.0f;
    float activity_kick_rad = 0.015f;
    float coupling_learning_rate = 0.002f;
    float maximum_coupling = 2.0f;
    float coupling_decay = 0.00001f;
    std::uint32_t temporal_association_window_steps = 39;
    float temporal_lag_learning_rate = 0.10f;
    float minimum_temporal_confidence = 0.20f;
};

class AutonomousPopulationPhase {
public:
    explicit AutonomousPopulationPhase(
        std::uint32_t populations,
        AutonomousPhaseParameters parameters = {})
        : parameters_(parameters),
          phase_(populations, 0.0f),
          activity_(populations, 0.0f),
          coupling_(static_cast<std::size_t>(populations) * populations, 0.0f),
          learned_lag_(static_cast<std::size_t>(populations) * populations, 0.0f),
          lag_cos_(static_cast<std::size_t>(populations) * populations, 0.0f),
          lag_sin_(static_cast<std::size_t>(populations) * populations, 0.0f),
          temporal_confidence_(static_cast<std::size_t>(populations) * populations, 0.0f),
          last_spike_step_(populations, invalid_step()) {
        if (populations < 2 || !std::isfinite(parameters_.dt_ms) ||
            parameters_.dt_ms <= 0.0f ||
            !std::isfinite(parameters_.intrinsic_frequency_hz) ||
            parameters_.intrinsic_frequency_hz <= 0.0f ||
            parameters_.coupling_learning_rate < 0.0f ||
            parameters_.maximum_coupling <= 0.0f ||
            parameters_.coupling_decay < 0.0f ||
            parameters_.temporal_association_window_steps == 0 ||
            parameters_.temporal_lag_learning_rate <= 0.0f ||
            parameters_.temporal_lag_learning_rate > 1.0f ||
            parameters_.minimum_temporal_confidence < 0.0f ||
            parameters_.minimum_temporal_confidence > 1.0f)
            throw std::invalid_argument("Invalid autonomous phase parameters");
    }

    std::uint32_t size() const noexcept {
        return static_cast<std::uint32_t>(phase_.size());
    }
    float phase(std::uint32_t p) const { return phase_.at(p); }
    float coupling(std::uint32_t source, std::uint32_t target) const {
        return coupling_.at(index(source, target));
    }
    float learned_lag(std::uint32_t source, std::uint32_t target) const {
        return learned_lag_.at(index(source, target));
    }
    float temporal_confidence(std::uint32_t source, std::uint32_t target) const {
        return temporal_confidence_.at(index(source, target));
    }

    // Spike counts are observed from the preceding neural integration step.
    // A spike burst advances the phase; no external phase clock is supplied.
    void step(const std::vector<std::uint32_t>& population_spikes,
              bool learn = true) {
        if (population_spikes.size() != phase_.size())
            throw std::invalid_argument("Phase activity vector has wrong size");
        const auto n = size();
        const float tau = 6.2831853071795864769f;
        const float omega = tau * parameters_.intrinsic_frequency_hz / 1000.0f;
        std::vector<float> next = phase_;
        for (std::uint32_t i = 0; i < n; ++i) {
            activity_[i] = 0.90f * activity_[i] +
                0.10f * static_cast<float>(population_spikes[i]);
            float velocity = omega +
                parameters_.activity_kick_rad * static_cast<float>(population_spikes[i]);
            for (std::uint32_t j = 0; j < n; ++j) {
                if (i == j) continue;
                const auto ij = index(j, i);
                const float delta = wrap(phase_[j] - phase_[i] -
                                         learned_lag_[ij]);
                velocity += coupling_[ij] * std::sin(delta);
            }
            next[i] = wrap(phase_[i] + velocity * parameters_.dt_ms);
        }
        phase_.swap(next);

        // Learn directed temporal associations from observed spike order.
        // A source event followed by a target event within the configured
        // window contributes its autonomous phase advance (omega * elapsed
        // time) to a circular estimate. Randomly varying delays cancel in
        // the resultant vector instead of being mistaken for a stable lag.
        if (learn) {
            // A source event is stored when it occurs; association is
            // evaluated later, when a target event arrives. The previous
            // implementation incorrectly required source and target to spike
            // on the same integration step, making delayed associations
            // impossible to learn.
            for (std::uint32_t target = 0; target < n; ++target) {
                if (population_spikes[target] == 0U) continue;
                for (std::uint32_t source = 0; source < n; ++source) {
                    if (source == target ||
                        last_spike_step_[source] == invalid_step()) continue;
                    const auto elapsed = steps_ - last_spike_step_[source];
                    if (elapsed == 0U ||
                        elapsed > parameters_.temporal_association_window_steps) continue;
                    const auto ij = index(source, target);
                    const float observed_lag = wrap(
                        omega * parameters_.dt_ms * static_cast<float>(elapsed));
                    const float alpha = parameters_.temporal_lag_learning_rate;
                    lag_cos_[ij] = (1.0f - alpha) * lag_cos_[ij] +
                                   alpha * std::cos(observed_lag);
                    lag_sin_[ij] = (1.0f - alpha) * lag_sin_[ij] +
                                   alpha * std::sin(observed_lag);
                    temporal_confidence_[ij] = std::clamp(
                        std::sqrt(lag_cos_[ij] * lag_cos_[ij] +
                                  lag_sin_[ij] * lag_sin_[ij]), 0.0f, 1.0f);
                    if (temporal_confidence_[ij] >=
                        parameters_.minimum_temporal_confidence) {
                        learned_lag_[ij] = wrap(std::atan2(lag_sin_[ij], lag_cos_[ij]));
                    }
                }
            }
            for (std::uint32_t source = 0; source < n; ++source)
                if (population_spikes[source] != 0U)
                    last_spike_step_[source] = steps_;
        }

        if (learn) {
            for (std::uint32_t i = 0; i < n; ++i) {
                for (std::uint32_t j = 0; j < n; ++j) {
                    if (i == j) continue;
                    const auto ij = index(i, j);
                    const float confidence = temporal_confidence_[ij];
                    coupling_[ij] = std::clamp(
                        coupling_[ij] * (1.0f - parameters_.coupling_decay) +
                        parameters_.coupling_learning_rate * confidence *
                            activity_[i] * activity_[j] *
                            std::cos(wrap(phase_[i] - phase_[j] -
                                          learned_lag_[ij])),
                        0.0f, parameters_.maximum_coupling);
                    // The directed lag is learned exclusively from spike timing
                    // above; do not overwrite it with instantaneous phase alignment.
                }
            }
        }
        ++steps_;
    }

    std::uint64_t steps() const noexcept { return steps_; }

private:
    static constexpr std::uint64_t invalid_step() noexcept {
        return std::numeric_limits<std::uint64_t>::max();
    }
    static float wrap(float x) {
        constexpr float tau = 6.2831853071795864769f;
        x = std::fmod(x, tau);
        return x < 0.0f ? x + tau : x;
    }
    std::size_t index(std::uint32_t a, std::uint32_t b) const {
        if (a >= size() || b >= size())
            throw std::out_of_range("Population phase index out of range");
        return static_cast<std::size_t>(a) * size() + b;
    }

    AutonomousPhaseParameters parameters_;
    std::vector<float> phase_;
    std::vector<float> activity_;
    std::vector<float> coupling_;
    std::vector<float> learned_lag_;
    std::vector<float> lag_cos_;
    std::vector<float> lag_sin_;
    std::vector<float> temporal_confidence_;
    std::vector<std::uint64_t> last_spike_step_;
    std::uint64_t steps_ = 0;
};

// Epoch-level activity-dependent structural plasticity. Activity statistics
// are supplied from observed spike counts; candidate growth is selected from
// co-active populations, not random edge replacement. Per-neuron resource
// limits total outgoing structural allocation. This models local synaptic
// structural plasticity; it is not a spatial dendritic-compartment model.
struct StructuralPlasticityParameters {
    float potentiation = 0.02f;
    float depression = 0.01f;
    float growth_threshold = 0.35f;
    float prune_threshold = 0.002f;
    std::uint32_t minimum_epochs_before_prune = 3;
    std::uint32_t maximum_out_degree = 16;
    float maximum_weight = 0.25f;
    float resource_per_excitatory_neuron = 1.0f;
};

struct StructuralPlasticityReport {
    std::uint32_t births = 0;
    std::uint32_t deaths = 0;
    std::uint32_t strengthened = 0;
    std::uint32_t weakened = 0;
    std::vector<std::uint32_t> next_edge_ages;
};

class ActivityStructuralPlasticity {
public:
    explicit ActivityStructuralPlasticity(
        StructuralPlasticityParameters parameters = {})
        : parameters_(parameters) {
        if (parameters_.potentiation < 0.0f || parameters_.depression < 0.0f ||
            parameters_.growth_threshold < 0.0f ||
            parameters_.prune_threshold < 0.0f ||
            parameters_.maximum_out_degree == 0 ||
            parameters_.maximum_weight <= 0.0f ||
            parameters_.resource_per_excitatory_neuron <= 0.0f)
            throw std::invalid_argument("Invalid structural plasticity parameters");
    }

    StructuralPlasticityReport apply(
        CSRTopology& graph,
        const std::vector<float>& activity,
        const std::vector<std::uint32_t>& age_epochs,
        bool enabled = true) const {
        graph.validate();
        const auto n = graph.neuron_count();
        if (activity.size() != n || age_epochs.size() != graph.synapse_count())
            throw std::invalid_argument("Structural activity/edge-age dimensions mismatch");
        StructuralPlasticityReport report;
        if (!enabled) return report;

        std::vector<std::vector<std::pair<std::uint32_t, float>>> rows(n);
        std::vector<std::vector<std::uint32_t>> ages(n);
        for (std::uint32_t s = 0; s < n; ++s) {
            for (std::uint32_t e = graph.row_offsets[s]; e < graph.row_offsets[s + 1]; ++e) {
                const auto t = graph.target_indices[e];
                float w = graph.weights[e];
                const float coactivity = std::clamp(activity[s] * activity[t], 0.0f, 1.0f);
                if (s < graph.excitatory_count) {
                    const float old = w;
                    w = std::clamp(w + parameters_.potentiation * coactivity -
                                      parameters_.depression * (1.0f - coactivity),
                                   0.0f, parameters_.maximum_weight);
                    if (w > old) ++report.strengthened;
                    else if (w < old) ++report.weakened;
                }
                const bool prune = s < graph.excitatory_count &&
                    w < parameters_.prune_threshold &&
                    age_epochs[e] >= parameters_.minimum_epochs_before_prune;
                if (prune) {
                    ++report.deaths;
                    continue;
                }
                rows[s].push_back({t, w});
                ages[s].push_back(age_epochs[e] + 1U);
            }
        }

        // Grow only E-source edges. Each source owns a finite local dendritic
        // allocation budget; one edge consumes one unit. The hard degree cap
        // remains an independent safety bound.
        const auto resource_capacity = static_cast<std::uint32_t>(std::max(1.0f,
            std::floor(parameters_.maximum_out_degree *
                       parameters_.resource_per_excitatory_neuron)));
        for (std::uint32_t s = 0; s < graph.excitatory_count; ++s) {
            const auto capacity = std::min(parameters_.maximum_out_degree,
                                           resource_capacity);
            if (rows[s].size() >= capacity) continue;
            std::vector<std::pair<float, std::uint32_t>> candidates;
            for (std::uint32_t t = 0; t < n; ++t) {
                if (t == s || activity[s] * activity[t] < parameters_.growth_threshold)
                    continue;
                const bool exists = std::any_of(rows[s].begin(), rows[s].end(),
                    [t](const auto& edge) { return edge.first == t; });
                if (!exists) candidates.push_back({activity[s] * activity[t], t});
            }
            std::sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) {
                    return a.first > b.first || (a.first == b.first && a.second < b.second);
                });
            for (const auto& candidate : candidates) {
                if (rows[s].size() >= capacity) break;
                rows[s].push_back({candidate.second, 0.01f});
                ages[s].push_back(0U);
                ++report.births;
            }
        }

        report.next_edge_ages.clear();
        graph.row_offsets.assign(static_cast<std::size_t>(n) + 1U, 0U);
        graph.target_indices.clear();
        graph.weights.clear();
        graph.delays.clear();
        for (std::uint32_t s = 0; s < n; ++s) {
            std::vector<std::size_t> order(rows[s].size());
            for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return rows[s][a].first < rows[s][b].first;
            });
            for (const auto i : order) {
                graph.target_indices.push_back(rows[s][i].first);
                graph.weights.push_back(s < graph.excitatory_count
                    ? std::max(rows[s][i].second, 0.0001f)
                    : -std::max(std::abs(rows[s][i].second), 0.0001f));
                graph.delays.push_back(1U);
                report.next_edge_ages.push_back(ages[s][i]);
            }
            graph.row_offsets[s + 1] =
                static_cast<std::uint32_t>(graph.target_indices.size());
        }
        graph.validate();
        return report;
    }

private:
    StructuralPlasticityParameters parameters_;
};

} // namespace spiking
