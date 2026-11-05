#include "topology/topology_generator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace spiking {
namespace {

// SplitMix64: a small, explicitly specified generator so the random stream
// does not depend on standard-library distribution implementations.
class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() noexcept {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31U);
    }

    // Uniform double strictly between zero and one.
    double unit_open() noexcept {
        constexpr double scale = 1.0 / 9007199254740992.0; // 2^53
        return (static_cast<double>(next() >> 11U) + 0.5) * scale;
    }

    // Unbiased integer sample in [0, bound).
    std::uint64_t bounded(std::uint64_t bound) noexcept {
        const std::uint64_t threshold = -bound % bound;

        for (;;) {
            const std::uint64_t value = next();
            if (value >= threshold) {
                return value % bound;
            }
        }
    }

private:
    std::uint64_t state_;
};

class DelaySampler {
public:
    explicit DelaySampler(const TopologyParameters& parameters)
        : rng_(parameters.delay_seed),
          minimum_delay_(parameters.minimum_delay),
          maximum_delay_(parameters.maximum_delay),
          distribution_(parameters.delay_distribution) {
        if (minimum_delay_ == 0 || maximum_delay_ < minimum_delay_) {
            throw std::invalid_argument("Invalid synaptic delay bounds");
        }

        if (distribution_ == DelayDistribution::Uniform) {
            return;
        }

        if (distribution_ != DelayDistribution::FibonacciWeighted) {
            throw std::invalid_argument("Unsupported synaptic delay distribution");
        }

        if (parameters.fibonacci_delays.empty() ||
            parameters.fibonacci_delays.size() !=
                parameters.fibonacci_weights.size()) {
            throw std::invalid_argument(
                "Fibonacci delay and weight arrays must be non-empty and equal in size");
        }

        double cumulative = 0.0;

        for (std::size_t i = 0;
             i < parameters.fibonacci_delays.size(); ++i) {
            const auto delay = parameters.fibonacci_delays[i];
            const double weight = parameters.fibonacci_weights[i];

            if (delay < minimum_delay_ || delay > maximum_delay_) {
                throw std::invalid_argument(
                    "Fibonacci delay is outside configured delay bounds");
            }

            if (!std::isfinite(weight) || weight < 0.0) {
                throw std::invalid_argument(
                    "Fibonacci delay weights must be finite and non-negative");
            }

            if (weight == 0.0) {
                continue;
            }

            cumulative += weight;
            if (!std::isfinite(cumulative)) {
                throw std::invalid_argument(
                    "Fibonacci delay weight total is not finite");
            }

            cumulative_weights_.push_back(cumulative);
            active_delays_.push_back(delay);
        }

        if (cumulative_weights_.empty() || cumulative <= 0.0) {
            throw std::invalid_argument(
                "Fibonacci delay weights must have a positive total");
        }

        total_weight_ = cumulative;
    }

    std::uint32_t sample() noexcept {
        if (distribution_ == DelayDistribution::Uniform) {
            const std::uint64_t range =
                static_cast<std::uint64_t>(maximum_delay_) -
                minimum_delay_ + 1ULL;

            return minimum_delay_ +
                static_cast<std::uint32_t>(rng_.bounded(range));
        }

        const double value = rng_.unit_open() * total_weight_;

        for (std::size_t i = 0; i < cumulative_weights_.size(); ++i) {
            if (value < cumulative_weights_[i]) {
                return active_delays_[i];
            }
        }

        return active_delays_.back();
    }

private:
    SplitMix64 rng_;
    std::uint32_t minimum_delay_;
    std::uint32_t maximum_delay_;
    DelayDistribution distribution_;
    std::vector<std::uint32_t> active_delays_;
    std::vector<double> cumulative_weights_;
    double total_weight_ = 0.0;
};

void validate_probability(double probability, const char* name) {
    if (!std::isfinite(probability) ||
        probability < 0.0 || probability > 1.0) {
        throw std::invalid_argument(
            std::string("Invalid connection probability: ") + name);
    }
}

void append_range(CSRTopology& graph,
                  SplitMix64& rng,
                  DelaySampler& delay_sampler,
                  std::uint32_t source,
                  std::uint32_t target_begin,
                  std::uint32_t target_end,
                  double probability,
                  float minimum_weight,
                  float maximum_weight,
                  std::uint64_t memory_budget_bytes,
                  std::uint64_t offset_storage_bytes) {
    if (probability == 0.0 || target_begin == target_end) {
        return;
    }

    const bool excitatory_source = source < graph.excitatory_count;
    const double log_failure = probability == 1.0
        ? 0.0
        : std::log1p(-probability);

    std::uint64_t candidate = 0;
    const std::uint64_t range_size =
        static_cast<std::uint64_t>(target_end) - target_begin;

    while (candidate < range_size) {
        if (probability != 1.0) {
            const double gap_value =
                std::floor(std::log(rng.unit_open()) / log_failure);

            const std::uint64_t remaining = range_size - candidate;
            if (!std::isfinite(gap_value) ||
                gap_value >= static_cast<double>(remaining)) {
                break;
            }

            candidate += static_cast<std::uint64_t>(gap_value);
        }

        const std::uint64_t edge_count = graph.target_indices.size();
        if (edge_count >= std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("CSR exceeds the 32-bit row-offset limit");
        }

        constexpr std::uint64_t bytes_per_edge =
            sizeof(std::uint32_t) + sizeof(float) +
            sizeof(std::uint32_t);
        if (offset_storage_bytes > memory_budget_bytes ||
            edge_count + 1 >
                (memory_budget_bytes - offset_storage_bytes) / bytes_per_edge) {
            throw std::length_error("CSR topology exceeds memory_budget_bytes");
        }

        const auto target =
            target_begin + static_cast<std::uint32_t>(candidate);
        const double weight_unit = rng.unit_open();
        const float magnitude = static_cast<float>(
            minimum_weight +
            weight_unit * (maximum_weight - minimum_weight));

        graph.target_indices.push_back(target);
        graph.weights.push_back(excitatory_source ? magnitude : -magnitude);
        graph.delays.push_back(delay_sampler.sample());

        ++candidate;
        if (probability != 1.0) {
            // The next loop iteration draws the next geometric gap.
        }
    }
}

} // namespace

CSRTopology generate_random_sparse_topology(
    const SimulationParameters& simulation,
    const TopologyParameters& topology,
    const TopologyGeneratorOptions& options) {
    const std::uint64_t total64 =
        static_cast<std::uint64_t>(simulation.excitatory_count) +
        simulation.inhibitory_count;

    if (simulation.excitatory_count == 0 ||
        simulation.inhibitory_count == 0 ||
        total64 == 0 ||
        total64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Invalid simulation population sizes for topology");
    }

    validate_probability(topology.p_ee, "p_ee");
    validate_probability(topology.p_ei, "p_ei");
    validate_probability(topology.p_ie, "p_ie");
    validate_probability(topology.p_ii, "p_ii");

    if (!std::isfinite(options.minimum_weight) ||
        !std::isfinite(options.maximum_weight) ||
        options.minimum_weight <= 0.0f ||
        options.maximum_weight < options.minimum_weight) {
        throw std::invalid_argument("Invalid positive synaptic weight magnitude range");
    }

    const std::uint64_t total = total64;
    const std::uint64_t offset_bytes =
        (total + 1) * sizeof(std::uint32_t);

    if (topology.memory_budget_bytes < offset_bytes) {
        throw std::length_error("CSR row offsets exceed topology memory budget");
    }

    CSRTopology graph;
    graph.excitatory_count = simulation.excitatory_count;
    graph.inhibitory_count = simulation.inhibitory_count;
    graph.minimum_delay = topology.minimum_delay;
    graph.maximum_delay = topology.maximum_delay;
    graph.row_offsets.resize(static_cast<std::size_t>(total) + 1);

    SplitMix64 rng(topology.topology_seed);
    DelaySampler delay_sampler(topology);

    for (std::uint32_t source = 0;
         source < static_cast<std::uint32_t>(total); ++source) {
        const bool excitatory_source = source < simulation.excitatory_count;

        const double p_exc_targets =
            excitatory_source ? topology.p_ee : topology.p_ie;
        const double p_inh_targets =
            excitatory_source ? topology.p_ei : topology.p_ii;

        const std::uint32_t excitatory_end = simulation.excitatory_count;
        const std::uint32_t total_count = static_cast<std::uint32_t>(total);

        // Split the source's own population range around itself to exclude
        // autapses without disturbing sorted target order.
        if (source < excitatory_end) {
            append_range(graph, rng, delay_sampler, source, 0, source, p_exc_targets,
                         options.minimum_weight, options.maximum_weight,
                         topology.memory_budget_bytes, offset_bytes);
            append_range(graph, rng, delay_sampler, source, source + 1, excitatory_end,
                         p_exc_targets, options.minimum_weight,
                         options.maximum_weight, topology.memory_budget_bytes,
                         offset_bytes);
            append_range(graph, rng, delay_sampler, source, excitatory_end, total_count,
                         p_inh_targets, options.minimum_weight,
                         options.maximum_weight, topology.memory_budget_bytes,
                         offset_bytes);
        } else {
            append_range(graph, rng, delay_sampler, source, 0, excitatory_end,
                         p_exc_targets, options.minimum_weight,
                         options.maximum_weight, topology.memory_budget_bytes,
                         offset_bytes);
            append_range(graph, rng, delay_sampler, source, excitatory_end, source,
                         p_inh_targets, options.minimum_weight,
                         options.maximum_weight, topology.memory_budget_bytes,
                         offset_bytes);
            append_range(graph, rng, delay_sampler, source, source + 1, total_count,
                         p_inh_targets, options.minimum_weight,
                         options.maximum_weight, topology.memory_budget_bytes,
                         offset_bytes);
        }

        graph.row_offsets[static_cast<std::size_t>(source) + 1] =
            static_cast<std::uint32_t>(graph.target_indices.size());
    }

    graph.validate();
    return graph;
}

} // namespace spiking
