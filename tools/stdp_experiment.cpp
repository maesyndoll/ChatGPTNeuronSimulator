#include "plasticity/stdp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace spiking;

constexpr float kInitialWeight = 0.5f;
constexpr float kDtMs = 1.0f;
constexpr float kTauMs = 20.0f;
constexpr float kMinimumWeight = 0.001f;
constexpr float kMaximumWeight = 1.0f;

void write_stage_a(const std::filesystem::path& output_dir) {
    const std::vector<int> lags_ms{
        -80, -40, -20, -10, -5, -2, -1, 0,
          1,   2,   5,  10, 20, 40, 80
    };

    StdpParameters p;
    p.minimum_weight = kMinimumWeight;
    p.maximum_weight = kMaximumWeight;
    p.presynaptic_tau_ms = kTauMs;
    p.postsynaptic_tau_ms = kTauMs;

    const std::vector<float> sources_weight{kInitialWeight};
    const std::vector<std::uint32_t> sources{0};
    const std::vector<std::uint32_t> targets{1};

    std::ofstream csv(output_dir / "stage_a_timing.csv");
    if (!csv) throw std::runtime_error("Could not create Stage A CSV");

    csv << "lag_ms,measured_delta_weight,expected_delta_weight,final_weight\n";
    csv << std::setprecision(10);

    for (const int lag : lags_ms) {
        auto weights = sources_weight;
        StdpTraces traces{{0.0f, 0.0f}, {0.0f, 0.0f}};

        const int first_neuron = lag < 0 ? 1 : 0;
        const int second_neuron = lag < 0 ? 0 : 1;
        const int steps = std::abs(lag);

        std::vector<std::uint8_t> events(2, 0);
        events[static_cast<std::size_t>(first_neuron)] = 1;
        apply_pair_stdp_step(
            weights, sources, targets, 2, 2, events, traces, kDtMs, p);

        for (int step = 1; step <= steps; ++step) {
            std::fill(events.begin(), events.end(), 0);
            if (step == steps)
                events[static_cast<std::size_t>(second_neuron)] = 1;

            apply_pair_stdp_step(
                weights, sources, targets, 2, 2, events, traces, kDtMs, p);
        }

        // At zero lag, both events occur at the same boundary.
        if (lag == 0) {
            std::fill(events.begin(), events.end(), 0);
            events[0] = 1;
            events[1] = 1;
            weights[0] = kInitialWeight;
            traces = StdpTraces{{0.0f, 0.0f}, {0.0f, 0.0f}};
            apply_pair_stdp_step(
                weights, sources, targets, 2, 2, events, traces, kDtMs, p);
        }

        float expected = 0.0f;
        if (lag > 0)
            expected = p.potentiation_rate *
                std::exp(-static_cast<float>(lag) / p.presynaptic_tau_ms);
        else if (lag < 0)
            expected = -p.depression_rate *
                std::exp(-static_cast<float>(-lag) / p.postsynaptic_tau_ms);

        csv << lag << ',' << weights[0] - kInitialWeight << ','
            << expected << ',' << weights[0] << '\n';
    }

    std::cout << "Stage A written: " << (output_dir / "stage_a_timing.csv") << '\n';
}

struct StabilityResult {
    float final_weight = 0.0f;
    double mean_weight = 0.0;
    float minimum_seen = 0.0f;
    float maximum_seen = 0.0f;
    double boundary_fraction = 0.0;
};

enum class BoundaryPolicy {
    HardClamp,
    WideRange,
    Reflecting
};

StabilityResult run_independent_activity(
    float pre_rate_hz,
    float post_rate_hz,
    float balance_ratio,
    float initial_weight,
    std::uint32_t seed,
    BoundaryPolicy boundary_policy = BoundaryPolicy::HardClamp) {

    constexpr int duration_ms = 300000;
    constexpr int steps = static_cast<int>(duration_ms / kDtMs);

    StdpParameters p;
    p.depression_rate = 0.00525f;
    p.presynaptic_tau_ms = kTauMs;
    p.postsynaptic_tau_ms = kTauMs;
    p.potentiation_rate =
        balance_ratio * p.depression_rate *
        p.postsynaptic_tau_ms / p.presynaptic_tau_ms;

    const bool use_wide_range =
        boundary_policy != BoundaryPolicy::HardClamp;
    p.minimum_weight = use_wide_range ? 1.0e-6f : kMinimumWeight;
    p.maximum_weight = use_wide_range ? 1000.0f : kMaximumWeight;

    std::vector<float> weights{initial_weight};
    const std::vector<std::uint32_t> sources{0};
    const std::vector<std::uint32_t> targets{1};
    StdpTraces traces{{0.0f, 0.0f}, {0.0f, 0.0f}};
    std::vector<std::uint8_t> events(2, 0);

    std::mt19937 rng(seed);
    const double scale =
        static_cast<double>(std::mt19937::max()) + 1.0;
    const double pre_probability = pre_rate_hz * kDtMs / 1000.0;
    const double post_probability = post_rate_hz * kDtMs / 1000.0;

    double weight_sum = 0.0;
    int boundary_steps = 0;
    float minimum_seen = initial_weight;
    float maximum_seen = initial_weight;

    for (int step = 0; step < steps; ++step) {
        const double pre_draw = static_cast<double>(rng()) / scale;
        const double post_draw = static_cast<double>(rng()) / scale;

        events[0] = pre_draw < pre_probability ? 1 : 0;
        events[1] = post_draw < post_probability ? 1 : 0;

        apply_pair_stdp_step(
            weights, sources, targets, 2, 2, events, traces, kDtMs, p);

        if (boundary_policy == BoundaryPolicy::Reflecting) {
            float reflected = weights[0];
            while (reflected < kMinimumWeight ||
                   reflected > kMaximumWeight) {
                if (reflected < kMinimumWeight)
                    reflected = 2.0f * kMinimumWeight - reflected;
                if (reflected > kMaximumWeight)
                    reflected = 2.0f * kMaximumWeight - reflected;
            }
            weights[0] = reflected;
        }

        const float weight = weights[0];
        weight_sum += weight;
        minimum_seen = std::min(minimum_seen, weight);
        maximum_seen = std::max(maximum_seen, weight);

        // Measure occupancy at the active policy's actual boundaries.
        // Reflecting uses the original interval after mapping; the other
        // policies use the limits supplied to the STDP update.
        const float boundary_min =
            boundary_policy == BoundaryPolicy::Reflecting
                ? kMinimumWeight
                : p.minimum_weight;
        const float boundary_max =
            boundary_policy == BoundaryPolicy::Reflecting
                ? kMaximumWeight
                : p.maximum_weight;

        if (weight <= boundary_min + 1.0e-6f ||
            weight >= boundary_max - 1.0e-6f)
            ++boundary_steps;
    }

    return {
        weights[0],
        weight_sum / steps,
        minimum_seen,
        maximum_seen,
        static_cast<double>(boundary_steps) / steps
    };
}

void write_stage_b(const std::filesystem::path& output_dir) {
    const std::vector<float> balance_ratios{
        0.950f, 0.975f, 1.000f, 1.025f, 1.050f
    };
    const std::vector<float> rates_hz{5.0f, 10.0f, 20.0f, 40.0f};
    constexpr int replicates = 20;
    constexpr float duration_seconds = 300.0f;

    std::ofstream csv(output_dir / "stage_b_stability.csv");
    if (!csv) throw std::runtime_error("Could not create Stage B CSV");

    csv << "balance_ratio,pre_rate_hz,post_rate_hz,seed,"
           "initial_weight,final_weight,mean_weight,drift_per_second,"
           "minimum_seen,maximum_seen,boundary_fraction\n";
    csv << std::setprecision(10);

    for (const float ratio : balance_ratios) {
        for (const float rate : rates_hz) {
            for (int replicate = 0; replicate < replicates; ++replicate) {
                const std::uint32_t seed =
                    1000u + static_cast<std::uint32_t>(replicate);

                const auto result = run_independent_activity(
                    rate, rate, ratio, kInitialWeight, seed);

                const double drift =
                    (result.final_weight - kInitialWeight) /
                    duration_seconds;

                csv << ratio << ',' << rate << ',' << rate << ','
                    << seed << ',' << kInitialWeight << ','
                    << result.final_weight << ',' << result.mean_weight << ','
                    << drift << ',' << result.minimum_seen << ','
                    << result.maximum_seen << ','
                    << result.boundary_fraction << '\n';
            }
        }
    }

    std::cout << "Stage B written: " << (output_dir / "stage_b_stability.csv")
              << '\n';
}


void write_initial_weight_experiment(
    const std::filesystem::path& output_dir) {

    const std::vector<float> initial_weights{0.1f, 0.5f, 0.9f};
    const std::vector<float> balance_ratios{0.975f, 1.0f, 1.025f};
    const std::vector<float> rates_hz{10.0f, 20.0f, 40.0f};

    constexpr int replicates = 20;
    constexpr float duration_seconds = 300.0f;

    std::ofstream csv(output_dir / "stage_b_initial_weight.csv");
    if (!csv)
        throw std::runtime_error("Could not create initial-weight CSV");

    csv << "initial_weight,balance_ratio,pre_rate_hz,post_rate_hz,seed,"
           "final_weight,mean_weight,drift_per_second,minimum_seen,"
           "maximum_seen,boundary_fraction\n";
    csv << std::setprecision(10);

    for (const float initial_weight : initial_weights) {
        for (const float ratio : balance_ratios) {
            for (const float rate : rates_hz) {
                for (int replicate = 0; replicate < replicates; ++replicate) {
                    // The same seed is reused for each initial weight.
                    // This pairs the activity realization across conditions.
                    const std::uint32_t seed =
                        5000u + static_cast<std::uint32_t>(replicate);

                    const auto result = run_independent_activity(
                        rate, rate, ratio, initial_weight, seed);

                    const double drift =
                        (result.final_weight - initial_weight) /
                        duration_seconds;

                    csv << initial_weight << ',' << ratio << ','
                        << rate << ',' << rate << ',' << seed << ','
                        << result.final_weight << ',' << result.mean_weight
                        << ',' << drift << ',' << result.minimum_seen << ','
                        << result.maximum_seen << ','
                        << result.boundary_fraction << '\n';
                }
            }
        }
    }

    std::cout << "Initial-weight experiment written: "
              << (output_dir / "stage_b_initial_weight.csv") << '\n';
}


void write_boundary_policy_experiment(
    const std::filesystem::path& output_dir) {

    const std::vector<float> ratios{0.975f, 1.0f, 1.025f};
    const std::vector<float> rates_hz{10.0f, 20.0f, 40.0f};

    struct PolicyEntry {
        BoundaryPolicy policy;
        const char* name;
    };

    const std::vector<PolicyEntry> policies{
        {BoundaryPolicy::HardClamp, "hard_clamp"},
        {BoundaryPolicy::WideRange, "wide_range"},
        {BoundaryPolicy::Reflecting, "reflecting"}
    };

    constexpr float initial_weight = 0.5f;
    constexpr float duration_seconds = 300.0f;
    constexpr int replicates = 20;

    std::ofstream csv(output_dir / "stage_c_boundary_policies.csv");
    if (!csv)
        throw std::runtime_error("Could not create boundary-policy CSV");

    csv << "policy,initial_weight,balance_ratio,rate_hz,seed,"
           "final_weight,mean_weight,drift_per_second,minimum_seen,"
           "maximum_seen,boundary_fraction\n";
    csv << std::setprecision(10);

    for (const auto& policy : policies) {
        for (const float ratio : ratios) {
            for (const float rate : rates_hz) {
                for (int replicate = 0; replicate < replicates; ++replicate) {
                    const std::uint32_t seed =
                        5000u + static_cast<std::uint32_t>(replicate);

                    const auto result = run_independent_activity(
                        rate, rate, ratio, initial_weight, seed,
                        policy.policy);

                    const double drift =
                        (result.final_weight - initial_weight) /
                        duration_seconds;

                    csv << policy.name << ',' << initial_weight << ','
                        << ratio << ',' << rate << ',' << seed << ','
                        << result.final_weight << ',' << result.mean_weight
                        << ',' << drift << ',' << result.minimum_seen << ','
                        << result.maximum_seen << ','
                        << result.boundary_fraction << '\n';
                }
            }
        }
    }

    std::cout << "Boundary-policy experiment written: "
              << (output_dir / "stage_c_boundary_policies.csv") << '\n';
}


void write_stage_d_response_surface(
    const std::filesystem::path& output_dir) {

    constexpr float initial_weight = 100.0f;
    constexpr float duration_seconds = 300.0f;
    constexpr int replicates = 20;

    const std::vector<float> rates_hz{10.0f, 20.0f, 40.0f};
    const std::vector<float> ratios{
        0.9850f, 0.9875f, 0.9900f, 0.9925f,
        0.9950f, 0.9975f, 1.0000f, 1.0025f,
        1.0050f, 1.0075f, 1.0100f, 1.0125f,
        1.0150f
    };

    std::ofstream csv(output_dir / "stage_d_response_surface.csv");
    if (!csv)
        throw std::runtime_error("Could not create Stage D CSV");

    csv << "initial_weight,balance_ratio,rate_hz,seed,"
           "final_weight,mean_weight,drift_per_second,"
           "minimum_seen,maximum_seen,boundary_fraction\n";
    csv << std::setprecision(10);

    for (const float rate : rates_hz) {
        for (const float ratio : ratios) {
            for (int replicate = 0; replicate < replicates; ++replicate) {
                const std::uint32_t seed =
                    5000u + static_cast<std::uint32_t>(replicate);

                const auto result = run_independent_activity(
                    rate, rate, ratio, initial_weight, seed,
                    BoundaryPolicy::WideRange);

                const double drift =
                    (result.final_weight - initial_weight) /
                    duration_seconds;

                csv << initial_weight << ',' << ratio << ','
                    << rate << ',' << seed << ','
                    << result.final_weight << ','
                    << result.mean_weight << ','
                    << drift << ','
                    << result.minimum_seen << ','
                    << result.maximum_seen << ','
                    << result.boundary_fraction << '\n';
            }
        }
    }

    std::cout << "Stage D response surface written: "
              << (output_dir / "stage_d_response_surface.csv") << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::filesystem::path output_dir =
            argc > 1 ? argv[1] : "stdp_analysis";
        std::filesystem::create_directories(output_dir);

        write_stage_a(output_dir);
        write_stage_b(output_dir);
        write_initial_weight_experiment(output_dir);
        write_boundary_policy_experiment(output_dir);
        write_stage_d_response_surface(output_dir);

        std::cout << "STDP experiments complete.\n"
                  << "  dt: 1 ms\n"
                  << "  Stage A: isolated spike pairs\n"
                  << "  Stage B: independent seeded spike trains, 300 seconds/run\n"
                  << "  Stage B: 5 balance ratios x 4 rates x 20 replicates\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "STDP experiment failed: " << error.what() << '\n';
        return 1;
    }
}
