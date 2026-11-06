#pragma once

#include "plasticity/stdp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace spiking {

// Expected weight change per repeated, periodically paired pre/post event.
// Positive lag means pre fires before post; negative lag means post fires first.
// This is the exact periodic-trace sum for the pair-based exponential rule,
// before weight clipping and network feedback are applied.
struct StdpTimingObservation {
    double lag_ms = 0.0;
    double target_delta_weight = 0.0;
};

struct StdpTimingPrediction {
    double delta_weight = 0.0;
    // Order: potentiation rate, depression rate, pre tau, post tau.
    std::array<double, 4> partials{};
};

struct StdpFitResult {
    StdpParameters parameters;
    double initial_loss = 0.0;
    double final_loss = 0.0;
    std::size_t iterations = 0;
    bool converged = false;
};

inline StdpTimingPrediction predict_periodic_stdp_pair(
    double lag_ms, double interval_ms, const StdpParameters& p) {
    if (!std::isfinite(lag_ms) || !std::isfinite(interval_ms) ||
        interval_ms <= 0.0 || std::abs(lag_ms) >= interval_ms ||
        !std::isfinite(p.potentiation_rate) || p.potentiation_rate < 0.0f ||
        !std::isfinite(p.depression_rate) || p.depression_rate < 0.0f ||
        !std::isfinite(p.presynaptic_tau_ms) || p.presynaptic_tau_ms <= 0.0f ||
        !std::isfinite(p.postsynaptic_tau_ms) || p.postsynaptic_tau_ms <= 0.0f) {
        throw std::invalid_argument("Invalid periodic STDP prediction inputs");
    }

    const double tp = p.presynaptic_tau_ms;
    const double tm = p.postsynaptic_tau_ms;
    const double ap = p.potentiation_rate;
    const double am = p.depression_rate;
    const double distance = std::abs(lag_ms);
    const bool causal = lag_ms >= 0.0;

    const double pre_decay = std::exp(-interval_ms / tp);
    const double post_decay = std::exp(-interval_ms / tm);
    const double pre_denominator = 1.0 - pre_decay;
    const double post_denominator = 1.0 - post_decay;

    // For causal pairs, current pre activity potentiates the current post
    // event; depression comes from the previous cycle's post trace.
    // For anti-causal pairs, the roles are reversed.
    const double pre_exponent = causal ? distance : interval_ms - distance;
    const double post_exponent = causal ? interval_ms - distance : distance;
    const double pre_factor =
        std::exp(-pre_exponent / tp) / pre_denominator;
    const double post_factor =
        std::exp(-post_exponent / tm) / post_denominator;

    const double pre_log_derivative =
        pre_exponent / (tp * tp) +
        interval_ms * pre_decay / (tp * tp * pre_denominator);
    const double post_log_derivative =
        post_exponent / (tm * tm) +
        interval_ms * post_decay / (tm * tm * post_denominator);

    StdpTimingPrediction result;
    result.delta_weight = ap * pre_factor - am * post_factor;
    result.partials[0] = pre_factor;
    result.partials[1] = -post_factor;
    result.partials[2] = ap * pre_factor * pre_log_derivative;
    result.partials[3] = -am * post_factor * post_log_derivative;
    return result;
}

namespace detail {
inline bool solve_stdp_4x4(
    std::array<std::array<double, 5>, 4>& matrix,
    std::array<double, 4>& solution) {
    for (std::size_t column = 0; column < 4; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < 4; ++row) {
            if (std::abs(matrix[row][column]) >
                std::abs(matrix[pivot][column])) pivot = row;
        }
        if (std::abs(matrix[pivot][column]) < 1.0e-18) return false;
        if (pivot != column) std::swap(matrix[pivot], matrix[column]);

        const double divisor = matrix[column][column];
        for (std::size_t j = column; j < 5; ++j)
            matrix[column][j] /= divisor;

        for (std::size_t row = 0; row < 4; ++row) {
            if (row == column) continue;
            const double factor = matrix[row][column];
            for (std::size_t j = column; j < 5; ++j)
                matrix[row][j] -= factor * matrix[column][j];
        }
    }
    for (std::size_t i = 0; i < 4; ++i) solution[i] = matrix[i][4];
    return true;
}
} // namespace detail

// Levenberg-Marquardt fit of all four differentiable STDP timing parameters.
// The scale vector keeps amplitudes and time constants numerically balanced.
inline StdpFitResult fit_stdp_timing_curve(
    const std::vector<StdpTimingObservation>& observations,
    const StdpParameters& initial,
    double interval_ms,
    std::size_t maximum_iterations = 200) {
    if (observations.size() < 4 || maximum_iterations == 0)
        throw std::invalid_argument("STDP fit requires at least four observations");

    StdpFitResult result;
    result.parameters = initial;
    const std::array<double, 4> scale{{0.01, 0.01, 20.0, 20.0}};
    const auto valid = [](const StdpParameters& p) {
        return p.potentiation_rate >= 1.0e-6f &&
               p.potentiation_rate <= 0.05f &&
               p.depression_rate >= 1.0e-6f &&
               p.depression_rate <= 0.05f &&
               p.presynaptic_tau_ms >= 2.0f &&
               p.presynaptic_tau_ms <= 100.0f &&
               p.postsynaptic_tau_ms >= 2.0f &&
               p.postsynaptic_tau_ms <= 100.0f;
    };
    if (!valid(result.parameters))
        throw std::invalid_argument("Initial STDP fit parameters are outside bounds");

    const auto loss = [&](const StdpParameters& p) {
        double sum = 0.0;
        for (const auto& observation : observations) {
            const double residual =
                predict_periodic_stdp_pair(
                    observation.lag_ms, interval_ms, p).delta_weight -
                observation.target_delta_weight;
            sum += residual * residual;
        }
        return sum;
    };

    result.initial_loss = loss(result.parameters);
    double current_loss = result.initial_loss;
    double damping = 1.0e-3;

    for (std::size_t iteration = 0; iteration < maximum_iterations; ++iteration) {
        std::array<std::array<double, 4>, 4> normal{};
        std::array<double, 4> gradient{};
        for (const auto& observation : observations) {
            const auto prediction = predict_periodic_stdp_pair(
                observation.lag_ms, interval_ms, result.parameters);
            const double residual =
                prediction.delta_weight - observation.target_delta_weight;
            std::array<double, 4> jacobian{};
            for (std::size_t j = 0; j < 4; ++j)
                jacobian[j] = prediction.partials[j] * scale[j];
            for (std::size_t row = 0; row < 4; ++row) {
                gradient[row] += jacobian[row] * residual;
                for (std::size_t column = 0; column < 4; ++column)
                    normal[row][column] += jacobian[row] * jacobian[column];
            }
        }

        std::array<std::array<double, 5>, 4> system{};
        for (std::size_t row = 0; row < 4; ++row) {
            for (std::size_t column = 0; column < 4; ++column)
                system[row][column] = normal[row][column];
            system[row][row] += damping;
            system[row][4] = gradient[row];
        }

        std::array<double, 4> scaled_step{};
        if (!detail::solve_stdp_4x4(system, scaled_step)) {
            damping *= 10.0;
            if (damping > 1.0e12) break;
            continue;
        }

        StdpParameters candidate = result.parameters;
        candidate.potentiation_rate = static_cast<float>(
            candidate.potentiation_rate - scale[0] * scaled_step[0]);
        candidate.depression_rate = static_cast<float>(
            candidate.depression_rate - scale[1] * scaled_step[1]);
        candidate.presynaptic_tau_ms = static_cast<float>(
            candidate.presynaptic_tau_ms - scale[2] * scaled_step[2]);
        candidate.postsynaptic_tau_ms = static_cast<float>(
            candidate.postsynaptic_tau_ms - scale[3] * scaled_step[3]);

        if (!valid(candidate)) {
            damping *= 10.0;
            if (damping > 1.0e12) break;
            continue;
        }

        const double candidate_loss = loss(candidate);
        if (candidate_loss < current_loss) {
            result.parameters = candidate;
            current_loss = candidate_loss;
            damping = std::max(1.0e-12, damping / 3.0);
            result.iterations = iteration + 1;
            if (candidate_loss < 1.0e-20 ||
                std::abs(result.final_loss - candidate_loss) < 1.0e-20) {
                result.converged = true;
                break;
            }
            result.final_loss = candidate_loss;
        } else {
            damping *= 10.0;
            if (damping > 1.0e12) break;
        }
    }

    result.final_loss = current_loss;
    if (current_loss < 1.0e-16) result.converged = true;
    return result;
}

} // namespace spiking
