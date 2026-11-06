#include "plasticity/stdp_timing_optimizer.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
bool near(double a, double b, double tolerance) {
    return std::abs(a - b) <= tolerance;
}
}

int main() {
    using namespace spiking;
    try {
        StdpParameters truth;
        truth.potentiation_rate = 0.008f;
        truth.depression_rate = 0.001f;
        truth.presynaptic_tau_ms = 20.0f;
        truth.postsynaptic_tau_ms = 20.0f;

        const std::vector<double> lags{1.5, -1.5, 5.0, -5.0};
        std::vector<StdpTimingObservation> observations;
        for (double lag : lags) {
            observations.push_back({
                lag, predict_periodic_stdp_pair(lag, 50.0, truth).delta_weight
            });
        }

        StdpParameters initial;
        initial.potentiation_rate = 0.005f;
        initial.depression_rate = 0.00525f;
        const auto fit = fit_stdp_timing_curve(observations, initial, 50.0);
        check(fit.converged, "nonlinear STDP fit did not converge");
        check(fit.final_loss < fit.initial_loss, "fit did not reduce objective");
        check(near(fit.parameters.potentiation_rate, 0.008, 2.0e-5),
              "potentiation parameter fit is inaccurate");
        check(near(fit.parameters.depression_rate, 0.001, 2.0e-5),
              "depression parameter fit is inaccurate");
        check(near(fit.parameters.presynaptic_tau_ms, 20.0, 2.0e-3),
              "presynaptic tau fit is inaccurate");
        check(near(fit.parameters.postsynaptic_tau_ms, 20.0, 2.0e-3),
              "postsynaptic tau fit is inaccurate");

        // Verify every analytical partial against a centered finite difference.
        const auto prediction = predict_periodic_stdp_pair(1.5, 50.0, truth);
        const double values[4] = {
            truth.potentiation_rate, truth.depression_rate,
            truth.presynaptic_tau_ms, truth.postsynaptic_tau_ms
        };
        for (std::size_t i = 0; i < 4; ++i) {
            StdpParameters plus = truth, minus = truth;
            const double h = i < 2 ? 1.0e-6 : 1.0e-4;
            float* plus_values[4] = {
                &plus.potentiation_rate, &plus.depression_rate,
                &plus.presynaptic_tau_ms, &plus.postsynaptic_tau_ms
            };
            float* minus_values[4] = {
                &minus.potentiation_rate, &minus.depression_rate,
                &minus.presynaptic_tau_ms, &minus.postsynaptic_tau_ms
            };
            *plus_values[i] = static_cast<float>(values[i] + h);
            *minus_values[i] = static_cast<float>(values[i] - h);
            const double numerical =
                (predict_periodic_stdp_pair(1.5, 50.0, plus).delta_weight -
                 predict_periodic_stdp_pair(1.5, 50.0, minus).delta_weight) /
                (2.0 * h);
            check(near(prediction.partials[i], numerical,
                       i < 2 ? 2.0e-5 : 2.0e-7),
                  "analytical partial does not match finite difference");
        }

        std::cout << "STDP timing optimizer tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "STDP timing optimizer test failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
