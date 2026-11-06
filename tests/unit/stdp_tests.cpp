#include "plasticity/stdp.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
bool near(float a, float b, float tolerance = 1.0e-6f) {
    return std::abs(a - b) <= tolerance;
}
}

int main() {
    using namespace spiking;
    const std::vector<std::uint32_t> sources{0, 0, 2};
    const std::vector<std::uint32_t> targets{1, 2, 0};
    const std::vector<std::uint8_t> no_spikes{0, 0, 0};
    StdpParameters p;
    p.potentiation_rate = 0.1f;
    p.depression_rate = 0.2f;
    p.presynaptic_tau_ms = 10.0f;
    p.postsynaptic_tau_ms = 10.0f;
    p.minimum_weight = 0.05f;
    p.maximum_weight = 0.9f;

    {
        std::vector<float> w{0.5f, 0.5f, -0.5f};
        StdpTraces traces{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
        auto pre = no_spikes; pre[0] = 1;
        apply_pair_stdp_step(w, sources, targets, 3, 2, pre, traces, 1.0f, p);
        auto post = no_spikes; post[1] = 1;
        apply_pair_stdp_step(w, sources, targets, 3, 2, post, traces, 1.0f, p);
        const float expected = 0.5f + 0.1f * std::exp(-0.1f);
        check(near(w[0], expected), "pre-before-post produces trace-based potentiation");
    }
    {
        std::vector<float> w{0.5f, 0.5f, -0.5f};
        StdpTraces traces{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
        auto post = no_spikes; post[1] = 1;
        apply_pair_stdp_step(w, sources, targets, 3, 2, post, traces, 1.0f, p);
        auto pre = no_spikes; pre[0] = 1;
        apply_pair_stdp_step(w, sources, targets, 3, 2, pre, traces, 1.0f, p);
        const float expected = 0.5f - 0.2f * std::exp(-0.1f);
        check(near(w[0], expected), "post-before-pre produces trace-based depression");
    }
    {
        std::vector<float> w{0.5f, 0.5f, -0.5f};
        StdpTraces traces{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
        auto both = no_spikes; both[0] = 1; both[1] = 1;
        apply_pair_stdp_step(w, sources, targets, 3, 2, both, traces, 1.0f, p);
        check(near(w[0], 0.5f), "simultaneous spikes have no zero-lag weight change");
        check(near(traces.presynaptic[0], 1.0f) &&
              near(traces.postsynaptic[1], 1.0f),
              "simultaneous spikes increment traces after updates");
    }
    {
        std::vector<float> w{0.899f, 0.5f, -0.5f};
        StdpTraces traces{{10.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
        auto post = no_spikes; post[1] = 1;
        apply_pair_stdp_step(w, sources, targets, 3, 2, post, traces, 1.0f, p);
        check(near(w[0], p.maximum_weight), "potentiation is clamped to maximum weight");
        check(near(w[2], -0.5f), "inhibitory-source weight is not modified");
    }


    {
        // Validate the complete pair-based STDP timing window against
        // its analytical exponential response.
        StdpParameters curve = p;
        curve.potentiation_rate = 0.01f;
        curve.depression_rate = 0.012f;
        curve.presynaptic_tau_ms = 20.0f;
        curve.postsynaptic_tau_ms = 25.0f;
        curve.minimum_weight = 0.05f;
        curve.maximum_weight = 0.9f;

        const std::vector<std::uint32_t> pair_sources{0};
        const std::vector<std::uint32_t> pair_targets{1};
        const std::vector<std::uint8_t> quiet{0, 0};
        const std::vector<int> lags_ms{1, 5, 10, 20, 40, 100};

        auto run_pair = [&](bool pre_first, int lag_ms) {
            std::vector<float> weights{0.5f};
            StdpTraces traces{{0.0f, 0.0f}, {0.0f, 0.0f}};

            auto first = quiet;
            first[pre_first ? 0 : 1] = 1;
            apply_pair_stdp_step(
                weights, pair_sources, pair_targets, 2, 2,
                first, traces, 1.0f, curve);

            for (int elapsed = 1; elapsed < lag_ms; ++elapsed) {
                apply_pair_stdp_step(
                    weights, pair_sources, pair_targets, 2, 2,
                    quiet, traces, 1.0f, curve);
            }

            auto second = quiet;
            second[pre_first ? 1 : 0] = 1;
            apply_pair_stdp_step(
                weights, pair_sources, pair_targets, 2, 2,
                second, traces, 1.0f, curve);

            return weights[0] - 0.5f;
        };

        float previous_ltp = 1.0f;
        float previous_ltd = -1.0f;

        for (const int lag : lags_ms) {
            const float ltp = run_pair(true, lag);
            const float ltd = run_pair(false, lag);

            const float expected_ltp =
                curve.potentiation_rate *
                std::exp(-static_cast<float>(lag) /
                         curve.presynaptic_tau_ms);
            const float expected_ltd =
                -curve.depression_rate *
                std::exp(-static_cast<float>(lag) /
                         curve.postsynaptic_tau_ms);

            check(near(ltp, expected_ltp, 2.0e-6f),
                  "STDP potentiation does not match its exponential timing curve");
            check(near(ltd, expected_ltd, 2.0e-6f),
                  "STDP depression does not match its exponential timing curve");

            check(ltp > 0.0f && ltp < previous_ltp,
                  "STDP potentiation should weaken with increasing pre-post interval");
            check(ltd < 0.0f && ltd > previous_ltd,
                  "STDP depression should weaken with increasing post-pre interval");

            previous_ltp = ltp;
            previous_ltd = ltd;
        }

        check(std::abs(run_pair(true, 100)) < 3.0e-4f,
              "STDP potentiation should approach zero at long intervals");
        check(std::abs(run_pair(false, 100)) < 3.0e-4f,
              "STDP depression should approach zero at long intervals");
    }

    if (failures == 0) {
        std::cout << "All STDP reference tests passed.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " STDP reference test(s) failed.\n";
    return EXIT_FAILURE;
}
