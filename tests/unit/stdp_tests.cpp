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

    if (failures == 0) {
        std::cout << "All STDP reference tests passed.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " STDP reference test(s) failed.\n";
    return EXIT_FAILURE;
}
