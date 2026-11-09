#include "runtime/module_api_v1.h"

#include <cstdlib>
#include <cstring>
#include <new>

namespace {
struct State {
    NeuromancerHostApiV1 host{};
    uint64_t observed_events = 0;
    bool reported_activity = false;
};

int on_load(const NeuromancerHostApiV1* host, void** instance) {
    if (host == nullptr || instance == nullptr ||
        host->abi_version != NEUROMANCER_MODULE_ABI_V1 ||
        host->set_edge_plasticity == nullptr) return 1;
    auto* state = new (std::nothrow) State{};
    if (state == nullptr) return 2;
    state->host = *host;
    *instance = state;
    if (state->host.log) state->host.log(state->host.user_data, 1,
                                         "edge_policy attached");
    return 0;
}

int configure(void* opaque, const char* key, const char* value) {
    auto* state = static_cast<State*>(opaque);
    if (state == nullptr || key == nullptr || value == nullptr) return 1;
    if (std::strcmp(key, "edge_range") != 0) return 2;
    char* end = nullptr;
    const unsigned long first = std::strtoul(value, &end, 10);
    if (end == value || *end != ':') return 3;
    const char* count_text = end + 1;
    const unsigned long count = std::strtoul(count_text, &end, 10);
    if (end == count_text || *end != ':' || (end[1] != '0' && end[1] != '1') ||
        end[2] != '\0') return 4;
    const uint8_t enabled = static_cast<uint8_t>(end[1] == '1');
    return state->host.set_edge_plasticity(
        state->host.user_data, static_cast<uint32_t>(first),
        static_cast<uint32_t>(count), enabled) == 0 ? 0 : 5;
}

void on_spikes(void* opaque, const NeuromancerSpikeEventV1*, uint32_t count) {
    auto* state = static_cast<State*>(opaque);
    if (state != nullptr) {
        state->observed_events += count;
        if (!state->reported_activity && state->observed_events >= 10) {
            state->reported_activity = true;
            if (state->host.log) state->host.log(state->host.user_data, 1,
                                                  "edge_policy receiving live spike events");
        }
    }
}

void on_unload(void* opaque) {
    auto* state = static_cast<State*>(opaque);
    if (state != nullptr) {
        if (state->host.log) state->host.log(state->host.user_data, 1,
                                             "edge_policy detached");
        delete state;
    }
}

const NeuromancerModuleV1 module = {
    NEUROMANCER_MODULE_ABI_V1,
    sizeof(NeuromancerModuleV1),
    "edge_policy",
    on_load,
    configure,
    on_spikes,
    on_unload
};
}

extern "C" const NeuromancerModuleV1* neuromancer_module_query_v1() {
    return &module;
}
