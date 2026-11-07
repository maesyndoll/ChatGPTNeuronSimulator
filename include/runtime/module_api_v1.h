#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NEUROMANCER_MODULE_ABI_V1 1u

typedef struct NeuromancerSpikeEventV1 {
    uint64_t step;
    uint32_t neuron_id;
} NeuromancerSpikeEventV1;

typedef struct NeuromancerHostApiV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    void* user_data;
    void (*log)(void* user_data, int level, const char* message);
    int (*set_edge_plasticity)(void* user_data, uint32_t first_edge,
                               uint32_t edge_count, uint8_t enabled);
} NeuromancerHostApiV1;

typedef struct NeuromancerModuleV1 {
    uint32_t abi_version;
    uint32_t struct_size;
    const char* name;
    int (*on_load)(const NeuromancerHostApiV1* host, void** instance);
    int (*configure)(void* instance, const char* key, const char* value);
    void (*on_spikes)(void* instance, const NeuromancerSpikeEventV1* events,
                      uint32_t event_count);
    void (*on_unload)(void* instance);
} NeuromancerModuleV1;

typedef const NeuromancerModuleV1* (*NeuromancerModuleQueryV1)(void);

#ifdef __cplusplus
}
#endif
