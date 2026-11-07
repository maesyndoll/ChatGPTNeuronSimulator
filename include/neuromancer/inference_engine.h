#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace neuromancer {

struct InferenceConfig {
    std::uint32_t input_neurons = 64;
    std::uint32_t output_a_neuron = 64;
    std::uint32_t output_b_neuron = 65;
    std::uint32_t simulation_steps = 100;
    std::uint32_t stimulus_start_step = 10;
    std::uint32_t stimulus_duration_steps = 4;
    float dt_ms = 0.5f;
    float spike_threshold_mv = -65.0f;
    float input_current = 500.0f;
    float active_feature_threshold = 0.5f;
};

struct InferenceResult {
    std::uint32_t predicted_class = UINT32_MAX;
    std::uint64_t output_a_spikes = 0;
    std::uint64_t output_b_spikes = 0;
    std::uint64_t input_spikes = 0;
    std::uint64_t dropped_events = 0;
    bool has_prediction = false;
};

class InferenceEngine {
public:
    // Loads and validates a persisted SPKCSR model. The model topology is
    // immutable during inference; each infer() call starts a fresh runtime.
    explicit InferenceEngine(std::string topology_file,
                             InferenceConfig config = {});

    InferenceResult infer(const std::vector<float>& features) const;

    std::uint32_t input_size() const noexcept;
    const std::string& model_path() const noexcept;

private:
    std::string topology_file_;
    InferenceConfig config_;
    std::uint32_t excitatory_count_ = 0;
    std::uint32_t inhibitory_count_ = 0;
};

} // namespace neuromancer
