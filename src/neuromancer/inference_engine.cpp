#include "neuromancer/inference_engine.h"

#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/topology_io.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace neuromancer {
namespace {

void parse_manifest(const std::string& path, InferenceConfig& config,
                    std::string& topology_file) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Could not open model manifest: " + path);

    std::string line;
    if (!std::getline(input, line) || line != "NEUROMANCER_MODEL 1")
        throw std::invalid_argument("Unsupported or missing Neuromancer model manifest header");

    bool have_features = false, have_outputs = false, have_topology = false;
    bool have_dt = false, have_threshold = false;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        std::string key;
        if (!(row >> key)) continue;
        if (key == "task" || key == "readout") {
            std::string value;
            if (!(row >> value)) throw std::invalid_argument("Missing value for manifest key: " + key);
        } else if (key == "input_features") {
            if (!(row >> config.input_neurons)) throw std::invalid_argument("Invalid input_features");
            have_features = true;
        } else if (key == "output_neurons") {
            if (!(row >> config.output_a_neuron >> config.output_b_neuron))
                throw std::invalid_argument("Invalid output_neurons");
            have_outputs = true;
        } else if (key == "dt_ms") {
            if (!(row >> config.dt_ms)) throw std::invalid_argument("Invalid dt_ms");
            have_dt = true;
        } else if (key == "spike_threshold_mv") {
            if (!(row >> config.spike_threshold_mv)) throw std::invalid_argument("Invalid spike_threshold_mv");
            have_threshold = true;
        } else if (key == "simulation_steps") {
            if (!(row >> config.simulation_steps)) throw std::invalid_argument("Invalid simulation_steps");
        } else if (key == "stimulus_start_step") {
            if (!(row >> config.stimulus_start_step)) throw std::invalid_argument("Invalid stimulus_start_step");
        } else if (key == "stimulus_duration_steps") {
            if (!(row >> config.stimulus_duration_steps)) throw std::invalid_argument("Invalid stimulus_duration_steps");
        } else if (key == "input_current") {
            if (!(row >> config.input_current)) throw std::invalid_argument("Invalid input_current");
        } else if (key == "active_feature_threshold") {
            if (!(row >> config.active_feature_threshold)) throw std::invalid_argument("Invalid active_feature_threshold");
        } else if (key == "topology_file") {
            if (!(row >> topology_file)) throw std::invalid_argument("Invalid topology_file");
            have_topology = true;
        } else {
            throw std::invalid_argument("Unknown model manifest key: " + key);
        }
        std::string extra;
        if (row >> extra) throw std::invalid_argument("Unexpected extra token on manifest line: " + key);
    }
    if (!input.eof()) throw std::runtime_error("Error reading model manifest");
    if (!have_features || !have_outputs || !have_topology || !have_dt || !have_threshold)
        throw std::invalid_argument("Model manifest is missing a required field");
    if (topology_file.empty()) throw std::invalid_argument("Manifest topology_file must not be empty");

    std::filesystem::path topology_path(topology_file);
    if (topology_path.is_relative())
        topology_path = std::filesystem::path(path).parent_path() / topology_path;
    topology_file = std::filesystem::absolute(topology_path).lexically_normal().string();
}

} // namespace

InferenceEngine::InferenceEngine(std::string manifest_file)
    : manifest_file_(std::filesystem::absolute(std::move(manifest_file)).lexically_normal().string()) {
    parse_manifest(manifest_file_, config_, topology_file_);

    if (config_.input_neurons == 0)
        throw std::invalid_argument("Inference model must have input neurons");
    if (config_.simulation_steps == 0 || config_.stimulus_duration_steps == 0 ||
        config_.stimulus_start_step + config_.stimulus_duration_steps > config_.simulation_steps)
        throw std::invalid_argument("Invalid inference stimulus or simulation window");
    if (!(config_.dt_ms > 0.0f) || !std::isfinite(config_.dt_ms) ||
        !std::isfinite(config_.spike_threshold_mv) || !std::isfinite(config_.input_current) ||
        !std::isfinite(config_.active_feature_threshold) ||
        config_.active_feature_threshold < 0.0f || config_.active_feature_threshold >= 1.0f)
        throw std::invalid_argument("Invalid inference configuration");

    const auto topology = spiking::load_csr_topology(topology_file_);
    excitatory_count_ = topology.excitatory_count;
    inhibitory_count_ = topology.inhibitory_count;
    const auto total = topology.neuron_count();
    if (config_.input_neurons > excitatory_count_ ||
        config_.output_a_neuron >= excitatory_count_ ||
        config_.output_b_neuron >= excitatory_count_ ||
        config_.output_a_neuron == config_.output_b_neuron ||
        config_.output_a_neuron < config_.input_neurons ||
        config_.output_b_neuron < config_.input_neurons ||
        config_.output_a_neuron >= total || config_.output_b_neuron >= total)
        throw std::invalid_argument("Input/output neuron IDs do not match model topology");
}

InferenceResult InferenceEngine::infer(const std::vector<float>& features) const {
    if (features.size() != config_.input_neurons)
        throw std::invalid_argument("Input feature vector has the wrong dimension");

    spiking::ModelParameters model;
    model.simulation.excitatory_count = excitatory_count_;
    model.simulation.inhibitory_count = inhibitory_count_;
    model.simulation.steps = config_.simulation_steps;
    model.simulation.dt = config_.dt_ms;
    model.simulation.threshold = config_.spike_threshold_mv;
    model.simulation.random_seed = 24001;
    model.simulation.initial_voltage_min = -65.0f;
    model.simulation.initial_voltage_max = -65.0f;
    model.simulation.excitatory_current_min = 0.0f;
    model.simulation.excitatory_current_max = 0.0f;
    model.simulation.inhibitory_current_min = 0.0f;
    model.simulation.inhibitory_current_max = 0.0f;
    model.topology.minimum_delay = 1;
    model.topology.maximum_delay = 1;
    model.topology.fibonacci_delays = {1};
    model.topology.fibonacci_weights = {1.0};

    spiking::CudaSimulationRuntime runtime(model, topology_file_, 0, true, false);
    std::uint64_t active_features = 0;
    for (std::uint32_t i = 0; i < config_.input_neurons; ++i) {
        const float value = features[i];
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
            throw std::invalid_argument("Input features must be finite values in [0, 1]");
        if (value <= config_.active_feature_threshold) continue;
        ++active_features;
        runtime.schedule_stimulus(i, config_.stimulus_start_step,
                                   config_.stimulus_duration_steps,
                                   config_.input_current * value);
    }
    if (active_features == 0) throw std::invalid_argument("Input vector has no active features");

    runtime.run_steps(config_.simulation_steps);
    const auto events = runtime.collect_spike_events();
    InferenceResult result;
    result.dropped_events = events.dropped_events;
    for (const auto& event : events.events) {
        if (event.neuron_id < config_.input_neurons) ++result.input_spikes;
        else if (event.neuron_id == config_.output_a_neuron) ++result.output_a_spikes;
        else if (event.neuron_id == config_.output_b_neuron) ++result.output_b_spikes;
    }
    if (result.dropped_events != 0)
        throw std::runtime_error("Spike event recorder dropped inference events");
    if (result.output_a_spikes != result.output_b_spikes) {
        result.has_prediction = true;
        result.predicted_class = result.output_a_spikes > result.output_b_spikes ? 0U : 1U;
    }
    return result;
}

std::uint32_t InferenceEngine::input_size() const noexcept { return config_.input_neurons; }
const std::string& InferenceEngine::model_path() const noexcept { return topology_file_; }
const InferenceConfig& InferenceEngine::config() const noexcept { return config_; }

} // namespace neuromancer
