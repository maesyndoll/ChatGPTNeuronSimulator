#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/csr_topology.h"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace spiking;

constexpr std::uint32_t kFeatures = 64;
constexpr std::uint32_t kOutputA = 64;
constexpr std::uint32_t kOutputB = 65;
constexpr std::uint32_t kInhibitory = 1;
constexpr std::uint32_t kTotalNeurons = 67;
constexpr std::uint32_t kTrainingSamplesPerClass = 8;
constexpr std::uint32_t kTestSamplesPerClass = 8;
constexpr std::uint32_t kStimulusStart = 10;
constexpr std::uint32_t kStimulusDuration = 4;
constexpr float kInputCurrent = 500.0f;
constexpr float kDtMs = 0.5f;
constexpr float kSpikeThresholdMv = -65.0f;
constexpr float kMinimumWeight = 0.01f;
constexpr float kDiscriminativeGain = 1.5f;
std::string kPackageDirectory = "neuromancer_v23_model";
std::string kTopologyPath;
std::string kManifestPath;
std::string kResultsPath;

struct Example {
    std::uint32_t label = 0;
    std::array<std::uint8_t, kFeatures> features{};
};

struct InferenceResult {
    std::uint32_t truth = 0;
    std::uint32_t prediction = 0;
    std::uint64_t output_a_spikes = 0;
    std::uint64_t output_b_spikes = 0;
    std::uint64_t input_spikes = 0;
    std::uint64_t dropped_events = 0;
};

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = kFeatures + 2U;
    p.simulation.inhibitory_count = kInhibitory;
    p.simulation.steps = 100;
    p.simulation.dt = kDtMs;
    p.simulation.threshold = kSpikeThresholdMv;
    p.simulation.random_seed = 23001;
    p.simulation.initial_voltage_min = -65.0f;
    p.simulation.initial_voltage_max = -65.0f;
    p.simulation.excitatory_current_min = 0.0f;
    p.simulation.excitatory_current_max = 0.0f;
    p.simulation.inhibitory_current_min = 0.0f;
    p.simulation.inhibitory_current_max = 0.0f;
    p.topology.minimum_delay = 1;
    p.topology.maximum_delay = 1;
    p.topology.fibonacci_delays = {1};
    p.topology.fibonacci_weights = {1.0};
    return p;
}

Example make_example(std::uint32_t label, std::uint32_t sample,
                     std::uint32_t seed) {
    Example example;
    example.label = label;
    const std::uint32_t preferred_begin = label == 0 ? 0U : 32U;
    const std::uint32_t other_begin = label == 0 ? 32U : 0U;

    for (std::uint32_t i = 0; i < 32; ++i)
        example.features[preferred_begin + i] = 1;

    std::mt19937 rng(seed + label * 1009U + sample * 9176U);
    std::uniform_int_distribution<std::uint32_t> feature(0, 31);
    const auto drop_a = feature(rng);
    auto drop_b = feature(rng);
    while (drop_b == drop_a) drop_b = feature(rng);
    const auto noise_a = feature(rng);
    auto noise_b = feature(rng);
    while (noise_b == noise_a) noise_b = feature(rng);

    example.features[preferred_begin + drop_a] = 0;
    example.features[preferred_begin + drop_b] = 0;
    example.features[other_begin + noise_a] = 1;
    example.features[other_begin + noise_b] = 1;
    return example;
}

std::vector<Example> make_training_set() {
    std::vector<Example> examples;
    examples.reserve(2U * kTrainingSamplesPerClass);
    for (std::uint32_t label = 0; label < 2; ++label) {
        for (std::uint32_t sample = 0; sample < kTrainingSamplesPerClass; ++sample)
            examples.push_back(make_example(label, sample, 23023U));
    }
    return examples;
}

std::vector<Example> make_test_set() {
    std::vector<Example> examples;
    examples.reserve(2U * kTestSamplesPerClass);
    for (std::uint32_t label = 0; label < 2; ++label) {
        for (std::uint32_t sample = kTrainingSamplesPerClass;
             sample < kTrainingSamplesPerClass + kTestSamplesPerClass; ++sample)
            examples.push_back(make_example(label, sample, 23023U));
    }
    return examples;
}

CSRTopology train_readout(const std::vector<Example>& training) {
    std::array<double, 2> class_counts{{0.0, 0.0}};
    std::array<std::array<double, kFeatures>, 2> active_counts{};

    for (const auto& example : training) {
        ++class_counts[example.label];
        for (std::uint32_t feature = 0; feature < kFeatures; ++feature)
            active_counts[example.label][feature] += example.features[feature] ? 1.0 : 0.0;
    }

    std::array<std::array<float, 2>, kFeatures> weights{};
    for (std::uint32_t feature = 0; feature < kFeatures; ++feature) {
        const double frequency_a = active_counts[0][feature] / class_counts[0];
        const double frequency_b = active_counts[1][feature] / class_counts[1];
        const double evidence_a = std::max(0.0, frequency_a - frequency_b);
        const double evidence_b = std::max(0.0, frequency_b - frequency_a);
        weights[feature][0] = kMinimumWeight +
            kDiscriminativeGain * static_cast<float>(evidence_a);
        weights[feature][1] = kMinimumWeight +
            kDiscriminativeGain * static_cast<float>(evidence_b);
    }

    CSRTopology graph;
    graph.excitatory_count = kFeatures + 2U;
    graph.inhibitory_count = kInhibitory;
    graph.minimum_delay = 1;
    graph.maximum_delay = 1;
    graph.row_offsets.assign(static_cast<std::size_t>(kTotalNeurons) + 1U, 0U);

    for (std::uint32_t source = 0; source < kFeatures; ++source) {
        graph.target_indices.push_back(kOutputA);
        graph.weights.push_back(weights[source][0]);
        graph.delays.push_back(1);
        graph.target_indices.push_back(kOutputB);
        graph.weights.push_back(weights[source][1]);
        graph.delays.push_back(1);
        graph.row_offsets[source + 1U] =
            static_cast<std::uint32_t>(graph.target_indices.size());
    }
    for (std::uint32_t source = kFeatures; source < kTotalNeurons; ++source)
        graph.row_offsets[source + 1U] =
            static_cast<std::uint32_t>(graph.target_indices.size());

    graph.validate();
    return graph;
}

void save_manifest() {
    std::ofstream out(kManifestPath);
    if (!out) throw std::runtime_error("Could not write Neuromancer model manifest");
    out << "NEUROMANCER_MODEL 1\n"
        << "task two_class_binary_feature_inference\n"
        << "input_features " << kFeatures << "\n"
        << "output_neurons " << kOutputA << ' ' << kOutputB << "\n"
        << "dt_ms " << kDtMs << "\n"
        << "spike_threshold_mv " << kSpikeThresholdMv << "\n"
        << "topology_file " << kTopologyPath << "\n"
        << "readout supervised_class_conditional_feature_frequency\n";
}

InferenceResult infer(const Example& example, const ModelParameters& model) {
    CudaSimulationRuntime runtime(
        model, kTopologyPath, 0, true, false);

    std::uint64_t active_features = 0;
    for (std::uint32_t feature = 0; feature < kFeatures; ++feature) {
        if (!example.features[feature]) continue;
        ++active_features;
        runtime.schedule_stimulus(
            feature, kStimulusStart, kStimulusDuration, kInputCurrent);
    }

    runtime.run_steps(model.simulation.steps);
    const auto events = runtime.collect_spike_events();

    InferenceResult result;
    result.truth = example.label;
    result.dropped_events = events.dropped_events;
    for (const auto& event : events.events) {
        if (event.neuron_id < kFeatures)
            ++result.input_spikes;
        else if (event.neuron_id == kOutputA)
            ++result.output_a_spikes;
        else if (event.neuron_id == kOutputB)
            ++result.output_b_spikes;
    }

    if (result.output_a_spikes == result.output_b_spikes)
        throw std::runtime_error("Output spike-count decoder produced a tie");
    result.prediction =
        result.output_a_spikes > result.output_b_spikes ? 0U : 1U;

    if (active_features == 0)
        throw std::runtime_error("Test example has no active input features");
    if (result.input_spikes == 0)
        throw std::runtime_error("Input encoder produced no presynaptic spikes");
    if (result.dropped_events != 0)
        throw std::runtime_error("Spike event recorder dropped inference events");
    return result;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("Usage: neuromancer_inference_prototype_v23 [package-directory]");
        if (argc == 2) kPackageDirectory = argv[1];
        std::filesystem::create_directories(kPackageDirectory);
        const auto package = std::filesystem::absolute(kPackageDirectory).lexically_normal();
        kTopologyPath = (package / "topology.csr").string();
        kManifestPath = (package / "model.nmc").string();
        kResultsPath = (package / "inference.csv").string();

        const auto model = make_model();
        const auto training = make_training_set();
        const auto testing = make_test_set();

        const auto learned_readout = train_readout(training);
        save_csr_topology(learned_readout, kTopologyPath);
        save_manifest();

        std::cout << "\nNEUROMANCER — TWO-CLASS SPIKING INFERENCE PROTOTYPE V23\n"
                  << "--------------------------------------------------------\n"
                  << "CUDA Izhikevich runtime; " << kFeatures
                  << " input features; 2 output neurons\n"
                  << "Training: " << training.size()
                  << " labeled examples; frozen supervised readout\n"
                  << "Inference: " << testing.size()
                  << " held-out examples; no learning during inference\n"
                  << "Input pulse: " << kInputCurrent << " for "
                  << kStimulusDuration << " steps; dt=" << kDtMs << " ms\n"
                  << "Output decoder: winner by output spike count\n\n";

        std::ofstream csv(kResultsPath);
        if (!csv) throw std::runtime_error("Could not write inference results CSV");
        csv << "true_class,predicted_class,output_a_spikes,output_b_spikes,input_spikes,dropped_events\n";

        std::uint32_t correct = 0;
        std::array<std::uint32_t, 4> confusion{};
        std::uint64_t total_input_spikes = 0;
        std::uint64_t total_output_spikes = 0;

        for (const auto& example : testing) {
            const auto result = infer(example, model);
            if (result.prediction == result.truth) ++correct;
            ++confusion[result.truth * 2U + result.prediction];
            total_input_spikes += result.input_spikes;
            total_output_spikes +=
                result.output_a_spikes + result.output_b_spikes;

            csv << result.truth << ',' << result.prediction << ','
                << result.output_a_spikes << ',' << result.output_b_spikes << ','
                << result.input_spikes << ',' << result.dropped_events << '\n';

            std::cout << "  true=" << result.truth
                      << " predicted=" << result.prediction
                      << " output_spikes=[" << result.output_a_spikes
                      << ',' << result.output_b_spikes << ']'
                      << " input_spikes=" << result.input_spikes << '\n';
        }
        csv.close();
        if (!csv) throw std::runtime_error("Failed while writing inference results CSV");

        const double accuracy =
            100.0 * static_cast<double>(correct) / static_cast<double>(testing.size());

        std::cout << "\nCONFUSION MATRIX (rows=true class 0/1; columns=predicted 0/1)\n"
                  << confusion[0] << "  " << confusion[1] << '\n'
                  << confusion[2] << "  " << confusion[3] << '\n'
                  << std::fixed << std::setprecision(1)
                  << "Accuracy: " << correct << '/' << testing.size()
                  << " (" << accuracy << "%)\n"
                  << "Total input spikes: " << total_input_spikes << '\n'
                  << "Total output spikes: " << total_output_spikes << '\n'
                  << "Saved model manifest: " << kManifestPath << '\n'
                  << "Saved CSR readout: " << kTopologyPath << '\n'
                  << "Results CSV: " << kResultsPath << '\n';

        require(correct == testing.size(),
                "Held-out inference accuracy was below 100%; inspect output recruitment");
        require(total_output_spikes > 0,
                "Inference produced no output spikes");

        std::cout << "\nPASS: trained readout loaded into fresh CUDA runtimes, "
                     "all held-out examples classified by output spike counts, "
                     "and plasticity remained disabled during inference.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "V23 inference prototype failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
