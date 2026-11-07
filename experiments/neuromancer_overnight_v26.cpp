#include "core/parameters.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "topology/csr_topology.h"
#include "topology/topology_io.h"

#include <algorithm>
#include <array>
#include <chrono>
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
constexpr std::uint32_t kInputs = 64;
constexpr std::uint32_t kOutputA = 64;
constexpr std::uint32_t kOutputB = 65;
constexpr std::uint32_t kExcitatory = 1024;
constexpr std::uint32_t kInhibitory = 256;
constexpr std::uint32_t kTotal = kExcitatory + kInhibitory;
constexpr std::uint32_t kStepsPerExample = 100;
constexpr std::uint32_t kStimulusStartOffset = 1;
constexpr std::uint32_t kTeacherLag = 3;
constexpr std::uint32_t kPulseSteps = 4;
constexpr float kInputCurrent = 500.0f;
constexpr float kDtMs = 0.5f;
constexpr float kThresholdMv = 30.0f;
constexpr std::uint32_t kTrainingSamplesPerClass = 16;
constexpr std::uint32_t kTestSamplesPerClass = 16;

struct Example {
    std::uint32_t label = 0;
    std::array<std::uint8_t, kInputs> features{};
};
struct Score {
    std::uint32_t correct = 0;
    std::uint32_t total = 0;
    std::uint64_t input_spikes = 0;
    std::uint64_t output_spikes = 0;
    std::uint64_t class0_a_spikes = 0;
    std::uint64_t class0_b_spikes = 0;
    std::uint64_t class1_a_spikes = 0;
    std::uint64_t class1_b_spikes = 0;
    std::uint32_t class0_correct = 0;
    std::uint32_t class1_correct = 0;
    std::uint64_t dropped_events = 0;
};

ModelParameters make_model() {
    ModelParameters p;
    p.simulation.excitatory_count = kExcitatory;
    p.simulation.inhibitory_count = kInhibitory;
    p.simulation.steps = kStepsPerExample;
    p.simulation.dt = kDtMs;
    p.simulation.threshold = kThresholdMv;
    p.simulation.random_seed = 26001;
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

CSRTopology make_initial_topology(std::uint32_t seed) {
    CSRTopology graph;
    graph.excitatory_count = kExcitatory;
    graph.inhibitory_count = kInhibitory;
    graph.minimum_delay = 1;
    graph.maximum_delay = 1;
    graph.row_offsets.reserve(kTotal + 1U);
    graph.row_offsets.push_back(0U);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::vector<float> row_weights(kTotal, 0.0f);

    for (std::uint32_t source = 0; source < kTotal; ++source) {
        std::fill(row_weights.begin(), row_weights.end(), 0.0f);
        for (std::uint32_t target = 0; target < kTotal; ++target) {
            if (source == target) continue;
            float probability = 0.0f;
            if (source < kExcitatory && target < kExcitatory) probability = 0.004f;
            else if (source < kExcitatory && target >= kExcitatory) probability = 0.010f;
            else if (source >= kExcitatory && target < kExcitatory) probability = 0.020f;
            else probability = 0.010f;
            if (unit(rng) < probability)
                row_weights[target] = source < kExcitatory ? 0.02f : -0.02f;
        }

        // A small, explicit input-to-readout candidate set makes the learning
        // assay identifiable while the remaining graph is globally random.
        if (source < kInputs) {
            row_weights[kOutputA] = 0.02f;
            row_weights[kOutputB] = 0.02f;
        }
        for (std::uint32_t target = 0; target < kTotal; ++target) {
            if (row_weights[target] == 0.0f) continue;
            graph.target_indices.push_back(target);
            graph.weights.push_back(row_weights[target]);
            graph.delays.push_back(1U);
        }
        graph.row_offsets.push_back(
            static_cast<std::uint32_t>(graph.target_indices.size()));
    }
    graph.validate();
    return graph;
}

Example make_example(std::uint32_t label, std::uint32_t sample,
                     std::uint32_t seed) {
    Example ex;
    ex.label = label;
    const std::uint32_t preferred = label == 0 ? 0U : 32U;
    const std::uint32_t other = label == 0 ? 32U : 0U;
    for (std::uint32_t i = 0; i < 32; ++i) ex.features[preferred + i] = 1U;

    std::mt19937 rng(seed + label * 1009U + sample * 9176U);
    std::uniform_int_distribution<std::uint32_t> pick(0U, 31U);
    const auto drop_a = pick(rng);
    auto drop_b = pick(rng);
    while (drop_b == drop_a) drop_b = pick(rng);
    const auto noise_a = pick(rng);
    auto noise_b = pick(rng);
    while (noise_b == noise_a) noise_b = pick(rng);
    ex.features[preferred + drop_a] = 0U;
    ex.features[preferred + drop_b] = 0U;
    ex.features[other + noise_a] = 1U;
    ex.features[other + noise_b] = 1U;
    return ex;
}

std::vector<Example> make_training_set() {
    std::vector<Example> examples;
    for (std::uint32_t label = 0; label < 2; ++label)
        for (std::uint32_t i = 0; i < kTrainingSamplesPerClass; ++i)
            examples.push_back(make_example(label, i, 26023U));
    return examples;
}
std::vector<Example> make_test_set() {
    std::vector<Example> examples;
    for (std::uint32_t label = 0; label < 2; ++label)
        for (std::uint32_t i = kTrainingSamplesPerClass;
             i < kTrainingSamplesPerClass + kTestSamplesPerClass; ++i)
            examples.push_back(make_example(label, i, 26023U));
    return examples;
}

void train_one(CudaSimulationRuntime& runtime, const Example& ex) {
    const auto base = runtime.completed_steps();
    for (std::uint32_t i = 0; i < kInputs; ++i) {
        if (!ex.features[i]) continue;
        runtime.schedule_stimulus(i, base + kStimulusStartOffset,
                                  kPulseSteps, kInputCurrent);
    }
    const auto teacher = ex.label == 0 ? kOutputA : kOutputB;
    runtime.schedule_stimulus(teacher, base + kStimulusStartOffset + kTeacherLag,
                              kPulseSteps, kInputCurrent);
    runtime.run_steps(kStepsPerExample);
    runtime.clear_stimuli();
}

Score evaluate(const std::string& topology_path, const ModelParameters& model) {
    Score score;
    for (const auto& ex : make_test_set()) {
        CudaSimulationRuntime runtime(model, topology_path, 0, true, false);
        for (std::uint32_t i = 0; i < kInputs; ++i) {
            if (ex.features[i])
                runtime.schedule_stimulus(i, 10U, kPulseSteps, kInputCurrent);
        }
        runtime.run_steps(kStepsPerExample);
        const auto events = runtime.collect_spike_events();
        std::uint64_t a = 0, b = 0;
        for (const auto& event : events.events) {
            if (event.neuron_id < kInputs) ++score.input_spikes;
            else if (event.neuron_id == kOutputA) { ++a; ++score.output_spikes; }
            else if (event.neuron_id == kOutputB) { ++b; ++score.output_spikes; }
        }
        score.dropped_events += events.dropped_events;
        if (ex.label == 0) {
            score.class0_a_spikes += a;
            score.class0_b_spikes += b;
        } else {
            score.class1_a_spikes += a;
            score.class1_b_spikes += b;
        }
        if (a != b && ((a > b ? 0U : 1U) == ex.label)) {
            ++score.correct;
            if (ex.label == 0) ++score.class0_correct;
            else ++score.class1_correct;
        }
        ++score.total;
    }
    return score;
}

void write_manifest(const std::filesystem::path& dir) {
    std::ofstream out(dir / "model.nmc");
    if (!out) throw std::runtime_error("Cannot write model.nmc");
    out << "NEUROMANCER_MODEL 1\n"
        << "task two_class_online_stdp_association\n"
        << "input_features " << kInputs << "\n"
        << "output_neurons " << kOutputA << ' ' << kOutputB << "\n"
        << "dt_ms " << kDtMs << "\n"
        << "spike_threshold_mv " << kThresholdMv << "\n"
        << "simulation_steps " << kStepsPerExample << "\n"
        << "stimulus_start_step 10\n"
        << "stimulus_duration_steps " << kPulseSteps << "\n"
        << "input_current " << kInputCurrent << "\n"
        << "active_feature_threshold 0.5\n"
        << "topology_file topology.csr\n"
        << "readout online_pair_stdp\n";
    if (!out) throw std::runtime_error("Failed writing model.nmc");
}

double mean_weight(const CSRTopology& graph) {
    if (graph.weights.empty()) return 0.0;
    double sum = 0.0;
    for (float w : graph.weights) sum += w;
    return sum / static_cast<double>(graph.weights.size());
}

void save_checkpoint(CudaSimulationRuntime& runtime,
                     const std::filesystem::path& dir) {
    const auto graph = runtime.collect_connectivity();
    save_csr_topology(graph, (dir / "topology.csr").string());
    write_manifest(dir);
}

void write_usage() {
    std::cout << "Usage:\n"
              << "  neuromancer_overnight_v26 --smoke-test <output-dir>\n"
              << "  neuromancer_overnight_v26 --hours <hours> --output <output-dir>\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        bool smoke = false;
        double hours = 8.0;
        std::filesystem::path output = "neuromancer_overnight_v26";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--smoke-test") {
                smoke = true;
                if (i + 1 < argc && argv[i + 1][0] != '-') output = argv[++i];
            } else if (arg == "--hours" && i + 1 < argc) {
                hours = std::stod(argv[++i]);
            } else if (arg == "--output" && i + 1 < argc) {
                output = argv[++i];
            } else {
                write_usage();
                return 2;
            }
        }
        if (!smoke && (!std::isfinite(hours) || hours <= 0.0 || hours > 168.0))
            throw std::invalid_argument("--hours must be in (0, 168]");

        std::filesystem::create_directories(output);
        output = std::filesystem::absolute(output).lexically_normal();
        const auto initial_path = output / "initial_topology.csr";
        const auto final_path = output / "topology.csr";
        const auto metrics_path = output / "training_metrics.csv";

        const auto model = make_model();
        auto initial = make_initial_topology(26031U);
        save_csr_topology(initial, initial_path.string());
        const auto baseline_score = evaluate(initial_path.string(), model);
        StdpParameters stdp;
        stdp.potentiation_rate = 0.005f;
        stdp.depression_rate = 0.00525f;
        stdp.presynaptic_tau_ms = 20.0f;
        stdp.postsynaptic_tau_ms = 20.0f;
        stdp.minimum_weight = 0.001f;
        stdp.maximum_weight = 1.0f;

        CudaSimulationRuntime runtime(model, initial_path.string(), 0, false, true, stdp);
        auto training = make_training_set();
        std::mt19937 order_rng(26041U);
        std::ofstream metrics(metrics_path);
        if (!metrics) throw std::runtime_error("Cannot open training_metrics.csv");
        metrics << "elapsed_seconds,epoch,completed_steps,excitatory_spikes,inhibitory_spikes,"
                   "excitatory_rate_hz,inhibitory_rate_hz,mean_voltage_mv,synapses,"
                   "mean_weight,heldout_accuracy,dropped_events\n";

        std::cout << "NEUROMANCER V26 — OVERNIGHT ONLINE STDP TRAINING\n"
                  << "Network: " << kExcitatory << " excitatory + " << kInhibitory
                  << " inhibitory neurons; " << initial.weights.size() << " initial synapses\n"
                  << "Training: 2 interleaved associations; " << training.size()
                  << " examples/epoch; causal teacher lag " << kTeacherLag
                  << " steps (" << kTeacherLag * kDtMs << " ms)\n"
                  << "Plasticity: pair STDP; A+=" << stdp.potentiation_rate
                  << " A-=" << stdp.depression_rate << "; tau=20 ms; weights ["
                  << stdp.minimum_weight << ',' << stdp.maximum_weight << "]\n"
                  << "Output package: " << output.string() << "\n"
                  << "Checkpoint interval: 60 seconds; training event capture disabled\n\n";

        const auto started = std::chrono::steady_clock::now();
        auto last_checkpoint = started;
        std::uint64_t epoch = 0;
        const auto limit = smoke ? std::chrono::seconds(0)
            : std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(hours * 3600.0));

        while (true) {
            ++epoch;
            std::shuffle(training.begin(), training.end(), order_rng);
            for (const auto& ex : training) train_one(runtime, ex);

            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - started).count();
            if (smoke || now - last_checkpoint >= std::chrono::seconds(60) ||
                (!smoke && now - started >= limit)) {
                const auto state = runtime.collect_metrics();
                const auto graph = runtime.collect_connectivity();
                if (!state.finite_state)
                    throw std::runtime_error("Non-finite neuron state detected; stopping training");
                save_csr_topology(graph, final_path.string());
                write_manifest(output);
                metrics << std::setprecision(10) << elapsed << ',' << epoch << ','
                        << state.completed_steps << ',' << state.excitatory_spikes << ','
                        << state.inhibitory_spikes << ',' << state.excitatory_rate_hz << ','
                        << state.inhibitory_rate_hz << ',' << state.mean_voltage_mv << ','
                        << graph.weights.size() << ',' << mean_weight(graph) << ",,"
                        << "0\n";
                metrics.flush();
                std::cout << "checkpoint epoch=" << epoch
                          << " steps=" << state.completed_steps
                          << " E_spikes=" << state.excitatory_spikes
                          << " I_spikes=" << state.inhibitory_spikes
                          << " mean_weight=" << mean_weight(graph)
                          << " elapsed_s=" << std::fixed << std::setprecision(1)
                          << elapsed << '\n';
                last_checkpoint = now;
            }
            if (smoke || now - started >= limit) break;
        }

        save_checkpoint(runtime, output);
        const auto score = evaluate(final_path.string(), model);
        const double accuracy = score.total == 0 ? 0.0
            : 100.0 * static_cast<double>(score.correct) / score.total;
        const auto final_metrics = runtime.collect_metrics();
        const auto final_graph = runtime.collect_connectivity();
        metrics << std::setprecision(10)
                << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
                << ',' << epoch << ',' << final_metrics.completed_steps << ','
                << final_metrics.excitatory_spikes << ',' << final_metrics.inhibitory_spikes
                << ',' << final_metrics.excitatory_rate_hz << ','
                << final_metrics.inhibitory_rate_hz << ',' << final_metrics.mean_voltage_mv
                << ',' << final_graph.weights.size() << ',' << mean_weight(final_graph) << ','
                << accuracy << ',' << score.dropped_events << '\n';
        metrics.flush();
        if (!final_metrics.finite_state || score.dropped_events != 0)
            throw std::runtime_error("Final integrity checks failed");

        const double baseline_accuracy = baseline_score.total == 0 ? 0.0
            : 100.0 * static_cast<double>(baseline_score.correct) / baseline_score.total;
        std::cout << "\nFROZEN-WEIGHT RECALL COMPARISON\n"
                  << "Untrained baseline: " << baseline_score.correct << '/'
                  << baseline_score.total << " (" << std::fixed << std::setprecision(1)
                  << baseline_accuracy << "%), output spikes=" << baseline_score.output_spikes << '\n'
                  << "Trained model: " << score.correct << '/' << score.total
                  << " (" << accuracy << "%), output spikes=" << score.output_spikes << '\n'
                  << "\nFINAL FROZEN-WEIGHT RECALL\n"
                  << "Held-out accuracy: " << score.correct << '/' << score.total
                  << " (" << std::fixed << std::setprecision(1) << accuracy << "%)\n"
                  << "Class 0: correct " << score.class0_correct << '/' << kTestSamplesPerClass
                  << ", output A spikes=" << score.class0_a_spikes
                  << ", output B spikes=" << score.class0_b_spikes << '\n'
                  << "Class 1: correct " << score.class1_correct << '/' << kTestSamplesPerClass
                  << ", output A spikes=" << score.class1_a_spikes
                  << ", output B spikes=" << score.class1_b_spikes << '\n'
                  << "Input spikes: " << score.input_spikes
                  << " output spikes: " << score.output_spikes
                  << " dropped events: " << score.dropped_events << '\n'
                  << "Saved inference package: " << (output / "model.nmc").string() << '\n'
                  << "Saved learned CSR: " << final_path.string() << '\n'
                  << "Training log: " << metrics_path.string() << '\n';
        constexpr std::uint32_t kMinimumCorrectPerClass = 12U;
        const bool both_classes_learned =
            score.class0_correct >= kMinimumCorrectPerClass &&
            score.class1_correct >= kMinimumCorrectPerClass;
        const bool inhibition_active = final_metrics.inhibitory_spikes > 0;
        const bool useful_recall = score.output_spikes > 0 &&
            score.correct > baseline_score.correct &&
            both_classes_learned && inhibition_active;
        if (!useful_recall) {
            std::cerr << "FAIL: acceptance criteria not met. Require trained recall "
                         "to beat baseline, at least 12/16 correct in each class, "
                         "nonzero output activity, and nonzero inhibitory activity. "
                         "Do not use for overnight training.\\n";
            return 2;
        }
        std::cout << (smoke ? "PASS: both classes learned; baseline, output, and "
                               "inhibitory-activity checks passed.\\n"
                            : "PASS: overnight training completed; all learning and "
                               "network-activity checks passed.\\n");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Neuromancer V26 error: " << e.what() << '\n';
        return 1;
    }
}
