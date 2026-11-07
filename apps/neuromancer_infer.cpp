#include "neuromancer/inference_engine.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<float> read_features(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Could not open input CSV: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Input CSV is empty");
    std::stringstream stream(line);
    std::vector<float> features;
    std::string token;
    while (std::getline(stream, token, ',')) {
        std::size_t consumed = 0;
        const float value = std::stof(token, &consumed);
        if (consumed != token.size()) throw std::runtime_error("Invalid numeric feature in input CSV");
        features.push_back(value);
    }
    return features;
}
void print_result(const neuromancer::InferenceResult& result) {
    std::cout << "output_spikes=[" << result.output_a_spikes << ',' << result.output_b_spikes << "]\n"
              << "input_spikes=" << result.input_spikes << '\n'
              << "dropped_events=" << result.dropped_events << '\n';
    if (result.has_prediction) std::cout << "predicted_class=" << result.predicted_class << '\n';
    else std::cout << "predicted_class=undecided (output spike-count tie)\n";
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--self-test") {
            neuromancer::InferenceEngine engine(argv[2]);
            std::vector<float> a(engine.input_size(), 0.0f), b(engine.input_size(), 0.0f);
            if (engine.input_size() != 64) throw std::runtime_error("Self-test currently requires 64 input features");
            for (std::size_t i = 0; i < 32; ++i) { a[i] = 1.0f; b[i + 32] = 1.0f; }
            const auto ra = engine.infer(a), rb = engine.infer(b);
            print_result(ra); print_result(rb);
            if (!ra.has_prediction || !rb.has_prediction || ra.predicted_class != 0 || rb.predicted_class != 1)
                throw std::runtime_error("Saved model failed the two-class API self-test");
            std::cout << "PASS: portable manifest and relative CSR topology loaded through public API.\n";
            return 0;
        }
        if (argc != 3) {
            std::cerr << "Usage:\n  " << argv[0] << " <model.nmc> <input.csv>\n  "
                      << argv[0] << " --self-test <model.nmc>\n";
            return 2;
        }
        neuromancer::InferenceEngine engine(argv[1]);
        const auto result = engine.infer(read_features(argv[2]));
        print_result(result);
        return result.has_prediction ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "Neuromancer inference error: " << e.what() << '\n';
        return 1;
    }
}
