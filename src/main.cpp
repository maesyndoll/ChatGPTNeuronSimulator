#include "core/parameters.h"
#include "ipc/unix_socket_server.h"
#include "ipc/parameter_json.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "runtime/simulation_controller.h"

#include <cuda_runtime.h>

#include <exception>
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>

#include <unistd.h>
#include <string>

namespace {

struct CommandLineOptions {
    bool server = false;
    bool small = false;
    int device = 0;
    std::string socket_path;
    std::string config_path;
};

CommandLineOptions parse_arguments(int argc, char** argv) {
    CommandLineOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--server") {
            options.server = true;
        } else if (arg == "--small") {
            options.small = true;
        } else if (arg == "--config") {
            if (++i >= argc)
                throw std::invalid_argument("--config requires a JSON file path");
            options.config_path = argv[i];
        } else if (arg == "--socket") {
            if (++i >= argc)
                throw std::invalid_argument("--socket requires a path");
            options.socket_path = argv[i];
        } else if (arg == "--device") {
            if (++i >= argc)
                throw std::invalid_argument("--device requires an integer");

            std::size_t consumed = 0;
            options.device = std::stoi(argv[i], &consumed);
            if (consumed != std::string(argv[i]).size() || options.device < 0)
                throw std::invalid_argument("--device must be a non-negative integer");
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: spiking_model [--config FILE] [--small] [--device N]\n"
                << "       spiking_model --server [--socket PATH] [--config FILE] [--small] [--device N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (!options.server &&
        (!options.socket_path.empty() || options.device != 0)) {
        // --device is also valid in one-shot mode; --socket is server-only.
        if (!options.socket_path.empty())
            throw std::invalid_argument("--socket requires --server");
    }

    return options;
}

} // namespace

int main(int argc, char** argv) {
    using namespace spiking;
    try {
        CommandLineOptions options = parse_arguments(argc, argv);

        ModelParameters parameters;
        if (options.small)
            parameters.simulation = SimulationParameters::small_test();
        if (!options.config_path.empty()) {
            std::ifstream config_file(options.config_path);
            if (!config_file)
                throw std::runtime_error("cannot open configuration file: " + options.config_path);
            nlohmann::json config;
            config_file >> config;
            parameters = ipc::parameters_from_json(config, parameters);
        }

        if (options.server) {
            if (options.socket_path.empty()) {
                options.socket_path =
                    "/tmp/spiking_model-" + std::to_string(getuid()) + ".sock";
            }

            SimulationController controller(
                parameters, options.device, 100, 10000, true);

            return ipc::run_unix_socket_server(
                controller, options.socket_path);
        }

        CudaSimulationRuntime runtime(parameters, options.device);
        cudaDeviceProp properties{};
        SPIKING_CUDA_CHECK(cudaGetDeviceProperties(&properties, runtime.device_id()));

        const auto& simulation = parameters.simulation;
        std::cout << "SpikingModel v0.4.0\n"
                  << "GPU: " << properties.name << '\n'
                  << "Compute capability: " << properties.major << '.'
                  << properties.minor << '\n'
                  << "Excitatory neurons: " << simulation.excitatory_count << '\n'
                  << "Inhibitory neurons: " << simulation.inhibitory_count << '\n'
                  << "Total neurons: " << simulation.neuron_count() << '\n'
                  << "Time step: " << simulation.dt << " ms\n"
                  << "Steps: " << simulation.steps << '\n';
        if (!options.config_path.empty())
            std::cout << "Configuration: " << options.config_path << '\n';
        std::cout << "Neuron seed: " << simulation.random_seed << '\n'
                  << "Topology seed: " << parameters.topology.topology_seed << '\n'
                  << "Delay seed: " << parameters.topology.delay_seed << '\n';

        runtime.run();
        const auto metrics = runtime.collect_metrics();
        std::cout << "\nSimulation results\n"
                  << "Completed steps: " << metrics.completed_steps << '\n'
                  << "Excitatory spikes: " << metrics.excitatory_spikes << '\n'
                  << "Inhibitory spikes: " << metrics.inhibitory_spikes << '\n'
                  << "Total spikes: "
                  << metrics.excitatory_spikes + metrics.inhibitory_spikes << '\n'
                  << "Excitatory mean firing rate: " << metrics.excitatory_rate_hz << " Hz\n"
                  << "Inhibitory mean firing rate: " << metrics.inhibitory_rate_hz << " Hz\n"
                  << "Mean membrane voltage: " << metrics.mean_voltage_mv << " mV\n"
                  << "Numerical state: " << (metrics.finite_state ? "VALID" : "INVALID") << '\n';
        return metrics.finite_state ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "SpikingModel runtime error: " << error.what() << '\n';
        return 1;
    }
}
