#include "core/parameters.h"
#include "ipc/unix_socket_server.h"
#include "ipc/parameter_json.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "runtime/simulation_controller.h"

#include <cuda_runtime.h>

#include <chrono>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace {

struct CommandLineOptions {
    bool server = false;
    bool small = false;
    bool log_results = true;
    int device = 0;
    std::string socket_path;
    std::string config_path;
    std::string log_path = "experiments/logs/experiment_log.jsonl";
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
        } else if (arg == "--log") {
            if (++i >= argc)
                throw std::invalid_argument("--log requires a JSONL file path");
            options.log_path = argv[i];
            options.log_results = true;
        } else if (arg == "--no-log") {
            options.log_results = false;
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
                << "Usage: spiking_model [--config FILE] [--small] [--device N] [--log FILE] [--no-log]\n"
                << "       spiking_model --server [--socket PATH] [--config FILE] [--small] [--device N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (!options.server && !options.socket_path.empty())
        throw std::invalid_argument("--socket requires --server");
    if (options.server && options.log_results)
        options.log_results = false;
    return options;
}

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&time, &utc);
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S")
        << '.' << std::setw(3) << std::setfill('0') << milliseconds.count() << 'Z';
    return out.str();
}

void append_experiment_record(
    const std::string& path,
    const spiking::ModelParameters& parameters,
    const spiking::SimulationMetrics& metrics,
    const cudaDeviceProp& gpu,
    int device_id,
    double elapsed_seconds,
    const std::string& config_path
) {
    namespace fs = std::filesystem;
    const fs::path log_path(path);
    if (!log_path.parent_path().empty())
        fs::create_directories(log_path.parent_path());

    nlohmann::json record = {
        {"timestamp_utc", utc_timestamp()},
        {"application", {
            {"name", "ChatGPTNeuronSimulator"},
            {"version", SPIKINGMODEL_VERSION}
        }},
        {"configuration_source", config_path.empty() ? "built-in defaults" : config_path},
        {"configuration", spiking::ipc::parameters_to_json(parameters)},
        {"hardware", {
            {"gpu_name", gpu.name},
            {"compute_capability", std::to_string(gpu.major) + "." + std::to_string(gpu.minor)},
            {"device_id", device_id}
        }},
        {"results", {
            {"completed_steps", metrics.completed_steps},
            {"excitatory_spikes", metrics.excitatory_spikes},
            {"inhibitory_spikes", metrics.inhibitory_spikes},
            {"total_spikes", metrics.excitatory_spikes + metrics.inhibitory_spikes},
            {"excitatory_mean_firing_rate_hz", metrics.excitatory_rate_hz},
            {"inhibitory_mean_firing_rate_hz", metrics.inhibitory_rate_hz},
            {"mean_membrane_voltage_mv", metrics.mean_voltage_mv},
            {"numerical_state_valid", metrics.finite_state},
            {"wall_time_seconds", elapsed_seconds}
        }}
    };

    std::ofstream output(path, std::ios::out | std::ios::app);
    if (!output)
        throw std::runtime_error("cannot open experiment log: " + path);
    output << record.dump() << '\n';
    output.flush();
    if (!output)
        throw std::runtime_error("failed writing experiment log: " + path);
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
            if (options.socket_path.empty())
                options.socket_path = "/tmp/spiking_model-" + std::to_string(getuid()) + ".sock";
            SimulationController controller(parameters, options.device, 100, 10000, true);
            return ipc::run_unix_socket_server(controller, options.socket_path);
        }

        CudaSimulationRuntime runtime(parameters, options.device);
        cudaDeviceProp properties{};
        SPIKING_CUDA_CHECK(cudaGetDeviceProperties(&properties, runtime.device_id()));

        const auto& simulation = parameters.simulation;
        std::cout << "SpikingModel v" << SPIKINGMODEL_VERSION << "\n"
                  << "GPU: " << properties.name << '\n'
                  << "Compute capability: " << properties.major << '.' << properties.minor << '\n'
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

        const auto start = std::chrono::steady_clock::now();
        runtime.run();
        const auto metrics = runtime.collect_metrics();
        const double elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();

        std::cout << "\nSimulation results\n"
                  << "Completed steps: " << metrics.completed_steps << '\n'
                  << "Excitatory spikes: " << metrics.excitatory_spikes << '\n'
                  << "Inhibitory spikes: " << metrics.inhibitory_spikes << '\n'
                  << "Total spikes: " << metrics.excitatory_spikes + metrics.inhibitory_spikes << '\n'
                  << "Excitatory mean firing rate: " << metrics.excitatory_rate_hz << " Hz\n"
                  << "Inhibitory mean firing rate: " << metrics.inhibitory_rate_hz << " Hz\n"
                  << "Mean membrane voltage: " << metrics.mean_voltage_mv << " mV\n"
                  << "Numerical state: " << (metrics.finite_state ? "VALID" : "INVALID") << '\n'
                  << "Wall time: " << elapsed_seconds << " s\n";

        if (options.log_results) {
            append_experiment_record(options.log_path, parameters, metrics, properties,
                                     runtime.device_id(), elapsed_seconds, options.config_path);
            std::cout << "Experiment log: " << options.log_path << '\n';
        }
        return metrics.finite_state ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "SpikingModel runtime error: " << error.what() << '\n';
        return 1;
    }
}
