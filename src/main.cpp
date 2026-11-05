#include "core/parameters.h"
#include "ipc/unix_socket_server.h"
#include "ipc/parameter_json.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "runtime/simulation_controller.h"

#include <cuda_runtime.h>

#include <algorithm>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
    std::uint32_t telemetry_window_steps = 0;
    std::string voltage_trace_path;
    std::vector<std::uint32_t> voltage_trace_neurons;
    std::uint32_t voltage_trace_interval_steps = 1;
};

struct WindowTelemetry {
    std::uint64_t start_step = 0;
    std::uint64_t end_step = 0;
    double start_time_ms = 0.0;
    double end_time_ms = 0.0;
    std::uint64_t excitatory_spikes = 0;
    std::uint64_t inhibitory_spikes = 0;
    double excitatory_rate_hz = 0.0;
    double inhibitory_rate_hz = 0.0;
    double mean_voltage_snapshot_mv = 0.0;
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
        } else if (arg == "--telemetry-window-steps") {
            if (++i >= argc)
                throw std::invalid_argument("--telemetry-window-steps requires an integer");
            std::size_t consumed = 0;
            const auto value = std::stoul(argv[i], &consumed);
            if (consumed != std::string(argv[i]).size() || value == 0 ||
                value > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("--telemetry-window-steps must be a positive 32-bit integer");
            options.telemetry_window_steps = static_cast<std::uint32_t>(value);
        } else if (arg == "--voltage-trace") {
            if (++i >= argc)
                throw std::invalid_argument("--voltage-trace requires a CSV file path");
            options.voltage_trace_path = argv[i];
        } else if (arg == "--voltage-trace-neuron") {
            if (++i >= argc)
                throw std::invalid_argument("--voltage-trace-neuron requires a neuron ID");
            std::size_t consumed = 0;
            const auto value = std::stoul(argv[i], &consumed);
            if (consumed != std::string(argv[i]).size() ||
                value > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("--voltage-trace-neuron must be a non-negative 32-bit integer");
            options.voltage_trace_neurons.push_back(static_cast<std::uint32_t>(value));
        } else if (arg == "--voltage-trace-interval") {
            if (++i >= argc)
                throw std::invalid_argument("--voltage-trace-interval requires an integer");
            std::size_t consumed = 0;
            const auto value = std::stoul(argv[i], &consumed);
            if (consumed != std::string(argv[i]).size() || value == 0 ||
                value > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("--voltage-trace-interval must be a positive 32-bit integer");
            options.voltage_trace_interval_steps = static_cast<std::uint32_t>(value);
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
                << "Usage: spiking_model [--config FILE] [--small] [--device N] [--log FILE] [--no-log] [--telemetry-window-steps N] [--voltage-trace FILE --voltage-trace-neuron ID ...] [--voltage-trace-interval N]\n"
                << "       spiking_model --server [--socket PATH] [--config FILE] [--small] [--device N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.voltage_trace_path.empty() &&
        !options.voltage_trace_neurons.empty())
        throw std::invalid_argument("--voltage-trace-neuron requires --voltage-trace");
    if (!options.voltage_trace_path.empty() &&
        options.voltage_trace_neurons.empty())
        throw std::invalid_argument("--voltage-trace requires at least one --voltage-trace-neuron");
    if (!options.voltage_trace_path.empty() && options.telemetry_window_steps != 0)
        throw std::invalid_argument("--voltage-trace and --telemetry-window-steps cannot be combined yet");
    if (options.server && !options.voltage_trace_path.empty())
        throw std::invalid_argument("--voltage-trace is not available in server mode");

    if (!options.server && !options.socket_path.empty())
        throw std::invalid_argument("--socket requires --server");
    if (options.server && options.log_results)
        options.log_results = false;
    return options;
}

void write_voltage_samples(
    std::ostream& output,
    const std::vector<spiking::VoltageSample>& samples) {
    for (const auto& sample : samples) {
        output << sample.step << ','
               << std::setprecision(12) << sample.time_ms << ','
               << sample.neuron_id << ','
               << std::setprecision(9) << sample.voltage_mv << ','
               << std::setprecision(9) << sample.recovery << ','
               << (sample.spiked_this_step ? 1 : 0) << '\n';
    }
    if (!output)
        throw std::runtime_error("failed writing voltage trace CSV");
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
    const std::string& config_path,
    const std::vector<WindowTelemetry>& telemetry
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

    if (!telemetry.empty()) {
        record["results"]["time_resolved_telemetry"] = nlohmann::json::array();
        for (std::size_t i = 0; i < telemetry.size(); ++i) {
            const auto& window = telemetry[i];
            record["results"]["time_resolved_telemetry"].push_back({
                {"window_index", i},
                {"start_step", window.start_step},
                {"end_step", window.end_step},
                {"start_time_ms", window.start_time_ms},
                {"end_time_ms", window.end_time_ms},
                {"excitatory_spikes", window.excitatory_spikes},
                {"inhibitory_spikes", window.inhibitory_spikes},
                {"excitatory_mean_firing_rate_hz", window.excitatory_rate_hz},
                {"inhibitory_mean_firing_rate_hz", window.inhibitory_rate_hz},
                {"mean_membrane_voltage_snapshot_mv", window.mean_voltage_snapshot_mv}
            });
        }
    }

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

        if (!options.voltage_trace_path.empty()) {
            const auto neuron_count = parameters.simulation.neuron_count();
            auto ids = options.voltage_trace_neurons;
            std::sort(ids.begin(), ids.end());
            if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
                throw std::invalid_argument("--voltage-trace-neuron IDs must be unique");
            for (const auto id : ids) {
                if (id >= neuron_count)
                    throw std::out_of_range("--voltage-trace-neuron ID exceeds the configured neuron count");
            }
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
        std::vector<WindowTelemetry> telemetry;

        if (!options.voltage_trace_path.empty()) {
            namespace fs = std::filesystem;
            const fs::path trace_path(options.voltage_trace_path);
            if (!trace_path.parent_path().empty())
                fs::create_directories(trace_path.parent_path());
            std::ofstream trace(trace_path, std::ios::out | std::ios::trunc);
            if (!trace)
                throw std::runtime_error("cannot open voltage trace CSV: " + options.voltage_trace_path);
            trace << "step,time_ms,neuron_id,voltage_mv,recovery,spiked_this_step\n";
            write_voltage_samples(trace,
                runtime.collect_voltage_samples(options.voltage_trace_neurons));

            std::uint64_t completed = 0;
            const std::uint64_t total = simulation.steps;
            while (completed < total) {
                const auto chunk = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(
                        total - completed, options.voltage_trace_interval_steps));
                runtime.run_steps(chunk);
                write_voltage_samples(trace,
                    runtime.collect_voltage_samples(options.voltage_trace_neurons));
                completed += chunk;
            }
            trace.flush();
            if (!trace)
                throw std::runtime_error("failed flushing voltage trace CSV");
            std::cout << "Voltage trace: " << options.voltage_trace_path
                      << " (neurons ";
            for (std::size_t i = 0; i < options.voltage_trace_neurons.size(); ++i)
                std::cout << (i == 0 ? "" : ",") << options.voltage_trace_neurons[i];
            std::cout << ", interval " << options.voltage_trace_interval_steps
                      << " steps)\n";
        } else if (options.telemetry_window_steps == 0) {
            runtime.run();
        } else {
            std::uint64_t previous_excitatory_spikes = 0;
            std::uint64_t previous_inhibitory_spikes = 0;
            const std::uint64_t total_steps = simulation.steps;
            std::uint64_t completed_window_steps = 0;

            while (completed_window_steps < total_steps) {
                const auto remaining = total_steps - completed_window_steps;
                const auto window_steps = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(remaining, options.telemetry_window_steps));

                runtime.run_steps(window_steps);
                const auto window_metrics = runtime.collect_metrics();

                const double window_duration_seconds =
                    static_cast<double>(window_steps) * simulation.dt / 1000.0;
                WindowTelemetry window;
                window.start_step = completed_window_steps;
                window.end_step = completed_window_steps + window_steps;
                window.start_time_ms =
                    static_cast<double>(window.start_step) * simulation.dt;
                window.end_time_ms =
                    static_cast<double>(window.end_step) * simulation.dt;
                window.excitatory_spikes =
                    window_metrics.excitatory_spikes - previous_excitatory_spikes;
                window.inhibitory_spikes =
                    window_metrics.inhibitory_spikes - previous_inhibitory_spikes;
                window.excitatory_rate_hz = window.excitatory_spikes /
                    (static_cast<double>(simulation.excitatory_count) *
                     window_duration_seconds);
                window.inhibitory_rate_hz = window.inhibitory_spikes /
                    (static_cast<double>(simulation.inhibitory_count) *
                     window_duration_seconds);
                window.mean_voltage_snapshot_mv = window_metrics.mean_voltage_mv;
                telemetry.push_back(window);

                previous_excitatory_spikes = window_metrics.excitatory_spikes;
                previous_inhibitory_spikes = window_metrics.inhibitory_spikes;
                completed_window_steps += window_steps;
            }
        }

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

        if (!telemetry.empty()) {
            std::cout << "Telemetry windows: " << telemetry.size() << '\n';
            for (std::size_t i = 0; i < telemetry.size(); ++i) {
                const auto& window = telemetry[i];
                std::cout << "  Window " << (i + 1)
                          << " [" << window.start_time_ms << ", "
                          << window.end_time_ms << ") ms: E "
                          << window.excitatory_rate_hz << " Hz, I "
                          << window.inhibitory_rate_hz << " Hz, voltage snapshot "
                          << window.mean_voltage_snapshot_mv << " mV\n";
            }
        }

        if (options.log_results) {
            append_experiment_record(options.log_path, parameters, metrics, properties,
                                     runtime.device_id(), elapsed_seconds, options.config_path,
                                     telemetry);
            std::cout << "Experiment log: " << options.log_path << '\n';
        }
        return metrics.finite_state ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "SpikingModel runtime error: " << error.what() << '\n';
        return 1;
    }
}
