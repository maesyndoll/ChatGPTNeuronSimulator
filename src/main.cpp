#include "core/parameters.h"
#include "ipc/unix_socket_server.h"
#include "ipc/parameter_json.h"
#include "runtime/cuda_simulation_runtime.cuh"
#include "runtime/active_neuron_telemetry.h"
#include "runtime/stdp_weight_analysis.h"
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
#include <cmath>
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
    std::string connectivity_export_path;
    std::string connectivity_export_after_path;
    bool enable_stdp = false;
    bool stdp_parameters_explicit = false;
    spiking::StdpParameters stdp_parameters{};
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
    std::uint64_t excitatory_active_neurons = 0;
    std::uint64_t inhibitory_active_neurons = 0;
    double excitatory_active_neuron_percent = 0.0;
    double inhibitory_active_neuron_percent = 0.0;
};

float parse_float_option(const char* value, const char* option_name) {
    std::size_t consumed = 0;
    const std::string text(value);
    const float parsed = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(parsed))
        throw std::invalid_argument(std::string(option_name) + " requires a finite number");
    return parsed;
}

void validate_stdp_parameters(const spiking::StdpParameters& p) {
    if (!std::isfinite(p.potentiation_rate) || p.potentiation_rate < 0.0f ||
        !std::isfinite(p.depression_rate) || p.depression_rate < 0.0f ||
        !std::isfinite(p.presynaptic_tau_ms) || p.presynaptic_tau_ms <= 0.0f ||
        !std::isfinite(p.postsynaptic_tau_ms) || p.postsynaptic_tau_ms <= 0.0f ||
        !std::isfinite(p.minimum_weight) || p.minimum_weight <= 0.0f ||
        !std::isfinite(p.maximum_weight) ||
        p.maximum_weight < p.minimum_weight)
        throw std::invalid_argument("Invalid STDP parameters");
}

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
        } else if (arg == "--enable-stdp") {
            options.enable_stdp = true;
        } else if (arg == "--stdp-a-plus") {
            if (++i >= argc) throw std::invalid_argument("--stdp-a-plus requires a number");
            options.stdp_parameters.potentiation_rate = parse_float_option(argv[i], "--stdp-a-plus");
            options.stdp_parameters_explicit = true;
        } else if (arg == "--stdp-a-minus") {
            if (++i >= argc) throw std::invalid_argument("--stdp-a-minus requires a number");
            options.stdp_parameters.depression_rate = parse_float_option(argv[i], "--stdp-a-minus");
            options.stdp_parameters_explicit = true;
        } else if (arg == "--stdp-tau-pre-ms") {
            if (++i >= argc) throw std::invalid_argument("--stdp-tau-pre-ms requires a number");
            options.stdp_parameters.presynaptic_tau_ms = parse_float_option(argv[i], "--stdp-tau-pre-ms");
            options.stdp_parameters_explicit = true;
        } else if (arg == "--stdp-tau-post-ms") {
            if (++i >= argc) throw std::invalid_argument("--stdp-tau-post-ms requires a number");
            options.stdp_parameters.postsynaptic_tau_ms = parse_float_option(argv[i], "--stdp-tau-post-ms");
            options.stdp_parameters_explicit = true;
        } else if (arg == "--stdp-min-weight") {
            if (++i >= argc) throw std::invalid_argument("--stdp-min-weight requires a number");
            options.stdp_parameters.minimum_weight = parse_float_option(argv[i], "--stdp-min-weight");
            options.stdp_parameters_explicit = true;
        } else if (arg == "--stdp-max-weight") {
            if (++i >= argc) throw std::invalid_argument("--stdp-max-weight requires a number");
            options.stdp_parameters.maximum_weight = parse_float_option(argv[i], "--stdp-max-weight");
            options.stdp_parameters_explicit = true;
        } else if (arg == "--connectivity-export") {
            if (++i >= argc)
                throw std::invalid_argument("--connectivity-export requires a CSV file path");
            options.connectivity_export_path = argv[i];
        } else if (arg == "--connectivity-export-after") {
            if (++i >= argc)
                throw std::invalid_argument("--connectivity-export-after requires a CSV file path");
            options.connectivity_export_after_path = argv[i];
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
                << "Usage: spiking_model [--config FILE] [--small] [--device N] [--log FILE] [--no-log] [--enable-stdp] [--stdp-a-plus X] [--stdp-a-minus X] [--stdp-tau-pre-ms X] [--stdp-tau-post-ms X] [--stdp-min-weight X] [--stdp-max-weight X] [--connectivity-export FILE] [--connectivity-export-after FILE] [--telemetry-window-steps N] [--voltage-trace FILE --voltage-trace-neuron ID ...] [--voltage-trace-interval N]\n"
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
    if (options.server && !options.connectivity_export_path.empty())
        throw std::invalid_argument("--connectivity-export is not available in server mode");
    if (options.server && !options.connectivity_export_after_path.empty())
        throw std::invalid_argument("--connectivity-export-after is not available in server mode");
    if (options.server && options.enable_stdp)
        throw std::invalid_argument("--enable-stdp is currently available only in standalone mode");
    if (options.stdp_parameters_explicit && !options.enable_stdp)
        throw std::invalid_argument("STDP parameter overrides require --enable-stdp");
    if (options.enable_stdp)
        validate_stdp_parameters(options.stdp_parameters);
    if (!options.connectivity_export_path.empty() &&
        options.connectivity_export_path == options.connectivity_export_after_path)
        throw std::invalid_argument("before/after connectivity exports must use different paths");
    if (options.server && !options.voltage_trace_path.empty())
        throw std::invalid_argument("--voltage-trace is not available in server mode");

    if (!options.server && !options.socket_path.empty())
        throw std::invalid_argument("--socket requires --server");
    if (options.server && options.log_results)
        options.log_results = false;
    return options;
}

void write_connectivity_csv(
    const std::string& path,
    const spiking::CSRTopology& topology) {
    namespace fs = std::filesystem;
    const fs::path output_path(path);
    if (!output_path.parent_path().empty())
        fs::create_directories(output_path.parent_path());

    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output)
        throw std::runtime_error("cannot open connectivity CSV: " + path);

    output << "source_neuron,target_neuron,source_population,target_population,weight,delay_steps\n";
    for (std::uint32_t source = 0; source < topology.neuron_count(); ++source) {
        const char* source_population =
            source < topology.excitatory_count ? "excitatory" : "inhibitory";
        for (std::uint32_t edge = topology.row_offsets[source];
             edge < topology.row_offsets[source + 1]; ++edge) {
            const auto target = topology.target_indices[edge];
            const char* target_population =
                target < topology.excitatory_count ? "excitatory" : "inhibitory";
            output << source << ',' << target << ','
                   << source_population << ',' << target_population << ','
                   << std::setprecision(9) << topology.weights[edge] << ','
                   << topology.delays[edge] << '\n';
        }
    }
    output.flush();
    if (!output)
        throw std::runtime_error("failed writing connectivity CSV: " + path);
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
    const std::vector<WindowTelemetry>& telemetry,
    bool stdp_enabled,
    const spiking::StdpParameters& stdp_parameters,
    const spiking::StdpWeightSummary* stdp_weight_summary
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

    record["results"]["stdp"] = {
        {"enabled", stdp_enabled},
        {"potentiation_rate", stdp_parameters.potentiation_rate},
        {"depression_rate", stdp_parameters.depression_rate},
        {"presynaptic_tau_ms", stdp_parameters.presynaptic_tau_ms},
        {"postsynaptic_tau_ms", stdp_parameters.postsynaptic_tau_ms},
        {"minimum_weight", stdp_parameters.minimum_weight},
        {"maximum_weight", stdp_parameters.maximum_weight}
    };
    if (stdp_weight_summary != nullptr) {
        record["results"]["stdp"]["weight_changes"] = {
            {"plastic_synapses", stdp_weight_summary->plastic_synapses},
            {"changed_synapses", stdp_weight_summary->changed_synapses},
            {"mean_initial_weight", stdp_weight_summary->mean_initial_weight},
            {"mean_final_weight", stdp_weight_summary->mean_final_weight},
            {"mean_absolute_change", stdp_weight_summary->mean_absolute_change},
            {"maximum_absolute_change", stdp_weight_summary->maximum_absolute_change}
        };
    }

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
                {"mean_membrane_voltage_snapshot_mv", window.mean_voltage_snapshot_mv},
                {"excitatory_active_neurons", window.excitatory_active_neurons},
                {"inhibitory_active_neurons", window.inhibitory_active_neurons},
                {"excitatory_active_neuron_percent", window.excitatory_active_neuron_percent},
                {"inhibitory_active_neuron_percent", window.inhibitory_active_neuron_percent}
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

        CudaSimulationRuntime runtime(
            parameters, options.device, true, options.enable_stdp,
            options.stdp_parameters);
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
                  << "Delay seed: " << parameters.topology.delay_seed << '\n'
                  << "STDP: " << (options.enable_stdp ? "ENABLED" : "disabled") << '\n';
        if (options.enable_stdp) {
            std::cout << "STDP A+: " << options.stdp_parameters.potentiation_rate
                      << ", A-: " << options.stdp_parameters.depression_rate
                      << ", tau_pre: " << options.stdp_parameters.presynaptic_tau_ms
                      << " ms, tau_post: " << options.stdp_parameters.postsynaptic_tau_ms
                      << " ms, weight bounds: [" << options.stdp_parameters.minimum_weight
                      << ", " << options.stdp_parameters.maximum_weight << "]\n";
        }

        CSRTopology initial_topology;
        bool have_initial_topology = false;
        if (options.enable_stdp || !options.connectivity_export_path.empty()) {
            initial_topology = runtime.collect_connectivity();
            have_initial_topology = true;
        }
        if (!options.connectivity_export_path.empty()) {
            write_connectivity_csv(options.connectivity_export_path, initial_topology);
            std::cout << "Initial connectivity export: "
                      << options.connectivity_export_path
                      << " (" << initial_topology.synapse_count() << " synapses)\n";
        }

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
            auto previous_spike_counts = runtime.collect_spike_count_snapshot();
            const std::uint64_t total_steps = simulation.steps;
            std::uint64_t completed_window_steps = 0;

            while (completed_window_steps < total_steps) {
                const auto remaining = total_steps - completed_window_steps;
                const auto window_steps = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(remaining, options.telemetry_window_steps));

                runtime.run_steps(window_steps);
                const auto window_metrics = runtime.collect_metrics();
                const auto current_spike_counts = runtime.collect_spike_count_snapshot();
                const auto active = count_active_neurons(previous_spike_counts,
                    current_spike_counts, simulation.excitatory_count);

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
                window.excitatory_active_neurons = active.excitatory;
                window.inhibitory_active_neurons = active.inhibitory;
                window.excitatory_active_neuron_percent = active.excitatory_percent;
                window.inhibitory_active_neuron_percent = active.inhibitory_percent;
                telemetry.push_back(window);

                previous_excitatory_spikes = window_metrics.excitatory_spikes;
                previous_inhibitory_spikes = window_metrics.inhibitory_spikes;
                previous_spike_counts = current_spike_counts;
                completed_window_steps += window_steps;
            }
        }

        spiking::StdpWeightSummary stdp_weight_summary;
        bool have_stdp_weight_summary = false;
        if (options.enable_stdp || !options.connectivity_export_after_path.empty()) {
            const auto final_topology = runtime.collect_connectivity();
            if (options.enable_stdp && have_initial_topology) {
                stdp_weight_summary =
                    summarize_stdp_weight_changes(initial_topology, final_topology);
                have_stdp_weight_summary = true;
                std::cout << "STDP weight changes: "
                          << stdp_weight_summary.changed_synapses << " / "
                          << stdp_weight_summary.plastic_synapses
                          << " excitatory synapses changed; mean |delta w| "
                          << stdp_weight_summary.mean_absolute_change
                          << ", max |delta w| "
                          << stdp_weight_summary.maximum_absolute_change << '\n';
            }
            if (!options.connectivity_export_after_path.empty()) {
                write_connectivity_csv(options.connectivity_export_after_path, final_topology);
                std::cout << "Final connectivity export: "
                          << options.connectivity_export_after_path << '\n';
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
                          << window.mean_voltage_snapshot_mv << " mV, active E "
                          << window.excitatory_active_neurons << " ("
                          << window.excitatory_active_neuron_percent
                          << "%), active I " << window.inhibitory_active_neurons
                          << " (" << window.inhibitory_active_neuron_percent << "%)\n";
            }
        }

        if (options.log_results) {
            append_experiment_record(options.log_path, parameters, metrics, properties,
                                     runtime.device_id(), elapsed_seconds, options.config_path,
                                     telemetry, options.enable_stdp,
                                     options.stdp_parameters,
                                     have_stdp_weight_summary ? &stdp_weight_summary : nullptr);
            std::cout << "Experiment log: " << options.log_path << '\n';
        }
        return metrics.finite_state ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "SpikingModel runtime error: " << error.what() << '\n';
        return 1;
    }
}
