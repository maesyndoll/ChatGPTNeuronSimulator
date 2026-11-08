#include "runtime/cuda_simulation_runtime.cuh"
#include "cuda/cuda_check.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::uint32_t steps = 2400;
    int device = 0;
    std::filesystem::path output = "neural-observatory-output";
};

Options parse_arguments(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--steps" || arg == "--device" || arg == "--output") {
            if (++i >= argc) {
                throw std::invalid_argument(arg + " requires a value");
            }
            const std::string value = argv[i];
            if (arg == "--steps") {
                std::size_t consumed = 0;
                const auto parsed = std::stoul(value, &consumed);
                if (consumed != value.size() || parsed == 0 ||
                    parsed > 100000000UL) {
                    throw std::invalid_argument("--steps must be in [1, 100000000]");
                }
                options.steps = static_cast<std::uint32_t>(parsed);
            } else if (arg == "--device") {
                std::size_t consumed = 0;
                const int parsed = std::stoi(value, &consumed);
                if (consumed != value.size() || parsed < 0) {
                    throw std::invalid_argument("--device must be a non-negative integer");
                }
                options.device = parsed;
            } else {
                options.output = value;
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Neural Observatory\n"
                << "  --steps N       Simulation steps (default: 2400)\n"
                << "  --device N      CUDA device index (default: 0)\n"
                << "  --output DIR    Output directory (default: neural-observatory-output)\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return options;
}

void write_report(const std::filesystem::path& directory,
                  const std::vector<spiking::SpikeEvent>& events,
                  const std::vector<std::uint64_t>& excitatory_bins,
                  const std::vector<std::uint64_t>& inhibitory_bins,
                  std::uint32_t steps,
                  std::uint32_t excitatory_count,
                  std::uint32_t neuron_count,
                  const spiking::SimulationMetrics& metrics,
                  const std::string& gpu_name) {
    const auto csv_path = directory / "spikes.csv";
    std::ofstream csv(csv_path);
    if (!csv) throw std::runtime_error("could not create " + csv_path.string());
    csv << "step,time_ms,neuron_id,population\n";
    for (const auto& event : events) {
        csv << event.step << ','
            << std::fixed << std::setprecision(3)
            << static_cast<double>(event.step) * 0.5 << ','
            << event.neuron_id << ','
            << (event.neuron_id < excitatory_count ? "excitatory" : "inhibitory")
            << '\n';
    }
    csv.close();

    const auto html_path = directory / "index.html";
    std::ofstream html(html_path);
    if (!html) throw std::runtime_error("could not create " + html_path.string());

    constexpr double width = 1200.0;
    constexpr double height = 860.0;
    constexpr double left = 76.0;
    constexpr double right = 28.0;
    constexpr double top = 100.0;
    constexpr double plot_height = 480.0;
    const double plot_width = width - left - right;
    const std::size_t maximum_rendered_events = 250000;
    const std::size_t stride = std::max<std::size_t>(
        1, (events.size() + maximum_rendered_events - 1) / maximum_rendered_events);

    html << "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
         << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
         << "<title>Neural Observatory</title><style>"
         << "body{margin:0;background:#080d18;color:#e8eef9;font:15px system-ui,sans-serif}"
         << "main{max-width:1280px;margin:32px auto;padding:0 22px}"
         << "h1{font-size:30px;margin:0 0 8px}.sub{color:#9aaac2;margin-bottom:24px}"
         << ".stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:20px 0}"
         << ".stat{background:#111b2c;border:1px solid #263750;border-radius:12px;padding:15px}"
         << ".stat b{display:block;font-size:24px;margin-top:5px}.stat span{color:#9aaac2;font-size:12px}"
         << ".panel{background:#0d1626;border:1px solid #263750;border-radius:14px;padding:12px;overflow-x:auto}"
         << "svg{display:block;width:100%;height:auto;min-width:760px}.axis{stroke:#34445d;stroke-width:1}"
         << ".grid{stroke:#1c2b40;stroke-width:1}.tick{fill:#9aaac2;font-size:12px}"
         << ".exc{fill:#42c7ff}.inh{fill:#ffad5c}.muted{color:#9aaac2;font-size:13px;margin:14px 2px}"
         << "a{color:#70cfff}</style></head><body><main>"
         << "<h1>Neural Observatory</h1>"
         << "<div class=\"sub\">GPU-resident spiking network · " << gpu_name
         << " · " << neuron_count << " neurons · " << steps << " integration steps</div>"
         << "<section class=\"stats\">"
         << "<div class=\"stat\"><span>Recorded spikes</span><b>" << events.size() << "</b></div>"
         << "<div class=\"stat\"><span>Excitatory spikes</span><b>" << metrics.excitatory_spikes << "</b></div>"
         << "<div class=\"stat\"><span>Inhibitory spikes</span><b>" << metrics.inhibitory_spikes << "</b></div>"
         << "<div class=\"stat\"><span>Mean membrane voltage</span><b>"
         << std::fixed << std::setprecision(2) << metrics.mean_voltage_mv << " mV</b></div>"
         << "<div class=\"stat\"><span>Excitatory firing rate</span><b>"
         << std::setprecision(2) << metrics.excitatory_rate_hz << " Hz</b></div>"
         << "<div class=\"stat\"><span>Inhibitory firing rate</span><b>"
         << std::setprecision(2) << metrics.inhibitory_rate_hz << " Hz</b></div>"
         << "</section><section class=\"panel\"><svg viewBox=\"0 0 "
         << width << ' ' << height << "\" role=\"img\" aria-label=\"Spike raster plot\">";

    for (int tick = 0; tick <= 10; ++tick) {
        const double x = left + plot_width * tick / 10.0;
        html << "<line class=\"grid\" x1=\"" << x << "\" y1=\"" << top
             << "\" x2=\"" << x << "\" y2=\"" << top + plot_height << "\"/>"
             << "<text class=\"tick\" x=\"" << x << "\" y=\""
             << top + plot_height + 24 << "\" text-anchor=\"middle\">"
             << (static_cast<std::uint64_t>(steps) * tick / 10) << "</text>";
    }
    for (int tick = 0; tick <= 4; ++tick) {
        const double y = top + plot_height * tick / 4.0;
        html << "<line class=\"grid\" x1=\"" << left << "\" y1=\"" << y
             << "\" x2=\"" << left + plot_width << "\" y2=\"" << y << "\"/>"
             << "<text class=\"tick\" x=\"" << left - 10 << "\" y=\"" << y + 4
             << "\" text-anchor=\"end\">"
             << (static_cast<std::uint64_t>(neuron_count) * tick / 4) << "</text>";
    }
    html << "<line class=\"axis\" x1=\"" << left << "\" y1=\"" << top
         << "\" x2=\"" << left << "\" y2=\"" << top + plot_height << "\"/>"
         << "<line class=\"axis\" x1=\"" << left << "\" y1=\"" << top + plot_height
         << "\" x2=\"" << left + plot_width << "\" y2=\"" << top + plot_height << "\"/>";

    for (std::size_t i = 0; i < events.size(); i += stride) {
        const auto& event = events[i];
        const double x = left + plot_width *
            static_cast<double>(event.step) / std::max<std::uint32_t>(1, steps);
        const double y = top + plot_height *
            static_cast<double>(event.neuron_id) / std::max<std::uint32_t>(1, neuron_count);
        html << "<circle class=\""
             << (event.neuron_id < excitatory_count ? "exc" : "inh")
             << "\" cx=\"" << x << "\" cy=\"" << y
             << "\" r=\"1.65\"><title>step " << event.step
             << ", neuron " << event.neuron_id << "</title></circle>";
    }

    html << "<text class=\"tick\" x=\"" << left + plot_width / 2
         << "\" y=\"" << top + plot_height + 52
         << "\" text-anchor=\"middle\">Simulation step</text>"
         << "<text class=\"tick\" transform=\"translate(18 "
         << top + plot_height / 2 << ") rotate(-90)\" text-anchor=\"middle\">Neuron ID</text>";

    const auto maximum_bin = std::max(
        *std::max_element(excitatory_bins.begin(), excitatory_bins.end()),
        *std::max_element(inhibitory_bins.begin(), inhibitory_bins.end()));
    const double activity_top = 690.0;
    const double activity_height = 82.0;
    const double bin_width = plot_width / static_cast<double>(excitatory_bins.size());
    const double bar_width = bin_width * 0.34;
    html << "<text class=\"tick\" x=\"" << left
         << "\" y=\"678\">Spike activity per time window</text>"
         << "<line class=\"axis\" x1=\"" << left << "\" y1=\"780\" x2=\""
         << left + plot_width << "\" y2=\"780\"/>";
    for (std::size_t bin = 0; bin < excitatory_bins.size(); ++bin) {
        const double x = left + bin_width * static_cast<double>(bin);
        const double exc_height = maximum_bin == 0 ? 0.0 :
            activity_height * static_cast<double>(excitatory_bins[bin]) /
            static_cast<double>(maximum_bin);
        const double inh_height = maximum_bin == 0 ? 0.0 :
            activity_height * static_cast<double>(inhibitory_bins[bin]) /
            static_cast<double>(maximum_bin);
        html << "<rect class=\"exc\" x=\"" << x + bin_width * 0.10
             << "\" y=\"" << 780.0 - exc_height << "\" width=\"" << bar_width
             << "\" height=\"" << exc_height << "\"/>"
             << "<rect class=\"inh\" x=\"" << x + bin_width * 0.54
             << "\" y=\"" << 780.0 - inh_height << "\" width=\"" << bar_width
             << "\" height=\"" << inh_height << "\"/>";
    }
    html << "<circle class=\"exc\" cx=\"930\" cy=\"830\" r=\"5\"/>"
         << "<text class=\"tick\" x=\"942\" y=\"834\">Excitatory</text>"
         << "<circle class=\"inh\" cx=\"1050\" cy=\"830\" r=\"5\"/>"
         << "<text class=\"tick\" x=\"1062\" y=\"834\">Inhibitory</text>"
         << "</svg></section><p class=\"muted\">Each dot is a GPU-recorded spike. "
         << "The raster is capped at " << maximum_rendered_events
         << " rendered points for browser responsiveness; the CSV contains every recorded event. "
         << "This experiment uses fixed synaptic weights; plasticity is not yet enabled.</p>"
         << "<p><a href=\"spikes.csv\">Download raw spike events (CSV)</a></p>"
         << "</main></body></html>";
    if (!html) throw std::runtime_error("failed while writing HTML report");
}

} // namespace

int main(int argc, char** argv) {
    using namespace spiking;
    try {
        const Options options = parse_arguments(argc, argv);

        ModelParameters parameters;
        parameters.simulation.excitatory_count = 2048;
        parameters.simulation.inhibitory_count = 512;
        parameters.simulation.steps = options.steps;
        parameters.simulation.dt = 0.5f;
        parameters.simulation.excitatory_current_min = 10.0f;
        parameters.simulation.excitatory_current_max = 14.0f;
        parameters.simulation.inhibitory_current_min = 8.0f;
        parameters.simulation.inhibitory_current_max = 12.0f;

        parameters.topology.p_ee = 0.02;
        parameters.topology.p_ei = 0.04;
        parameters.topology.p_ie = 0.04;
        parameters.topology.p_ii = 0.02;
        parameters.topology.minimum_delay = 1;
        parameters.topology.maximum_delay = 13;
        parameters.topology.fibonacci_delays = {1, 2, 3, 5, 8, 13};
        parameters.topology.fibonacci_weights = {0.25, 0.20, 0.18, 0.15, 0.12, 0.10};
        parameters.topology.memory_budget_bytes = 256ULL * 1024ULL * 1024ULL;

        CudaSimulationRuntime runtime(parameters, options.device);
        cudaDeviceProp properties{};
        SPIKING_CUDA_CHECK(cudaGetDeviceProperties(&properties, runtime.device_id()));

        std::cout << "Neural Observatory\n"
                  << "GPU: " << properties.name << " (sm_"
                  << properties.major << properties.minor << ")\n"
                  << "Neurons: " << parameters.simulation.neuron_count()
                  << " (" << parameters.simulation.excitatory_count
                  << " excitatory, " << parameters.simulation.inhibitory_count
                  << " inhibitory)\n"
                  << "Steps: " << options.steps << " at "
                  << parameters.simulation.dt << " ms\n"
                  << "Running GPU spike propagation...\n";

        std::vector<SpikeEvent> events;
        std::vector<std::uint64_t> excitatory_bins(48, 0);
        std::vector<std::uint64_t> inhibitory_bins(48, 0);
        constexpr std::uint32_t batch_steps = 100;

        std::uint32_t completed = 0;
        while (completed < options.steps) {
            const auto count = std::min(batch_steps, options.steps - completed);
            runtime.run_steps(count);
            completed += count;

            const auto batch = runtime.collect_spike_events();
            for (const auto& event : batch.events) {
                events.push_back(event);
                const auto bin = std::min<std::size_t>(
                    47, static_cast<std::size_t>(
                        (static_cast<std::uint64_t>(event.step) * 48) /
                        (static_cast<std::uint64_t>(options.steps) + 1)));
                if (event.neuron_id < parameters.simulation.excitatory_count) {
                    ++excitatory_bins[bin];
                } else {
                    ++inhibitory_bins[bin];
                }
            }

            if (completed % 500 == 0 || completed == options.steps) {
                const auto metrics = runtime.collect_metrics();
                std::cout << "  " << completed << '/' << options.steps
                          << " steps | spikes " << metrics.excitatory_spikes
                          << " E + " << metrics.inhibitory_spikes
                          << " I | mean V " << std::fixed << std::setprecision(2)
                          << metrics.mean_voltage_mv << " mV\n";
            }
        }

        const auto metrics = runtime.collect_metrics();
        std::filesystem::create_directories(options.output);
        write_report(options.output, events, excitatory_bins, inhibitory_bins,
                     options.steps, parameters.simulation.excitatory_count,
                     parameters.simulation.neuron_count(), metrics, properties.name);

        std::cout << "\nExperiment complete\n"
                  << "Recorded spike events: " << events.size() << '\n'
                  << "Numerical state: " << (metrics.finite_state ? "VALID" : "INVALID") << '\n'
                  << "HTML report: " << (options.output / "index.html") << '\n'
                  << "Raw events: " << (options.output / "spikes.csv") << '\n';
        return metrics.finite_state ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "Neural Observatory error: " << error.what() << '\n';
        return 1;
    }
}
