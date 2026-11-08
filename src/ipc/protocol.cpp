#include "ipc/protocol.h"

#include "ipc/parameter_json.h"

#include <limits>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace spiking::ipc {
namespace {

using Json = nlohmann::json;

std::uint64_t topology_revision(const CSRTopology& graph) {
    // FNV-1a over the canonical CSR arrays. Unlike completed_steps, this
    // changes only when connectivity, delay, or weight data changes.
    std::uint64_t hash = 14695981039346656037ull;
    const auto mix = [&hash](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    };
    mix(&graph.excitatory_count, sizeof(graph.excitatory_count));
    mix(&graph.inhibitory_count, sizeof(graph.inhibitory_count));
    if (!graph.row_offsets.empty())
        mix(graph.row_offsets.data(), graph.row_offsets.size() * sizeof(graph.row_offsets[0]));
    if (!graph.target_indices.empty())
        mix(graph.target_indices.data(), graph.target_indices.size() * sizeof(graph.target_indices[0]));
    if (!graph.delays.empty())
        mix(graph.delays.data(), graph.delays.size() * sizeof(graph.delays[0]));
    if (!graph.weights.empty())
        mix(graph.weights.data(), graph.weights.size() * sizeof(graph.weights[0]));
    return hash;
}

std::string status_name(ControllerStatus status) {
    switch (status) {
        case ControllerStatus::Ready: return "ready";
        case ControllerStatus::Running: return "running";
        case ControllerStatus::Paused: return "paused";
        case ControllerStatus::Faulted: return "faulted";
        case ControllerStatus::ShuttingDown: return "shutting_down";
    }
    return "unknown";
}

std::uint32_t read_positive_u32(const Json& value, const char* field) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument(std::string(field) + " must be an integer");
    }

    std::uint64_t number = 0;
    if (value.is_number_unsigned()) {
        number = value.get<std::uint64_t>();
    } else {
        const auto signed_number = value.get<std::int64_t>();
        if (signed_number <= 0) {
            throw std::invalid_argument(std::string(field) + " must be positive");
        }
        number = static_cast<std::uint64_t>(signed_number);
    }

    if (number == 0 ||
        number > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            std::string(field) + " must be in the uint32 positive range");
    }
    return static_cast<std::uint32_t>(number);
}

Json update_result_json(const ParameterUpdateResult& result) {
    Json errors = Json::array();
    for (const auto& error : result.errors) {
        errors.push_back({
            {"field", error.field},
            {"message", error.message}
        });
    }

    return {
        {"accepted", result.accepted},
        {"requires_reinitialization", result.requires_reinitialization},
        {"parameter_version", std::to_string(result.parameter_version)},
        {"simulation_generation", std::to_string(result.simulation_generation)},
        {"errors", std::move(errors)}
    };
}

Json configuration_json(const ControllerConfigurationSnapshot& snapshot) {
    return {
        {"requested", parameters_to_json(snapshot.requested_parameters)},
        {"active", parameters_to_json(snapshot.active_parameters)},
        {"has_pending_reinitialization", snapshot.has_pending_reinitialization},
        {"parameter_version", std::to_string(snapshot.parameter_version)},
        {"simulation_generation", std::to_string(snapshot.simulation_generation)}
    };
}

const char* cell_kind_name(tissue::CellKind kind) {
    switch (kind) {
        case tissue::CellKind::ExcitatoryNeuron: return "excitatory_neuron";
        case tissue::CellKind::InhibitoryNeuron: return "inhibitory_neuron";
        case tissue::CellKind::Astrocyte: return "astrocyte";
        case tissue::CellKind::Microglia: return "microglia";
        case tissue::CellKind::Oligodendrocyte: return "oligodendrocyte";
    }
    return "unknown";
}

const char* port_direction_name(tissue::PortDirection direction) {
    return direction == tissue::PortDirection::Output ? "output" : "input";
}

Json tissue_snapshot_json(const TissueSnapshot& snapshot) {
    Json modules = Json::array();
    for (const auto& entry : snapshot.model.modules()) {
        const auto& module = entry.second;
        Json populations = Json::array();
        for (const auto& population : module.populations) {
            populations.push_back({
                {"id", std::to_string(population.id)},
                {"kind", cell_kind_name(population.kind)},
                {"size", population.size},
                {"mean_rate_hz", population.mean_rate_hz},
                {"damage_fraction", population.damage_fraction},
                {"metabolic_reserve", population.metabolic_reserve},
                {"immune_activation", population.immune_activation}
            });
        }

        Json ports = Json::array();
        for (const auto& port : module.ports) {
            ports.push_back({
                {"id", std::to_string(port.id)},
                {"population_id", std::to_string(port.population_id)},
                {"direction", port_direction_name(port.direction)},
                {"connection_capacity", port.connection_capacity}
            });
        }

        modules.push_back({
            {"id", std::to_string(module.id)},
            {"name", module.name},
            {"generation", std::to_string(module.generation)},
            {"topology", {
                {"excitatory_count", module.topology.excitatory_count},
                {"inhibitory_count", module.topology.inhibitory_count},
                {"synapse_count", module.topology.synapse_count()}
            }},
            {"populations", std::move(populations)},
            {"ports", std::move(ports)}
        });
    }

    Json adapters = Json::array();
    for (const auto& adapter : snapshot.model.adapters()) {
        adapters.push_back({
            {"id", std::to_string(adapter.id)},
            {"source_module_id", std::to_string(adapter.source_module_id)},
            {"source_port_id", std::to_string(adapter.source_port_id)},
            {"target_module_id", std::to_string(adapter.target_module_id)},
            {"target_port_id", std::to_string(adapter.target_port_id)},
            {"gain", adapter.gain},
            {"delay_steps", adapter.delay_steps},
            {"enabled", adapter.enabled}
        });
    }

    return {
        {"revision", std::to_string(snapshot.revision)},
        {"modules", std::move(modules)},
        {"adapters", std::move(adapters)}
    };
}

Json telemetry_json(const TelemetrySnapshot& snapshot) {
    const auto& m = snapshot.metrics;
    return {
        {"sequence", std::to_string(snapshot.sequence)},
        {"completed_steps", std::to_string(m.completed_steps)},
        {"excitatory_spikes", std::to_string(m.excitatory_spikes)},
        {"inhibitory_spikes", std::to_string(m.inhibitory_spikes)},
        {"last_step_excitatory_spikes", m.last_step_excitatory_spikes},
        {"last_step_inhibitory_spikes", m.last_step_inhibitory_spikes},
        {"mean_voltage_mv", m.mean_voltage_mv},
        {"excitatory_rate_hz", m.excitatory_rate_hz},
        {"inhibitory_rate_hz", m.inhibitory_rate_hz},
        {"finite_state", m.finite_state}
    };
}


Json spike_events_json(const SpikeEventSnapshot& snapshot) {
    Json events = Json::array();
    for (const auto& event : snapshot.events) {
        events.push_back({
            {"sequence", std::to_string(event.sequence)},
            {"step", std::to_string(event.step)},
            {"neuron_id", event.neuron_id}
        });
    }

    return {
        {"first_sequence", std::to_string(snapshot.first_sequence)},
        {"total_events", std::to_string(snapshot.total_events)},
        {"dropped_events", std::to_string(snapshot.dropped_events)},
        {"simulation_generation",
         std::to_string(snapshot.simulation_generation)},
        {"events", std::move(events)}
    };
}

Json topology_snapshot_json(const CSRTopology& graph, std::uint64_t step) {
    // The runtime may contain millions of neurons and synapses. The visual
    // surface receives a bounded, deterministic induced subgraph: source
    // neurons are sampled across the full population and every returned edge
    // is copied verbatim from the live CSR arrays.
    // The temporal embedding renderer can now handle a larger induced
    // subgraph. Keep this transport snapshot bounded until viewport-driven
    // LOD and incremental topology streaming are available.
    constexpr std::size_t kMaximumSources = 512;
    constexpr std::size_t kMaximumNodes = 2048;
    constexpr std::size_t kMaximumEdges = 12000;
    const std::uint32_t total = graph.neuron_count();
    const std::size_t source_count = std::min<std::size_t>(kMaximumSources, total);
    std::vector<std::uint32_t> nodes;
    nodes.reserve(kMaximumNodes);
    std::vector<std::uint32_t> sources;
    sources.reserve(source_count);
    for (std::size_t i = 0; i < source_count; ++i) {
        const auto source = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(i) * total) / source_count);
        sources.push_back(source);
        nodes.push_back(source);
    }

    struct EdgeView { std::uint32_t source, target, delay; float weight; };
    std::vector<EdgeView> edges;
    edges.reserve(kMaximumEdges);
    for (const auto source : sources) {
        for (std::uint32_t edge = graph.row_offsets[source];
             edge < graph.row_offsets[source + 1] && edges.size() < kMaximumEdges;
             ++edge) {
            const auto target = graph.target_indices[edge];
            const auto found = std::find(nodes.begin(), nodes.end(), target);
            if (found == nodes.end()) {
                if (nodes.size() >= kMaximumNodes) continue;
                nodes.push_back(target);
            }
            edges.push_back({source, target, graph.delays[edge], graph.weights[edge]});
        }
        if (edges.size() >= kMaximumEdges) break;
    }

    Json node_array = Json::array();
    for (const auto id : nodes) node_array.push_back(id);
    Json edge_array = Json::array();
    for (const auto& edge : edges) {
        edge_array.push_back({
            {"source", edge.source}, {"target", edge.target},
            {"weight", edge.weight}, {"delay", edge.delay},
            {"excitatory", edge.weight > 0.0f}
        });
    }
    return {
        {"revision", std::to_string(topology_revision(graph))},
        {"completed_steps", std::to_string(step)},
        {"total_neurons", total},
        {"total_synapses", graph.synapse_count()},
        {"rendered_neurons", nodes.size()},
        {"rendered_synapses", edges.size()},
        {"excitatory_count", graph.excitatory_count},
        {"inhibitory_count", graph.inhibitory_count},
        {"nodes", std::move(node_array)},
        {"edges", std::move(edge_array)}
    };
}

Json success_response(const Json& request_id, Json result) {
    return {
        {"protocol_version", kProtocolVersion},
        {"request_id", request_id},
        {"ok", true},
        {"result", std::move(result)}
    };
}

Json error_response(
    const Json& request_id,
    const std::string& code,
    const std::string& message
) {
    return {
        {"protocol_version", kProtocolVersion},
        {"request_id", request_id},
        {"ok", false},
        {"error", {
            {"code", code},
            {"message", message}
        }}
    };
}

} // namespace

Json telemetry_event_json(const TelemetrySnapshot& snapshot) {
    return {
        {"protocol_version", kProtocolVersion},
        {"event", "telemetry_update"},
        {"telemetry", telemetry_json(snapshot)}
    };
}

Json spike_events_event_json(const SpikeEventSnapshot& snapshot) {
    return {
        {"protocol_version", kProtocolVersion},
        {"event", "spike_events"},
        {"spike_events", spike_events_json(snapshot)}
    };
}

Json handle_request(SimulationController& controller, const Json& request) {
    Json request_id = nullptr;

    try {
        if (!request.is_object()) {
            throw std::invalid_argument("request must be a JSON object");
        }

        if (request.contains("request_id")) {
            request_id = request.at("request_id");
        }

        if (!request_id.is_string() || request_id.get<std::string>().empty()) {
            return error_response(
                nullptr, "invalid_request",
                "request_id must be a non-empty string");
        }

        if (!request.contains("protocol_version") ||
            (!request.at("protocol_version").is_number_unsigned() &&
             !(request.at("protocol_version").is_number_integer() &&
               request.at("protocol_version").get<std::int64_t>() >= 0))) {
            return error_response(
                request_id, "invalid_request",
                "protocol_version must be a non-negative integer");
        }

        const auto version = request.at("protocol_version").get<std::uint64_t>();
        if (version != kProtocolVersion) {
            return error_response(
                request_id, "unsupported_protocol",
                "unsupported protocol_version; this server supports version 1");
        }

        if (!request.contains("command") ||
            !request.at("command").is_string()) {
            return error_response(
                request_id, "invalid_request",
                "command must be a string");
        }

        if (!request.contains("payload") || !request.at("payload").is_object()) {
            return error_response(
                request_id, "invalid_request",
                "payload must be a JSON object");
        }

        const std::string command = request.at("command").get<std::string>();
        const Json& payload = request.at("payload");

        if (command == "get_tissue") {
            const auto snapshot = controller.tissue_snapshot();
            if (payload.contains("after_revision")) {
                const auto& value = payload.at("after_revision");
                if (!value.is_string()) {
                    throw std::invalid_argument(
                        "after_revision must be a decimal string");
                }
                const auto text = value.get<std::string>();
                if (text.empty() ||
                    text.find_first_not_of("0123456789") != std::string::npos) {
                    throw std::invalid_argument(
                        "after_revision must be a decimal string");
                }
                const auto requested_revision = std::stoull(text);
                if (requested_revision == snapshot.revision) {
                    return success_response(request_id, {
                        {"unchanged", true},
                        {"revision", std::to_string(snapshot.revision)}
                    });
                }
            }
            Json result = tissue_snapshot_json(snapshot);
            result["unchanged"] = false;
            return success_response(request_id, std::move(result));
        }

        if (command == "get_status") {
            return success_response(request_id, {
                {"status", status_name(controller.status())},
                {"is_running", controller.is_running()}
            });
        }

        if (command == "get_parameters") {
            return success_response(
                request_id, configuration_json(controller.configuration_snapshot()));
        }

        if (command == "get_telemetry") {
            return success_response(
                request_id, telemetry_json(controller.telemetry_snapshot()));
        }

        if (command == "get_topology") {
            const auto graph = controller.topology_snapshot();
            return success_response(request_id, topology_snapshot_json(
                graph, controller.telemetry_snapshot().metrics.completed_steps));
        }

        if (command == "get_spike_events") {
            std::uint64_t after_sequence = 0;
            if (payload.contains("after_sequence")) {
                const auto& value = payload.at("after_sequence");
                if (!value.is_string()) {
                    throw std::invalid_argument(
                        "after_sequence must be a decimal string");
                }
                const auto text = value.get<std::string>();
                if (text.empty() ||
                    text.find_first_not_of("0123456789") != std::string::npos) {
                    throw std::invalid_argument(
                        "after_sequence must be a decimal string");
                }
                after_sequence = std::stoull(text);
            }
            return success_response(request_id, spike_events_json(
                controller.spike_events_snapshot(after_sequence)));
        }

        if (command == "subscribe_spike_events") {
            return success_response(request_id, {{"subscribed", true}});
        }

        if (command == "unsubscribe_spike_events") {
            return success_response(request_id, {{"subscribed", false}});
        }

        if (command == "subscribe_telemetry") {
            return success_response(request_id, {
                {"subscribed", true}
            });
        }

        if (command == "unsubscribe_telemetry") {
            return success_response(request_id, {
                {"subscribed", false}
            });
        }

        if (command == "apply_parameters") {
            if (!payload.contains("parameters")) {
                throw std::invalid_argument(
                    "apply_parameters requires payload.parameters");
            }

            const auto current = controller.configuration_snapshot();
            const auto candidate = parameters_from_json(
                payload.at("parameters"), current.requested_parameters);
            const auto result = controller.apply_parameters(candidate);
            return success_response(request_id, update_result_json(result));
        }

        if (command == "reset_parameters") {
            return success_response(
                request_id,
                update_result_json(controller.reset_parameters_to_defaults()));
        }

        if (command == "reinitialize") {
            controller.reinitialize_simulation();
            return success_response(request_id, {
                {"status", status_name(controller.status())},
                {"configuration",
                 configuration_json(controller.configuration_snapshot())},
                {"telemetry", telemetry_json(controller.telemetry_snapshot())}
            });
        }

        if (command == "start") {
            controller.start();
            return success_response(request_id, {
                {"status", status_name(controller.status())}
            });
        }

        if (command == "pause") {
            controller.pause();
            return success_response(request_id, {
                {"status", status_name(controller.status())},
                {"telemetry", telemetry_json(controller.telemetry_snapshot())}
            });
        }

        if (command == "stimulate_neurons") {
            if (!payload.contains("neuron_ids") || !payload.at("neuron_ids").is_array() ||
                payload.at("neuron_ids").empty())
                throw std::invalid_argument("stimulate_neurons requires a non-empty neuron_ids array");
            if (!payload.contains("amplitude") || !payload.at("amplitude").is_number())
                throw std::invalid_argument("stimulate_neurons requires numeric amplitude");
            if (!payload.contains("duration_steps"))
                throw std::invalid_argument("stimulate_neurons requires duration_steps");
            std::vector<std::uint32_t> neuron_ids;
            neuron_ids.reserve(payload.at("neuron_ids").size());
            for (const auto& id : payload.at("neuron_ids")) {
                if (!id.is_number_integer())
                    throw std::invalid_argument("each neuron_id must be a non-negative integer");
                std::uint64_t value = 0;
                if (id.is_number_unsigned()) value = id.get<std::uint64_t>();
                else {
                    const auto signed_value = id.get<std::int64_t>();
                    if (signed_value < 0)
                        throw std::invalid_argument("neuron_id must be non-negative");
                    value = static_cast<std::uint64_t>(signed_value);
                }
                if (value > std::numeric_limits<std::uint32_t>::max())
                    throw std::invalid_argument("neuron_id exceeds uint32 range");
                neuron_ids.push_back(static_cast<std::uint32_t>(value));
            }
            const double amplitude = payload.at("amplitude").get<double>();
            if (!std::isfinite(amplitude) || amplitude == 0.0)
                throw std::invalid_argument("amplitude must be finite and non-zero");
            const auto duration = read_positive_u32(payload.at("duration_steps"), "duration_steps");
            controller.stimulate_neurons(neuron_ids, static_cast<float>(amplitude), duration);
            return success_response(request_id, {{"accepted", true},
                {"selected_neurons", neuron_ids.size()},
                {"amplitude", amplitude}, {"duration_steps", duration}});
        }

        if (command == "step") {
            if (!payload.contains("steps")) {
                throw std::invalid_argument("step requires payload.steps");
            }
            const auto steps = read_positive_u32(payload.at("steps"), "steps");
            controller.step(steps);
            return success_response(request_id, {
                {"status", status_name(controller.status())},
                {"telemetry", telemetry_json(controller.telemetry_snapshot())}
            });
        }

        return error_response(
            request_id, "unknown_command", "unknown command: " + command);

    } catch (const std::invalid_argument& error) {
        return error_response(request_id, "invalid_parameters", error.what());
    } catch (const std::exception& error) {
        return error_response(request_id, "command_failed", error.what());
    } catch (...) {
        return error_response(
            request_id, "internal_error", "unknown internal server error");
    }
}

} // namespace spiking::ipc
