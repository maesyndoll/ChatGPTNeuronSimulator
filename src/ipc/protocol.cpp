#include "ipc/protocol.h"

#include "ipc/parameter_json.h"
#include "topology/topology_io.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace spiking::ipc {
namespace {

using Json = nlohmann::json;

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


Json connectivity_json(const TopologyStatistics& stats) {
    const auto distribution = [](const TopologyDistributionStats& value) {
        return Json{{"count", std::to_string(value.count)},
                    {"minimum", value.minimum}, {"maximum", value.maximum},
                    {"mean", value.mean}};
    };
    return {
        {"excitatory_count", stats.excitatory_count},
        {"inhibitory_count", stats.inhibitory_count},
        {"neuron_count", static_cast<std::uint64_t>(stats.excitatory_count) + stats.inhibitory_count},
        {"synapse_count", std::to_string(stats.synapse_count)},
        {"density", stats.density},
        {"population_edges", {
            {"excitatory_to_excitatory", std::to_string(stats.excitatory_to_excitatory)},
            {"excitatory_to_inhibitory", std::to_string(stats.excitatory_to_inhibitory)},
            {"inhibitory_to_excitatory", std::to_string(stats.inhibitory_to_excitatory)},
            {"inhibitory_to_inhibitory", std::to_string(stats.inhibitory_to_inhibitory)}
        }},
        {"outgoing_degree", distribution(stats.outgoing_degree)},
        {"incoming_degree", distribution(stats.incoming_degree)},
        {"weights", distribution(stats.weights)},
        {"delays", distribution(stats.delays)}
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

        if (command == "get_connectivity") {
            return success_response(
                request_id, connectivity_json(controller.connectivity_statistics()));
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
            // Give new subscribers the latest snapshot immediately. Without
            // this, dashboards remain empty until the next periodic telemetry
            // interval even though the controller already has valid metrics.
            return success_response(request_id, {
                {"subscribed", true},
                {"telemetry", telemetry_json(controller.telemetry_snapshot())}
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
