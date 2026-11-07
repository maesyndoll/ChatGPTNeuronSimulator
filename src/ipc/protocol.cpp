#include "ipc/protocol.h"

#include "ipc/parameter_json.h"
#include "topology/topology_io.h"

#include <limits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

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

Json self_observation_json(const SelfObservationReport& report) {
    Json ranked = Json::array();
    for (const auto& neuron : report.highest_degree_neurons) {
        ranked.push_back({
            {"neuron_id", neuron.neuron_id},
            {"outgoing_degree", neuron.outgoing_degree},
            {"incoming_degree", neuron.incoming_degree},
            {"activity_count", std::to_string(neuron.activity_count)}
        });
    }
    return {
        {"graph_fingerprint", std::to_string(report.graph_fingerprint)},
        {"neuron_count", report.neuron_count},
        {"synapse_count", std::to_string(report.synapse_count)},
        {"mean_outgoing_degree", report.mean_outgoing_degree},
        {"mean_incoming_degree", report.mean_incoming_degree},
        {"isolated_neurons", report.isolated_neurons},
        {"high_degree_neurons", report.high_degree_neurons},
        {"inactive_neurons", report.inactive_neurons},
        {"reciprocal_pairs", std::to_string(report.reciprocal_pairs)},
        {"weak_edges", std::to_string(report.weak_edges)},
        {"ranked_neurons", std::move(ranked)}
    };
}

Json topology_visual_json(const TopologyVisualSnapshot& snapshot) {
    Json neuron_ids = Json::array();
    for (const auto neuron_id : snapshot.neuron_ids)
        neuron_ids.push_back(neuron_id);
    Json edges = Json::array();
    for (const auto& edge : snapshot.edges) {
        edges.push_back({
            {"source", edge.source}, {"target", edge.target},
            {"weight", edge.weight}
        });
    }
    return {
        {"excitatory_count", snapshot.excitatory_count},
        {"inhibitory_count", snapshot.inhibitory_count},
        {"synapse_count", std::to_string(snapshot.synapse_count)},
        {"sampled_neuron_count", snapshot.neuron_ids.size()},
        {"sampled_edge_count", snapshot.edges.size()},
        {"neuron_ids", std::move(neuron_ids)},
        {"edges", std::move(edges)}
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

        if (command == "get_topology_visual") {
            return success_response(request_id, topology_visual_json(
                controller.topology_visual_snapshot(256U, 8U)));
        }

        if (command == "get_self_observation") {
            return success_response(request_id, self_observation_json(
                controller.self_observation_snapshot()));
        }


        if (command == "get_topology_proposal") {
            const auto topology = controller.connectivity_snapshot();
            const auto spikes = controller.spike_events_snapshot(0, 4096);
            std::vector<ObservedSpike> recent;
            recent.reserve(spikes.events.size());
            std::vector<std::uint64_t> activity(topology.neuron_count(), 0);
            for (const auto& event : spikes.events) {
                recent.push_back({event.step, event.neuron_id});
                if (event.neuron_id < activity.size()) ++activity[event.neuron_id];
            }
            TopologyAdvisorOptions options;
            const auto proposal = propose_topology_update(
                topology, recent, activity, options);
            return success_response(request_id, {
                {"base_graph_fingerprint", std::to_string(proposal.base_graph_fingerprint)},
                {"candidate_graph_fingerprint", std::to_string(proposal.candidate_graph_fingerprint)},
                {"proposed_added_edges", proposal.proposed_added_edges},
                {"proposed_pruned_edges", proposal.proposed_pruned_edges},
                {"candidate_synapse_count", std::to_string(proposal.candidate.synapse_count())},
                {"source_unchanged", true},
                {"applied", false},
                {"evidence_spikes", recent.size()}
            });
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


        if (command == "attach_graft") {
            if (!payload.contains("package_path") ||
                !payload.at("package_path").is_string() ||
                payload.at("package_path").get<std::string>().empty())
                throw std::invalid_argument("attach_graft requires a non-empty package_path");
            bool plasticity_enabled = false;
            if (payload.contains("plasticity_enabled")) {
                if (!payload.at("plasticity_enabled").is_boolean())
                    throw std::invalid_argument("plasticity_enabled must be boolean");
                plasticity_enabled = payload.at("plasticity_enabled").get<bool>();
            }
            const auto info = controller.attach_graft_package(
                payload.at("package_path").get<std::string>(), plasticity_enabled);
            return success_response(request_id, {
                {"attached", true}, {"id", info.id}, {"name", info.name},
                {"excitatory_count", info.excitatory_count},
                {"inhibitory_count", info.inhibitory_count},
                {"synapse_count", info.synapse_count},
                {"status", status_name(controller.status())},
                {"configuration", configuration_json(controller.configuration_snapshot())},
                {"telemetry", telemetry_json(controller.telemetry_snapshot())}
            });
        }

        if (command == "detach_graft") {
            controller.detach_graft_package();
            return success_response(request_id, {
                {"attached", false},
                {"status", status_name(controller.status())},
                {"configuration", configuration_json(controller.configuration_snapshot())},
                {"telemetry", telemetry_json(controller.telemetry_snapshot())}
            });
        }

        if (command == "get_graft_status") {
            return success_response(request_id, {
                {"attached", controller.graft_attached()}
            });
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

        if (command == "schedule_stimuli") {
            if (!payload.contains("stimuli") || !payload.at("stimuli").is_array())
                throw std::invalid_argument("schedule_stimuli requires payload.stimuli array");
            if (controller.is_running())
                throw std::invalid_argument("pause the simulation before submitting a visual frame");
            const auto& input = payload.at("stimuli");
            if (input.size() > 64)
                throw std::invalid_argument("schedule_stimuli accepts at most 64 visual cells");

            std::vector<ExternalStimulus> stimuli;
            stimuli.reserve(input.size());
            for (const auto& item : input) {
                if (!item.is_object() || !item.contains("neuron_id") ||
                    !item.contains("start_step") || !item.contains("duration_steps") ||
                    !item.contains("amplitude"))
                    throw std::invalid_argument("each stimulus requires neuron_id, start_step, duration_steps, amplitude");
                const auto id_value = item.at("neuron_id").get<std::int64_t>();
                if (id_value < 0 || static_cast<std::uint64_t>(id_value) >
                    std::numeric_limits<std::uint32_t>::max())
                    throw std::invalid_argument("neuron_id is outside uint32 range");
                const auto start_value = item.at("start_step").get<std::uint64_t>();
                const auto duration = read_positive_u32(item.at("duration_steps"), "duration_steps");
                const float amplitude = item.at("amplitude").get<float>();
                if (!std::isfinite(amplitude) || amplitude < 0.0f || amplitude > 100.0f)
                    throw std::invalid_argument("stimulus amplitude must be finite and in [0, 100]");
                stimuli.push_back({static_cast<std::uint32_t>(id_value), start_value, duration, amplitude});
            }
            controller.schedule_stimuli(stimuli);
            return success_response(request_id, {{"scheduled", stimuli.size()}});
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
