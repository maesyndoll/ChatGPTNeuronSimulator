#pragma once

#include "runtime/simulation_controller.h"

#include <nlohmann/json.hpp>

#include <cstdint>

namespace spiking::ipc {

inline constexpr std::uint32_t kProtocolVersion = 1;

// Build asynchronous telemetry and spike-event notifications for subscribed clients.
nlohmann::json telemetry_event_json(const TelemetrySnapshot& snapshot);
nlohmann::json spike_events_event_json(const SpikeEventSnapshot& snapshot);

// Dispatch one complete JSON request and return one complete JSON response.
// The caller owns line framing and transport.
nlohmann::json handle_request(
    SimulationController& controller,
    const nlohmann::json& request
);

} // namespace spiking::ipc
