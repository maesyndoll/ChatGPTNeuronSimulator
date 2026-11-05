#pragma once

#include "runtime/simulation_controller.h"

#include <string>

namespace spiking::ipc {

// Blocks while serving one local client at a time.
// Returns zero after a graceful SIGINT/SIGTERM shutdown.
int run_unix_socket_server(
    SimulationController& controller,
    const std::string& socket_path
);

} // namespace spiking::ipc
