#pragma once

#include "core/parameters.h"

#include <nlohmann/json.hpp>

namespace spiking::ipc {

// Serialize all model parameters. uint64_t values are decimal strings.
nlohmann::json parameters_to_json(const ModelParameters& parameters);

// Apply a partial JSON update to a base configuration.
// Unspecified fields retain their existing values.
// The resulting complete configuration is validated before return.
ModelParameters parameters_from_json(
    const nlohmann::json& update,
    const ModelParameters& base
);

} // namespace spiking::ipc
