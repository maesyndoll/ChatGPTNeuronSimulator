#include "ipc/parameter_json.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const char* message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

} // namespace

int main() {
    using namespace spiking;
    using namespace spiking::ipc;

    try {
        const ModelParameters defaults;
        const auto encoded = parameters_to_json(defaults);
        const auto decoded = parameters_from_json(encoded, ModelParameters{});

        require(decoded.simulation.excitatory_count ==
                    defaults.simulation.excitatory_count,
                "excitatory population did not round-trip");
        require(decoded.simulation.dt == defaults.simulation.dt,
                "timestep did not round-trip");
        require(decoded.topology.topology_seed == defaults.topology.topology_seed,
                "topology seed did not round-trip");
        require(decoded.topology.delay_seed == defaults.topology.delay_seed,
                "delay seed did not round-trip");
        require(decoded.topology.memory_budget_bytes ==
                    defaults.topology.memory_budget_bytes,
                "memory budget did not round-trip");
        require(decoded.topology.delay_distribution ==
                    defaults.topology.delay_distribution,
                "delay distribution did not round-trip");

        // Verify lossless handling of the full uint64 range.
        ModelParameters large = defaults;
        large.topology.master_seed = std::numeric_limits<std::uint64_t>::max();
        large.topology.topology_seed = 9007199254740993ULL;
        const auto large_json = parameters_to_json(large);
        require(large_json["topology"]["master_seed"] ==
                    "18446744073709551615",
                "uint64 maximum was not serialized as an exact decimal string");
        const auto large_round_trip =
            parameters_from_json(large_json, ModelParameters{});
        require(large_round_trip.topology.master_seed ==
                    std::numeric_limits<std::uint64_t>::max(),
                "uint64 maximum did not round-trip");
        require(large_round_trip.topology.topology_seed == 9007199254740993ULL,
                "value above JavaScript safe integer did not round-trip");

        // Partial updates must preserve every unspecified setting.
        nlohmann::json partial = {
            {"simulation", {{"dt", 0.25}}},
            {"topology", {{"master_seed", "42"}}}
        };
        const auto updated = parameters_from_json(partial, defaults);
        require(updated.simulation.dt == 0.25f,
                "partial update did not apply timestep");
        require(updated.topology.master_seed == 42,
                "partial update did not apply master seed");
        require(updated.simulation.steps == defaults.simulation.steps,
                "partial update changed an unspecified simulation field");
        require(updated.topology.topology_seed == defaults.topology.topology_seed,
                "partial update changed an unspecified topology field");

        // Enum and array updates.
        nlohmann::json structured = {
            {"topology", {
                {"delay_distribution", "Uniform"},
                {"fibonacci_delays", nlohmann::json::array({1, 2, 3})},
                {"fibonacci_weights", nlohmann::json::array({0.5, 0.3, 0.2})}
            }}
        };
        const auto structured_result = parameters_from_json(structured, defaults);
        require(structured_result.topology.delay_distribution ==
                    DelayDistribution::Uniform,
                "enum update failed");
        require(structured_result.topology.fibonacci_delays.size() == 3,
                "array update failed");

        require_throws([&] {
            parameters_from_json(
                nlohmann::json{{"simulation", {{"unknown_field", 1}}}},
                defaults
            );
        }, "unknown parameter was accepted");

        require_throws([&] {
            parameters_from_json(
                nlohmann::json{{"topology", {{"master_seed", 42}}}},
                defaults
            );
        }, "numeric uint64 value was accepted instead of a decimal string");

        require_throws([&] {
            parameters_from_json(
                nlohmann::json{{"topology", {{"master_seed", "18446744073709551616"}}}},
                defaults
            );
        }, "uint64 overflow was accepted");

        require_throws([&] {
            parameters_from_json(
                nlohmann::json{{"simulation", {{"steps", -1}}}},
                defaults
            );
        }, "negative unsigned parameter was accepted");

        require_throws([&] {
            parameters_from_json(
                nlohmann::json{{"simulation", {{"dt", 0.0}}}},
                defaults
            );
        }, "out-of-range parameter was accepted");

        require_throws([&] {
            parameters_from_json(
                nlohmann::json{{"topology", {{"delay_distribution", "Unknown"}}}},
                defaults
            );
        }, "unknown enum value was accepted");

        std::cout << "IPC parameter JSON tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "IPC parameter JSON test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
