#include "ipc/parameter_json.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace spiking::ipc {
namespace {

using Json = nlohmann::json;

void require_object(const Json& value, const char* name) {
    if (!value.is_object()) {
        throw std::invalid_argument(std::string(name) + " must be a JSON object");
    }
}

void reject_unknown_fields(
    const Json& object,
    std::initializer_list<const char*> allowed,
    const char* group
) {
    for (auto it = object.begin(); it != object.end(); ++it) {
        bool found = false;
        for (const char* name : allowed) {
            if (it.key() == name) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::invalid_argument(
                std::string("unknown ") + group + " parameter: " + it.key()
            );
        }
    }
}

std::uint64_t read_uint64_string(const Json& value, const char* field) {
    if (!value.is_string()) {
        throw std::invalid_argument(
            std::string(field) + " must be a decimal string"
        );
    }

    const std::string text = value.get<std::string>();
    if (text.empty()) {
        throw std::invalid_argument(std::string(field) + " must not be empty");
    }

    std::uint64_t result = 0;
    for (char c : text) {
        if (c < '0' || c > '9') {
            throw std::invalid_argument(
                std::string(field) + " must contain only decimal digits"
            );
        }
        const std::uint64_t digit = static_cast<std::uint64_t>(c - '0');
        if (result > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
            throw std::invalid_argument(
                std::string(field) + " exceeds the uint64 range"
            );
        }
        result = result * 10 + digit;
    }
    return result;
}

std::uint32_t read_uint32(const Json& value, const char* field) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument(std::string(field) + " must be an integer");
    }

    std::uint64_t result = 0;
    if (value.is_number_unsigned()) {
        result = value.get<std::uint64_t>();
    } else {
        const auto signed_value = value.get<std::int64_t>();
        if (signed_value < 0) {
            throw std::invalid_argument(std::string(field) + " must not be negative");
        }
        result = static_cast<std::uint64_t>(signed_value);
    }

    if (result > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string(field) + " exceeds the uint32 range");
    }
    return static_cast<std::uint32_t>(result);
}

double read_number(const Json& value, const char* field) {
    if (!value.is_number()) {
        throw std::invalid_argument(std::string(field) + " must be a number");
    }
    const double result = value.get<double>();
    if (!std::isfinite(result)) {
        throw std::invalid_argument(std::string(field) + " must be finite");
    }
    return result;
}

float read_float(const Json& value, const char* field) {
    const double number = read_number(value, field);
    if (number > std::numeric_limits<float>::max() ||
        number < -std::numeric_limits<float>::max()) {
        throw std::invalid_argument(std::string(field) + " exceeds the float range");
    }
    return static_cast<float>(number);
}

void apply_simulation(Json const& j, SimulationParameters& s) {
    require_object(j, "simulation");
    reject_unknown_fields(j, {
        "excitatory_count", "inhibitory_count", "steps", "dt", "threshold",
        "random_seed", "excitatory_current_min", "excitatory_current_max",
        "inhibitory_current_min", "inhibitory_current_max",
        "initial_voltage_min", "initial_voltage_max"
    }, "simulation");

#define READ_U32(name) \
    if (j.contains(#name)) s.name = read_uint32(j.at(#name), #name)
#define READ_FLOAT(name) \
    if (j.contains(#name)) s.name = read_float(j.at(#name), #name)

    READ_U32(excitatory_count);
    READ_U32(inhibitory_count);
    READ_U32(steps);
    READ_FLOAT(dt);
    READ_FLOAT(threshold);
    READ_U32(random_seed);
    READ_FLOAT(excitatory_current_min);
    READ_FLOAT(excitatory_current_max);
    READ_FLOAT(inhibitory_current_min);
    READ_FLOAT(inhibitory_current_max);
    READ_FLOAT(initial_voltage_min);
    READ_FLOAT(initial_voltage_max);

#undef READ_U32
#undef READ_FLOAT
}

void apply_topology(Json const& j, TopologyParameters& t) {
    require_object(j, "topology");
    reject_unknown_fields(j, {
        "p_ee", "p_ei", "p_ie", "p_ii",
        "master_seed", "topology_seed", "delay_seed",
        "delay_distribution", "minimum_delay", "maximum_delay",
        "fibonacci_delays", "fibonacci_weights", "memory_budget_bytes"
    }, "topology");

#define READ_NUMBER(name) \
    if (j.contains(#name)) t.name = read_number(j.at(#name), #name)
#define READ_U32(name) \
    if (j.contains(#name)) t.name = read_uint32(j.at(#name), #name)
#define READ_U64(name) \
    if (j.contains(#name)) t.name = read_uint64_string(j.at(#name), #name)

    READ_NUMBER(p_ee);
    READ_NUMBER(p_ei);
    READ_NUMBER(p_ie);
    READ_NUMBER(p_ii);
    READ_U64(master_seed);
    READ_U64(topology_seed);
    READ_U64(delay_seed);
    READ_U32(minimum_delay);
    READ_U32(maximum_delay);
    READ_U64(memory_budget_bytes);

#undef READ_NUMBER
#undef READ_U32
#undef READ_U64

    if (j.contains("delay_distribution")) {
        const auto& value = j.at("delay_distribution");
        if (!value.is_string()) {
            throw std::invalid_argument("delay_distribution must be a string");
        }
        const std::string name = value.get<std::string>();
        if (name == "Uniform") {
            t.delay_distribution = DelayDistribution::Uniform;
        } else if (name == "FibonacciWeighted") {
            t.delay_distribution = DelayDistribution::FibonacciWeighted;
        } else {
            throw std::invalid_argument(
                "delay_distribution must be Uniform or FibonacciWeighted"
            );
        }
    }

    if (j.contains("fibonacci_delays")) {
        const auto& values = j.at("fibonacci_delays");
        if (!values.is_array()) {
            throw std::invalid_argument("fibonacci_delays must be an array");
        }
        std::vector<std::uint32_t> parsed;
        parsed.reserve(values.size());
        for (const auto& value : values) {
            parsed.push_back(read_uint32(value, "fibonacci_delays[]"));
        }
        t.fibonacci_delays = std::move(parsed);
    }

    if (j.contains("fibonacci_weights")) {
        const auto& values = j.at("fibonacci_weights");
        if (!values.is_array()) {
            throw std::invalid_argument("fibonacci_weights must be an array");
        }
        std::vector<double> parsed;
        parsed.reserve(values.size());
        for (const auto& value : values) {
            parsed.push_back(read_number(value, "fibonacci_weights[]"));
        }
        t.fibonacci_weights = std::move(parsed);
    }
}

void validate_or_throw(const ModelParameters& parameters) {
    const auto errors = validate_parameters(parameters);
    if (errors.empty()) {
        return;
    }

    std::string message = "parameter validation failed:";
    for (const auto& error : errors) {
        message += " ";
        message += error.field;
        message += ": ";
        message += error.message;
        message += ";";
    }
    throw std::invalid_argument(message);
}

} // namespace

Json parameters_to_json(const ModelParameters& p) {
    return {
        {"simulation", {
            {"excitatory_count", p.simulation.excitatory_count},
            {"inhibitory_count", p.simulation.inhibitory_count},
            {"steps", p.simulation.steps},
            {"dt", p.simulation.dt},
            {"threshold", p.simulation.threshold},
            {"random_seed", p.simulation.random_seed},
            {"excitatory_current_min", p.simulation.excitatory_current_min},
            {"excitatory_current_max", p.simulation.excitatory_current_max},
            {"inhibitory_current_min", p.simulation.inhibitory_current_min},
            {"inhibitory_current_max", p.simulation.inhibitory_current_max},
            {"initial_voltage_min", p.simulation.initial_voltage_min},
            {"initial_voltage_max", p.simulation.initial_voltage_max}
        }},
        {"topology", {
            {"p_ee", p.topology.p_ee},
            {"p_ei", p.topology.p_ei},
            {"p_ie", p.topology.p_ie},
            {"p_ii", p.topology.p_ii},
            {"master_seed", std::to_string(p.topology.master_seed)},
            {"topology_seed", std::to_string(p.topology.topology_seed)},
            {"delay_seed", std::to_string(p.topology.delay_seed)},
            {"delay_distribution",
                p.topology.delay_distribution == DelayDistribution::Uniform
                    ? "Uniform" : "FibonacciWeighted"},
            {"minimum_delay", p.topology.minimum_delay},
            {"maximum_delay", p.topology.maximum_delay},
            {"fibonacci_delays", p.topology.fibonacci_delays},
            {"fibonacci_weights", p.topology.fibonacci_weights},
            {"memory_budget_bytes", std::to_string(p.topology.memory_budget_bytes)}
        }}
    };
}

ModelParameters parameters_from_json(
    const Json& update,
    const ModelParameters& base
) {
    require_object(update, "parameters");
    reject_unknown_fields(update, {"simulation", "topology"}, "parameter group");

    ModelParameters result = base;

    if (update.contains("simulation")) {
        apply_simulation(update.at("simulation"), result.simulation);
    }
    if (update.contains("topology")) {
        apply_topology(update.at("topology"), result.topology);
    }

    validate_or_throw(result);
    return result;
}

} // namespace spiking::ipc
