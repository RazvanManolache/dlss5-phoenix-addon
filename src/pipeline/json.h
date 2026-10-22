#pragma once
#include <type_traits>
#include "options.h"
#include "../../third_party/nlohmann/json.hpp"
namespace phoenix_pipeline {
inline Options options_from_json(const nlohmann::json& source, Options options = {}) {
    if (!source.is_object()) throw std::invalid_argument("Pipeline settings must be an object");
    visit_options(options, [&](const char* name, auto& value) {
        if (source.contains(name)) value = source.at(name).template get<std::remove_reference_t<decltype(value)>>();
    });
    if (!options.valid()) throw std::invalid_argument("Invalid Phoenix pipeline settings");
    return options;
}
inline nlohmann::json options_json(Options options) {
    nlohmann::json result = nlohmann::json::object();
    visit_options(options, [&](const char* name, auto& value) { result[name] = value; });
    return result;
}
} // namespace phoenix_pipeline
