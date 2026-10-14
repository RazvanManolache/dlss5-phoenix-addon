#pragma once
#include "../../native/support/routing.h"
#include "../../../third_party/nlohmann/json.hpp"

namespace phoenix_compat {
using Settings = phoenix_addon::phoenix::Settings;
using SettingsJson = nlohmann::json;
inline SettingsJson settings_json(const Settings& settings) {
    auto copy = settings;
    SettingsJson result = SettingsJson::object();
    phoenix_addon::phoenix::visit_settings(copy, [&](const char* key, auto& field) {
        result[key] = field;
    });
    return result;
}
// Capability restrictions apply at every ingress, including old INIs/presets.
inline bool normalize_capabilities(Settings& settings) {
    if (settings.budget_mode != 1) return false;
    settings.budget_mode = 0; // Companion GPU-time budget; not hidden-window FPS.
    return true;
}
inline bool helper_settings_changed(const Settings& previous, const Settings& desired) {
    auto before = settings_json(previous), after = settings_json(desired);
    for (const auto* key : {"Split", "ComparisonView", "ScreenshotRequest", "CaptureBurst"}) {
        before.erase(key);
        after.erase(key);
    }
    return before != after;
}
} // namespace phoenix_compat
