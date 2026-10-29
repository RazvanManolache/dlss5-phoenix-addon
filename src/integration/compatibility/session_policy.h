#pragma once
#include "../../native/support/routing.h"
#include "../../native/support/overrides.h"
#include "../../../third_party/nlohmann/json.hpp"
#include <optional>

namespace phoenix_compat {
using Settings = phoenix_addon::phoenix::Settings;
using SettingsJson = nlohmann::json;
inline bool needs_nvidia_dll(const Settings& settings) {
    return settings.mode == phoenix_addon::phoenix::nvidia ||
           (settings.mode == phoenix_addon::phoenix::split &&
            (settings.source_a == 1 || settings.source_b == 1));
}
inline Settings in_process_settings(const Settings& requested) {
    auto effective = requested;
    // NVIDIA-only rendering does not run the selected Phoenix executor.
    // Keep that saved choice intact for the next Phoenix selection.
    if (effective.mode == phoenix_addon::phoenix::nvidia)
        effective.execution_device = "native";
    return effective;
}
inline SettingsJson settings_json(const Settings& settings) {
    auto copy = settings;
    SettingsJson result = SettingsJson::object();
    phoenix_addon::phoenix::visit_settings(copy, [&](const char* key, auto& field) {
        result[key] = field;
    });
    return result;
}
inline SettingsJson live_tuning_json(const Settings& settings) {
    auto copy = settings;
    SettingsJson result = SettingsJson::object();
    phoenix_addon::phoenix::visit_live_tuning(copy, [&](const char* key, auto& field) { result[key] = field; });
    return result;
}
inline void apply_live_tuning_patch(Settings& settings, const SettingsJson& patch) {
    using namespace phoenix_addon::phoenix;
    using phoenix_addon::check;
    check(patch.is_object() && !patch.empty() && patch.size() <= 32, "Invalid live tuning patch");
    auto next = settings;
    for (auto item = patch.begin(); item != patch.end(); ++item) {
        bool found = false;
        visit_live_tuning(next, [&](const char* key, auto& field) {
            if (item.key() != key) return;
            found = true;
            using T = std::decay_t<decltype(field)>;
            if constexpr (std::is_same_v<T, bool>) check(item->is_boolean(), "Boolean live control required");
            else if constexpr (std::is_integral_v<T>) check(item->is_number_integer(), "Integer live control required");
            else check(item->is_number(), "Numeric live control required");
            check(parse_override(item->dump(), field), "Live tuning value out of range");
        });
        check(found, "Live tuning cannot change model or resource settings");
    }
    check(next.valid(), "Invalid live tuning values; existing controls retained");
    copy_live_tuning(settings, next);
}
// Capability restrictions apply at every ingress, including old INIs/presets.
inline bool normalize_capabilities(Settings& settings) {
    if (settings.budget_mode != 1) return false;
    settings.budget_mode = 0; // Companion GPU-time budget; not hidden-window FPS.
    return true;
}
inline bool helper_settings_changed(const Settings& previous, const Settings& desired) {
    auto before = settings_json(previous), after = settings_json(desired);
    for (const auto* key : {"Split", "ComparisonView", "ScreenshotRequest", "CaptureBurst", "ComparisonLabels", "ComparisonLines"}) {
        before.erase(key);
        after.erase(key);
    }
    auto copy = previous;
    phoenix_addon::phoenix::visit_live_tuning(copy, [&](const char* key, auto&) {
        before.erase(key); after.erase(key);
    });
    return before != after;
}
inline bool hud_only_change(const Settings& previous, const Settings& desired) {
    auto before = settings_json(previous), after = settings_json(desired);
    if (before == after) return false;
    // Overlay decoration never changes the image consumed by the runner.
    for (const auto* key : {"ComparisonLabels", "ComparisonLines"}) {
        before.erase(key);
        after.erase(key);
    }
    return before == after;
}
inline bool portable_view_only(const Settings& previous,const Settings& desired){
    auto before=settings_json(previous),after=settings_json(desired);
    for(const auto* key:{"Divider","Comparison","SourceA","SourceB","DifferenceGain","ComparisonLabels","ComparisonLines",
        "CompareLoupe","CompareZoom","LoupeX","LoupeY"}){before.erase(key);after.erase(key);}
    return before==after;
}
inline int helper_presentation_mode(const Settings& settings) {
    // Comparison sources/divider/difference are composed once by the evaluator.
    // The transport must not replace its left half with an unrelated input.
    return settings.mode == phoenix_addon::phoenix::split ? phoenix_addon::phoenix::ours : settings.mode;
}
inline void migrate_processing_switch(bool& enabled, Settings& settings) {
    enabled = enabled && settings.uplift;
    // Compatibility has one user-facing switch. Never send a second hidden
    // bypass gate to a helper that otherwise reports successful processing.
    settings.uplift = true;
}
inline bool apply_live_selection(Settings& applied, const Settings& desired) {
    const bool changed = phoenix_addon::phoenix::copy_live_selection(applied, desired);
    phoenix_addon::phoenix::normalize_direct_comparison(applied);
    applied.uplift = true;
    return changed;
}
inline void merge_settings_patch(Settings& draft, const Settings& applied,
                                 const std::vector<std::string>& keys) {
    const auto fields = settings_json(applied);
    phoenix_addon::phoenix::visit_settings(draft, [&](const char* key, auto& field) {
        if (std::find(keys.begin(), keys.end(), key) != keys.end())
            field = fields.at(key).get<std::decay_t<decltype(field)>>();
    });
}
struct DeferredTimeout {
    std::optional<uint64_t> since;
    void reset() { since.reset(); }
    bool expired(uint64_t tick) {
        if (!since) since = tick;
        return tick >= *since && tick - *since >= 10000;
    }
};
inline bool backbuffer_requires_reset(bool have_engine_motion, bool requested_reset) {
    return requested_reset || !have_engine_motion;
}
} // namespace phoenix_compat
