#pragma once
#include "routing.h"

namespace phoenix_addon::phoenix {
// Read current settings on every query so checkbox changes reveal their child
// controls in the same draw. Visibility never clears saved settings.
struct PanelVisibility {
    const Settings& settings;
    bool neural() const {
        return settings.mode != input;
    }
    bool comparison() const {
        return settings.mode == split;
    }
    bool pair_sources() const {
        return comparison() && settings.comparison != 0;
    }
    bool divider() const {
        return comparison() && settings.comparison <= 1;
    }
    bool difference_gain() const {
        return comparison() && settings.comparison >= 2;
    }
    bool strengths() const {
        return neural() && settings.controls.enabled;
    }
    bool guides() const {
        return neural() && settings.controls.advanced;
    }
    // NVIDIA overrides drive the self-drive feature in split views too, so
    // the sliders stay visible wherever the native route runs; fullscreen
    // Input is the only mode where they do nothing.
    bool native() const {
        return uses_native(settings.mode);
    }
    bool native_values() const {
        return native() && settings.native_options;
    }
    bool experiments() const {
        return settings.mode == ours;
    }
    bool depth() const {
        return experiments() && settings.experimental;
    }
    bool budget() const {
        return depth() && settings.automatic_budget;
    }
    bool history_cadence() const {
        return neural() && !settings.force_reset;
    }
    bool own_details() const {
        return uses_ours(settings.mode);
    }
    // Wipe furniture (divider lines, side labels, difference caption) belongs
    // only to an actually-split frame. Saved comparison presets survive a
    // Backend switch, so the overlay must key off the displayed route, never
    // the preset alone; otherwise the wipe lingers over fullscreen routes.
    bool wipe_hud() const {
        return settings.mode == split;
    }
    bool hotkey(unsigned action) const {
        return action != 1 && action != 3 ? true : action == 1 ? neural() : depth();
    }
};
}
