#pragma once
#include "controls.h"

namespace phoenix_addon {
// UI owns only these supported strengths. Host resource/state/mode contracts
// are deliberately not overridable, including the first-frame takeover reset.
struct PhoenixLiveControls {
    bool enabled = false;
    float tone = 1, structure = 1;
    bool advanced = false, auto_mask = false;
    float skin = -1, motion_x = 1, motion_y = 1;
    int depth = -1;
    bool reset_requested = false;

    bool valid() const noexcept {
        return std::isfinite(tone) && std::isfinite(structure) && tone >= 0 && tone <= 2 &&
               structure >= 0 && structure <= 2 && std::isfinite(skin) && skin >= -1 && skin <= 2 &&
               std::isfinite(motion_x) && motion_x >= -4 && motion_x <= 4 &&
               std::isfinite(motion_y) && motion_y >= -4 && motion_y <= 4 && depth >= -1 &&
               depth <= 1;
    }
    Controls apply(Controls controls) const {
        check(valid(), "Phoenix DLSS5 live controls are outside their supported finite ranges");
        if (enabled) {
            controls.tone = tone;
            controls.structure = structure;
        }
        if (advanced) {
            controls.skin = skin;
            controls.auto_mask = auto_mask ? 1 : 0;
            // RenoDX-faithful motion guides: the host owns the MV unit
            // convention (RE9 reports render-extent scales); the override
            // is a multiplier, neutral at 1. Absolute replacement
            // collapses reprojection to near-zero motion and shimmers.
            controls.motion_x *= motion_x;
            controls.motion_y *= motion_y;
            if (depth >= 0)
                controls.inverted = depth;
        }
        if (reset_requested)
            controls.reset = 1;
        return controls;
    }
    void recorded() noexcept {
        reset_requested = false;
    }
};
}
