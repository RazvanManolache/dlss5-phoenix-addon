#pragma once
#include "policy.h"
#include <array>
#include <cmath>
#include <cstring>
#include <string>

namespace phoenix_addon {
inline uint64_t native_feature_key(const void* handle) {
    check(handle != nullptr, "phoenix null native feature handle");
    uint64_t value = 0;
    std::memcpy(&value, handle, sizeof(value));
    return value;
}
// x/y are pixel coordinates of the rectangle origin; width/height are its extent.
struct Subrect {
    int x = 0, y = 0, width = 0, height = 0;
};
struct Controls {
    Subrect color, motion, output;
    float motion_x = 1, motion_y = 1, intensity = 1, tone = 1, structure = 1, skin = -1;
    unsigned style = 0;
    int reset = 0, inverted = 1, enabled = 1, auto_mask = 0, ui_correction = 0;
};
inline Controls effective_controls(Controls controls, bool first_record,
                                   bool observed_fresh_feature, bool cold_restart = false) {
    if (first_record) {
        // A thread-migration rebuild restarts our candidate history cold while
        // the engine stays warm: the local reset is honestly set, the
        // engine's parameters are never touched, and native history is
        // unaffected. Takeover quality dips for a frame; nothing corrupts.
        check(controls.reset != 0 || observed_fresh_feature || cold_restart,
              "phoenix takeover needs native Reset or observed first feature evaluation");
        if (observed_fresh_feature || cold_restart)
            controls.reset = 1;
    }
    return controls;
}

// Get failure must restore the native default, even if an implementation wrote
// its output before failing. The reader supplies the NGX success predicate.
template <class Parameters, class Success>
Controls read_controls(const Parameters& parameters, Success success) {
    Controls controls;
    auto get = [&](const char* key, auto& value) {
        const auto fallback = value;
        if (!success(parameters.Get(key, &value)))
            value = fallback;
    };
    auto rect = [&](const char* prefix, Subrect& rectangle) {
        const std::string name = std::string("DLSSNR.") + prefix + "Subrect";
        get((name + "BaseX").c_str(), rectangle.x);
        get((name + "BaseY").c_str(), rectangle.y);
        get((name + "Width").c_str(), rectangle.width);
        get((name + "Height").c_str(), rectangle.height);
    };
    rect("Color", controls.color);
    rect("MVec", controls.motion);
    rect("Output", controls.output);
    get("DLSSNR.MVecScaleX", controls.motion_x);
    get("DLSSNR.MVecScaleY", controls.motion_y);
    get("DLSSNR.Intensity", controls.intensity);
    get("DLSSNR.LocalToneStrength", controls.tone);
    get("DLSSNR.LocalStructureStrength", controls.structure);
    get("DLSSNR.SkinStructureStrength", controls.skin);
    get("DLSSNR.Style", controls.style);
    get("DLSSNR.Reset", controls.reset);
    get("DLSSNR.DepthInverted", controls.inverted);
    get("DLSSNR.Enabled", controls.enabled);
    get("DLSSNR.UseAutoMask", controls.auto_mask);
    get("DLSSNR.UICorrection", controls.ui_correction);
    // Native 0x1a96a unconditionally forces ScalingRatio to 1 after Get.
    return controls;
}

// The native stem counter is an independent QWORD, not a Present/frame ID.
// Failed/abandoned recordings poison the Session, so no reserved value is reused.
class Sampling {
    uint64_t next_ = 0, pending_ = 0;
    bool reserved_ = false, failed_ = false;

  public:
    uint64_t begin(bool reset) {
        check(!failed_ && !reserved_, "phoenix sampling failed/pending");
        pending_ = reset ? 0 : next_;
        reserved_ = true;
        return pending_;
    }
    void commit() {
        check(!failed_ && reserved_, "phoenix sampling commit without record");
        next_ = pending_ + uint64_t(1);
        reserved_ = false;
    }
    void invalidate() noexcept {
        failed_ = true;
        reserved_ = false;
    }
    uint64_t next() const noexcept {
        return next_;
    }
};

using Rectangle = std::array<float, 6>;
struct ResolvedControls {
    Rectangle color{}, history{}, motion{};
    float motion_x = 0, motion_y = 0, tone = 0, structure = 0, style = 0, skin = -1,
          skin_structure = -1;
    uint32_t inverted = 0;
    uint64_t sampling = 0;
    bool recurrent = false;
    uint64_t control_mask_handle = 0;
};

// HDR control override (looking, not measuring): the game may ask its
// native NR for a mode no captured plan covers (RE9 menus send intensity
// 2, style 2, UI correction on). The kernels never see these values --
// resolve_controls refuses before they could -- so overriding them runs
// OUR kernels at settings WE support while native runs at the game's.
// enabled is deliberately untouched: a disabled host path is never
// overridden into existence. Default off; without an explicit config
// declaration the refusal stands. Pair results record the override.
inline Controls hdr_overridden_controls(Controls controls, bool override) {
    if (override) {
        controls.intensity = 1;
        controls.style = 0;
        controls.ui_correction = 0;
    }
    return controls;
}
inline void full_extent(const Subrect& rectangle, unsigned width, unsigned height) {
    check(rectangle.x == 0 && rectangle.y == 0 &&
              (!rectangle.width || rectangle.width == int(width)) &&
              (!rectangle.height || rectangle.height == int(height)),
          "phoenix first-live scope requires full-resource subrects");
}

// Validate the supported full-frame control contract, normalize motion
// scales into texture coordinates, and suppress temporal bindings on reset.
// The stem/head patchers copy these resolved values into fixed ABI slots.
inline ResolvedControls resolve_controls(const Controls& controls, unsigned width, unsigned height,
                                         bool motion_present, bool history_valid,
                                         uint64_t sampling) {
    check(width && height && width <= 16384 && height <= 16384, "phoenix invalid active extent");
    full_extent(controls.color, width, height);
    full_extent(controls.output, width, height);
    if (motion_present)
        full_extent(controls.motion, width, height);
    check(std::isfinite(controls.motion_x) && std::isfinite(controls.motion_y) &&
              std::isfinite(controls.tone) && std::isfinite(controls.structure) &&
              std::isfinite(controls.skin),
          "phoenix non-finite frame control");
    if (!(controls.enabled != 0 && controls.intensity == 1 && controls.style == 0 &&
          controls.ui_correction == 0)) {
        const std::string detail =
            "phoenix unsupported disabled/blend/style/UI host path: enabled=" +
            std::to_string(controls.enabled) + " intensity=" + std::to_string(controls.intensity) +
            " style=" + std::to_string(controls.style) +
            " ui_correction=" + std::to_string(controls.ui_correction);
        check(false, detail.c_str());
    }
    // Registered trained model: registry +20..22 are all 1. Style 0's tone
    // table has mask 0, so no native host color transform is requested.
    ResolvedControls resolved;
    resolved.sampling = sampling;
    resolved.recurrent = history_valid && controls.reset == 0 && motion_present;
    resolved.color = {0, 0, float(width), float(height), 1.0f / float(width), 1.0f / float(height)};
    if (resolved.recurrent) {
        resolved.history = resolved.color;
        resolved.motion = resolved.color;
    }
    resolved.motion_x = controls.motion_x / float(width);
    resolved.motion_y = controls.motion_y / float(height);
    resolved.tone = controls.tone;
    resolved.structure = controls.structure;
    resolved.inverted = controls.inverted != 0;
    if (controls.auto_mask != 0) {
        resolved.skin = controls.skin < 0 ? controls.structure : controls.skin;
        resolved.skin_structure = controls.structure;
    }
    return resolved;
}

template <class T> inline void put(uint8_t* bytes, size_t offset, const T& value) {
    std::memcpy(bytes + offset, &value, sizeof(value));
}
inline void patch_stem(uint8_t* bytes, size_t size, const ResolvedControls& controls) {
    check(bytes && size == 264, "phoenix unexpected stem ABI");
    if (!controls.recurrent || controls.sampling == 0) {
        put(bytes, 8, uint64_t(0));
        put(bytes, 16, uint64_t(0));
    }
    put(bytes, 24, uint64_t(0));
    put(bytes, 32, controls.control_mask_handle);
    put(bytes, 40, controls.history);
    put(bytes, 64, controls.motion);
    put(bytes, 88, Rectangle{});
    put(bytes, 112, controls.control_mask_handle ? controls.color : Rectangle{});
    put(bytes, 136, controls.color);
    put(bytes, 160, controls.motion_x);
    put(bytes, 164, controls.motion_y);
    put(bytes, 168, controls.inverted);
    put(bytes, 172, controls.tone);
    put(bytes, 176, controls.structure);
    put(bytes, 180, controls.style);
    put(bytes, 184, controls.skin);
    put(bytes, 188, controls.skin_structure);
    put(bytes, 192, uint32_t(1));
    put(bytes, 196, 0.0625f);
    put(bytes, 200, controls.sampling);
}
inline void patch_head(uint8_t* bytes, size_t size, const ResolvedControls& controls) {
    check(bytes && size == 184, "phoenix unexpected head ABI");
    if (!controls.recurrent) {
        put(bytes, 88, uint64_t(0));
        put(bytes, 96, uint64_t(0));
    }
    put(bytes, 64, controls.color);
    put(bytes, 112, uint32_t(1));
    put(bytes, 116, controls.history);
    put(bytes, 140, controls.motion);
    put(bytes, 164, controls.motion_x);
    put(bytes, 168, controls.motion_y);
}
} // namespace production_game41
