#pragma once
// Control model for the in-game CNR panel: the OptiScaler-menu-style overlay
// hosted by our own .asi (F8), with backend selection (game native / NVIDIA
// DLL / CNR ours), the 3-way side-by-side view, execution control (skip
// presets, UI-correction forcing, trace capture) and timing readout.
//
// Everything here is plain C++ with no D3D12, window or imgui dependency so
// the mapping, the ms history and the ini round-trip stay headless-testable.
// The ASI's hotkeys (F9/F10/F12) and the panel widgets both write this
// model; the ASI applies the model's state to its globals from one place,
// so the two control paths can never disagree about what is selected.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace phoenix_addon::panel {

// Backend options mirror the ASI Mode order (game=0, nvidia=1, ours=2,
// ours_full=3) so the panel index converts with a cast. `ours` shows the
// three-way split (frame before NR | NVIDIA | ours); `ours_full` covers the
// whole frame with our reconstruction.
constexpr unsigned kBackendCount = 4;
constexpr const char* kBackendLabels[kBackendCount] = {
    "Game native (NR off)",
    "NVIDIA DLSS5 DLL",
    "CNR ours (3-way split)",
    "CNR ours (fullscreen)",
};
constexpr const char* kBackendHelp[kBackendCount] = {
    "Plugin records nothing; the frame is byte-for-byte the game's own.",
    "The DLL evaluates alone. The reference picture.",
    "Left: before NR. Middle: NVIDIA DLSS5. Right: ours at intensity=1 style=0 ui_correction=0.",
    "Our reconstruction fullscreen. Same run settings as the 3-way right band.",
};
// Short names for config files (ReShade.ini [CNR_LIVE] Backend). Unknown or
// absent values select the NVIDIA reference; tests pin the mapping.
constexpr const char* kBackendNames[kBackendCount] = {"game", "nvidia", "ours", "ours_full"};
inline int backend_from_name(const char* name) noexcept {
    if (!name)
        return 1;
    for (unsigned iteration_index = 0; iteration_index < kBackendCount; ++iteration_index) {
        const char* input_cursor = name;
        const char* candidate_cursor = kBackendNames[iteration_index];
        while (*input_cursor && *input_cursor == *candidate_cursor) {
            ++input_cursor;
            ++candidate_cursor;
        }
        if (*input_cursor == 0 && *candidate_cursor == 0)
            return int(iteration_index);
    }
    return 1;
}

// Skip presets mirror the ASI's F9 cycle exactly: the masks here ARE the
// F9 masks, shared so the hotkey and the panel cannot drift apart. Preset
// 1 skips encoder_128 (group 3) + decoder_128 (group 11); preset 2 skips
// vit_1024 (group 7). Tests pin the launch counts against the policy.
constexpr unsigned kSkipPresetCount = 3;
constexpr uint32_t kSkipPresetMasks[kSkipPresetCount] = {0, (1u << 3) | (1u << 11), 1u << 7};
constexpr const char* kSkipPresetLabels[kSkipPresetCount] = {
    "Full network",
    "Skip encoder_128 + decoder_128",
    "Skip vit_1024",
};

// UI-correction forcing: -1 follows the game, 0/1 overwrite the parameter
// before native Evaluate reads it. Mirrors the ASI's F12 cycle.
constexpr int kUicValues[3] = {-1, 0, 1};
constexpr const char* kUicLabels[3] = {
    "Auto (game decides)",
    "Force 0",
    "Force 1",
};

inline unsigned cycle_skip(unsigned preset) noexcept {
    return (preset + 1) % kSkipPresetCount;
}
inline unsigned cycle_backend(unsigned backend) noexcept {
    return (backend + 1) % kBackendCount;
}
inline int cycle_uic(int value) noexcept {
    return value < 0 ? 0 : (value == 0 ? 1 : -1);
}

// Recent per-frame GPU milliseconds, oldest to newest, for the panel's
// trend plot. Same acceptance window as the ASI accumulators: a wrapped
// or bogus pair is dropped, not plotted.
struct MsHistory {
    static constexpr unsigned kCapacity = 120;
    float samples[kCapacity]{};
    unsigned head = 0;  // next write position
    unsigned count = 0; // valid samples, <= kCapacity

    void push(double duration_ms) noexcept {
        if (!(duration_ms >= 0.0) || duration_ms > 1000.0)
            return;
        samples[head] = float(duration_ms);
        head = (head + 1) % kCapacity;
        if (count < kCapacity)
            ++count;
    }
    unsigned size() const noexcept {
        return count;
    }
    double average() const noexcept {
        if (!count)
            return 0.0;
        double total = 0.0;
        for (unsigned iteration_index = 0; iteration_index < count; ++iteration_index)
            total += samples[iteration_index];
        return total / double(count);
    }
    float ordered_min() const noexcept {
        if (!count)
            return 0.0f;
        float value = samples[0];
        for (unsigned iteration_index = 1; iteration_index < count; ++iteration_index)
            if (samples[iteration_index] < value)
                value = samples[iteration_index];
        return value;
    }
    float ordered_max() const noexcept {
        if (!count)
            return 0.0f;
        float value = samples[0];
        for (unsigned iteration_index = 1; iteration_index < count; ++iteration_index)
            if (samples[iteration_index] > value)
                value = samples[iteration_index];
        return value;
    }
    // Oldest sample first, so the plot reads left to right. `out` must
    // hold at least `size()` floats.
    void copy_ordered(float* out) const noexcept {
        if (!out || !count)
            return;
        const unsigned oldest = (head + kCapacity - count) % kCapacity;
        for (unsigned iteration_index = 0; iteration_index < count; ++iteration_index)
            out[iteration_index] = samples[(oldest + iteration_index) % kCapacity];
    }
};

// What the panel persists. Defaults match the ASI's own: NVIDIA backend,
// full network, game-driven UI correction, panel hidden.
struct PanelConfig {
    int backend = 1;
    int skip = 0;
    int uic = -1;
    bool visible = false;
    // Neural-rendering options, shared with the DLSS5-Feeder panel: same names,
    // ranges and defaults, so the two control paths cannot disagree about what
    // a value means. The feeder's upscaling/preset have no phoenix counterpart (no
    // upscale stage, no runtime plan switch) and are deliberately absent.
    int uplift = 1;
    int style = 0;
    float intensity = 1.0f, structure = 0.99f, tone = 0.45f, skin = 0.99f;
    int automask = 1, uicorr = 1;
    float paper_white = 1.0f;
    float transfer_strength = 1.0f, color_strength = 1.0f;
    int depth_mode = 0;
    float mvec_scale_x = 1.0f, mvec_scale_y = 1.0f;
};

constexpr float kNrParamMax = 2.0f;
constexpr float kNrPaperMax = 16.0f;
constexpr float kNrMvecMax = 4.0f;
constexpr const char* kNrStyleLabels[3] = {"Default", "Natural", "Cinematic"};
constexpr const char* kNrDepthLabels[3] = {
    "Use game NGX flag",
    "Force normal depth",
    "Force inverted depth",
};

// Flat key=value ini (no sections). Forgiving by design: unknown keys,
// malformed lines and out-of-range values are ignored and the field keeps
// its default, because a hand-edited config must never break loading.
// Tests pin the exact behavior.
inline PanelConfig parse_ini(const char* text) noexcept {
    PanelConfig config;
    if (!text)
        return config;
    const char* line = text;
    for (;;) {
        const char* end = line;
        while (*end && *end != '\n')
            ++end;
        const char* stop = end;
        while (stop > line && (stop[-1] == '\r' || stop[-1] == ' ' || stop[-1] == '\t'))
            --stop;
        const char* begin = line;
        while (begin < stop && (*begin == ' ' || *begin == '\t'))
            ++begin;
        if (begin < stop && *begin != '#' && *begin != ';' && *begin != '[') {
            const char* equals = begin;
            while (equals < stop && *equals != '=')
                ++equals;
            const char* key_end = equals;
            while (key_end > begin && (key_end[-1] == ' ' || key_end[-1] == '\t'))
                --key_end;
            const char* value = equals;
            if (value < stop)
                ++value;
            while (value < stop && (*value == ' ' || *value == '\t'))
                ++value;
            auto integer = [](const char* first, const char* last, int& out) {
                if (first >= last)
                    return false;
                bool negative = false;
                if (*first == '-') {
                    negative = true;
                    ++first;
                }
                if (first >= last)
                    return false;
                int value = 0;
                for (const char* cursor = first; cursor != last; ++cursor) {
                    if (*cursor < '0' || *cursor > '9')
                        return false;
                    value = value * 10 + (*cursor - '0');
                }
                out = negative ? -value : value;
                return true;
            };
            auto key_is = [&](const char* name) {
                const char* cursor = begin;
                while (cursor != key_end && *name && *cursor == *name) {
                    ++cursor;
                    ++name;
                }
                return cursor == key_end && *name == 0;
            };
            // Floats accept integral text too ("1" is 1.0), so both parsers run
            // and each key family reads its own result.
            auto real = [](const char* first, const char* last, float& out) {
                if (first >= last || last - first >= 64)
                    return false;
                char buf[64];
                const size_t len = size_t(last - first);
                for (size_t iteration_index = 0; iteration_index < len; ++iteration_index)
                    buf[iteration_index] = first[iteration_index];
                buf[len] = 0;
                char* end = nullptr;
                const double parsed_value = std::strtod(buf, &end);
                if (end != buf + len || !std::isfinite(parsed_value))
                    return false;
                out = float(parsed_value);
                return true;
            };
            int parsed = 0;
            const bool is_int = integer(value, stop, parsed);
            float number = 0.0f;
            const bool is_num = real(value, stop, number);
            if (is_int) {
                if (key_is("backend") && parsed >= 0 && parsed < int(kBackendCount))
                    config.backend = parsed;
                if (key_is("skip") && parsed >= 0 && parsed < int(kSkipPresetCount))
                    config.skip = parsed;
                if (key_is("uic") && parsed >= -1 && parsed <= 1)
                    config.uic = parsed;
                if (key_is("visible") && (parsed == 0 || parsed == 1))
                    config.visible = parsed == 1;
                if (key_is("nr_uplift") && (parsed == 0 || parsed == 1))
                    config.uplift = parsed;
                if (key_is("nr_style") && parsed >= 0 && parsed <= 2)
                    config.style = parsed;
                if (key_is("nr_automask") && (parsed == 0 || parsed == 1))
                    config.automask = parsed;
                if (key_is("nr_uicorr") && (parsed == 0 || parsed == 1))
                    config.uicorr = parsed;
                if (key_is("nr_depth_mode") && parsed >= 0 && parsed <= 2)
                    config.depth_mode = parsed;
            }
            if (is_num) {
                if (key_is("nr_intensity") && number >= 0.0f && number <= kNrParamMax)
                    config.intensity = number;
                if (key_is("nr_structure") && number >= 0.0f && number <= kNrParamMax)
                    config.structure = number;
                if (key_is("nr_tone") && number >= 0.0f && number <= kNrParamMax)
                    config.tone = number;
                if (key_is("nr_skin") && number >= 0.0f && number <= kNrParamMax)
                    config.skin = number;
                if (key_is("nr_paper_white") && number >= 0.0f && number <= kNrPaperMax)
                    config.paper_white = number;
                if (key_is("nr_transfer_strength") && number >= 0.0f && number <= kNrParamMax)
                    config.transfer_strength = number;
                if (key_is("nr_color_strength") && number >= 0.0f && number <= kNrParamMax)
                    config.color_strength = number;
                if (key_is("nr_mvec_scale_x") && number >= 0.0f && number <= kNrMvecMax)
                    config.mvec_scale_x = number;
                if (key_is("nr_mvec_scale_y") && number >= 0.0f && number <= kNrMvecMax)
                    config.mvec_scale_y = number;
            }
        }
        if (!*end)
            return config;
        line = end + 1;
    }
}

inline std::string serialize_ini(const PanelConfig& config) {
    std::string out;
    out += "backend=" + std::to_string(config.backend) + "\n";
    out += "skip=" + std::to_string(config.skip) + "\n";
    out += "uic=" + std::to_string(config.uic) + "\n";
    out += "visible=" + std::string(config.visible ? "1" : "0") + "\n";
    out += "nr_uplift=" + std::to_string(config.uplift) + "\n";
    out += "nr_style=" + std::to_string(config.style) + "\n";
    out += "nr_automask=" + std::to_string(config.automask) + "\n";
    out += "nr_uicorr=" + std::to_string(config.uicorr) + "\n";
    out += "nr_depth_mode=" + std::to_string(config.depth_mode) + "\n";
    char buf[64];
    const auto flo = [&](const char* key, float value) {
        std::snprintf(buf, sizeof(buf), "%.6f", double(value));
        out += key;
        out += "=";
        out += buf;
        out += "\n";
    };
    flo("nr_intensity", config.intensity);
    flo("nr_structure", config.structure);
    flo("nr_tone", config.tone);
    flo("nr_skin", config.skin);
    flo("nr_paper_white", config.paper_white);
    flo("nr_transfer_strength", config.transfer_strength);
    flo("nr_color_strength", config.color_strength);
    flo("nr_mvec_scale_x", config.mvec_scale_x);
    flo("nr_mvec_scale_y", config.mvec_scale_y);
    return out;
}

// DXGI format values from dxgiformat.h (the model stays free of windows
// headers; the host casts both ways).
constexpr uint32_t kDxgiR8G8B8A8Unorm = 28;
constexpr uint32_t kDxgiR8G8B8A8UnormSrgb = 29;

// The panel draws into the engine's NR output, so its extent is whatever
// the output is this frame. An sRGB target needs the linear view for the
// RTV (the same mapping OptiScaler's own menu applies); everything else
// renders as-is.
inline uint32_t render_target_format(uint32_t target_format) noexcept {
    return target_format == kDxgiR8G8B8A8UnormSrgb ? kDxgiR8G8B8A8Unorm : target_format;
}

// The engine output's identity for the RTV cache: a new resource, size or
// format rebuilds the panel's render-target view. Compared every recorded
// frame; rebuilding is rare (resolution or output changes only).
struct OutputIdentity {
    uint64_t resource = 0;
    uint32_t width = 0, height = 0, format = 0;
};
inline bool output_changed(const OutputIdentity& cached, const OutputIdentity& current) noexcept {
    return cached.resource != current.resource || cached.width != current.width ||
           cached.height != current.height || cached.format != current.format;
}

} // namespace production_game41::panel
