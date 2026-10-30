#pragma once
#include "live_controls.h"
#include "budget.h"
#include "device_selection.h"
#include "kernel_style.h"
#include "resolution.h"
#include "processing.h"
#include "../../pipeline/options.h"
#include "../model/registry.h"
#include <variant>
#include <array>
#include <vector>
#include <tuple>

namespace phoenix_addon::phoenix {
inline constexpr int kMaxOffloadAgeFrames = 6000, kMaxOffloadAgeMs = 120000;
enum Mode { input = 0, nvidia = 1, split = 2, ours = 3 };
inline constexpr const char* names[] = {"Input (NR bypass)", "NVIDIA DLSS5 DLL",
                                        "Input | NVIDIA DLSS5 | Phoenix DLSS5", "Phoenix DLSS5"};
inline bool uses_native(int mode) {
    return mode == nvidia || mode == split;
}
inline bool uses_ours(int mode) {
    return mode == ours || mode == split;
}
// Comparison wipe modes (see phoenix_comparison.h): the pair wipe (1)
// displays the previous frame's snapshot, because the game's NR writes may
// still be in flight on another queue (Streamline interposer, async compute)
// when the snapshot copy records, and a current-frame snapshot can then read
// a stale pre-NR buffer (dark smear in the NVIDIA half). Difference modes
// (2-5) need temporal alignment for forensics and keep the current frame.
// The D3D12 reorder itself is verified by gameplay captures, which menus
// cannot exercise (static scenes hide the race).
inline bool wipe_uses_previous_frame(unsigned mode) noexcept {
    return mode == 1;
}
// Effective self-drive value: the configured override when its gate is on,
// otherwise the reference host default (1.0 strengths, Default style, skin
// -1, masks/correction on). Headless-tested; keeps one branch out of the
// recording path.
template <class T> inline T host_strength(bool specified, T value, T reference_default) noexcept {
    return specified ? value : reference_default;
}
struct Settings {
    int mode = ours;
    std::string execution_device = "native";
    std::string native_kernel_style = "auto";
    std::string worker_engine = "auto";
    int offload_age_frames = 4, offload_age_ms = 100;
    int cpu_threads = 0;
    int resolution_mode = 0, resolution_quality = 0, resolution_aspect = 0;
    int upscaler = 0; // Spatial, NVIDIA DLSS SR, or Intel XeSS.
    int resolution_ratio_width = 16, resolution_ratio_height = 9;
    std::string model_source = "configured", model_folder, model_dll, model_id;
    std::string nvidia_dll_folder; // Optional shared DLL fallback; local selection wins.
    std::string model_library_directory; // Explicit shared native-model index root.
    PhoenixLiveControls controls;
    Processing processing;
    phoenix_pipeline::Options pipeline;
    int nvidia_model = -1; // Match Phoenix, DLL default, Original, Lightweight FP8.
    bool native_options = false;
    float intensity = 1, divider = 2.f / 3.f;
    int style = 0, ui_correction = 0;
    // The master neural-rendering switch: false skips
    // every native override block, leaving the game's own NR controls alone.
    bool uplift = true;
    // Self-drive host recipe (mirrors the reference NR host): explicit
    // exposure globals, global tone, and the jitter source for our own NR
    // feature. The game-feature overrides above stay untouched.
    float host_exposure = 1, host_exposure_scale = 1, host_globaltone = 1;
    int host_jitter = 1;
    // Reversible HDR bridge for native NR: gamut/magnitude compression,
    // sRGB in FP16, opaque alpha. Off uses the older UNORM8 edit transfer.
    int host_bridge = 1;
    float host_white_nits = 500;
    // Self-drive NR feature contract flags (create-time, recreate on change):
    // HostHdr declares the proxy domain to the DLL (0 = SDR, 1 = HDR; pairs
    // with the linear bridge above), NRAutoExposure keeps the DLL's own
    // exposure tracking. Both default to the probe-verified recipe.
    int host_hdr = 0;
    bool nr_auto_exposure = true;
    // Self-drive NR feature id (create-time, recreates on change): 18 drives
    // the dlssnr runtime's own NR network; 13 drives Ray Reconstruction
    // through the NGX core. Retained for old profiles; direct self-drive
    // refuses 13 because it lacks the pre-denoise RR guide contract.
    int nr_feature = 18;
    // One-shot screenshot request. Increment the token for a new capture.
    int screenshot_request = 0;
    // Reference-grid unjitter for our own network (meter + normalize taps):
    // on resamples the snapshot onto the jitter-free grid, off feeds the
    // jittered snapshot with the raw game motion. Off is the default: the
    // reconstructed kernels receive no jitter offset, so shifting color
    // without shifting motion misregisters history at the jitter period
    // (measured: unjittered static-scene breathing ~2x worse than off).
    bool host_unjitter = false;
    int reset_after_switch = 1, reset_every = 0;
    bool experimental = false;
    int transformer_blocks = 8;
    bool automatic_budget = false;
    float budget_ms = 5;
    int budget_mode = 0; // 0: DLSS5 Phoenix GPU milliseconds; 1: game presentation FPS.
    float desired_fps = 60;
    int refinement_passes = 1;
    std::array<int, 8> block_repeats{{1, 1, 1, 1, 1, 1, 1, 1}};
    int minimum_blocks = 1;
    int comparison = 0, source_a = 1, source_b = 2;
    float difference_gain = 1;
    bool comparison_labels = true, comparison_lines = true;
    bool force_reset = false, external_reload = false;
    int log_level = 1;
    // Offline capture burst length (0 = idle, at most 64 frames). Frames
    // retire one at a time; GPU readbacks are released after the final frame.
    // Consumed on arm: the tick resets it to zero so a burst never
    // repeats by itself.
    int capture_burst = 0;
    unsigned blocks_for(int route) const noexcept {
        return route == ours && experimental ? unsigned(transformer_blocks) : 8u;
    }
    double budget_target_ms() const noexcept {
        return budget_mode == 1 ? 1000.0 / desired_fps : budget_ms;
    }
    bool valid() const {
        for (int repeat_count : block_repeats)
            if (repeat_count < 0 || repeat_count > 4)
                return false;
        return pipeline.valid() && processing.valid() && phoenix_kernel::valid_style(native_kernel_style) && nvidia_model >= -1 && nvidia_model <= 2 && upscaler >= 0 && upscaler <= 2 && mode >= input && mode <= ours &&
               controls.valid() && resolution_mode >= 0 && resolution_mode <= 1 &&
               resolution_quality >= 0 && resolution_quality <= 2 && resolution_aspect >= 0 &&
               resolution_aspect <= 5 && resolution_ratio_width >= 1 &&
               resolution_ratio_width <= 100 && resolution_ratio_height >= 1 &&
               resolution_ratio_height <= 100 && valid_execution_device(execution_device) &&
               (model_source == "configured" || model_source == "model_folder" ||
                model_source == "nvidia_dll") &&
               (model_id.empty() || std::any_of(phoenix_model::definitions.begin(), phoenix_model::definitions.end(),
                    [&](const auto& model) { return model.id == model_id; })) &&
               model_folder.size() < 4096 && model_folder.find('\0') == std::string::npos &&
               model_dll.size() < 4096 && model_dll.find('\0') == std::string::npos &&
               nvidia_dll_folder.size() < 4096 && nvidia_dll_folder.find('\0') == std::string::npos &&
               model_library_directory.size() < 4096 && model_library_directory.find('\0') == std::string::npos &&
               (worker_engine == "auto" || worker_engine == "portable" ||
                worker_engine == "matrix" || worker_engine == "fp8") &&
               offload_age_frames >= 1 && offload_age_frames <= kMaxOffloadAgeFrames &&
               offload_age_ms >= 1 && offload_age_ms <= kMaxOffloadAgeMs && cpu_threads >= 0 &&
               cpu_threads <= 65535 && std::isfinite(intensity) && intensity >= 0 &&
               intensity <= 2 && std::isfinite(divider) && divider >= .1f && divider <= .9f &&
               style >= 0 && style <= 2 && ui_correction >= 0 && ui_correction <= 1 &&
               reset_after_switch >= 1 && reset_after_switch <= 120 && reset_every >= 0 &&
               reset_every <= 600 && transformer_blocks >= 1 && transformer_blocks <= 8 &&
               minimum_blocks >= 1 && minimum_blocks <= transformer_blocks &&
               std::isfinite(budget_ms) && budget_ms >= .1f && budget_ms <= 100.f &&
               budget_mode >= 0 && budget_mode <= 1 && std::isfinite(desired_fps) &&
               desired_fps >= 10.f && desired_fps <= 360.f && refinement_passes >= 1 &&
               refinement_passes <= 10 && host_jitter >= 0 && host_jitter <= 1 && host_bridge >= 0 &&
               host_bridge <= 1 && host_hdr >= 0 && host_hdr <= 1 &&
               std::isfinite(host_white_nits) && host_white_nits >= 1 && host_white_nits <= 10000 &&
               std::isfinite(host_exposure) && host_exposure >= 0.05f && host_exposure <= 4.0f &&
               std::isfinite(host_exposure_scale) && host_exposure_scale >= 0.05f &&
               host_exposure_scale <= 4.0f && std::isfinite(host_globaltone) &&
               host_globaltone >= 0 && host_globaltone <= 2 &&
               (nr_feature == 13 || nr_feature == 18) && screenshot_request >= 0 &&
               comparison >= 0 && comparison <= 5 && source_a >= 0 && source_a <= 2 &&
               source_b >= 0 && source_b <= 2 && std::isfinite(difference_gain) &&
               difference_gain >= 1 && difference_gain <= 64 && log_level >= 0 && log_level <= 2 &&
               capture_burst >= 0 && capture_burst <= 64;
    }
};
inline bool normalize_direct_comparison(Settings& settings) {
    if (settings.mode != split || settings.comparison != 0) return false;
    settings.comparison = 1; // Direct feeds expose pairs, not the legacy three-way view.
    return true;
}
// Presentation choices apply independently of pending model/quality edits.
inline bool copy_live_selection(Settings& target, const Settings& source) {
    auto fields = [](auto& value) {
        return std::tie(value.mode, value.comparison, value.source_a, value.source_b,
                        value.divider, value.difference_gain,
                        value.comparison_labels, value.comparison_lines);
    };
    const bool view_changed=phoenix_pipeline::copy_comparison(target.pipeline,source.pipeline);
    const bool changed = view_changed||fields(target) != fields(source);
    fields(target) = fields(source);
    return changed;
}
// Per-frame controls only: never include model, executor or feature-create settings.
template <class Visitor> void visit_live_tuning(Settings& settings, Visitor visit) {
    visit("OverrideStrengths", settings.controls.enabled);
    visit("Tone", settings.controls.tone);
    visit("Structure", settings.controls.structure);
    visit("Advanced", settings.controls.advanced);
    visit("AutoMask", settings.controls.auto_mask);
    visit("Skin", settings.controls.skin);
    visit("MotionX", settings.controls.motion_x);
    visit("MotionY", settings.controls.motion_y);
    visit("Depth", settings.controls.depth);
    visit("PhoenixAppearance", settings.controls.appearance);
    visit("PhoenixIntensity", settings.controls.intensity);
    visit("PhoenixStyle", settings.controls.style);
    visit("NativeOptions", settings.native_options);
    visit("Intensity", settings.intensity);
    visit("Style", settings.style);
    visit("UICorrection", settings.ui_correction);
    visit("HostExposure", settings.host_exposure);
    visit("HostExposureScale", settings.host_exposure_scale);
    visit("HostGlobalTone", settings.host_globaltone);
    visit("HostJitter", settings.host_jitter);
    visit("HostWhiteNits", settings.host_white_nits);
    visit("HostUnjitter", settings.host_unjitter);
}
inline bool copy_live_tuning(Settings& target, const Settings& source) {
    auto fields = [](auto& s) {
        return std::tie(s.controls.enabled, s.controls.tone, s.controls.structure,
                        s.controls.advanced, s.controls.auto_mask, s.controls.skin,
                        s.controls.motion_x, s.controls.motion_y, s.controls.depth,
                        s.controls.appearance,s.controls.intensity,s.controls.style,
                        s.native_options, s.intensity, s.style, s.ui_correction,
                        s.host_exposure, s.host_exposure_scale, s.host_globaltone,
                        s.host_jitter, s.host_white_nits, s.host_unjitter);
    };
    const bool changed = fields(target) != fields(source);
    fields(target) = fields(source);
    return changed;
}
template <class Visitor> void visit_settings(Settings& settings, Visitor visit) {
    phoenix_pipeline::visit_options(settings.pipeline, visit);
    visit_processing(settings.processing, visit);
    visit("NvidiaModel", settings.nvidia_model);
    visit("ModelSource", settings.model_source);
    visit("ModelId", settings.model_id);
    visit("ModelFolder", settings.model_folder);
    visit("ModelDll", settings.model_dll);
    visit("NvidiaDllFolder", settings.nvidia_dll_folder);
    visit("ModelLibraryDirectory", settings.model_library_directory);
    visit("ExecutionDevice", settings.execution_device);
    visit("NativeKernelStyle", settings.native_kernel_style);
    visit("WorkerEngine", settings.worker_engine);
    visit("OffloadAgeFrames", settings.offload_age_frames);
    visit("OffloadAgeMs", settings.offload_age_ms);
    visit("CpuThreads", settings.cpu_threads);
    visit("ResolutionMode", settings.resolution_mode);
    visit("Upscaler", settings.upscaler);
    visit("ResolutionQuality", settings.resolution_quality);
    visit("ResolutionAspect", settings.resolution_aspect);
    visit("ResolutionRatioWidth", settings.resolution_ratio_width);
    visit("ResolutionRatioHeight", settings.resolution_ratio_height);
    visit("Backend", settings.mode);
    visit_live_tuning(settings, visit);
    visit("Divider", settings.divider);
    visit("NeuralUplift", settings.uplift);
    visit("HostBridge", settings.host_bridge);
    visit("HostHdr", settings.host_hdr);
    visit("NRAutoExposure", settings.nr_auto_exposure);
    visit("NrFeature", settings.nr_feature);
    visit("ScreenshotRequest", settings.screenshot_request);
    visit("ResetAfterSwitch", settings.reset_after_switch);
    visit("ResetEvery", settings.reset_every);
    visit("Experimental", settings.experimental);
    visit("TransformerBlocks", settings.transformer_blocks);
    visit("AutomaticBudget", settings.automatic_budget);
    visit("BudgetMs", settings.budget_ms);
    visit("MinimumBlocks", settings.minimum_blocks);
    visit("BudgetMode", settings.budget_mode);
    visit("DesiredFPS", settings.desired_fps);
    visit("RefinementPasses", settings.refinement_passes);
    const char* repeat_names[]{"Block1Repeats", "Block2Repeats", "Block3Repeats", "Block4Repeats",
                               "Block5Repeats", "Block6Repeats", "Block7Repeats", "Block8Repeats"};
    for (unsigned iteration_index = 0; iteration_index < 8; ++iteration_index)
        visit(repeat_names[iteration_index], settings.block_repeats[iteration_index]);
    visit("Comparison", settings.comparison);
    visit("SourceA", settings.source_a);
    visit("SourceB", settings.source_b);
    visit("DifferenceGain", settings.difference_gain);
    visit("ComparisonLabels", settings.comparison_labels);
    visit("ComparisonLines", settings.comparison_lines);
    visit("ForceReset", settings.force_reset);
    visit("ExternalReload", settings.external_reload);
    visit("LogLevel", settings.log_level);
    visit("CaptureBurst", settings.capture_burst);
}

// Per-section defaults (pure, CPU-tested): every overlay option belongs to
// exactly one section, so each option reaches its default through its
// section button. Values mirror the struct defaults above.
inline void reset_neural_section(Settings& settings) {
    settings.uplift = true;
    settings.controls.enabled = false;
    settings.controls.tone = 1;
    settings.controls.structure = 1;
    settings.controls.advanced = false;
    settings.controls.auto_mask = false;
    settings.controls.skin = -1;
    settings.controls.motion_x = 1;
    settings.controls.motion_y = 1;
    settings.controls.depth = -1;
    settings.controls.appearance=false;
    settings.controls.intensity=1;
    settings.controls.style=0;
    settings.host_unjitter = false;
}
inline void reset_native_section(Settings& settings) {
    settings.nvidia_model = -1;
    settings.native_options = false;
    settings.intensity = 1;
    settings.style = 0;
    settings.ui_correction = 0;
    settings.host_exposure = 1;
    settings.host_exposure_scale = 1;
    settings.host_globaltone = 1;
    settings.host_jitter = 1;
    settings.host_bridge = 1;
    settings.host_white_nits = 500;
    settings.host_hdr = 0;
    settings.nr_auto_exposure = true;
    settings.nr_feature = 18;
}
inline void reset_comparison_section(Settings& settings) {
    settings.comparison = 0;
    settings.source_a = 1;
    settings.source_b = 2;
    settings.divider = 2.f / 3.f;
    settings.difference_gain = 1;
    settings.comparison_labels = true;
    settings.comparison_lines = true;
}
inline void reset_history_section(Settings& settings) {
    settings.force_reset = false;
    settings.reset_after_switch = 1;
    settings.reset_every = 0;
}
// RenoDX reference values for every NR-driving option (pure, CPU-tested):
// the DLL comparison baseline. Strengths at 1.0 with skin off, Default
// style, correction on, host scales at 1x, scene exposure untouched. The
// bridge select is a path choice, not an NR value, and stays as configured.
inline void apply_nr_reference(Settings& settings) {
    settings.uplift = true;
    settings.controls.enabled = false;
    settings.controls.tone = 1;
    settings.controls.structure = 1;
    settings.controls.advanced = false;
    settings.controls.auto_mask = true;
    settings.controls.skin = -1;
    settings.controls.motion_x = 1;
    settings.controls.motion_y = 1;
    settings.controls.depth = -1;
    settings.native_options = true;
    settings.intensity = 1;
    settings.style = 0;
    settings.ui_correction = 1;
    settings.host_exposure = 1;
    settings.host_exposure_scale = 1;
    settings.host_globaltone = 1;
    settings.host_jitter = 1;
}
// Observe-mode config rules (pure, CPU-tested): a record-only run never
// claims frames, runs routing tests, skips groups, or exceeds the record
// cap. Past the cap the detour keeps forwarding silently.
inline bool observe_config_valid(bool claim, bool has_routing_period, unsigned routing_period,
                                 bool has_skip_groups, unsigned frame_limit) noexcept {
    if (claim)
        return false;
    if (has_routing_period && routing_period != 0)
        return false;
    if (has_skip_groups)
        return false;
    return frame_limit >= 1 && frame_limit <= 4096;
}

// Native must receive its real parameter object (it has private fields).
// Capture every edited key before changing any, and restore after Evaluate.
// Missing typed getters refuse the operation instead of guessing a default.
template <class Parameters, class Success> class ParameterEdits {
    using Value = std::variant<int, unsigned, float>;
    struct Edit {
        const char* key;
        Value before, after;
    };
    Parameters& parameters_;
    Success success_;
    std::vector<Edit> edits_;
    bool applied_ = false;

  public:
    ParameterEdits(Parameters& parameters, Success success)
        : parameters_(parameters), success_(success) {}
    ParameterEdits(const ParameterEdits&) = delete;
    template <class T> void add(const char* key, T after) {
        check(!applied_, "Phoenix DLSS5 cannot extend an applied parameter transaction");
        T before{};
        check(success_(parameters_.Get(key, &before)),
              "Phoenix DLSS5 native override lacks a restorable typed getter");
        edits_.push_back({key, before, after});
    }
    void apply() {
        check(!applied_, "Phoenix DLSS5 duplicate parameter transaction");
        applied_ = true;
        try {
            for (const auto& edit : edits_)
                std::visit([&](auto value) { parameters_.Set(edit.key, value); }, edit.after);
        } catch (...) {
            restore();
            throw;
        }
    }
    void restore() {
        if (!applied_)
            return;
        for (auto lookup_iterator = edits_.rbegin(); lookup_iterator != edits_.rend();
             ++lookup_iterator)
            std::visit([&](auto value) { parameters_.Set(lookup_iterator->key, value); },
                       lookup_iterator->before);
        applied_ = false;
    }
    ~ParameterEdits() noexcept {
        try {
            restore();
        } catch (...) {
        }
    }
};
}
