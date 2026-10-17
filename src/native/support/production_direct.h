#pragma once
// Production direct, post-SR/RR integration. No NR DLL or synthetic NGX feature is created.
// The host serializes all methods and supplies observed states and queue ownership.
#include "../session/session.h"
#include "routing.h"
#include "compute_pass.h"
#include "executor.h"
#include "guides.h"
#include "resolution_bridge.h"
#include "temporal_upscaler.h"
#include <cstdint>
#include <cstring>
#include <d3dcompiler.h>

namespace phoenix_addon::phoenix::direct {
inline bool extent_supported(unsigned width, unsigned height) {
    return phoenix_generated::Geometry::supported(width, height);
}
inline bool source_format(DXGI_FORMAT format) {
    return format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R11G11B10_FLOAT;
}
inline bool motion_format(DXGI_FORMAT format) {
    return format == DXGI_FORMAT_R16G16_FLOAT || format == DXGI_FORMAT_R32G32_FLOAT;
}
inline bool comparison_supported(const Settings& settings, bool have_nvidia_output) {
    if (settings.mode != split)
        return true;
    if (settings.comparison < 1 || settings.comparison > 5)
        return false;
    const bool uses_nr = (settings.source_a == 1 || settings.source_b == 1);
    // Source 1 is the self-hosted native NR image; without a live NR feature
    // it has no pixels and must be refused rather than mislabeled. With one,
    // only the two-way wipe is supported: the resolve transfer composes whole
    // sides, not difference views.
    if (!uses_nr)
        return true;
    return have_nvidia_output && settings.comparison == 1;
}
// Offline capture frame (pure, CPU-tested): everything the replay probe
// needs to re-run record() without the game. Color/motion/output are tight
// row-major half blobs; meta is a key=value text sidecar (no JSON
// dependency, hand-parseable). bpp disambiguates UNORM8 from FP16 inputs.
struct CaptureMeta {
    unsigned width = 0, height = 0, color_format = 0, motion_format = 0;
    unsigned color_bpp = 0, motion_bpp = 0, frame = 0, blocks = 8;
    unsigned refinement_passes = 1;
    std::array<int, 8> block_repeats{{1, 1, 1, 1, 1, 1, 1, 1}};
    bool default_repetitions() const {
        return refinement_passes == 1 &&
               block_repeats == std::array<int, 8>{{1, 1, 1, 1, 1, 1, 1, 1}};
    }
    int reset = 0, enabled = 1, automask = 0, style = 0, inverted = 0, uicorrection = 0;
    int srgb = 0, unjitter = 1, mode = 3, reference_feed = 0;
    float jitter_x = 0, jitter_y = 0, motion_x = 1, motion_y = 1;
    float intensity = 1, tone = 1, structure = 1, skin = -1;
};
struct CaptureFrame {
    CaptureMeta meta;
    std::vector<uint8_t> color, motion, output;
    // Same-command-list samples, retired by the existing capture fence.
    std::array<std::vector<uint8_t>, 4> lineage; // proxy, native, phoenix, feed
    std::vector<uint8_t> exposure;
    int source_a = 0, source_b = 0;
    float divider = 0.5f;
};
inline std::string capture_meta_write(const CaptureMeta& metadata) {
    std::string text;
    auto put = [&](const char* key, const std::string& value) {
        text += key;
        text += '=';
        text += value;
        text += '\n';
    };
    auto write_unsigned = [&](const char* key, unsigned value) { put(key, std::to_string(value)); };
    auto write_integer = [&](const char* key, int value) { put(key, std::to_string(value)); };
    auto write_float = [&](const char* key, float value) {
        put(key, std::to_string(double(value)));
    };
    write_unsigned("width", metadata.width);
    write_unsigned("height", metadata.height);
    write_unsigned("color_format", metadata.color_format);
    write_unsigned("motion_format", metadata.motion_format);
    write_unsigned("color_bpp", metadata.color_bpp);
    write_unsigned("motion_bpp", metadata.motion_bpp);
    write_unsigned("frame", metadata.frame);
    write_unsigned("blocks", metadata.blocks);
    write_unsigned("refinement_passes", metadata.refinement_passes);
    for (unsigned block_index = 0; block_index < 8; ++block_index)
        write_unsigned(("block_repeat_" + std::to_string(block_index + 1)).c_str(),
                       unsigned(metadata.block_repeats[block_index]));
    write_integer("reset", metadata.reset);
    write_integer("enabled", metadata.enabled);
    write_integer("automask", metadata.automask);
    write_integer("style", metadata.style);
    write_integer("inverted", metadata.inverted);
    write_integer("uicorrection", metadata.uicorrection);
    write_integer("srgb", metadata.srgb);
    write_integer("unjitter", metadata.unjitter);
    write_integer("mode", metadata.mode);
    write_integer("reference_feed", metadata.reference_feed);
    write_float("jitter_x", metadata.jitter_x);
    write_float("jitter_y", metadata.jitter_y);
    write_float("motion_x", metadata.motion_x);
    write_float("motion_y", metadata.motion_y);
    write_float("intensity", metadata.intensity);
    write_float("tone", metadata.tone);
    write_float("structure", metadata.structure);
    write_float("skin", metadata.skin);
    return text;
}
// Read the stable key=value capture format, retaining defaults for optional
// fields. Reject unsupported pass/repetition counts before any GPU replay.
inline bool capture_meta_parse(const std::string& text, CaptureMeta& metadata,
                               bool allow_repetitions = false) {
    metadata = CaptureMeta{};
    size_t pos = 0;
    auto read_unsigned = [&](const char* field_name, bool& success, unsigned long& unsigned_value) {
        const std::string key = std::string(field_name) + '=';
        size_t value_begin = text.find(key, pos);
        if (value_begin == std::string::npos) {
            success = false;
            return;
        }
        value_begin += key.size();
        size_t value_end = text.find('\n', value_begin);
        const std::string token = text.substr(
            value_begin, value_end == std::string::npos ? value_end : value_end - value_begin);
        try {
            unsigned_value = std::stoul(token);
        } catch (...) {
            success = false;
        }
    };
    auto read_float = [&](const char* field_name, bool& success, double& floating_value) {
        const std::string key = std::string(field_name) + '=';
        size_t value_begin = text.find(key, pos);
        if (value_begin == std::string::npos) {
            success = false;
            return;
        }
        value_begin += key.size();
        size_t value_end = text.find('\n', value_begin);
        const std::string token = text.substr(
            value_begin, value_end == std::string::npos ? value_end : value_end - value_begin);
        try {
            floating_value = std::stod(token);
        } catch (...) {
            success = false;
        }
    };
    bool success = true;
    unsigned long unsigned_value = 0;
    double floating_value = 0;
    read_unsigned("width", success, unsigned_value);
    metadata.width = unsigned(unsigned_value);
    read_unsigned("height", success, unsigned_value);
    metadata.height = unsigned(unsigned_value);
    read_unsigned("color_format", success, unsigned_value);
    metadata.color_format = unsigned(unsigned_value);
    read_unsigned("motion_format", success, unsigned_value);
    metadata.motion_format = unsigned(unsigned_value);
    read_unsigned("color_bpp", success, unsigned_value);
    metadata.color_bpp = unsigned(unsigned_value);
    read_unsigned("motion_bpp", success, unsigned_value);
    metadata.motion_bpp = unsigned(unsigned_value);
    read_unsigned("frame", success, unsigned_value);
    metadata.frame = unsigned(unsigned_value);
    read_unsigned("blocks", success, unsigned_value);
    metadata.blocks = unsigned(unsigned_value);
    if (text.find("refinement_passes=") != std::string::npos) {
        read_unsigned("refinement_passes", success, unsigned_value);
        metadata.refinement_passes = unsigned(unsigned_value);
    }
    if (metadata.refinement_passes < 1 || metadata.refinement_passes > 4)
        success = false;
    for (unsigned block_index = 0; block_index < 8; ++block_index) {
        const auto name = "block_repeat_" + std::to_string(block_index + 1);
        if (text.find(name + '=') != std::string::npos) {
            read_unsigned(name.c_str(), success, unsigned_value);
            metadata.block_repeats[block_index] = int(unsigned_value);
        }
        if (metadata.block_repeats[block_index] < 0 || metadata.block_repeats[block_index] > 4)
            success = false;
    }
    read_float("reset", success, floating_value);
    metadata.reset = int(floating_value);
    read_float("enabled", success, floating_value);
    metadata.enabled = int(floating_value);
    read_float("automask", success, floating_value);
    metadata.automask = int(floating_value);
    read_float("style", success, floating_value);
    metadata.style = int(floating_value);
    read_float("inverted", success, floating_value);
    metadata.inverted = int(floating_value);
    read_float("uicorrection", success, floating_value);
    metadata.uicorrection = int(floating_value);
    read_float("srgb", success, floating_value);
    metadata.srgb = int(floating_value);
    read_float("unjitter", success, floating_value);
    metadata.unjitter = int(floating_value);
    read_float("mode", success, floating_value);
    metadata.mode = int(floating_value);
    if (text.find("reference_feed=") != std::string::npos) {
        read_float("reference_feed", success, floating_value);
        metadata.reference_feed = int(floating_value);
        if (metadata.reference_feed != 0 && metadata.reference_feed != 1)
            success = false;
    }
    read_float("jitter_x", success, floating_value);
    metadata.jitter_x = float(floating_value);
    read_float("jitter_y", success, floating_value);
    metadata.jitter_y = float(floating_value);
    read_float("motion_x", success, floating_value);
    metadata.motion_x = float(floating_value);
    read_float("motion_y", success, floating_value);
    metadata.motion_y = float(floating_value);
    read_float("intensity", success, floating_value);
    metadata.intensity = float(floating_value);
    read_float("tone", success, floating_value);
    metadata.tone = float(floating_value);
    read_float("structure", success, floating_value);
    metadata.structure = float(floating_value);
    read_float("skin", success, floating_value);
    metadata.skin = float(floating_value);
    if (!success || !metadata.width || !metadata.height || !metadata.color_bpp ||
        !metadata.motion_bpp)
        return false;
    // Legacy parity/profiling probes must explicitly opt into the expanded
    // architecture; otherwise refuse rather than silently replay one pass.
    if (!allow_repetitions && !metadata.default_repetitions())
        return false;
    (void)pos;
    return true;
}
// Consecutive-frame exposure flicker detector (pure, CPU-tested). The GPU
// meter writes one float per recorded frame; the host feeds each retired
// value here. A relative step over kFlipStep counts a flip: steady scenes
// sit at zero flips, while a jitter-period pump (brightness breathing at the
// Halton cadence) accumulates one flip per oscillation. max_delta_pct keeps
// the worst step for the summary line.
struct ExposureStability {
    static constexpr float kFlipStep = 0.02f;
    float last = -1;
    unsigned flips = 0;
    float max_delta_pct = 0;
    // Returns true on a counted flip. Non-finite and non-positive samples
    // are dropped without touching the baseline: a lost readback must not
    // read as a scene cut.
    bool push(float exposure) {
        if (!(exposure > 0) || !(exposure < 1e7))
            return false;
        if (last <= 0) {
            last = exposure;
            return false;
        }
        const float delta = exposure > last ? exposure / last - 1 : last / exposure - 1;
        if (!(delta > 0)) {
            last = exposure;
            return false;
        }
        // Absurd spikes (unretired readback, degenerate frame) are dropped
        // without touching the baseline: a single lost sample must not read
        // as a scene cut, and true cuts re-anchor through history resets.
        if (delta > 50)
            return false;
        if (delta * 100 > max_delta_pct)
            max_delta_pct = delta * 100;
        last = exposure;
        if (delta <= kFlipStep)
            return false;
        ++flips;
        return true;
    }
    void reset() noexcept {
        last = -1;
        flips = 0;
        max_delta_pct = 0;
    }
};
// Pure wipe-side plan (CPU-tested). NVIDIA source 1 is remapped onto the two
// slots the output shader reads (0 = input snapshot, 2 = network result) and
// the NR image is staged into the slot the other side does not need. Slot 2
// reads the network result texture itself, never the wipe destination:
// staging into the destination is overwritten by the compose pass and would
// show DLSS5 Phoenix mislabeled as NVIDIA.
struct NrWipeStage {
    bool stage;
    bool to_input;
    unsigned effective_source_a, effective_source_b;
    unsigned resolve;
};
inline NrWipeStage nr_wipe_stage(int mode, int comparison, int source_a, int source_b,
                                 bool have_nvidia_output) {
    NrWipeStage plan{false, false, unsigned(source_a), unsigned(source_b), 0};
    if (!have_nvidia_output || mode != split || comparison == 0)
        return plan;
    if (source_a != 1 && source_b != 1)
        return plan;
    plan.stage = true;
    unsigned effective_source_a = plan.effective_source_a,
             effective_source_b = plan.effective_source_b;
    if (effective_source_a == 1 && effective_source_b == 1)
        effective_source_a = effective_source_b = 0;
    else {
        if (effective_source_a == 1)
            effective_source_a = (effective_source_b == 2) ? 0 : 2;
        if (effective_source_b == 1)
            effective_source_b = (effective_source_a == 2) ? 0 : 2;
    }
    plan.effective_source_a = effective_source_a;
    plan.effective_source_b = effective_source_b;
    plan.to_input = (source_a == 1 && source_b == 1) || (source_a == 1 && source_b == 2) ||
                    (source_b == 1 && int(effective_source_a) == 2);
    // NR occupies slot 0 (input snapshot) when staged there, else slot 2
    // (network result). Flag exactly the wipe sides reading that slot so the
    // resolve pass transfers the NR answer onto those sides.
    const unsigned slot = plan.to_input ? 0u : 2u;
    if (effective_source_a == slot)
        plan.resolve |= 1u;
    if (effective_source_b == slot)
        plan.resolve |= 2u;
    return plan;
}
// A marker is useful for externally controlled test runs. Normal gameplay must
// not probe the filesystem or depend on an extra file to start processing.
template <typename ReadyFilePresent>
inline bool recording_allowed(bool automatic, bool require_ready_file,
                              ReadyFilePresent&& file_present) {
    return automatic && (!require_ready_file || file_present());
}
inline const char* startup_status(bool configured, bool automatic, bool gate, bool direct,
                                  bool running, bool terminal, bool observe) {
    if (terminal)
        return "Stopped: see error below";
    if (!configured || !automatic)
        return "Not armed: Phoenix.json or launch configuration required";
    if (!gate)
        return "Paused: requested test readiness file absent";
    if (observe)
        return "Observation only";
    if (running)
        return direct ? "Running DLSS5 Phoenix directly (no NR provider)" : "Running";
    return direct ? "Waiting for supported game DLSS frame" : "Waiting for NR provider input";
}
// Meter only the fitted image. Letterbox pixels are storage padding, not
// dark scene content: including them biases a logarithmic average severely.
struct ExposureConstants {
    unsigned width, height, reset;
    unsigned jitter_x_bits, jitter_y_bits;
    unsigned left, top, image_width, image_height;
};
inline ExposureConstants exposure_constants(const Constants& common, const Resolution& geometry) {
    return {common.width,         common.height,        common.reset,
            common.jitter_x_bits, common.jitter_y_bits, geometry.left,
            geometry.top,         geometry.image_width, geometry.image_height};
}
inline constexpr char meter_shader[] = R"(

Texture2D<float4> scene : register(t0);
Texture2D<float> previous : register(t1);
RWTexture2D<float> exposure : register(u0);
cbuffer Params : register(b0) {
    uint width, height, reset;
    uint jitter_x_bits, jitter_y_bits;
    uint image_left, image_top, image_width, image_height;
};
groupshared float logs[256];
[numthreads(16, 16, 1)] void main(uint3 thread_id : SV_GroupThreadID,
                                  uint lane_index : SV_GroupIndex) {
    if (reset & 4) {
        if (lane_index == 0)
            exposure[uint2(0, 0)] = 1.0;
        return;
    }
    // 4096 taps (each thread averages its 4x4 cell) instead of 256: the
    // estimate is unbiased either way, but per-frame game jitter moves
    // content under sparse taps, so 256 taps let the exposure breathe at
    // the jitter period (fullscreen shimmer on HDR scenes, where the
    // normalization gain is far from 1). 16x taps quarter the flicker.
    float acc = 0;
    float2 jitter_pixels = float2(asfloat(jitter_x_bits), asfloat(jitter_y_bits));
    for (uint tap_y = 0; tap_y < 4; ++tap_y)
        for (uint tap_x = 0; tap_x < 4; ++tap_x) {
            // Same unjitter as the normalize pass: tap the reference grid so
            // the estimate does not wander with the jitter period.
            float2 image_origin = float2(image_left, image_top);
            float2 image_last = min(image_origin + float2(image_width, image_height),
                                    float2(width, height)) - 1;
            float2 sample_position = image_origin +
                (float2(thread_id.xy * 4u + uint2(tap_x, tap_y)) + .5) *
                    float2(image_width, image_height) / 64 + jitter_pixels;
            // Clamp every bilinear tap to the image, including jittered edge taps.
            sample_position = clamp(sample_position, image_origin, image_last);
            float2 texel_base = floor(sample_position);
            float2 fraction = sample_position - texel_base;
            int2 top_left = int2(texel_base);
            int2 bottom_right = int2(min(texel_base + 1.0, image_last));
            float l00 = dot(max(scene.Load(int3(top_left, 0)).rgb, 0), float3(.2126, .7152, .0722));
            float l10 = dot(max(scene.Load(int3(bottom_right.x, top_left.y, 0)).rgb, 0),
                            float3(.2126, .7152, .0722));
            float l01 = dot(max(scene.Load(int3(top_left.x, bottom_right.y, 0)).rgb, 0),
                            float3(.2126, .7152, .0722));
            float l11 =
                dot(max(scene.Load(int3(bottom_right, 0)).rgb, 0), float3(.2126, .7152, .0722));
            acc += log(max(lerp(lerp(l00, l10, fraction.x), lerp(l01, l11, fraction.x), fraction.y),
                           1e-6));
        }
    logs[lane_index] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint reduction_width = 128; reduction_width > 0; reduction_width >>= 1) {
        if (lane_index < reduction_width)
            logs[lane_index] += logs[lane_index + reduction_width];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane_index == 0) {
        float raw = .23 / exp(logs[0] / 4096);
        float old = previous.Load(int3(0, 0, 0));
        // Degenerate frames (all-black loads: raw is +inf) carry no
        // information: hold the previous estimate instead of flashing.
        if (!(raw > 0.0) || raw > 100.0)
            raw = (old > 0.0) ? old : 1.0;
        float filtered_exposure = raw;
        if (old > 0.0 && (reset & 1) == 0) {
            // Jitter-period pump guard: raw alternates with the tap weights, so
            // chasing it frame-to-frame breathes fullscreen brightness at the
            // Halton cadence. Crawl toward it (period-2 averages out) and snap
            // only on scene cuts or history resets.
            filtered_exposure =
                (raw > old * 1.3 || raw < old * 0.7) ? raw : old + 0.05 * (raw - old);
        }
        exposure[uint2(0, 0)] = clamp(filtered_exposure, 1e-6, 1e6);
    }
})";
inline constexpr char normalize_shader[] = R"(

Texture2D<float4> scene : register(t0);
Texture2D<float> exposure : register(t1);
RWTexture2D<float4> normalized : register(u0);
cbuffer Params : register(b0) {
    uint width, height, reset, mode;
    uint jitter_x_bits, jitter_y_bits;
    float divider, gain;
};
[numthreads(16, 16, 1)] void main(uint3 pixel : SV_DispatchThreadID) {
    if (pixel.x >= width || pixel.y >= height)
        return;
    // Unjitter: the engine shifts every frame subpixel (a/b carry the
    // jitter in pixels) while the network history rests on the reference
    // grid. Resampling the snapshot onto the reference grid keeps history
    // consistent without relying on warp precision; the presented frame
    // lives on that grid too, so no re-jitter is needed downstream.
    // Manual 4-tap bilinear: no sampler in this root signature.
    if (reset & 4) {
        normalized[pixel.xy] = float4(ReferenceEncode(scene.Load(int3(pixel.xy, 0)).rgb), 1);
        return;
    }
    float2 sample_position =
        float2(pixel.xy) + float2(asfloat(jitter_x_bits), asfloat(jitter_y_bits));
    float2 texel_base = max(floor(sample_position), 0.0);
    float2 fraction = sample_position - texel_base;
    int2 top_left = int2(min(texel_base, float2(width - 1, height - 1)));
    int2 bottom_right = int2(min(texel_base + 1.0, float2(width - 1, height - 1)));
    float3 s00 = scene.Load(int3(top_left, 0)).rgb;
    float3 s10 = scene.Load(int3(bottom_right.x, top_left.y, 0)).rgb;
    float3 s01 = scene.Load(int3(top_left.x, bottom_right.y, 0)).rgb;
    float3 s11 = scene.Load(int3(bottom_right, 0)).rgb;
    float3 raw = lerp(lerp(s00, s10, fraction.x), lerp(s01, s11, fraction.x), fraction.y);
    float3 exposure_color = raw * exposure.Load(int3(0, 0, 0));
    // Bound the network domain: exposure-normalized HDR highlights reach
    // ~100x, far outside any display-referred training distribution, and a
    // windowed-attention model answers with per-window mush (blocky
    // blotches around brights, muddy darks). Neutwo magnitude compression
    // keeps everything in distribution. The scale tap is the NEAREST
    // snapshot texel (not the unjittered bilinear value), so write_
    // recomputes bit-identical scales from the same snapshot: the pair is
    // exact even though the colors ride the reference grid.
    float3 nearest_exposure_color =
        scene.Load(int3(pixel.xy, 0)).rgb * exposure.Load(int3(0, 0, 0));
    float maximum_channel =
        max(max(nearest_exposure_color.r, nearest_exposure_color.g), nearest_exposure_color.b);
    float3 encoded_color = (maximum_channel > 1e-8)
                               ? exposure_color * rsqrt(maximum_channel * maximum_channel + 1.0)
                               : exposure_color;
    if (reset & 2)
        encoded_color =
            lerp(12.92 * encoded_color, 1.055 * pow(max(encoded_color, 0), 1.0 / 2.4) - .055,
                 step(.0031308, encoded_color));
    normalized[pixel.xy] = float4(encoded_color, 1);
})";
inline constexpr char output_shader[] = R"(

Texture2D<float4> scene : register(t0);
Texture2D<float4> neural : register(t1);
Texture2D<float> exposure : register(t2);
RWTexture2D<float4> destination : register(u0);
cbuffer Params : register(b0) {
    uint width, height, reset, mode;
    uint source_a, source_b;
    float divider, gain;
    uint resolve;
};
[numthreads(16, 16, 1)] void main(uint3 pixel : SV_DispatchThreadID) {
    if (pixel.x >= width || pixel.y >= height)
        return;
    // A native wipe write replaces only its selected side. Both networks
    // use the identical transfer below, including the metered DLSS5 Phoenix path.
    if (resolve & 8) {
        bool left = pixel.x < uint(width * divider);
        if ((left && (resolve & 1) == 0) || (!left && (resolve & 2) == 0))
            return;
    }
    float4 input = scene.Load(int3(pixel.xy, 0));
    float exposure_scale = max(exposure.Load(int3(0, 0, 0)), 1e-6);
    float3 neural_rgb = neural.Load(int3(pixel.xy, 0)).rgb;
    if (reset & 2)
        neural_rgb = lerp(neural_rgb / 12.92, pow(max((neural_rgb + .055) / 1.055, 0), 2.4),
                          step(.04045, neural_rgb));
    // Invert the normalize-side Neutwo compression with the same scale
    // recomputed from the snapshot (exact pair).
    float3 src = input.rgb * exposure_scale;
    float maximum_channel = max(max(src.r, src.g), src.b);
    float3 linear_rgb = (maximum_channel > 1e-8)
                            ? neural_rgb * sqrt(maximum_channel * maximum_channel + 1.0)
                            : neural_rgb;
    float4 result =
        float4((reset & 4) ? ReferenceDecode(input.rgb, neural.Load(int3(pixel.xy, 0)).rgb)
                           : linear_rgb / exposure_scale,
               input.a);
    if (mode != 0) {
        // Sides flagged for NR resolve are drawn provisionally here (DLSS5 Phoenix
        // transfer over the network output) and overwritten by the resolve
        // pass below, which transfers the genuine NR answer.
        float4 left_image = source_a == 0 ? input : result,
               right_image = source_b == 0 ? input : result;
        if (mode == 1)
            result = pixel.x < uint(width * divider) ? left_image : right_image;
        else {
            float3 difference = (left_image.rgb - right_image.rgb) * exposure_scale * gain;
            float3 visualization = saturate(abs(difference));
            if (mode == 3)
                visualization = saturate(.5 + .5 * difference);
            if (mode == 4) {
                float heat_value =
                    saturate(max(abs(difference.r), max(abs(difference.g), abs(difference.b))));
                visualization = float3(saturate(4 * heat_value - 2),
                                       saturate(4 * heat_value) - saturate(4 * heat_value - 3),
                                       saturate(4 * heat_value) - saturate(4 * heat_value - 1));
            }
            if (mode == 5)
                visualization = saturate(abs(dot(difference, float3(.2126, .7152, .0722)))).xxx;
            result = float4(visualization / exposure_scale, input.a);
        }
    }
    destination[pixel.xy] = result;
})";
inline constexpr const char* encode_shader = R"(
Texture2D<float4> scene : register(t0);
RWTexture2D<float4> destination : register(u0);
cbuffer Params : register(b0) {
    uint width, height, reset, mode;
    uint reserved_source_a, reserved_source_b;
    float divider, gain;
    uint resolve;
};
[numthreads(16, 16, 1)] void main(uint3 pixel : SV_DispatchThreadID) {
    if (pixel.x >= width || pixel.y >= height)
        return;
    // Fixed-white Neutwo proxy: hue-preserving [0,inf)->[0,1) roll-off, then
    // sRGB encode. The NR model is shown display-referred light, never raw
    // HDR. White metering is a documented follow-up; Neutwo degrades
    // gracefully under a fixed white where a knee would crush highlights.
    float3 scene_rgb = max(scene.Load(int3(pixel.xy, 0)).rgb, 0.0);
    float maximum_channel = max(scene_rgb.r, max(scene_rgb.g, scene_rgb.b));
    float3 prox =
        (maximum_channel > 1e-6)
            ? scene_rgb * ((maximum_channel * rsqrt(maximum_channel * maximum_channel + 1.0)) /
                           maximum_channel)
            : scene_rgb;
    float3 encoded_rgb = saturate(prox);
    float3 s = lerp(encoded_rgb * 12.92, 1.055 * pow(max(encoded_rgb, 1e-6), 1.0 / 2.4) - 0.055,
                    step(0.0031308, encoded_rgb));
    destination[pixel.xy] = float4(s, 1.0);
})";
inline constexpr const char* bridge_common = R"(
// Reversible BT.709 gamut and magnitude compression. Recompute the
// inverse scale from the same original pixel to preserve scene exposure.

float SanitizeFinite(float value, float fallback) {
    return ((asuint(value) & 0x7F800000u) == 0x7F800000u) ? fallback : value;
}
float3 SanitizeFinite3(float3 value, float3 fallback) {
    return float3(SanitizeFinite(value.x, fallback.x), SanitizeFinite(value.y, fallback.y),
                  SanitizeFinite(value.z, fallback.z));
}
float SafeDivide(float numerator, float denominator, float fallback) {
    return abs(denominator) > 1e-8 ? numerator / denominator : fallback;
}
float3 BT709ToLMS(float3 color) {
    const float3x3 transform_matrix = {0.30569589, 0.62271286, 0.04528636, 0.15776262, 0.76968599,
                                       0.08807030, 0.01933082, 0.11919478, 0.95053215};
    return mul(transform_matrix, color);
}
float3 LMSToBT709(float3 color) {
    const float3x3 transform_matrix = {5.62059812,  -4.57145756, 0.15577924,
                                       -1.15555585, 2.25800438,  -0.15415806,
                                       0.03059913,  -0.19018011, 1.06820532};
    return mul(transform_matrix, color);
}
float LuminanceBT709(float3 color) {
    return dot(color, float3(.21263901, .71516868, .07219232));
}
float3 D65NeutralBT709(float3 lms, float luminance) {
    float3 d65 = LMSToBT709(max(lms, 1e-8));
    return d65 * (luminance / max(LuminanceBT709(d65), 1e-8));
}
float GamutScaleBT709(float3 source_rgb, float3 lms) {
    float3 color = SanitizeFinite3(source_rgb, 0.0);
    float luminance = LuminanceBT709(color);
    float gamut_scale = 1.0;
    if (luminance > 1e-8) {
        float3 neutral_rgb = D65NeutralBT709(lms, luminance);
        if (color.r < 0.0 && neutral_rgb.r > color.r)
            gamut_scale = min(gamut_scale, SafeDivide(neutral_rgb.r, neutral_rgb.r - color.r, 1.0));
        if (color.g < 0.0 && neutral_rgb.g > color.g)
            gamut_scale = min(gamut_scale, SafeDivide(neutral_rgb.g, neutral_rgb.g - color.g, 1.0));
        if (color.b < 0.0 && neutral_rgb.b > color.b)
            gamut_scale = min(gamut_scale, SafeDivide(neutral_rgb.b, neutral_rgb.b - color.b, 1.0));
        gamut_scale = saturate(SanitizeFinite(gamut_scale, 1.0));
    }
    return gamut_scale;
}
float3 GamutCompressBT709(float3 source_rgb, float3 lms, float gamut_scale) {
    float3 color = SanitizeFinite3(source_rgb, 0.0);
    float luminance = LuminanceBT709(color);
    float3 outc = color;
    if (luminance > 1e-8) {
        float3 neutral_rgb = D65NeutralBT709(lms, luminance);
        outc = SanitizeFinite3(neutral_rgb + (color - neutral_rgb) * saturate(gamut_scale),
                               max(neutral_rgb, 0.0));
    }
    return outc;
}
float3 GamutDecompressBT709(float3 source_rgb, float3 lms, float gamut_scale) {
    float3 color = SanitizeFinite3(source_rgb, 0.0);
    float luminance = LuminanceBT709(color);
    float3 outc = color;
    if (luminance > 1e-8) {
        float3 neutral_rgb = D65NeutralBT709(lms, luminance);
        float safe_scale = max(gamut_scale, 1e-6);
        outc = SanitizeFinite3(neutral_rgb + (color - neutral_rgb) / safe_scale, color);
    }
    return outc;
}
float Neutwo(float magnitude) {
    return magnitude * rsqrt(magnitude * magnitude + 1.0);
}
float NeutwoScale(float3 color) {
    float maximum_channel = max(max(abs(color.r), abs(color.g)), abs(color.b));
    float outv = 1.0;
    if (maximum_channel > 1e-8)
        outv = SanitizeFinite(Neutwo(maximum_channel) / maximum_channel, 1.0);
    return outv;
}
float SrgbEncodeSafe(float value) {
    value = saturate(SanitizeFinite(value, 0.0));
    return value <= 0.0031308 ? value * 12.92 : 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}
float3 SrgbEncodeSafe3(float3 color) {
    return float3(SrgbEncodeSafe(color.r), SrgbEncodeSafe(color.g), SrgbEncodeSafe(color.b));
}
float SrgbDecodeSafe(float value) {
    value = saturate(SanitizeFinite(value, 0.0));
    return value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4);
}
float3 SrgbDecodeSafe3(float3 color) {
    return float3(SrgbDecodeSafe(color.r), SrgbDecodeSafe(color.g), SrgbDecodeSafe(color.b));
}
float3 ReferenceEncode(float3 original) {
    float3 src = SanitizeFinite3(original, 0.0);
    float3 lms = BT709ToLMS(float3(.18, .18, .18));
    float gamut_scale = GamutScaleBT709(src, lms);
    float3 comp = GamutCompressBT709(src, lms, gamut_scale);
    return SrgbEncodeSafe3(max(comp * NeutwoScale(comp), 0.0));
}
float3 ReferenceDecode(float3 original, float3 neural) {
    float3 lms = BT709ToLMS(float3(.18, .18, .18));
    float gamut_scale = GamutScaleBT709(original, lms);
    float magnitude_scale = NeutwoScale(GamutCompressBT709(original, lms, gamut_scale));
    float3 graded = SrgbDecodeSafe3(neural);
    return GamutDecompressBT709((abs(magnitude_scale) > 1e-8) ? graded / magnitude_scale : graded,
                                lms, gamut_scale);
}
)";
inline constexpr const char* encode_linear_shader = R"(
Texture2D<float4> scene : register(t0);
RWTexture2D<float4> destination : register(u0);
cbuffer Params : register(b0) {
    uint width, height, reset, mode;
    uint reserved_source_a, reserved_source_b;
    float divider, gain;
    uint resolve;
};
[numthreads(16, 16, 1)] void main(uint3 pixel : SV_DispatchThreadID) {
    if (pixel.x >= width || pixel.y >= height)
        return;
    // Reference-host HdrMode-1 encode (linear HDR BT.709, scene units):
    // gamut-compress, Neutwo magnitude scale, sRGB. There is deliberately
    // NO white divisor: the host may use arbitrary scene/pre-exposed units
    // and an absolute paper normalization breaks the DLL's input stats
    // (milky veil + shimmer). Opaque output, like the reference host.
    float3 scene_rgb = SanitizeFinite3(scene.Load(int3(pixel.xy, 0)).rgb, 0.0);
    float3 lms = BT709ToLMS(float3(0.18, 0.18, 0.18));
    float gamut_scale = GamutScaleBT709(scene_rgb, lms);
    float3 compressed_rgb = GamutCompressBT709(scene_rgb, lms, gamut_scale);
    float magnitude_scale = NeutwoScale(compressed_rgb);
    float3 neural_input_linear = max(compressed_rgb * magnitude_scale, 0.0);
    destination[pixel.xy] = float4(SrgbEncodeSafe3(neural_input_linear), 1.0);
})";
inline constexpr const char* resolve_shader = R"(
Texture2D<float4> original : register(t0);
Texture2D<float4> proxy : register(t1);
Texture2D<float4> model : register(t2);
RWTexture2D<float4> destination : register(u0);
cbuffer Params : register(b0) {
    uint width, height, reset, mode;
    uint reserved_source_a, reserved_source_b;
    float divider, gain;
    uint resolve;
};
float3 SrgbToLinear(float3 encoded_rgb) {
    return lerp(encoded_rgb / 12.92, pow(max((encoded_rgb + .055) / 1.055, 0.0), 2.4),
                step(.04045, encoded_rgb));
}
[numthreads(16, 16, 1)] void main(uint3 pixel : SV_DispatchThreadID) {
    if (pixel.x >= width || pixel.y >= height)
        return;
    // Fullscreen NVIDIA resolves everywhere; a two-way wipe resolves only the
    // flagged NR sides (side A left of the divider, side B right of it), so
    // the DLSS5 Phoenix side keeps the genuine network result.
    bool leftSide = pixel.x < uint(width * divider);
    bool side = (((resolve & 1) != 0) && leftSide) || (((resolve & 2) != 0) && !leftSide);
    bool selected_nvidia =
        (mode == 1) ? side
                    : (mode == 2 && (((resolve & 1) && leftSide) || ((resolve & 2) && !leftSide)));
    if (!selected_nvidia)
        return;
    float4 original_pixel = original.Load(int3(pixel.xy, 0));
    float4 proxy_pixel = proxy.Load(int3(pixel.xy, 0));
    // Resolve flag 0x4 selects the reversible gamut/Neutwo bridge. Both
    // proxy paths carry sRGB, with opaque alpha. The inverse recomputes its
    // scale from the original; no metadata travels through model alpha.
    bool reference_bridge = (resolve & 4) != 0;
    if (reference_bridge) {
        // Reference-host HdrMode-1 decode: the bridge state is recomputed
        // from the original (exactly like the encode side); the neural
        // verdict is sRGB-decoded, un-scaled and gamut-decompressed. No
        // original blend, no guard: the bridge is reversible by design and
        // the DLL's verdict stands on its own, like the reference host.
        float3 lms = BT709ToLMS(float3(0.18, 0.18, 0.18));
        float gamut_scale = GamutScaleBT709(original_pixel.rgb, lms);
        float magnitude_scale =
            NeutwoScale(GamutCompressBT709(original_pixel.rgb, lms, gamut_scale));
        float3 graded = SrgbDecodeSafe3(model.Load(int3(pixel.xy, 0)).rgb);
        float3 unscaled = (abs(magnitude_scale) > 1e-8) ? graded / magnitude_scale : graded;
        float3 recon = GamutDecompressBT709(unscaled, lms, gamut_scale);
        destination[pixel.xy] = float4(recon, original_pixel.a);
        return;
    }
    float3 proxy_linear = SrgbToLinear(proxy_pixel.rgb);
    // The DLL always runs its SDR pipeline (Hdr=0, like the reference
    // host's untouched defaults), so its output is sRGB-encoded: decode it.
    float3 neural_linear = SrgbToLinear(model.Load(int3(pixel.xy, 0)).rgb);
    float edit_strength = 1.0;
    // Classic transfer at fixed paper white 1.0: the normalized edit lands on
    // the linear original, held near the model's verdict by a two-sided
    // luminance guard. One scalar over the triple, so hue cannot bend.
    float3 edit = (neural_linear - proxy_linear) * edit_strength;
    float3 sum = max(original_pixel.rgb + edit, 0.0);
    float original_luminance = dot(original_pixel.rgb, float3(.2126, .7152, .0722)) + 1.0 / 512.0;
    float luminance_ratio = (dot(neural_linear, float3(.2126, .7152, .0722)) + 1.0 / 512.0) /
                            (dot(proxy_linear, float3(.2126, .7152, .0722)) + 1.0 / 512.0);
    float want = clamp(original_luminance * luminance_ratio,
                       max(original_luminance / 2.5 - 0.1, 0.0), original_luminance * 2.5 + 0.1);
    float result_luminance = dot(sum, float3(.2126, .7152, .0722));
    float3 final = (result_luminance > 1e-5 && original_luminance > 1e-5)
                       ? sum * (want / result_luminance)
                       : sum;
    destination[pixel.xy] = float4(final, original_pixel.a);
})";
inline const std::string& bridge_encode_source() {
    static const std::string shader_source = std::string(bridge_common) + encode_linear_shader;
    return shader_source;
}
inline const std::string& bridge_resolve_source() {
    static const std::string shader_source = std::string(bridge_common) + resolve_shader;
    return shader_source;
}
// Optional neural-only timing boundaries. Host transfer work stays in the
// enclosing route metric, so native and DLSS5 Phoenix network timings are comparable.
inline void (*network_timing_begin)() = nullptr;
inline void (*network_timing_end)() = nullptr;
class Resources final : public NVSDK_NGX_Parameter {
    ID3D12Resource* color_;
    ID3D12Resource* motion_;

  public:
    Resources(ID3D12Resource* color, ID3D12Resource* motion) : color_(color), motion_(motion) {}
#define PHOENIX_EMPTY(T)                                                                           \
    void Set(const char*, T) override {}                                                           \
    NVSDK_NGX_Result Get(const char*, T*) const override {                                         \
        return NVSDK_NGX_Result_Fail;                                                              \
    }
    PHOENIX_EMPTY(unsigned long long)
    PHOENIX_EMPTY(float)
    PHOENIX_EMPTY(double)
    PHOENIX_EMPTY(unsigned)
    PHOENIX_EMPTY(int)
    PHOENIX_EMPTY(ID3D11Resource*)
#undef PHOENIX_EMPTY
    void Set(const char*, ID3D12Resource*) override {}
    void Set(const char*, void*) override {}
    NVSDK_NGX_Result Get(const char* key, ID3D12Resource** out) const override {
        *out = !strcmp(key, "DLSSNR.Color")  ? color_
               : !strcmp(key, "DLSSNR.MVec") ? motion_
                                             : nullptr;
        return *out ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
    }
    NVSDK_NGX_Result Get(const char* key, void** out) const override {
        ID3D12Resource* value = nullptr;
        auto status = Get(key, &value);
        *out = value;
        return status;
    }
    void Reset() override {}
};
class Runner {
    // Destroyed last: callbacks from every owned resource can close this generation.
    phoenix_native::memory::SessionLifetime allocation_lifetime_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Fence> fence_;
    ComPtr<ID3D12Resource> input_, motion_, normalized_, network_input_, exposure_[2], output_;
    // Flicker forensics: a 4-byte readback of the metered exposure, copied
    // per recorded frame and retired through the submission fence. A single
    // in-flight sample is enough (one pending frame at a time); a sample
    // overwritten before retirement is dropped, never faked.
    ComPtr<ID3D12Resource> exposure_read_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT exposure_fp_{};
    uint64_t exposure_fence_ = 0;
    bool exposure_armed_ = false;
    ExposureStability exposure_stability_;
    // Offline input-capture burst: fence-gated readbacks of the owned
    // snapshots plus the presented output. One frame in flight; the host
    // drains each retired frame to disk and re-arms by countdown.
    ComPtr<ID3D12Resource> cap_color_, cap_motion_, cap_output_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT cap_color_fp_{}, cap_motion_fp_{};
    unsigned cap_color_bpp_ = 0, cap_motion_bpp_ = 0;
    std::string cap_base_;
    unsigned cap_left_ = 0, cap_index_ = 0;
    uint64_t cap_fence_ = 0;
    bool cap_armed_ = false, cap_ready_ = false;
    CaptureFrame cap_staged_;
    std::array<ComPtr<ID3D12Resource>, 4> cap_lineage_;
    ComPtr<ID3D12Resource> cap_exposure_;
    std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT, 4> cap_lineage_fp_{};
    bool cap_has_lineage_ = false;
    float host_jx_ = 0, host_jy_ = 0;
    ComPtr<ID3D12Resource> engine_output_, engine_motion_, engine_depth_;
    // NR proxy (display-referred sRGB, the model's native domain) and the
    // fullscreen-resolve scratch surface. The proxy rests readable; NR reads
    // it without transitions, exactly like the owned model inputs.
    ComPtr<ID3D12Resource> proxy_, resolve_target_;
    // HDR bridge proxy (fp16 sRGB-encoded light, opaque alpha).
    // Created alongside the LDR proxy; selected per frame, never recreated.
    ComPtr<ID3D12Resource> proxy16_;
    // Self-hosted native NR image (UAV state between evaluates). Null when the
    // NR feature is unavailable; NVIDIA sources are refused in that case.
    ID3D12Resource* nr_ = nullptr;
    struct Lease {
        uint64_t fence;
        ComPtr<ID3D12Resource> output, motion, depth;
    };
    std::vector<Lease> leases_;
    std::unique_ptr<FrameDispatch> dispatch_;
    std::unique_ptr<NeuralExecutor> external_;
    std::unique_ptr<GuideAdapter> guides_;
    std::string execution_device_ = "native", worker_engine_ = "auto";
    int offload_age_frames_ = 4, offload_age_ms_ = 100, cpu_threads_ = 0;
    unsigned passes_ = 1;
    std::unique_ptr<phoenix_native::Network> network_;
    std::vector<std::unique_ptr<phoenix_native::Network>> refinement_;
    std::array<int, 8> block_repeats_{{1, 1, 1, 1, 1, 1, 1, 1}};
    Pass meter_, normalize_, write_, encode_, resolve_, encode_lin_;
    // Independent immutable descriptor pairs for native pre/post processing.
    // Reusing the DLSS5 Phoenix pass heaps would overwrite descriptors in the same list.
    Pass native_meter_, native_normalize_, native_write_, native_fullscreen_;
    ComPtr<ID3D12Resource> native_exposure_[2];
    uint64_t native_prepared_ = 0;
    unsigned native_slot_ = 0;
    Constants native_constants_{};
    bool shared_native_ = false;
    // Linear bridge select + white divisor (white_nits/100), set per frame
    // from the panel before encode; resolve_fullscreen reads the same flag.
    bool host_bridge_ = false;
    bool bridge_announced_ = false;
    float host_white_div_ = 5.0f;
    unsigned width_, height_;
    Resolution geometry_;
    std::unique_ptr<ResolutionBridge> resolution_bridge_;
    std::unique_ptr<TemporalUpscaler> upscaler_;
    ComPtr<ID3D12Resource> upscale_depth_;
    Controls upscale_controls_;
    int upscaler_method_ = 0;
    bool bridge_only_ = false;
    int resolution_mode_ = 0, resolution_quality_ = 0, resolution_aspect_ = 0;
    int resolution_ratio_width_ = 16, resolution_ratio_height_ = 9;
    DXGI_FORMAT format_, motion_format_;
    uint64_t submitted_ = 0;
    ID3D12GraphicsCommandList* pending_ = nullptr;
    // Claimed game list with backend-1 work recorded but no network pending.
    // Tracked only so submit-time breadcrumbs fire; no fence semantics.
    ID3D12GraphicsCommandList* claimed_ = nullptr;
    bool failed_ = false, initialized_ = false, srgb_ = false;
    bool network_recorded_ = false;

    ComPtr<ID3D12Resource> tracked_image(ID3D12Device* device, unsigned width, unsigned height,
                                         DXGI_FORMAT format, D3D12_RESOURCE_STATES state,
                                         bool writable = true) {
        auto resource = phoenix_native::image(device, width, height, format, state, writable);
        phoenix_native::memory::resource(device, resource.Get(), allocation_lifetime_.generation,
                                         width == 1 && height == 1
                                             ? phoenix_native::memory::Kind::exposure
                                             : phoenix_native::memory::Kind::bridge);
        return resource;
    }
    ComPtr<ID3D12Resource> tracked_buffer(ID3D12Device* device, uint64_t bytes,
                                          D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state) {
        auto resource = phoenix_native::buffer(device, bytes, heap, state);
        phoenix_native::memory::resource(device, resource.Get(), allocation_lifetime_.generation,
                                         phoenix_native::memory::Kind::capture);
        return resource;
    }
    void prepare_capture_readbacks() {
        if (cap_color_)
            return;
        auto bpp = [](DXGI_FORMAT format) -> unsigned {
            switch (format) {
            case DXGI_FORMAT_R16G16B16A16_FLOAT:
                return 8;
            case DXGI_FORMAT_R32G32B32A32_FLOAT:
                return 16;
            case DXGI_FORMAT_R8G8B8A8_UNORM:
            case DXGI_FORMAT_R11G11B10_FLOAT:
                return 4;
            case DXGI_FORMAT_R16G16_FLOAT:
                return 4;
            case DXGI_FORMAT_R32G32_FLOAT:
                return 8;
            default:
                return 0;
            }
        };
        cap_color_bpp_ = bpp(format_);
        cap_motion_bpp_ = bpp(motion_format_);
        check(cap_color_bpp_ && cap_motion_bpp_, "Direct capture bytes-per-pixel unknown");
        auto foot = [&](DXGI_FORMAT format, D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint) {
            D3D12_RESOURCE_DESC description{};
            description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            description.Width = width_;
            description.Height = height_;
            description.DepthOrArraySize = 1;
            description.MipLevels = 1;
            description.Format = format;
            description.SampleDesc.Count = 1;
            description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            UINT64 total = 0;
            device_->GetCopyableFootprints(&description, 0, 1, 0, &footprint, nullptr, nullptr,
                                           &total);
            return total;
        };
        auto color_readback =
            tracked_buffer(device_.Get(), foot(format_, cap_color_fp_), D3D12_HEAP_TYPE_READBACK,
                           D3D12_RESOURCE_STATE_COPY_DEST);
        auto output_readback =
            tracked_buffer(device_.Get(), foot(format_, cap_color_fp_), D3D12_HEAP_TYPE_READBACK,
                           D3D12_RESOURCE_STATE_COPY_DEST);
        auto motion_readback =
            tracked_buffer(device_.Get(), foot(motion_format_, cap_motion_fp_),
                           D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        cap_color_ = std::move(color_readback);
        cap_output_ = std::move(output_readback);
        cap_motion_ = std::move(motion_readback);
    }
    // Called only after the capture submission fence has retired. CPU snapshots
    // are independent copies, so GPU readbacks can be freed before consumption.
    void release_capture_readbacks() {
        if (cap_left_ || cap_armed_)
            return;
        cap_color_.Reset();
        cap_motion_.Reset();
        cap_output_.Reset();
        for (auto& readback : cap_lineage_)
            readback.Reset();
        cap_exposure_.Reset();
        cap_has_lineage_ = false;
    }

  public:
    Runner(ID3D12Device* device, ID3D12CommandQueue* queue, const Json& plan,
           const std::vector<uint8_t>& weights, const std::vector<phoenix_kernel::Binary>& kernels,
           DXGI_FORMAT format, DXGI_FORMAT motion_format, bool srgb = false, unsigned passes = 1,
           std::array<int, 8> repeats = {{1, 1, 1, 1, 1, 1, 1, 1}},
           std::unique_ptr<NeuralExecutor> executor = {},
           const Settings* execution_settings = nullptr,
           std::shared_ptr<phoenix_native::memory::Generation> accounting = {},
           bool allow_gpu_upload = true, unsigned output_width = 0, unsigned output_height = 0,
           std::filesystem::path upscaler_directory = {})
        : allocation_lifetime_{accounting}, device_(device), queue_(queue),
          external_(std::move(executor)), meter_(device, meter_shader, accounting),
          normalize_(device, (std::string(bridge_common) + normalize_shader).c_str(), accounting),
          write_(device, (std::string(bridge_common) + output_shader).c_str(), accounting),
          encode_(device, encode_shader, accounting),
          resolve_(device, bridge_resolve_source().c_str(), accounting),
          encode_lin_(device, bridge_encode_source().c_str(), accounting),
          native_meter_(device, meter_shader, accounting),
          native_normalize_(device, (std::string(bridge_common) + normalize_shader).c_str(),
                            accounting),
          native_write_(device, (std::string(bridge_common) + output_shader).c_str(), accounting),
          native_fullscreen_(device, (std::string(bridge_common) + output_shader).c_str(),
                             accounting),
          width_(plan.at("width")), height_(plan.at("height")), format_(format),
          motion_format_(motion_format), srgb_(srgb) {
        bridge_only_ = execution_settings && execution_settings->mode == nvidia;
        geometry_ = fit_resolution(output_width ? output_width : width_,
                                   output_height ? output_height : height_, width_, height_);
        if (execution_settings) {
            upscaler_method_ = execution_settings->upscaler;
            resolution_mode_ = execution_settings->resolution_mode;
            resolution_quality_ = execution_settings->resolution_quality;
            resolution_aspect_ = execution_settings->resolution_aspect;
            resolution_ratio_width_ = execution_settings->resolution_ratio_width;
            resolution_ratio_height_ = execution_settings->resolution_ratio_height;
        }
        if (geometry_.scaled())
            resolution_bridge_ =
                std::make_unique<ResolutionBridge>(device, geometry_, format, accounting);
        if (geometry_.scaled() && upscaler_method_)
            upscaler_ = std::make_unique<TemporalUpscaler>(
                device, geometry_, upscaler_method_, accounting, std::move(upscaler_directory));
        check((external_ || bridge_only_
                   ? (width_ > 0 && height_ > 0 && width_ <= 8192 && height_ <= 8192)
                   : extent_supported(width_, height_)) &&
                  source_format(format) && direct::motion_format(motion_format),
              "Direct input geometry/format unsupported");
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
        phoenix_native::hr(
            device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)));
        check((support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != 0,
              "Direct output format lacks typed stores");
        input_ = tracked_image(device, width_, height_, format,
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, geometry_.scaled());
        if (!bridge_only_ || geometry_.scaled())
            motion_ =
                tracked_image(device, width_, height_, motion_format,
                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, geometry_.scaled());
        // Encode writes the proxy through a UAV (it rests readable between
        // frames), so the UAV flag is required despite the SRV rest state.
        if (!bridge_only_)
            proxy_ = tracked_image(device, width_, height_, DXGI_FORMAT_R8G8B8A8_UNORM,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
        proxy16_ = tracked_image(device, width_, height_, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
        // Scratch resolve target: UAV-written, so it needs the UAV flag even
        // though nothing else writes it.
        resolve_target_ = tracked_image(device, width_, height_, format,
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
        if (!bridge_only_) {
            normalized_ = tracked_image(device, width_, height_, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
            network_input_ = tracked_image(device, width_, height_, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, false);
            output_ = tracked_image(device, width_, height_, format,
                                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true);
            for (auto& exposure_resource : exposure_)
                exposure_resource =
                    tracked_image(device, 1, 1, DXGI_FORMAT_R32_FLOAT,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
        }
        for (auto& exposure_resource : native_exposure_)
            exposure_resource = tracked_image(device, 1, 1, DXGI_FORMAT_R32_FLOAT,
                                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
        if (!bridge_only_) {
            D3D12_RESOURCE_DESC exposure_description{};
            exposure_description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            exposure_description.Width = 1;
            exposure_description.Height = 1;
            exposure_description.DepthOrArraySize = 1;
            exposure_description.MipLevels = 1;
            exposure_description.Format = DXGI_FORMAT_R32_FLOAT;
            exposure_description.SampleDesc.Count = 1;
            exposure_description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            UINT64 total = 0;
            device->GetCopyableFootprints(&exposure_description, 0, 1, 0, &exposure_fp_, nullptr,
                                          nullptr, &total);
            exposure_read_ = tracked_buffer(device, total, D3D12_HEAP_TYPE_READBACK,
                                            D3D12_RESOURCE_STATE_COPY_DEST);
        }
        phoenix_native::hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)));
        phoenix_native::memory::track(fence_.Get(), accounting,
                                      phoenix_native::memory::Kind::fence);
        check(passes >= 1 && passes <= 4, "Direct refinement count outside 1..4");
        passes_ = passes;
        block_repeats_ = repeats;
        leases_.reserve(16);
        if (bridge_only_) {
            if (geometry_.scaled())
                guides_ =
                    std::make_unique<GuideAdapter>(device, width_, height_, accounting, geometry_);
            if (accounting)
                accounting->constructed = true;
            return;
        }
        if (external_) {
            external_->account(accounting);
            check(execution_settings && execution_settings->execution_device != "native",
                  "External executor needs an explicit device identity");
            execution_device_ = execution_settings->execution_device;
            worker_engine_ = execution_settings->worker_engine;
            offload_age_frames_ = execution_settings->offload_age_frames;
            offload_age_ms_ = execution_settings->offload_age_ms;
            cpu_threads_ = execution_settings->cpu_threads;
            guides_ =
                std::make_unique<GuideAdapter>(device, width_, height_, accounting, geometry_);
            if (accounting)
                accounting->constructed = true;
            return;
        }
        dispatch_ = std::make_unique<FrameDispatch>();
        validate_phoenix_depth_plan(plan);
        check(passes >= 1 && passes <= 4, "Direct refinement count outside 1..4");
        block_repeats_ = repeats;
        dispatch_->configure_repeats(device, plan, repeats);
        leases_.reserve(16);
        network_ = std::make_unique<phoenix_native::Network>(
            device, dispatch_->api(), plan, weights, kernels, network_input_.Get(), motion_.Get(),
            accounting, allow_gpu_upload);
        // Each pass owns its previous-frame history and immutable descriptor
        // tables. The shared input snapshot is updated in queue order only.
        for (unsigned iteration_index = 1; iteration_index < passes; ++iteration_index)
            refinement_.push_back(std::make_unique<phoenix_native::Network>(
                device, dispatch_->api(), plan, weights, kernels, network_input_.Get(),
                motion_.Get(), accounting, allow_gpu_upload, network_.get()));
        if (accounting) {
            accounting->constructed = true;
            accounting->host_container_capacity_bytes = network_->host_container_capacity_bytes();
            for (const auto& pass : refinement_)
                accounting->host_container_capacity_bytes += pass->host_container_capacity_bytes();
        }
    }
    unsigned passes() const {
        return passes_;
    }
    bool external() const {
        return bool(external_);
    }
    Json execution_status() const {
        auto result = external_ ? external_->status() : Json{{"kind", "native"}};
        result["metered_exposure_scale"] = exposure_stability_.last;
        if (resolution_bridge_)
            result["resolution_bridge"] = resolution_bridge_->status();
        return result;
    }
    bool depth_matches(ID3D12Resource* depth) const {
        return !guides_ || guides_->matches(depth);
    }
    bool architecture_matches(const Settings& settings) const {
        if (bridge_only_ != (settings.mode == nvidia) || upscaler_method_ != settings.upscaler ||
            resolution_mode_ != settings.resolution_mode ||
            resolution_quality_ != settings.resolution_quality ||
            resolution_aspect_ != settings.resolution_aspect ||
            resolution_ratio_width_ != settings.resolution_ratio_width ||
            resolution_ratio_height_ != settings.resolution_ratio_height)
            return false;
        if (bridge_only_)
            return true;
        return passes() == unsigned(settings.refinement_passes) &&
               block_repeats_ == settings.block_repeats &&
               execution_device_ == settings.execution_device &&
               (!external_ || (worker_engine_ == settings.worker_engine &&
                               offload_age_frames_ == settings.offload_age_frames &&
                               offload_age_ms_ == settings.offload_age_ms &&
                               ((execution_device_ != "cpu" && execution_device_ != "sycl:cpu") ||
                                cpu_threads_ == settings.cpu_threads)));
    }
    bool ready() const {
        auto done = fence_->GetCompletedValue();
        return !failed_ && !pending_ && done != UINT64_MAX && submitted_ - done < 8;
    }
    bool idle() const {
        return ready() && completed() == submitted_;
    }
    bool matches(ID3D12CommandQueue* queue, const D3D12_RESOURCE_DESC& color,
                 const D3D12_RESOURCE_DESC& motion_description) const {
        return queue == queue_.Get() && color.Width == geometry_.output_width &&
               color.Height == geometry_.output_height && color.Format == format_ &&
               motion_description.Width == geometry_.output_width &&
               motion_description.Height == geometry_.output_height &&
               motion_description.Format == motion_format_;
    }
    unsigned inference_width() const {
        return width_;
    }
    unsigned inference_height() const {
        return height_;
    }
    const Resolution& resolution() const {
        return geometry_;
    }
    void copy_input(ID3D12GraphicsCommandList* cmd, ID3D12Resource* source,
                    D3D12_RESOURCE_STATES state, ID3D12Resource* destination, bool motion = false) {
        if (resolution_bridge_ && (source->GetDesc().Width != destination->GetDesc().Width ||
                                   source->GetDesc().Height != destination->GetDesc().Height)) {
            resolution_bridge_->begin_frame(completed(), submitted_ + 1);
            resolution_bridge_->reduce(cmd, source, state, destination, motion);
            return;
        }
        phoenix_native::transition(cmd, source, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        phoenix_native::transition(cmd, destination, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(destination, source);
        phoenix_native::transition(cmd, destination, D3D12_RESOURCE_STATE_COPY_DEST,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(cmd, source, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
    }
    ID3D12Resource* prepare_native_motion(ID3D12GraphicsCommandList* cmd, ID3D12Resource* motion,
                                          D3D12_RESOURCE_STATES state) {
        if (!resolution_bridge_)
            return motion;
        engine_motion_ = motion;
        copy_input(cmd, motion, state, motion_.Get(), true);
        return motion_.Get();
    }
    ID3D12Resource* prepare_native_depth(ID3D12GraphicsCommandList* cmd, ID3D12Resource* depth,
                                         D3D12_RESOURCE_STATES state) {
        if (!resolution_bridge_)
            return depth;
        if (!guides_)
            guides_ = std::make_unique<GuideAdapter>(device_.Get(), width_, height_,
                                                     allocation_lifetime_.generation, geometry_);
        return guides_->native_depth(cmd, motion_.Get(), depth, state);
    }
    void prepare_upscale(ID3D12GraphicsCommandList* cmd, const Controls& controls,
                         ID3D12Resource* depth, D3D12_RESOURCE_STATES state) {
        upscale_controls_ = controls;
        if (upscaler_) {
            engine_depth_ = depth;
            upscale_depth_ = depth ? prepare_native_depth(cmd, depth, state) : nullptr;
        }
    }
    std::string upscale_status() const {
        return upscaler_ ? upscaler_->status()
                         : (resolution_bridge_ ? "Spatial upscaling active" : "Normal resolution");
    }
    void present(ID3D12GraphicsCommandList* cmd, ID3D12Resource* low, ID3D12Resource* game,
                 D3D12_RESOURCE_STATES state) {
        if (resolution_bridge_) {
            ID3D12Resource* reconstructed = nullptr;
            if (upscaler_) {
                phoenix_native::transition(cmd, low, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                try {
                    reconstructed = upscaler_->record(
                        cmd, low, motion_.Get(), upscale_depth_.Get(), upscale_controls_.reset != 0,
                        upscale_controls_.inverted != 0, upscale_controls_.motion_x,
                        upscale_controls_.motion_y, host_jx_, host_jy_);
                } catch (...) {
                    phoenix_native::transition(cmd, low,
                                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                               D3D12_RESOURCE_STATE_COPY_SOURCE);
                    throw;
                }
                phoenix_native::transition(cmd, low, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
            }
            if (reconstructed)
                resolution_bridge_->present_full(cmd, reconstructed, game, state);
            else
                resolution_bridge_->present(cmd, low, game, state);
            return;
        }
        phoenix_native::transition(cmd, game, state, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(game, low);
        phoenix_native::transition(cmd, game, D3D12_RESOURCE_STATE_COPY_DEST, state);
    }
    void set_nr(ID3D12Resource* nvidia_output) {
        nr_ = nvidia_output;
    }
    bool nr_armed() const {
        return nr_ != nullptr;
    }
    ID3D12Resource* proxy_image() {
        return (shared_native_ || host_bridge_) ? proxy16_.Get() : proxy_.Get();
    }
    void set_host_bridge(bool bridge, float white_div) {
        if (bridge != host_bridge_) {
            host_bridge_ = bridge;
            bridge_announced_ = false;
        }
        host_white_div_ = white_div;
    }
    // Display-referred proxy encode (fixed-white Neutwo + sRGB) from the raw
    // engine color. Bracketed like the self-drive evaluate: the color returns
    // to its tracked state, the proxy rests readable for NR/resolve.
    // Preserve the original DLSS5 Phoenix feed exactly and offer that same feed
    // to the real DLL. Exposure histories reset together on route changes.
    void encode_shared_proxy(ID3D12GraphicsCommandList* cmd, ID3D12Resource* color,
                             D3D12_RESOURCE_STATES color_state, const Controls& controls,
                             const Settings& settings) {
        check(cmd && color, "Shared native feed contract missing");
        note_claimed(cmd, color);
        shared_native_ = true;
        copy_input(cmd, color, color_state, input_.Get());
        native_slot_ = unsigned(native_prepared_ % 2);
        auto* exposure_texture = native_exposure_[native_slot_].Get();
        auto* old = native_exposure_[1 - native_slot_].Get();
        native_constants_ = {width_,
                             height_,
                             unsigned(controls.reset || !native_prepared_) | (srgb_ ? 2u : 0u) |
                                 (settings.host_bridge ? 4u : 0u),
                             0,
                             0,
                             0,
                             settings.divider,
                             1,
                             0};
        auto native_constants = native_constants_;
        const float jitter_x = settings.host_unjitter ? host_jx_ : 0.f,
                    jitter_y = settings.host_unjitter ? host_jy_ : 0.f;
        memcpy(&native_constants.jitter_x_bits, &jitter_x, 4);
        memcpy(&native_constants.jitter_y_bits, &jitter_y, 4);
        phoenix_native::transition(cmd, exposure_texture,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        native_meter_.record(cmd, {input_.Get(), old, nullptr}, exposure_texture,
                             exposure_constants(native_constants, geometry_), true, native_slot_);
        phoenix_native::transition(cmd, exposure_texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(cmd, proxy16_.Get(),
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        native_normalize_.record(cmd, {input_.Get(), exposure_texture, nullptr}, proxy16_.Get(),
                                 native_constants, false, native_slot_);
        phoenix_native::transition(cmd, proxy16_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ++native_prepared_;
    }
    void write_shared_native(ID3D12GraphicsCommandList* cmd, ID3D12Resource* nvidia_output,
                             ID3D12Resource* output, unsigned sides = 0) {
        check(shared_native_ && native_prepared_, "Native resolve without shared feed");
        auto constants = native_constants_;
        constants.resolve = sides ? (sides | 8u) : 0u;
        phoenix_native::transition(cmd, nvidia_output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        (sides ? native_write_ : native_fullscreen_)
            .record(cmd, {input_.Get(), nvidia_output, native_exposure_[native_slot_].Get()},
                    output, constants, false, native_slot_);
        phoenix_native::transition(cmd, nvidia_output,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    // Fullscreen NVIDIA: resolve the NR answer onto the engine color through
    // the owned scratch surface (the engine texture may not allow UAV stores).
    void resolve_fullscreen(ID3D12GraphicsCommandList* cmd, ID3D12Resource* color,
                            D3D12_RESOURCE_STATES color_state, ID3D12Resource* proxy,
                            ID3D12Resource* nvidia_output) {
        check(cmd && color && proxy && nvidia_output, "Direct fullscreen resolve contract missing");
        crumb("resolve_fullscreen: enter");
        if (shared_native_) {
            write_shared_native(cmd, nvidia_output, resolve_target_.Get());
        } else {
            Constants constants{
                width_, height_, 0, 1, 0, 0, 0.5f, 1.0f, unsigned(3 | (host_bridge_ ? 4 : 0))};
            phoenix_native::transition(cmd, nvidia_output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resolve_.record(cmd, {input_.Get(), proxy, nvidia_output}, resolve_target_.Get(),
                            constants, false, 1);
            phoenix_native::transition(cmd, nvidia_output,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        phoenix_native::transition(cmd, resolve_target_.Get(),
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        present(cmd, resolve_target_.Get(), color, color_state);
        phoenix_native::transition(cmd, resolve_target_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        crumb("resolve_fullscreen: exit");
    }
    uint64_t submitted() const {
        return submitted_;
    }
    uint64_t completed() const {
        auto completed_fence = fence_->GetCompletedValue();
        return completed_fence == UINT64_MAX ? 0 : std::min(completed_fence, submitted_);
    }
    // Flicker forensics: retire the metered exposure once its fence passes.
    // Called once per direct frame; a sample overwritten before retirement
    // is dropped, never faked.
    void poll_exposure() {
        if (!exposure_armed_)
            return;
        const auto done = fence_->GetCompletedValue();
        if (done == UINT64_MAX || done < exposure_fence_)
            return;
        exposure_armed_ = false;
        void* mapped = nullptr;
        if (FAILED(exposure_read_->Map(0, nullptr, &mapped)) || !mapped)
            return;
        float value = 0;
        std::memcpy(&value, static_cast<uint8_t*>(mapped) + exposure_fp_.Offset, 4);
        exposure_read_->Unmap(0, nullptr);
        exposure_stability_.push(value);
    }
    unsigned exposure_flips() const {
        return exposure_stability_.flips;
    }
    float exposure_max_delta_pct() const {
        return exposure_stability_.max_delta_pct;
    }
    float exposure_last_value() const {
        return exposure_stability_.last;
    }
    // Offline input-capture burst. arm_capture starts a countdown; record()
    // copies the owned snapshots and the presented output into readback
    // heaps; poll_captures retires each frame through the submission fence
    // and tight-packs it; take_capture moves it out for the host to write.
    void arm_capture(const std::string& base, unsigned frames) {
        if (!frames || cap_left_ || cap_armed_ || cap_ready_)
            return;
        prepare_capture_readbacks();
        cap_base_ = base;
        cap_left_ = frames;
        cap_index_ = 0;
    }
    bool capture_armed() const {
        return cap_left_ > 0;
    }
    unsigned capture_left() const {
        return cap_left_;
    }
    const std::string& capture_base() const {
        return cap_base_;
    }
    void capture_lineage(ID3D12GraphicsCommandList* cmd, unsigned index, ID3D12Resource* tex,
                         D3D12_RESOURCE_STATES state) {
        if (!cap_left_ || cap_armed_ || cap_ready_ || !cap_has_lineage_)
            return;
        const auto desc = tex->GetDesc();
        check(desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT && desc.Width == width_ &&
                  desc.Height == height_,
              "Lineage capture requires matching RGBA16F");
        if (!cap_lineage_[index]) {
            UINT64 total = 0;
            device_->GetCopyableFootprints(&desc, 0, 1, 0, &cap_lineage_fp_[index], nullptr,
                                           nullptr, &total);
            cap_lineage_[index] = tracked_buffer(device_.Get(), total, D3D12_HEAP_TYPE_READBACK,
                                                 D3D12_RESOURCE_STATE_COPY_DEST);
        }
        phoenix_native::transition(cmd, tex, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = cap_lineage_[index].Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = cap_lineage_fp_[index];
        src.pResource = tex;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        phoenix_native::transition(cmd, tex, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
    }
    void capture_inputs(ID3D12GraphicsCommandList* cmd, const Controls& controls,
                        const Settings& settings, unsigned blocks) {
        if (!cap_left_ || cap_armed_ || cap_ready_)
            return;
        auto region = [&](ID3D12Resource* tex, ID3D12Resource* read,
                          const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint) {
            phoenix_native::transition(cmd, tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
            dst.pResource = read;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = footprint;
            src.pResource = tex;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            phoenix_native::transition(cmd, tex, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        };
        region(input_.Get(), cap_color_.Get(), cap_color_fp_);
        region(motion_.Get(), cap_motion_.Get(), cap_motion_fp_);
        CaptureMeta metadata;
        metadata.width = width_;
        metadata.height = height_;
        metadata.color_format = unsigned(format_);
        metadata.motion_format = unsigned(motion_format_);
        metadata.color_bpp = cap_color_bpp_;
        metadata.motion_bpp = cap_motion_bpp_;
        metadata.frame = cap_index_;
        metadata.blocks = blocks;
        metadata.refinement_passes = unsigned(settings.refinement_passes);
        metadata.block_repeats = settings.block_repeats;
        metadata.reset = controls.reset;
        metadata.enabled = controls.enabled;
        metadata.automask = controls.auto_mask;
        metadata.style = controls.style;
        metadata.inverted = controls.inverted;
        metadata.uicorrection = controls.ui_correction;
        metadata.srgb = srgb_ ? 1 : 0;
        metadata.unjitter = settings.host_unjitter ? 1 : 0;
        metadata.mode = settings.mode;
        metadata.reference_feed = settings.host_bridge ? 1 : 0;
        metadata.jitter_x = host_jx_;
        metadata.jitter_y = host_jy_;
        metadata.motion_x = controls.motion_x;
        metadata.motion_y = controls.motion_y;
        metadata.intensity = controls.intensity;
        metadata.tone = controls.tone;
        metadata.structure = controls.structure;
        metadata.skin = controls.skin;
        cap_staged_.meta = metadata;
        cap_staged_.source_a = settings.source_a;
        cap_staged_.source_b = settings.source_b;
        cap_staged_.divider = settings.divider;
        cap_has_lineage_ =
            !external_ && nr_ && settings.mode == split && (shared_native_ || settings.host_bridge);
        if (cap_has_lineage_) {
            capture_lineage(cmd, 0, proxy_image(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            capture_lineage(cmd, 1, nr_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }
    void capture_output(ID3D12GraphicsCommandList* cmd, ID3D12Resource* color,
                        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST) {
        if (!cap_left_ || cap_armed_ || cap_ready_)
            return;
        phoenix_native::transition(cmd, color, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = cap_output_.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = cap_color_fp_;
        src.pResource = color;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        phoenix_native::transition(cmd, color, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        cap_fence_ = submitted_ + 1;
        cap_armed_ = true;
        --cap_left_;
        ++cap_index_;
    }
    // After the capture fence completes, remove row-pitch padding from each
    // readback into tightly packed CPU snapshots. Preserve the frame metadata
    // so replay uses the same controls, jitter, formats and refinement passes.
    void poll_captures() {
        if (!cap_armed_)
            return;
        const auto done = fence_->GetCompletedValue();
        if (done == UINT64_MAX || done < cap_fence_)
            return;
        cap_armed_ = false;
        auto tight = [&](ID3D12Resource* read, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint,
                         unsigned width, unsigned height, unsigned bpp, std::vector<uint8_t>& out) {
            void* mapped = nullptr;
            if (FAILED(read->Map(0, nullptr, &mapped)) || !mapped)
                return false;
            out.resize(size_t(width) * height * bpp);
            for (unsigned row_index = 0; row_index < height; ++row_index)
                std::memcpy(out.data() + size_t(row_index) * width * bpp,
                            static_cast<uint8_t*>(mapped) + footprint.Offset +
                                size_t(row_index) * footprint.Footprint.RowPitch,
                            size_t(width) * bpp);
            read->Unmap(0, nullptr);
            return true;
        };
        CaptureFrame frame;
        frame.meta = cap_staged_.meta;
        frame.source_a = cap_staged_.source_a;
        frame.source_b = cap_staged_.source_b;
        frame.divider = cap_staged_.divider;
        bool complete =
            tight(cap_color_.Get(), cap_color_fp_, width_, height_, cap_color_bpp_, frame.color) &&
            tight(cap_motion_.Get(), cap_motion_fp_, width_, height_, cap_motion_bpp_,
                  frame.motion) &&
            tight(cap_output_.Get(), cap_color_fp_, width_, height_, cap_color_bpp_, frame.output);
        if (complete && cap_has_lineage_)
            complete = tight(cap_exposure_.Get(), exposure_fp_, 1, 1, 4, frame.exposure);
        if (complete && cap_has_lineage_)
            for (unsigned index = 0; index < cap_lineage_.size() && complete; ++index)
                complete = tight(cap_lineage_[index].Get(), cap_lineage_fp_[index], width_, height_,
                                 8, frame.lineage[index]);
        if (complete) {
            cap_staged_ = std::move(frame);
            cap_ready_ = true;
        }
        release_capture_readbacks();
    }
    // Logical buffer bytes, excluding the small always-on exposure telemetry.
    uint64_t capture_readback_bytes() const {
        uint64_t bytes = 0;
        for (auto* readback :
             {cap_color_.Get(), cap_motion_.Get(), cap_output_.Get(), cap_exposure_.Get()})
            if (readback)
                bytes += readback->GetDesc().Width;
        for (const auto& readback : cap_lineage_)
            if (readback)
                bytes += readback->GetDesc().Width;
        return bytes;
    }
    bool take_capture(CaptureFrame& out) {
        if (!cap_ready_)
            return false;
        out = std::move(cap_staged_);
        cap_ready_ = false;
        return true;
    }
    uint64_t launches() const {
        return dispatch_ ? dispatch_->launches() : 0;
    }
    bool pending(ID3D12GraphicsCommandList* cmd) const {
        return pending_ == cmd;
    }
    bool pending_in_batch(UINT count, ID3D12CommandList* const* lists) const {
        for (UINT iteration_index = 0; iteration_index < count; ++iteration_index)
            if (lists[iteration_index] == pending_ || lists[iteration_index] == claimed_)
                return true;
        return false;
    }
    void note_claimed(ID3D12GraphicsCommandList* cmd, ID3D12Resource* color = nullptr,
                      ID3D12Resource* motion = nullptr) {
        check(!pending_ || pending_ == cmd, "Bridge recording overlaps another command list");
        const auto done = completed();
        leases_.erase(std::remove_if(leases_.begin(), leases_.end(),
                                     [&](const Lease& lease) { return lease.fence <= done; }),
                      leases_.end());
        claimed_ = pending_ = cmd;
        if (color)
            engine_output_ = color;
        if (motion)
            engine_motion_ = motion;
    }
    void set_host_jitter(float jitter_x, float jitter_y) {
        host_jx_ = jitter_x * geometry_.motion_x();
        host_jy_ = jitter_y * geometry_.motion_y();
    }
    unsigned group_count() const {
        return network_ ? unsigned(network_->group_ends().size()) : 0;
    }
    void invalidate() {
        failed_ = true;
        if (external_)
            external_->invalidate();
        if (network_)
            network_->invalidate();
        for (auto& refinement_runner : refinement_)
            refinement_runner->invalidate();
    }
    // Only an owner that has proven retirement of prior submissions may drop
    // this unsubmitted recording. A recycled list pointer must not retain a
    // failed claim and accidentally match an unrelated game submission.
    void abandon_unsubmitted(ID3D12GraphicsCommandList* cmd) {
        if (pending_ == cmd) {
            invalidate();
            pending_ = nullptr;
            network_recorded_ = false;
        }
        if (claimed_ == cmd) claimed_ = nullptr;
    }
    void validate_submit(ID3D12CommandQueue* queue, UINT count,
                         ID3D12CommandList* const* lists) const {
        unsigned hits = 0;
        for (UINT iteration_index = 0; iteration_index < count; ++iteration_index)
            hits += pending_ && lists[iteration_index] == pending_;
        check(!hits || (hits == 1 && queue == queue_.Get() && !failed_),
              "Direct frame wrong/duplicate submission");
    }
    bool after_submit(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists) {
        if (claimed_)
            for (UINT iteration_index = 0; iteration_index < count; ++iteration_index)
                if (lists[iteration_index] == claimed_)
                    claimed_ = nullptr;
        for (UINT iteration_index = 0; iteration_index < count; ++iteration_index)
            if (pending_ && lists[iteration_index] == pending_) {
                check(queue == queue_.Get(), "Direct submission queue changed");
                phoenix_native::hr(queue->Signal(fence_.Get(), submitted_ + 1));
                leases_.push_back({submitted_ + 1, std::move(engine_output_),
                                   std::move(engine_motion_), std::move(engine_depth_)});
                if (network_recorded_) {
                    network_->commit_any_thread(pending_);
                    for (auto& refinement_runner : refinement_)
                        refinement_runner->commit_any_thread(pending_);
                }
                network_recorded_ = false;
                pending_ = nullptr;
                ++submitted_;
                return true;
            }
        return false;
    }
    // Capture color/motion into owned textures, normalize the network feed,
    // run each configured refinement pass, then compose the selected output.
    // Resource transitions and fence leases protect snapshots used by GPU work.
    void record(ID3D12GraphicsCommandList* cmd, ID3D12Resource* color, ID3D12Resource* motion,
                FrameEvidence& evidence, Controls controls, const Settings& settings,
                unsigned blocks, ID3D12Resource* depth = nullptr,
                D3D12_RESOURCE_STATES depth_state = D3D12_RESOURCE_STATE_COMMON) {
        check((ready() || (pending_ == cmd && !failed_ && !network_recorded_)) && cmd && color &&
                  motion && evidence.states_verified && evidence.restore_bindings,
              "Direct frame contract missing");
        check(settings.valid() && architecture_matches(settings),
              "Direct refinement configuration changed; rebuild after fence retirement");
        check(comparison_supported(settings, nr_ != nullptr),
              "Direct comparison needs a live NR feature for NVIDIA sources");
        check(matches(evidence.queue, color->GetDesc(), motion->GetDesc()),
              "Direct geometry changed");
        for (auto* resource : {color, motion}) {
            auto description = resource->GetDesc();
            check(description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                      description.MipLevels == 1 && description.DepthOrArraySize == 1 &&
                      description.SampleDesc.Count == 1,
                  "Direct resource layout unsupported");
        }
        check(color != motion, "Direct aliased inputs");
        controls.reset = controls.reset || !submitted_;
        const auto resolved =
            resolve_controls(controls, width_, height_, true, submitted_ > 0,
                             controls.reset || !network_ ? 0 : network_->next_sampling_index());
        if (network_)
            network_->validate_command_any_thread(cmd, queue_.Get());
        check(!external_ || !settings.host_unjitter,
              "Worker execution requires reference-grid unjitter off");
        check(depth_matches(depth), "Direct depth geometry changed; retire executor first");
        const auto done = completed();
        leases_.erase(std::remove_if(leases_.begin(), leases_.end(),
                                     [&](const Lease& lease) { return lease.fence <= done; }),
                      leases_.end());
        engine_output_ = color;
        engine_motion_ = motion;
        engine_depth_ = depth;
        pending_ = cmd;
        try {
            copy_input(cmd, color, evidence.color_state, input_.Get());
            copy_input(cmd, motion, evidence.motion_state, motion_.Get(), true);
            prepare_upscale(cmd, controls, depth, depth_state);
            if (cap_left_)
                capture_inputs(cmd, controls, settings, blocks);
            // NVIDIA wipe sides remapped onto the two slots the output shader
            // reads (0 = input snapshot, 2 = own result). Sides flagged in
            // the plan are overwritten by the resolve pass below, which
            // transfers the genuine NR answer (evaluated on the proxy).
            const unsigned compose = settings.mode == split ? unsigned(settings.comparison) : 0u;
            const auto wipe = nr_wipe_stage(settings.mode, int(compose), settings.source_a,
                                            settings.source_b, nr_ != nullptr);
            check(!wipe.stage || compose, "Direct NR resolve outside a wipe");
            unsigned effective_source_a = wipe.effective_source_a,
                     effective_source_b = wipe.effective_source_b;
            Constants constants{width_,
                                height_,
                                unsigned(controls.reset != 0) | (srgb_ ? 2u : 0u) |
                                    (settings.host_bridge ? 4u : 0u),
                                compose,
                                effective_source_a,
                                effective_source_b,
                                settings.divider,
                                settings.difference_gain,
                                wipe.resolve};
            auto* exposure_texture = exposure_[submitted_ % 2].Get();
            auto* previous = exposure_[(submitted_ + 1) % 2].Get();
            phoenix_native::transition(cmd, exposure_texture,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            const unsigned slot = unsigned(submitted_ % 2);
            // Meter and normalize resample onto the reference grid; they
            // take the frame jitter (bit-cast into a/b) while the output
            // pass keeps the wipe indices there.
            uint32_t jbx = 0, jby = 0;
            {
                const float ujx = settings.host_unjitter ? host_jx_ : 0.0f;
                const float ujy = settings.host_unjitter ? host_jy_ : 0.0f;
                memcpy(&jbx, &ujx, 4);
                memcpy(&jby, &ujy, 4);
            }
            const Constants native_constants{
                constants.width,   constants.height, constants.reset,  constants.mode, jbx, jby,
                constants.divider, constants.gain,   constants.resolve};
            meter_.record(cmd, {input_.Get(), previous, nullptr}, exposure_texture,
                          exposure_constants(native_constants, geometry_), true, slot);
            phoenix_native::transition(cmd, exposure_texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            // Retire the metered value for the flicker detector: 4 bytes to
            // the readback heap, fenced with this frame's submission.
            phoenix_native::transition(cmd, exposure_texture,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            {
                D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
                dst.pResource = exposure_read_.Get();
                dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst.PlacedFootprint = exposure_fp_;
                src.pResource = exposure_texture;
                src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            phoenix_native::transition(cmd, exposure_texture, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            exposure_fence_ = submitted_ + 1;
            exposure_armed_ = true;
            if (cap_left_ && !cap_armed_ && !cap_ready_ && cap_has_lineage_) {
                if (!cap_exposure_)
                    cap_exposure_ = tracked_buffer(device_.Get(), 256, D3D12_HEAP_TYPE_READBACK,
                                                   D3D12_RESOURCE_STATE_COPY_DEST);
                phoenix_native::transition(cmd, exposure_texture,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
                dst.pResource = cap_exposure_.Get();
                dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                dst.PlacedFootprint = exposure_fp_;
                src.pResource = exposure_texture;
                src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                phoenix_native::transition(cmd, exposure_texture, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            phoenix_native::transition(cmd, normalized_.Get(),
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            normalize_.record(cmd, {input_.Get(), exposure_texture, nullptr}, normalized_.Get(),
                              native_constants, false, slot);
            phoenix_native::transition(cmd, normalized_.Get(),
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            // Match the established Session HDR bridge-to-NVAPI texture boundary.
            copy_input(cmd, normalized_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       network_input_.Get());
            capture_lineage(cmd, 3, network_input_.Get(),
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (!external_ && network_timing_begin)
                network_timing_begin();
            ID3D12Resource* result = nullptr;
            if (external_) {
                ExecutorFrame frame;
                frame.command = cmd;
                frame.color = network_input_.Get();
                frame.motion = motion_.Get();
                frame.depth = depth;
                frame.depth_state = depth_state;
                frame.device_depth = true;
                frame.guides = guides_->record(cmd, motion_.Get(), depth, depth_state,
                                               controls.motion_x, controls.motion_y);
                frame.controls = controls;
                frame.settings = &settings;
                frame.transformer_blocks = blocks;
                frame.serial = submitted_ + 1;
                frame.completed_serial = done;
                result = external_->record(frame);
            } else
                for (unsigned pass = 0; pass < passes(); ++pass) {
                    if (pass)
                        copy_input(cmd, result, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   network_input_.Get());
                    dispatch_->prepare_repeats(cmd);
                    FrameDispatch::Scope scope(*dispatch_, cmd, resolved, blocks);
                    auto* net = pass ? refinement_[pass - 1].get() : network_.get();
                    result = net->record(cmd, !resolved.recurrent, 0, queue_.Get(), true);
                    scope.finish();
                }
            if (!external_)
                network_recorded_ = true;
            if (!external_ && network_timing_end)
                network_timing_end();
            if (!external_)
                capture_lineage(cmd, 2, result, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (initialized_)
                phoenix_native::transition(cmd, output_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (external_ && result == network_input_.Get() && settings.mode == ours) {
                phoenix_native::transition(cmd, input_.Get(),
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COPY_SOURCE);
                phoenix_native::transition(cmd, output_.Get(),
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_COPY_DEST);
                cmd->CopyResource(output_.Get(), input_.Get());
                phoenix_native::transition(cmd, input_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                phoenix_native::transition(cmd, output_.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            } else
                write_.record(cmd, {input_.Get(), result, exposure_texture}, output_.Get(),
                              constants, false, slot);
            // Resolve the genuine NR answer over the flagged wipe sides. The
            // NR image rests in UAV after its evaluate; it returns there so
            // the next evaluate finds the state it expects.
            if (wipe.resolve && compose == 1) {
                check(nr_ != nullptr, "Direct NR resolve without an image");
                phoenix_native::uav(cmd);
                if (shared_native_) {
                    write_shared_native(cmd, nr_, output_.Get(), wipe.resolve);
                } else {
                    phoenix_native::transition(cmd, nr_, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    auto resolve_constants = constants;
                    if (host_bridge_)
                        resolve_constants.resolve |= 4u;
                    resolve_.record(cmd, {input_.Get(), proxy_image(), nr_}, output_.Get(),
                                    resolve_constants, false, slot);
                    phoenix_native::transition(cmd, nr_,
                                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                }
            }
            phoenix_native::transition(cmd, output_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            present(cmd, output_.Get(), color, evidence.color_state);
            if (cap_left_)
                capture_output(cmd, output_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE);
            evidence.restore_bindings(evidence.restore_context, cmd);
            initialized_ = true;
        } catch (...) {
            invalidate();
            throw;
        }
    }
};
} // namespace production_game41::phoenix::direct
