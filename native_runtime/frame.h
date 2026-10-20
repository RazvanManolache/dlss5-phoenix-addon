#pragma once
#include "model.h"
#include "parallel_rows.h"
#include "frame_contract.h"
#include "appearance.h"
namespace phoenix_cpp {
struct Controls {
    float intensity = 1, tone = 1, structure = 1, skin = -1, history_strength = 1;
    unsigned seed = 0, style = 0;
    bool auto_mask = true;
    bool advancing_noise = false;
    bool legacy_output = false;
    static Controls from(const Json& configuration) {
        Controls controls;
        controls.intensity = configuration.value("intensity", 1.f);
        controls.tone = configuration.value("local_tone", 1.f);
        controls.structure = configuration.value("local_structure", 1.f);
        controls.skin = configuration.value("skin_structure", -1.f);
        controls.seed = configuration.value("seed", 0u);
        controls.style = configuration.value("style", 0u);
        controls.auto_mask = configuration.value("auto_mask", true);
        const auto sampling = configuration.value("sampling_mode", std::string("fixed"));
        demand(sampling == "fixed" || sampling == "advancing", "Unknown sampling mode");
        controls.advancing_noise = sampling == "advancing";
        demand(std::isfinite(controls.intensity) && controls.intensity >= 0 &&
                   controls.intensity <= 2 && controls.style <= 2,
               "Invalid frame controls");
        demand(std::isfinite(controls.tone) && controls.tone >= 0 && controls.tone <= 2 &&
                   std::isfinite(controls.structure) && controls.structure >= 0 &&
                   controls.structure <= 2 && std::isfinite(controls.skin) && controls.skin >= -1 &&
                   controls.skin <= 2,
               "Invalid conditioning controls");
        return controls;
    }
};
struct FrameState {
    unsigned width, height, padded_width, padded_height;
    std::vector<float> history;
    // Noise is already half-rounded by the model contract. Retain those exact
    // bits instead of expanding them into a second, 32-bit cached representation.
    std::vector<uint16_t> noise;
    unsigned noise_seed = UINT32_MAX;
    phx::Kernel<HostMath> math{};
    RowExecutor rows;
    const std::vector<float>* history_view = nullptr;
    const std::vector<float>& source_history() const {
        return history_view ? *history_view : history;
    }
    FrameState(unsigned frame_width, unsigned frame_height, unsigned working_width,
               unsigned working_height,
               unsigned threads = std::min(16u,
                                           std::max(1u, std::thread::hardware_concurrency() / 2)))
        : width(frame_width), height(frame_height), padded_width(working_width),
          padded_height(working_height), rows(threads) {}
    static int mirror(int coordinate, int extent) {
        return reflect_once_then_clamp(coordinate, extent);
    }
    static uint32_t pcg(uint32_t state) {
        return ((state >> ((state >> 28) + 4)) ^ state) * 0x108ef2d9u;
    }
    static float unit(uint32_t state) {
        auto permuted_bits = pcg(state);
        return float(((permuted_bits >> 30) ^ (permuted_bits >> 8)) + 1) * 0x1p-24f;
    }
    // Generate deterministic Gaussian RGB noise with Box-Muller sampling.
    // Cache it by seed so repeated frames do not regenerate the padded noise image.
    void prepare_noise(unsigned seed) {
        if (noise_seed == seed && !noise.empty())
            return;
        noise.resize(size_t(padded_width) * padded_height * 3);
        rows.run(padded_height, [&](unsigned pixel_y) {
            for (unsigned pixel_x = 0; pixel_x < padded_width; ++pixel_x) {
                auto pixel_hash = pcg(pixel_x * 0x8da6b343u ^ seed * 0x9e3779b9u ^
                                      pixel_y * 0xd8163841u ^ 0x243f6a88u);
                auto mixed_hash = (pixel_hash >> 22) ^ pixel_hash;
                float uniform_0 = unit(mixed_hash * 0x2c9277b5u + 0xac564b05u),
                      uniform_1 = unit(mixed_hash * 0xfa6dc5f9u + 0x4712a88eu),
                      uniform_2 = unit(mixed_hash * 0xcaa5b80du + 0x21dd796bu),
                      uniform_3 = unit(mixed_hash * 0x83232c31u + 0x3463e0acu);
                float radius_0 = std::sqrt(-2.f * std::log(uniform_0)),
                      radius_1 = std::sqrt(-2.f * std::log(uniform_2));
                auto pixel_offset = (size_t(pixel_y) * padded_width + pixel_x) * 3;
                noise[pixel_offset] =
                    uint16_t(math.half_bits(radius_0 * std::cos(uniform_1 * 6.28318530718f)));
                noise[pixel_offset + 1] =
                    uint16_t(math.half_bits(radius_0 * std::sin(uniform_1 * 6.28318530718f)));
                noise[pixel_offset + 2] =
                    uint16_t(math.half_bits(radius_1 * std::cos(uniform_3 * 6.28318530718f)));
            }
        });
        noise_seed = seed;
    }
    // Bilinear history lookup in pixel-center coordinates, clamped at image edges.
    std::array<float, 3> sample_rgb(float sample_x, float sample_y) const {
        sample_x = std::clamp(sample_x - .5f, 0.f, float(width - 1));
        sample_y = std::clamp(sample_y - .5f, 0.f, float(height - 1));
        auto left_column = unsigned(sample_x), top_row = unsigned(sample_y),
             right_column = std::min(left_column + 1, width - 1),
             bottom_row = std::min(top_row + 1, height - 1);
        float horizontal_fraction = sample_x - left_column, vertical_fraction = sample_y - top_row;
        auto read_channel = [&](unsigned sample_x, unsigned sample_y, unsigned channel) {
            return source_history()[(size_t(sample_y) * width + sample_x) * 3 + channel];
        };
        std::array<float, 3> rgb;
        for (unsigned channel = 0; channel < 3; ++channel) {
            float top = read_channel(left_column, top_row, channel) * (1 - horizontal_fraction) +
                        read_channel(right_column, top_row, channel) * horizontal_fraction,
                  bottom =
                      read_channel(left_column, bottom_row, channel) * (1 - horizontal_fraction) +
                      read_channel(right_column, bottom_row, channel) * horizontal_fraction;
            rgb[channel] = top * (1 - vertical_fraction) + bottom * vertical_fraction;
        }
        return rgb;
    }
    // Approximate bicubic history sampling with five bilinear RGB taps. Combine
    // the middle cubic weights into one filtered tap and keep the four axial taps;
    // the multiplication/addition order below is retained for temporal parity.
    std::array<float, 3> five_rgb(float sample_x, float sample_y) const {
        float center_x = std::floor(sample_x - .5f) + .5f,
              center_y = std::floor(sample_y - .5f) + .5f,
              horizontal_fraction = std::clamp(sample_x - center_x, 0.f, 1.f),
              vertical_fraction = std::clamp(sample_y - center_y, 0.f, 1.f);
        auto weights = [](float fraction) {
            float t2 = fraction * fraction, t3 = t2 * fraction, w0 = (fraction + t3) * -.5f + t2,
                  w1 = 1.5f * t3 - 2.5f * t2 + 1, w3 = .5f * (t3 - t2), w2 = 1 - w0 - w1 - w3;
            return std::array<float, 4>{w0, w1, w2, w3};
        };
        auto horizontal_weights = weights(horizontal_fraction),
             vertical_weights = weights(vertical_fraction);
        float middle_weight_x = horizontal_weights[1] + horizontal_weights[2],
              middle_weight_y = vertical_weights[1] + vertical_weights[2],
              mx = center_x + horizontal_weights[2] / middle_weight_x,
              my = center_y + vertical_weights[2] / middle_weight_y;
        float w0 = horizontal_weights[0] * middle_weight_y,
              w1 = vertical_weights[0] * middle_weight_x, w2 = middle_weight_x * middle_weight_y,
              w3 = vertical_weights[3] * middle_weight_x,
              w4 = horizontal_weights[3] * middle_weight_y, weight_sum = w0 + w1 + w2 + w3 + w4;
        auto p0 = sample_rgb(center_x - 1, my), p1 = sample_rgb(mx, center_y - 1),
             p2 = sample_rgb(mx, my), p3 = sample_rgb(mx, center_y + 2),
             p4 = sample_rgb(center_x + 2, my);
        std::array<float, 3> rgb;
        for (unsigned channel = 0; channel < 3; ++channel)
            rgb[channel] = (p0[channel] * w0 + p1[channel] * w1 + p2[channel] * w2 +
                            p3[channel] * w3 + p4[channel] * w4) /
                           weight_sum;
        return rgb;
    }
    // Build two history views: depth-dilated motion for network conditioning, and
    // center-pixel motion for final blending. Their validity masks stay independent.
    void warp(const float* motion, const float* depth, const float* validity, bool reversed,
              std::vector<float>& conditioning_history, std::vector<float>& resolve_history,
              std::vector<uint8_t>& conditioning_valid, std::vector<uint8_t>& resolve_valid) {
        auto pixel_count = size_t(width) * height;
        conditioning_history.resize(pixel_count * 3);
        resolve_history.resize(pixel_count * 3);
        conditioning_valid.assign(pixel_count, 0);
        resolve_valid.assign(pixel_count, 0);
        if (source_history().empty() || !motion)
            return;
        rows.run(height, [&](unsigned pixel_y) {
            for (unsigned pixel_x = 0; pixel_x < width; ++pixel_x) {
                size_t pixel_index = size_t(pixel_y) * width + pixel_x, closest_pixel = pixel_index;
                float closest_depth = depth ? depth[pixel_index] : 0;
                if (depth)
                    for (int neighbor_offset_y : {-1, 1})
                        for (int neighbor_offset_x : {-1, 1}) {
                            auto neighbor_x = std::clamp(int(pixel_x) + neighbor_offset_x, 0,
                                                         int(width) - 1),
                                 neighbor_y = std::clamp(int(pixel_y) + neighbor_offset_y, 0,
                                                         int(height) - 1);
                            size_t neighbor_index = size_t(neighbor_y) * width + neighbor_x;
                            bool nearer = reversed ? depth[neighbor_index] > closest_depth
                                                   : depth[neighbor_index] < closest_depth;
                            if (nearer && (!validity || validity[neighbor_index] > .9999f)) {
                                closest_pixel = neighbor_index;
                                closest_depth = depth[neighbor_index];
                            }
                        }
                float conditioning_x = (float(pixel_x) + .5f) + motion[closest_pixel * 2],
                      conditioning_y = (float(pixel_y) + .5f) + motion[closest_pixel * 2 + 1],
                      resolve_x = (float(pixel_x) + .5f) + motion[pixel_index * 2],
                      resolve_y = (float(pixel_y) + .5f) + motion[pixel_index * 2 + 1];
                auto inside = [&](float sample_x, float sample_y) {
                    return std::isfinite(sample_x) && std::isfinite(sample_y) && sample_x >= .5f &&
                           sample_x <= float(width) - .5f && sample_y >= .5f &&
                           sample_y <= float(height) - .5f;
                };
                bool valid = !validity || validity[pixel_index] > .9999f;
                // A missing optional mask means all pixels are eligible; it does
                // not make out-of-frame motion valid history. Reject those samples
                // consistently with an explicitly supplied all-one mask.
                conditioning_valid[pixel_index] =
                    valid && (!validity || validity[closest_pixel] > .9999f) &&
                    inside(conditioning_x, conditioning_y);
                resolve_valid[pixel_index] = valid && inside(resolve_x, resolve_y);
                std::array<float, 3> conditioning_rgb{}, resolve_rgb{};
                if (conditioning_valid[pixel_index])
                    conditioning_rgb = five_rgb(conditioning_x, conditioning_y);
                if (resolve_valid[pixel_index])
                    resolve_rgb = conditioning_valid[pixel_index] && conditioning_x == resolve_x &&
                                          conditioning_y == resolve_y
                                      ? conditioning_rgb
                                      : five_rgb(resolve_x, resolve_y);
                for (unsigned channel = 0; channel < 3; ++channel) {
                    conditioning_history[pixel_index * 3 + channel] = conditioning_rgb[channel];
                    resolve_history[pixel_index * 3 + channel] = resolve_rgb[channel];
                }
            }
        });
    }
    // Pack 16 half features per padded pixel: noise[0..2], constant[3], centered
    // current RGB[4..6], history RGB[7..9], controls[10..14], and zero[15].
    // Mirror-padding and intermediate half-rounding match the model input contract.
    std::vector<uint16_t> features(const float* color, const Controls& controls,
                                   const std::vector<float>& prior,
                                   const std::vector<uint8_t>& valid,
                                   const float* control_mask = nullptr) {
        prepare_noise(controls.seed);
        std::vector<uint16_t> result(size_t(padded_width) * padded_height * 16);
        std::array<float, 5> conditioning = {
            controls.style / 128.f, controls.tone, controls.auto_mask ? 1.f : controls.structure,
            controls.auto_mask ? (controls.skin < 0 ? controls.structure : controls.skin) : -1.f,
            controls.auto_mask ? controls.structure : -1.f};
        rows.run(padded_height, [&](unsigned pixel_y) {
            for (unsigned pixel_x = 0; pixel_x < padded_width; ++pixel_x) {
                size_t padded_pixel_index = size_t(pixel_y) * padded_width + pixel_x,
                       source_pixel_index =
                           size_t(mirror(pixel_y, height)) * width + mirror(pixel_x, width);
                auto pixel_features = result.data() + padded_pixel_index * 16;
                for (unsigned feature_index = 0; feature_index < 3; ++feature_index)
                    pixel_features[feature_index] = noise[padded_pixel_index * 3 + feature_index];
                pixel_features[3] = uint16_t(math.half_bits(1));
                for (unsigned feature_index = 0; feature_index < 3; ++feature_index) {
                    auto centre = [&](float value) {
                        return centered_half_feature(math, value);
                    };
                    pixel_features[4 + feature_index] = uint16_t(
                        math.half_bits(centre(color[source_pixel_index * 3 + feature_index])));
                    pixel_features[7 + feature_index] =
                        valid[source_pixel_index]
                            ? uint16_t(math.half_bits(
                                  centre(prior[source_pixel_index * 3 + feature_index])))
                            : pixel_features[4 + feature_index];
                }
                auto pixel_conditioning = conditioning;
                if (control_mask) {
                    pixel_conditioning[1] = controls.tone * control_mask[source_pixel_index * 4 + 1];
                    pixel_conditioning[2] = controls.structure * control_mask[source_pixel_index * 4 + 2];
                    pixel_conditioning[3] = pixel_conditioning[4] = 0.f;
                }
                for (unsigned feature_index = 0; feature_index < 5; ++feature_index)
                    pixel_features[10 + feature_index] =
                        uint16_t(math.half_bits(pixel_conditioning[feature_index]));
                pixel_features[15] = 0;
            }
        });
        return result;
    }
    // The four-channel head predicts RGB corrections plus a history gate.
    // Update half-rounded history separately from display intensity, so changing
    // intensity does not silently change the temporal state fed to the next frame.
    std::vector<float> resolve(const float* color, const std::vector<float>& head,
                               const Controls& controls, const std::vector<float>& prior,
                               const std::vector<uint8_t>& valid, std::vector<float>& next,
                               const float* control_mask = nullptr, bool defer_display = false) {
        size_t pixel_count = size_t(width) * height;
        std::vector<float> output(pixel_count * 3);
        next.resize(pixel_count * 3);
        rows.run(height, [&](unsigned pixel_y) {
            for (unsigned pixel_x = 0; pixel_x < width; ++pixel_x) {
                auto pixel_index = size_t(pixel_y) * width + pixel_x,
                     head_offset = (size_t(pixel_y) * padded_width + pixel_x) * 4;
                float activation_gate = resolve_gate(head[head_offset + 3],
                                                    valid[pixel_index] != 0,
                                                    controls.history_strength);
                float frame_luma = 0, out_luma = 0;
                const float intensity = controls.intensity *
                    (control_mask ? control_mask[pixel_index * 4] : 1.f);
                const float luminance_weights[] = {.2126f, .7152f, .0722f};
                for (unsigned channel = 0; channel < 3; ++channel) {
                    float current_value = color[pixel_index * 3 + channel],
                          network = std::clamp(current_value + head[head_offset + channel] * .25f,
                                               0.f, 1.f);
                    network += (prior[pixel_index * 3 + channel] - network) * activation_gate;
                    next[pixel_index * 3 + channel] = math.half(network);
                    float blended_value =
                        std::clamp(current_value + std::min(intensity, 1.f) *
                                                       (network - current_value),
                                   0.f, 1.f);
                    output[pixel_index * 3 + channel] = blended_value;
                    frame_luma += current_value * luminance_weights[channel];
                    out_luma += blended_value * luminance_weights[channel];
                }
                if (intensity > 1) {
                    float ratio = (out_luma + 1.f / 512) / (frame_luma + 1.f / 512),
                          safe_luminance_ratio = std::max(ratio, 1e-6f),
                          intensity_gain = std::clamp(
                              std::pow(safe_luminance_ratio, intensity), .5f, 2.f);
                    for (unsigned channel = 0; channel < 3; ++channel)
                        output[pixel_index * 3 + channel] =
                            std::clamp(output[pixel_index * 3 + channel] * intensity_gain /
                                           safe_luminance_ratio,
                                       0.f, 1.f);
                }
                if (!controls.legacy_output) {
                    RGB original{}, neural{};
                    for (unsigned c = 0; c < 3; ++c) {
                        original[c] = color[pixel_index * 3 + c];
                        neural[c] = next[pixel_index * 3 + c];
                    }
                    const auto displayed = defer_display ? neural : compose_display(original, neural,
                        pixel_index, controls.intensity, controls.style, controls.tone,
                        DisplayInputs{control_mask});
                    for (unsigned c = 0; c < 3; ++c) output[pixel_index * 3 + c] = displayed[c];
                }
            }
        });
        return output;
    }
};
}
