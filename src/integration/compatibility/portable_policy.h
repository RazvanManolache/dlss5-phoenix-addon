#pragma once
#include "session_policy.h"
#include <filesystem>
#include <cmath>

namespace phoenix_compat {
struct PortableOptions {
    std::filesystem::path root;
    std::string device = "auto";
    bool allow_unvalidated = false;
};
inline std::string portable_backend(const std::string& device) {
    if (!phoenix_addon::phoenix::valid_execution_device(device) || device == "native" ||
        device == "sycl:cpu")
        throw std::runtime_error("Choose a portable device; SYCL CPU requires the isolated helper.");
    return device == "cpu" ? "cpu" : device.substr(0, device.find(':'));
}
inline SettingsJson portable_metadata(const Settings& settings, bool reset) {
    auto controls = settings.controls.apply(phoenix_addon::Controls{});
    auto repeats = settings.block_repeats;
    for (unsigned index = settings.blocks_for(settings.mode); index < repeats.size(); ++index)
        repeats[index] = 0;
    return {{"reset", reset}, {"has_motion", false}, {"has_depth", false},
        {"has_history_validity", false}, {"repeats", repeats},
        {"passes", settings.refinement_passes}, {"controls", {
            {"intensity", 1.f}, {"style", 0},
            {"local_tone", controls.tone}, {"local_structure", controls.structure},
            {"skin_structure", controls.skin}, {"auto_mask", controls.auto_mask != 0}}}};
}
inline std::vector<float> portable_color(const std::vector<uint8_t>& rgba) {
    if (rgba.empty() || rgba.size() % 4)
        throw std::runtime_error("Incomplete portable input image");
    std::vector<float> result(rgba.size() / 4 * 3);
    for (size_t pixel = 0; pixel < rgba.size() / 4; ++pixel)
        for (size_t channel = 0; channel < 3; ++channel)
            result[pixel * 3 + channel] = rgba[pixel * 4 + channel] / 255.f;
    return result;
}
// Mirrors comparison.h's shader, using this evaluation's input/output pair.
inline std::vector<uint8_t> portable_compose(const std::vector<uint8_t>& input,
                                            const std::vector<float>& output,
                                            unsigned width, const Settings& settings,
                                            const std::vector<float>* nvidia = nullptr) {
    if (!width || input.empty() || input.size() % (size_t(width) * 4) ||
        output.size() != input.size() / 4 * 3 || (needs_nvidia_dll(settings) && (!nvidia || nvidia->size()!=output.size())))
        throw std::runtime_error("Invalid portable comparison sources or geometry");
    std::vector<uint8_t> result(input.size());
    auto saturate = [](float value) { return std::clamp(value, 0.f, 1.f); };
    for (size_t pixel = 0; pixel < input.size() / 4; ++pixel) {
        std::array<float, 3> a{}, b{}, display{}, difference{};
        for (unsigned channel = 0; channel < 3; ++channel) {
            const float enhanced = output[pixel * 3 + channel];
            if (!std::isfinite(enhanced)) throw std::runtime_error("Nonfinite portable output");
            const float normal = input[pixel * 4 + channel] / 255.f;
            auto source=[&](int index) {return index==0?normal:index==1&&nvidia?nvidia->at(pixel*3+channel):enhanced;};
            a[channel] = source(settings.source_a);
            b[channel] = source(settings.source_b);
            display[channel] = settings.mode==phoenix_addon::phoenix::nvidia&&nvidia?nvidia->at(pixel*3+channel):enhanced;
            difference[channel] = (a[channel] - b[channel]) * settings.difference_gain;
        }
        if (settings.mode == phoenix_addon::phoenix::input)
            for (unsigned channel = 0; channel < 3; ++channel)
                display[channel] = input[pixel * 4 + channel] / 255.f;
        else if (settings.mode == phoenix_addon::phoenix::split) {
            const auto mode = settings.comparison == 0 ? 1 : settings.comparison;
            if (mode == 1) display = pixel % width < unsigned(width * settings.divider) ? a : b;
            else for (unsigned channel = 0; channel < 3; ++channel)
                display[channel] = mode == 3 ? .5f + .5f * difference[channel] : std::abs(difference[channel]);
            if (mode == 4) {
                const float v = saturate(std::max({std::abs(difference[0]), std::abs(difference[1]), std::abs(difference[2])}));
                display = {saturate(4*v-2), saturate(4*v)-saturate(4*v-3), saturate(4*v)-saturate(4*v-1)};
            }
            if (mode == 5) display.fill(std::abs(difference[0]*.2126f + difference[1]*.7152f + difference[2]*.0722f));
        }
        for (unsigned channel = 0; channel < 3; ++channel)
            result[pixel * 4 + channel] = uint8_t(std::lround(saturate(display[channel]) * 255));
        result[pixel * 4 + 3] = input[pixel * 4 + 3];
    }
    return result;
}
}
