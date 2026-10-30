#pragma once
#include "../support/kernel_style.h"
#include "../../../third_party/nlohmann/json.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
namespace phoenix_bundle {
// Optional style metadata must not make the native-default UI unusable when
// an external bundle omits it or contains an incomplete optional entry.
inline nlohmann::json available_kernel_styles(const nlohmann::json& manifest, unsigned architecture) {
    auto result = nlohmann::json::array();
    if (!manifest.contains("kernel_styles") || !manifest.at("kernel_styles").is_object()) return result;
    for (const auto& item : manifest.at("kernel_styles").items()) {
        if (!phoenix_kernel::style_supported(item.key(), architecture) || item.key() == "auto" ||
            !item.value().is_object() || !item.value().contains("architectures") ||
            !item.value().contains("kernels") || !item.value().at("kernels").is_array() ||
            item.value().at("kernels").size() != 41) continue;
        const auto& targets = item.value().at("architectures");
        if (targets.is_array() && std::find(targets.begin(), targets.end(), "sm_" + std::to_string(architecture)) != targets.end())
            result.push_back(item.key());
    }
    return result;
}
// Variants share every native launch contract. Only the pinned binary changes.
inline const nlohmann::json& style_catalog(const nlohmann::json& manifest,
                                          const std::string& style, unsigned architecture) {
    if (manifest.value("kernel_style", std::string("auto")) != "auto")
        throw std::runtime_error("A standalone style build cannot replace the native default catalogue");
    if (!phoenix_kernel::valid_style(style) || !phoenix_kernel::style_supported(style, architecture))
        throw std::runtime_error("Kernel style is not supported on the processing GPU");
    const auto& native = manifest.at("kernels");
    if (style == "auto") return native;
    if (!manifest.contains("kernel_styles") || !manifest.at("kernel_styles").contains(style))
        throw std::runtime_error("Selected kernel style is not included in this native bundle");
    const auto& variant = manifest.at("kernel_styles").at(style);
    const auto target = "sm_" + std::to_string(architecture);
    const auto& targets = variant.at("architectures");
    if (std::find(targets.begin(), targets.end(), target) == targets.end())
        throw std::runtime_error("Selected kernel style has no image for the processing GPU");
    const auto& records = variant.at("kernels");
    if (native.size() != 41 || records.size() != native.size())
        throw std::runtime_error("Incomplete kernel-style catalogue");
    std::unordered_set<std::string> seen;
    for (const auto& row : records) {
        const auto name = row.at("name").get<std::string>();
        const auto original = std::find_if(native.begin(), native.end(), [&](const auto& item) { return item.at("name") == name; });
        if (!seen.insert(name).second || original == native.end())
            throw std::runtime_error("Duplicate or unknown kernel-style entry");
        for (const char* key : {"entry", "parameter_bytes", "block", "large_input_entry", "large_input_min_pixels"})
            if (row.contains(key) != original->contains(key) || (row.contains(key) && row.at(key) != original->at(key)))
                throw std::runtime_error("Kernel-style launch ABI differs from native");
    }
    return records;
}
}
