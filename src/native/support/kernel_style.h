#pragma once
#include <string_view>
namespace phoenix_kernel {
inline constexpr unsigned style_architecture(std::string_view style) {
    return style == "rtx20" ? 75 : style == "rtx30" ? 86 : style == "rtx40" ? 89 : 0;
}
inline constexpr bool valid_style(std::string_view style) {
    return style == "auto" || style_architecture(style) != 0;
}
inline constexpr bool style_supported(std::string_view style, unsigned architecture) {
    if (style == "auto") return true;
    const auto requested = style_architecture(style);
    return requested && (architecture == 75 || architecture == 86 || architecture == 89 || architecture == 120) &&
           requested < architecture;
}
inline constexpr const char* style_label(std::string_view style) {
    return style == "auto" ? "Automatic (native, recommended)" : style == "rtx20" ? "RTX 20-style fallback" :
           style == "rtx30" ? "RTX 30-style fallback" : style == "rtx40" ? "RTX 40-style fallback" : "Unknown";
}
}
