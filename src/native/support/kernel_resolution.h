// Select a compiled kernel for the actual neural input, independently of the
// display/output resolution. Selection happens when a network is created.
#pragma once
#include <cstdint>
#include <string>

namespace phoenix_kernel {
inline bool valid_resolution_variant(const std::string& default_entry,
                                     const std::string& large_input_entry,
                                     uint64_t large_input_min_pixels) {
    if (large_input_entry.empty())
        return large_input_min_pixels == 0;
    return large_input_entry != default_entry && large_input_min_pixels > 0 &&
           large_input_min_pixels <= uint64_t(16384) * 16384;
}

inline const std::string& entry_for_resolution(const std::string& default_entry,
                                             const std::string& large_input_entry,
                                             uint64_t large_input_min_pixels,
                                             unsigned width, unsigned height) {
    const uint64_t input_pixels = uint64_t(width) * height;
    if (!large_input_entry.empty() && large_input_min_pixels != 0 &&
        input_pixels >= large_input_min_pixels)
        return large_input_entry;
    return default_entry;
}
} // namespace phoenix_kernel
