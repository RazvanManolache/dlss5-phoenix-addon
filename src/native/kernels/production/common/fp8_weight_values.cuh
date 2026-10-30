#pragma once
#include "architecture_policy.cuh"
#include <stdint.h>

#if PHOENIX_STYLE_ARCH < 890 || defined(PHOENIX_ARCHITECTURE_TEST)
namespace phoenix_architecture {
namespace weight_values {
// E4M3 has only 256 encodings. This is a floating-point format vocabulary,
// not trained weights or a per-model expanded copy. Compute its 512 bytes at
// compile time; the module loader installs the immutable data once.
constexpr uint16_t half_bits(unsigned code) {
    const unsigned magnitude = code & 127u, sign = (code & 128u) << 8;
    if (magnitude == 127u) return uint16_t(sign | 0x7e00u);
    if (magnitude >= 8u) return uint16_t(sign | ((magnitude << 7) + 0x2000u));
    constexpr uint16_t subnormals[8] = {
        0, 0x1800, 0x1c00, 0x1e00, 0x2000, 0x2100, 0x2200, 0x2300};
    return uint16_t(sign | subnormals[magnitude]);
}
struct Table { uint16_t value[256]; };
constexpr Table make_table() {
    Table result{};
    for (unsigned code = 0; code < 256; ++code) result.value[code] = half_bits(code);
    return result;
}
// Deliberately global/read-only, not __constant__: a warp's different weight
// bytes must not serialize through the constant-memory broadcast mechanism.
__device__ const Table decoded = make_table();
} // namespace weight_values

__device__ __forceinline__ void decode_fp8_weight_fragment(
    uint32_t word, uint32_t& even, uint32_t& odd) {
    even = uint32_t(weight_values::decoded.value[word & 255u]) |
           (uint32_t(weight_values::decoded.value[(word >> 16) & 255u]) << 16);
    odd = uint32_t(weight_values::decoded.value[(word >> 8) & 255u]) |
          (uint32_t(weight_values::decoded.value[word >> 24]) << 16);
}
} // namespace phoenix_architecture
#endif
