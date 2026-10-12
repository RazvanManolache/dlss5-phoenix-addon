// E4M3-grid publication in FP32 storage. Keep ties-to-even, finite saturation,
// and the portable zero-sign rule without computing a floating-point scale.
float e4m3(float value) {
    uint raw = floatBitsToUint(value);
    uint magnitude = raw & 0x7fffffffu;
    // Integer sign handling also preserves negative underflow on drivers that
    // flush float32 subnormals during comparisons. Negative zero stays canonical.
    uint sign = magnitude != 0u && magnitude <= 0x7f800000u ? raw & 0x80000000u : 0u;
    if (magnitude >= 0x43e00000u)
        return uintBitsToFloat(sign | 0x43e00000u);
    if (magnitude >= 0x3c800000u)
        return uintBitsToFloat(sign |
                               ((magnitude + 0x7ffffu + ((magnitude >> 20u) & 1u)) & 0x7ff00000u));
    if (magnitude < 0x3a800000u)
        return uintBitsToFloat(sign);
    uint shift = 141u - (magnitude >> 23u);
    uint mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    uint rounded = (mantissa + ((1u << (shift - 1u)) - 1u) + ((mantissa >> shift) & 1u)) >> shift;
    return uintBitsToFloat(sign | floatBitsToUint(float(rounded) * 0.001953125));
}
