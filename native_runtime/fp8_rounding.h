#pragma once
namespace phoenix_cpp {
// Publish on the E4M3 grid while retaining FP32 storage. Normals discard twenty
// mantissa bits with a ties-to-even bias; subnormals use the fixed 1/512 step.
// Saturation and zero signs match the portable arithmetic, including NaN -> +448.
template <class FloatBits> PHX_DEVICE static float round_e4m3_bits(float input) {
    const uint32_t sign = input < 0 ? 0x80000000u : 0;
    const uint32_t magnitude = FloatBits::bits(input) & 0x7fffffffu;
    if (magnitude >= 0x43e00000u)
        return FloatBits::value(sign | 0x43e00000u);
    if (magnitude >= 0x3c800000u)
        return FloatBits::value(sign |
                                ((magnitude + 0x7ffffu + ((magnitude >> 20) & 1u)) & 0x7ff00000u));
    if (magnitude < 0x3a800000u)
        return FloatBits::value(sign);
    const unsigned shift = 141u - (magnitude >> 23);
    const uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    const uint32_t rounded =
        (mantissa + ((1u << (shift - 1u)) - 1u) + ((mantissa >> shift) & 1u)) >> shift;
    return FloatBits::value(sign | FloatBits::bits(float(rounded) * 0.001953125f));
}
}
