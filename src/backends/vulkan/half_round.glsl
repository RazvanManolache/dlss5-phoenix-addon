// Phoenix FP32 -> FP16-grid rounding with integer round-to-nearest-even.
// Kept in FP32 storage: no driver-dependent half cast/pack round-trip folding.
// Specialization is opt-in and tested per driver. Unknown devices keep the
// integer implementation, including its exact NaN and subnormal behavior.
layout(constant_id = 3) const bool PHOENIX_HARDWARE_HALF = false;
float half_round(float value) {
    uint bits = floatBitsToUint(value), sign = bits & 0x80000000u, mag = bits & 0x7fffffffu;
    if (mag >= 0x7f800000u)
        return value; // infinities and NaNs
    if (PHOENIX_HARDWARE_HALF) {
        // Keep the packed integer observable. Some drivers fold a direct
        // pack/unpack round trip and accidentally remove its rounding boundary.
        uint packed = packHalf2x16(vec2(value, 0.0)) & 65535u;
        uint half_sign = (packed & 0x8000u) << 16u;
        uint half_exponent = (packed >> 10u) & 31u, half_mantissa = packed & 1023u;
        if (half_exponent == 0u)
            return uintBitsToFloat(half_sign |
                                   floatBitsToUint(float(half_mantissa) * 5.9604644775390625e-8));
        if (half_exponent == 31u)
            return uintBitsToFloat(half_sign | 0x7f800000u | (half_mantissa << 13u));
        return uintBitsToFloat(half_sign | ((half_exponent + 112u) << 23u) |
                               (half_mantissa << 13u));
    }
    if (mag >= 0x38800000u) {
        uint rounded = mag + 0xfffu + ((mag >> 13u) & 1u);
        return uintBitsToFloat(sign |
                               (rounded >= 0x47800000u ? 0x7f800000u : rounded & 0xffffe000u));
    }
    if (mag < 0x33000000u)
        return uintBitsToFloat(sign);
    uint shift = 126u - (mag >> 23u), mant = (mag & 0x7fffffu) | 0x800000u;
    uint rounded = (mant + ((1u << (shift - 1u)) - 1u) + ((mant >> shift) & 1u)) >> shift;
    return uintBitsToFloat(sign | floatBitsToUint(float(rounded) * 5.9604644775390625e-8));
}
uint phoenix_half_bits(float value) {
    if (PHOENIX_HARDWARE_HALF && (floatBitsToUint(value) & 0x7fffffffu) < 0x7f800000u)
        return packHalf2x16(vec2(value, 0.0)) & 65535u;
    uint bits = floatBitsToUint(half_round(value)), sign = (bits >> 16u) & 0x8000u,
         mag = bits & 0x7fffffffu;
    if (mag >= 0x7f800000u)
        return sign | 0x7c00u | (mag > 0x7f800000u ? 0x200u : 0u);
    if (mag == 0u)
        return sign;
    uint exp = mag >> 23u;
    if (exp < 113u)
        return sign | (((mag & 0x7fffffu) | 0x800000u) >> (126u - exp));
    return sign | ((exp - 112u) << 10u) | ((mag >> 13u) & 0x3ffu);
}
uint phoenix_pack_half(vec2 v) {
    return phoenix_half_bits(v.x) | (phoenix_half_bits(v.y) << 16u);
}
float phoenix_half_value(uint bits) {
    if (PHOENIX_HARDWARE_HALF && (bits & 0x7c00u) != 0x7c00u)
        return unpackHalf2x16(bits & 65535u).x;
    uint sign = (bits & 0x8000u) << 16u, exp = (bits >> 10u) & 31u, mant = bits & 1023u;
    if (exp == 0u)
        return uintBitsToFloat(sign | floatBitsToUint(float(mant) * 5.9604644775390625e-8));
    if (exp == 31u)
        return uintBitsToFloat(sign | 0x7f800000u | (mant << 13u));
    return uintBitsToFloat(sign | ((exp + 112u) << 23u) | (mant << 13u));
}
vec2 phoenix_unpack_half(uint bits) {
    return vec2(phoenix_half_value(bits & 65535u), phoenix_half_value(bits >> 16u));
}
