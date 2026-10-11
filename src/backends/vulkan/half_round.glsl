// Phoenix FP32 -> FP16-grid rounding with integer round-to-nearest-even.
// Kept in FP32 storage: no driver-dependent half cast/pack round-trip folding.
float half_round(float value) {
    uint bits = floatBitsToUint(value), sign = bits & 0x80000000u, mag = bits & 0x7fffffffu;
    if (mag >= 0x7f800000u)
        return value; // infinities and NaNs
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
