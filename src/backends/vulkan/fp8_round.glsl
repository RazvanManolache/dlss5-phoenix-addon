// Publish on the finite E4M3 grid while retaining FP32 storage. Powers of two
// make scaling exact; roundEven supplies the model's ties-to-even boundary.
float e4m3(float value) {
    uint raw_bits = floatBitsToUint(value);
    uint magnitude_bits = raw_bits & 0x7fffffffu;
    uint clamped_bits = min(magnitude_bits, 0x43e00000u); // Finite saturation at 448.
    uint exponent = max(clamped_bits >> 23u, 121u) - 3u;
    // Below the normal E4M3 range, every value uses the fixed 2^-9 step.
    float step = uintBitsToFloat(exponent << 23u);
    float reciprocal = uintBitsToFloat((254u - exponent) << 23u);
    float rounded = roundEven(uintBitsToFloat(clamped_bits) * reciprocal) * step;
    // Preserve negative underflow even when a driver flushes FP32 subnormals.
    // Input zero and NaN keep the portable canonical positive sign.
    uint sign = magnitude_bits != 0u && magnitude_bits <= 0x7f800000u ? raw_bits & 0x80000000u : 0u;
    return uintBitsToFloat(floatBitsToUint(rounded) | sign);
}
