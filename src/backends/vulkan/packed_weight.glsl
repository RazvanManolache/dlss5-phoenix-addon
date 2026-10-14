// E4M3 weight bytes are loaded through aligned words on every Vulkan GPU.
// Decoding is exact; multiplication and accumulation retain the existing path.
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer WeightWords {
    uint v[];
};
// Decode a finite E4M3 byte exactly. Matrix loaders can reuse one aligned word
// for four weights rather than repeat the word fetch for each byte.
float decoded_weight_byte(uint code) {
    uint e = (code >> 3u) & 15u, m = code & 7u, sign = (code & 128u) << 24u;
    uint bits =
        e != 0u ? ((e + 120u) << 23u) | (m << 20u) : floatBitsToUint(float(m) * 0.001953125);
    return uintBitsToFloat(sign | bits);
}
float matrix_b(uint index) {
    if ((operation_flags() & 0x2000000u) == 0u)
        return float(pc.b.v[index]);
    uint code = (WeightWords(uint64_t(pc.b)).v[index >> 2u] >> ((index & 3u) * 8u)) & 255u;
    return decoded_weight_byte(code);
}
