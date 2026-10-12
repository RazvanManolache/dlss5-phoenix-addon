#include "gpu_rows.cuh"

// Used inside fused attention after its Q*K^T tile has reached shared memory.
template <class Probability>
__device__ __forceinline__ void normalize_window_row8(const float* scores, const float* bias,
                                                      Probability* probabilities, unsigned lane,
                                                      bool split_heads) {
    phx::Kernel<GpuMath> math{};
    float weights[8];
#pragma unroll
    for (unsigned index = 0; index < 8; ++index) {
        const unsigned key = lane + index * 8;
        float ramp =
            split_heads
                ? math.half(fmaf(math.half(scores[key] + bias[key]), .044921875f, 1.30078125f))
                : math.half(
                      fmaf(scores[key], .044921875f, fmaf(bias[key], .044921875f, 1.30078125f)));
        ramp = fminf(fmaxf(ramp, 1.03125f), 1.5693359375f);
        weights[index] = math.half_value((math.half_bits(ramp) & 1023u) << 5u);
    }
    float total;
    if (split_heads) {
        const float even = ((weights[0] + weights[2]) + weights[4]) + weights[6];
        const float odd = ((weights[1] + weights[3]) + weights[5]) + weights[7];
        const float even_total = ((row_member(even, 0) + row_member(even, 1)) +
                                  (row_member(even, 2) + row_member(even, 3))) +
                                 ((row_member(even, 4) + row_member(even, 5)) +
                                  (row_member(even, 6) + row_member(even, 7)));
        const float odd_total =
            ((row_member(odd, 0) + row_member(odd, 1)) +
             (row_member(odd, 2) + row_member(odd, 3))) +
            ((row_member(odd, 4) + row_member(odd, 5)) + (row_member(odd, 6) + row_member(odd, 7)));
        total = even_total + odd_total;
    } else {
        const float even = math.add_half(
            math.add_half(math.add_half(weights[0], weights[2]), weights[4]), weights[6]);
        const float odd = math.add_half(
            math.add_half(math.add_half(weights[1], weights[3]), weights[5]), weights[7]);
        const float parity =
            math.add_half(math.add_half(math.add_half(even, row_member(even, lane % 4 + 4)), odd),
                          row_member(odd, lane % 4 + 4));
        total = math.add_half(math.add_half(row_member(parity, 0), row_member(parity, 1)),
                              math.add_half(row_member(parity, 2), row_member(parity, 3)));
    }
    const float inverse = math.reciprocal(fmaxf(total, .00006103515625f));
#pragma unroll
    for (unsigned index = 0; index < 8; ++index)
        if constexpr (sizeof(Probability) == 1)
            probabilities[lane + index * 8] = __nv_cvt_float_to_fp8(
                math.half(weights[index] * inverse), __NV_SATFINITE, __NV_E4M3);
        else
            probabilities[lane + index * 8] =
                __float2half_rn(math.quantize_e4m3(math.half(weights[index] * inverse)));
}
