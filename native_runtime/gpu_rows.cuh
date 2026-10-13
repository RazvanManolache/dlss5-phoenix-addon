#pragma once
// Eight lanes own a row, keeping four/eight values per lane in registers.
// Shuffle only exchanges partials: the reference reduction and rounding order
// remain explicit rather than replacing them with an unordered warp sum.
__device__ __forceinline__ float row_member(float value, unsigned lane) {
#if defined(PHX_CUDA)
    return __shfl_sync(__activemask(), value, lane, 8);
#else
    return __shfl(value, lane, 8);
#endif
}

__device__ __forceinline__ void normalize_head_row8(phx::Operation operation, uint64_t row,
                                                    unsigned lane) {
    phx::Kernel<GpuMath> math{operation};
    float values[4];
    const unsigned channels[4] = {lane, lane + 16, lane + 8, lane + 24};
    float partial = 0;
#pragma unroll
    for (unsigned index = 0; index < 4; ++index) {
        values[index] = math.load_query_or_key(unsigned(row), channels[index]);
        partial = fmaf(values[index], values[index], partial);
    }
    const float even = math.half((row_member(partial, 0) + row_member(partial, 2)) +
                                 (row_member(partial, 4) + row_member(partial, 6)));
    const float odd = math.half((row_member(partial, 1) + row_member(partial, 3)) +
                                (row_member(partial, 5) + row_member(partial, 7)));
    const float total = fmaxf(math.half(even + odd), 0.00006198883056640625f);
    float root = sqrtf(total);
    const float error = fmaf(-root, root, total);
    root += error / (2.f * root);
    float scale = isinf(total) ? 0.f : math.reciprocal(root);
    if (operation.reduction_count)
        scale *= math.load_float(operation.auxiliary,
                                 (row / operation.column_count) % operation.batch_count);
#pragma unroll
    for (unsigned index = 0; index < 4; ++index)
        math.store(row * 32 + channels[index], math.quantize_e4m3(values[index] * scale));
}

__global__ void normalize_heads8(phx::Operation operation, uint64_t count, uint64_t first) {
    const uint64_t row = first + uint64_t(blockIdx.x) * 32 + threadIdx.x / 8;
    if (row < first + count)
        normalize_head_row8(operation, row, threadIdx.x % 8);
}

__global__ void normalize_global_scores8(phx::Operation operation, uint64_t count, uint64_t first) {
    const uint64_t row = first + uint64_t(blockIdx.x) * 32 + threadIdx.x / 8;
    if (row >= first + count)
        return;
    const unsigned lane = threadIdx.x % 8;
    phx::Kernel<GpuMath> math{operation};
    const unsigned stride =
        operation.left_batch_stride ? operation.left_batch_stride : operation.column_count;
    const unsigned blocks = (operation.column_count + 63) / 64;
    const uint64_t base = row * stride;
    const float zero = math.global_attention_weight(0.f);
    float total = 0;
    for (unsigned block = 0; block < blocks; ++block) {
        float pairs[4];
#pragma unroll
        for (unsigned pair = 0; pair < 4; ++pair) {
            unsigned key = block * 64 + lane + pair * 16;
            const float left = key < operation.column_count
                                   ? math.global_attention_weight(
                                         math.load_float(operation.left_input, base + key))
                                   : zero;
            const float right = key + 8 < operation.column_count
                                    ? math.global_attention_weight(
                                          math.load_float(operation.left_input, base + key + 8))
                                    : zero;
            pairs[pair] = math.add_half(left, right);
            if (key < stride)
                math.store(base + key,
                           key < operation.column_count ? math.quantize_e4m3(left) : 0.f);
            if (key + 8 < stride)
                math.store(base + key + 8,
                           key + 8 < operation.column_count ? math.quantize_e4m3(right) : 0.f);
        }
        const float column =
            math.add_half(math.add_half(math.add_half(pairs[0], pairs[1]), pairs[2]), pairs[3]);
        const float even =
            math.add_half(math.add_half(math.add_half(row_member(column, 0), row_member(column, 2)),
                                        row_member(column, 4)),
                          row_member(column, 6));
        const float odd =
            math.add_half(math.add_half(math.add_half(row_member(column, 1), row_member(column, 3)),
                                        row_member(column, 5)),
                          row_member(column, 7));
        total = math.add_half(total, math.add_half(even, odd));
    }
    const unsigned padding = blocks * 64 - operation.column_count;
    if (padding)
        total = math.add_half(total, -math.multiply_half(zero, float(padding)));
    if (lane == 0)
        static_cast<float*>(operation.auxiliary)[row] = math.half(1.f / total);
    for (unsigned key = blocks * 64 + lane; key < stride; key += 8)
        math.store(base + key, 0.f);
}

// The local attention row has 64 keys. Eight lanes retain eight weights each,
// rather than one thread holding the entire row and nested reduction arrays.
// Shuffle exchanges preserve the reference tree, including half-rounded sums.
__global__ void normalize_window_scores8(phx::Operation operation, uint64_t count, uint64_t first) {
    const uint64_t row = first + uint64_t(blockIdx.x) * 32 + threadIdx.x / 8;
    if (row >= first + count)
        return;
    const unsigned lane = threadIdx.x % 8;
    const uint64_t base = row * 64,
                   bias_base = (row / 64 % operation.batch_count) * 4096 + row % 64 * 64;
    const bool split_heads = operation.batch_count == 16,
               has_bias = (operation.flags & 0x2000u) != 0;
    phx::Kernel<GpuMath> math{operation};
    float weights[8];
#pragma unroll
    for (unsigned index = 0; index < 8; ++index) {
        const unsigned key = lane + index * 8;
        float ramp =
            split_heads
                ? math.half(fmaf(
                      math.half(math.load_float(operation.left_input, base + key) +
                                (has_bias ? math.load_float(operation.right_input, bias_base + key)
                                          : 0.f)),
                      .044921875f, 1.30078125f))
                : math.half(
                      fmaf(math.load_float(operation.left_input, base + key), .044921875f,
                           fmaf((has_bias ? math.load_float(operation.right_input, bias_base + key)
                                          : 0.f),
                                .044921875f, 1.30078125f)));
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
        math.store(base + lane + index * 8,
                   math.quantize_e4m3(math.half(weights[index] * inverse)));
}
