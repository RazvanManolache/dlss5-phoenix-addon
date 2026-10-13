// The attention inputs and normalized probabilities have already been published
// as E4M3. Use their original precision directly, with FP32 accumulators and the
// same half-rounded normalization tree. Padding shared rows avoids bank aliases.
__device__ __forceinline__ void attention_mma8(float* sum, const uint32_t* left,
                                               const uint32_t* right) {
#if __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(sum[0]), "+f"(sum[1]), "+f"(sum[2]), "+f"(sum[3])
                 : "r"(left[0]), "r"(left[1]), "r"(left[2]), "r"(left[3]), "r"(right[0]),
                   "r"(right[1]));
#endif
}

__device__ __forceinline__ void window_attention_fp8_tile(phx::Operation operation, uint64_t batch,
                                                          float* scores, uint8_t* probabilities) {
    constexpr unsigned SharedStride = 68;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned row_group = lane / 4, reduction_group = lane % 4;
    const auto* query = static_cast<const __half*>(operation.left_input) + batch * 2048;
    const auto* key = static_cast<const __half*>(operation.right_input) + batch * 2048;
    const auto* value = static_cast<const __half*>(operation.auxiliary) + batch * 2048;
    float sums[8][4] = {};
    uint32_t left[4];
#pragma unroll
    for (unsigned fragment = 0; fragment < 4; ++fragment) {
        const unsigned row = warp * 16 + row_group + (fragment % 2) * 8;
        const unsigned reduction = reduction_group * 4 + (fragment / 2) * 16;
        left[fragment] = pack_half4_fp8(query + row * 32 + reduction);
    }
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 8; ++column_tile) {
        const auto* source = key + (column_tile * 8 + row_group) * 32 + reduction_group * 4;
        const uint32_t right[2] = {pack_half4_fp8(source), pack_half4_fp8(source + 16)};
        attention_mma8(sums[column_tile], left, right);
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) {
            const unsigned row = warp * 16 + row_group + (element / 2) * 8;
            const unsigned column = column_tile * 8 + reduction_group * 2 + element % 2;
            scores[row * 64 + column] = sums[column_tile][element];
        }
    }
    __syncthreads();
    for (unsigned row = threadIdx.x / 8; row < 64; row += blockDim.x / 8) {
        const auto* bias = static_cast<const float*>(operation.secondary_output) +
                           ((batch % operation.column_count) * 64 + row) * 64;
        normalize_window_row8(scores + row * 64, bias, probabilities + row * SharedStride,
                              threadIdx.x % 8, operation.column_count == 16);
    }
    __syncthreads();
    // V outlives scores. Transpose it into their unused second half so each
    // MMA B fragment can read four adjacent bytes; context uses the first half.
    auto* transposed_value = reinterpret_cast<uint8_t*>(scores + 2048);
    for (unsigned group = threadIdx.x; group < 512; group += blockDim.x) {
        const uint32_t packed = pack_half4_fp8(value + group * 4);
#pragma unroll
        for (unsigned channel = 0; channel < 4; ++channel)
            transposed_value[((group * 4) % 32 + channel) * SharedStride + group / 8] =
                uint8_t(packed >> (channel * 8));
    }
    __syncthreads();
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
#pragma unroll
        for (unsigned element = 0; element < 4; ++element)
            sums[column_tile][element] = 0;
    for (unsigned reduction = 0; reduction < 64; reduction += 32) {
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment) {
            const unsigned row = warp * 16 + row_group + (fragment % 2) * 8;
            const unsigned offset = reduction + reduction_group * 4 + (fragment / 2) * 16;
            left[fragment] =
                *reinterpret_cast<const uint32_t*>(probabilities + row * SharedStride + offset);
        }
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile) {
            const auto* source = transposed_value + (column_tile * 8 + row_group) * SharedStride +
                                 reduction + reduction_group * 4;
            const uint32_t right[2] = {*reinterpret_cast<const uint32_t*>(source),
                                       *reinterpret_cast<const uint32_t*>(source + 16)};
            attention_mma8(sums[column_tile], left, right);
        }
    }
    auto publish = operation;
    publish.flags = phx::StoreHalf | (5u << 8);
    phx::Kernel<GpuMath> math{publish};
#pragma unroll
    for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) {
            const unsigned row = warp * 16 + row_group + (element / 2) * 8;
            const unsigned column = column_tile * 8 + reduction_group * 2 + element % 2;
            const uint64_t target =
                ((batch / operation.column_count) * 64 + row) * operation.column_count * 32 +
                (batch % operation.column_count) * 32 + column;
            math.store(target, sums[column_tile][element]);
        }
}

__global__ void window_attention_fp8(phx::Operation operation, uint64_t first_batch) {
    __shared__ __align__(32) float scores[4096];
    __shared__ __align__(32) uint8_t probabilities[64 * 68];
    window_attention_fp8_tile(operation, first_batch + blockIdx.x, scores, probabilities);
}
