// Direct E4M3 tensor instructions for inputs already published on the FP8 grid.
// Fragment coordinates follow NVIDIA's documented mma.m16n8k32 lane mapping:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k32
// Packed model bytes enter registers directly; no expanded weight cache is kept.
__device__ __forceinline__ uint32_t pack_half4_fp8(const __half* source) {
    const auto* pairs = reinterpret_cast<const __half2*>(source);
    const unsigned low =
        __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(pairs[0]), __NV_SATFINITE, __NV_E4M3);
    const unsigned high =
        __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(pairs[1]), __NV_SATFINITE, __NV_E4M3);
    return low | (high << 16);
}

// Evaluate the reference matrix epilogue before a paired half write. Keep
// residual multiply/add separate, exactly as Kernel::store does.
__device__ __forceinline__ float fp8_matrix_published_value(const phx::Kernel<GpuMath>& math,
                                                            const phx::Operation& operation,
                                                            uint64_t output_index, float value) {
    if (operation.flags & phx::MatrixResidual) {
        const float skip = (operation.flags & phx::MatrixSkipHalf)
                               ? math.load_half(operation.auxiliary, output_index)
                               : math.load_float(operation.auxiliary, output_index);
        const float product = skip * math.load_float(operation.secondary_output,
                                                     output_index % operation.column_count);
        value = value + product;
    }
    return math.publish((operation.flags >> 8u) & 15u, value);
}

template <unsigned ColumnTiles, unsigned PublishMode = UINT32_MAX>
__device__ __forceinline__ void matrix_direct_fp8_tile(phx::Operation operation, uint64_t tile) {
    // Fix known matrix/output semantics before device compilation. The common
    // half/FP8 and gated-half/FP8 variants omit unrelated publication branches.
    operation.family = phx::MatrixMultiply;
    if constexpr (PublishMode != UINT32_MAX)
        operation.flags = (operation.flags & ~0xf00u) | (PublishMode << 8u);
    const unsigned column_tiles =
        (operation.column_count + ColumnTiles * 8 - 1) / (ColumnTiles * 8);
    const unsigned row_tiles = (operation.row_count + 63) / 64;
    const uint64_t batch = tile / (uint64_t(column_tiles) * row_tiles);
    const unsigned first_row = unsigned(tile / column_tiles % row_tiles) * 64;
    const unsigned first_column = unsigned(tile % column_tiles) * ColumnTiles * 8;
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned row_group = lane / 4, reduction_group = lane % 4;
    const auto* activations =
        static_cast<const __half*>(operation.left_input) + batch * operation.left_batch_stride;
    const auto* weights =
        static_cast<const uint8_t*>(operation.right_input) + batch * operation.right_batch_stride;
    float sums[ColumnTiles][4] = {};
    for (unsigned reduction = 0; reduction < operation.reduction_count; reduction += 32) {
        uint32_t left[4];
#pragma unroll
        for (unsigned fragment = 0; fragment < 4; ++fragment) {
            const unsigned row = first_row + warp * 16 + row_group + (fragment % 2) * 8;
            const unsigned offset = reduction + reduction_group * 4 + (fragment / 2) * 16;
            left[fragment] =
                row < operation.row_count
                    ? pack_half4_fp8(activations + uint64_t(row) * operation.left_row_stride +
                                     offset)
                    : 0;
        }
#pragma unroll
        for (unsigned column_tile = 0; column_tile < ColumnTiles; ++column_tile) {
            const unsigned column = first_column + column_tile * 8 + row_group;
            uint32_t right[2] = {};
            if (column < operation.column_count) {
                const auto* source = weights + uint64_t(column) * operation.right_row_stride +
                                     reduction + reduction_group * 4;
                right[0] = *reinterpret_cast<const uint32_t*>(source);
                right[1] = *reinterpret_cast<const uint32_t*>(source + 16);
            }
#if __CUDA_ARCH__ >= 890
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                         "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+f"(sums[column_tile][0]), "+f"(sums[column_tile][1]),
                           "+f"(sums[column_tile][2]), "+f"(sums[column_tile][3])
                         : "r"(left[0]), "r"(left[1]), "r"(left[2]), "r"(left[3]), "r"(right[0]),
                           "r"(right[1]));
#endif
        }
    }
    phx::Kernel<GpuMath> math{operation};
    if constexpr (PublishMode == phx::PublishFusedGateFp8 ||
                  PublishMode == phx::PublishHalfThenFp8 ||
                  PublishMode == phx::PublishGateThenHalfFp8) {
        // Each lane owns two adjacent columns in each of two rows. Publish
        // those values independently, then write their two halves as one word.
        // Odd/tail layouts retain the scalar contract without an overrun.
        const bool paired_output = (operation.flags & phx::StoreHalf) &&
                                   operation.output_row_stride % 2 == 0 &&
                                   operation.output_batch_stride % 2 == 0 &&
                                   (reinterpret_cast<uintptr_t>(operation.output) & 3u) == 0;
        if (paired_output) {
#pragma unroll
            for (unsigned column_tile = 0; column_tile < ColumnTiles; ++column_tile)
#pragma unroll
                for (unsigned row_in_lane = 0; row_in_lane < 2; ++row_in_lane) {
                    const unsigned row = first_row + warp * 16 + row_group + row_in_lane * 8;
                    const unsigned column = first_column + column_tile * 8 + reduction_group * 2;
                    const uint64_t output_index = batch * operation.output_batch_stride +
                                                  uint64_t(row) * operation.output_row_stride +
                                                  column;
                    if (row >= operation.row_count || column >= operation.column_count)
                        continue;
                    const unsigned first_element = row_in_lane * 2;
                    if (column + 1 < operation.column_count) {
                        const uint32_t low = math.half_bits(fp8_matrix_published_value(
                            math, operation, output_index, sums[column_tile][first_element]));
                        const uint32_t high = math.half_bits(
                            fp8_matrix_published_value(math, operation, output_index + 1,
                                                       sums[column_tile][first_element + 1]));
                        auto* destination = static_cast<uint16_t*>(operation.output) + output_index;
                        *reinterpret_cast<uint32_t*>(destination) = low | (high << 16u);
                    } else {
                        math.store(output_index, sums[column_tile][first_element]);
                    }
                }
            return;
        }
    }
#pragma unroll
    for (unsigned column_tile = 0; column_tile < ColumnTiles; ++column_tile)
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) {
            const unsigned row = first_row + warp * 16 + row_group + (element / 2) * 8;
            const unsigned column =
                first_column + column_tile * 8 + reduction_group * 2 + element % 2;
            if (row < operation.row_count && column < operation.column_count)
                math.store(batch * operation.output_batch_stride +
                               uint64_t(row) * operation.output_row_stride + column,
                           sums[column_tile][element]);
        }
}

template <unsigned ColumnTiles, unsigned PublishMode = UINT32_MAX>
__global__ void matrix_direct_fp8(phx::Operation operation, uint64_t first_tile) {
    matrix_direct_fp8_tile<ColumnTiles, PublishMode>(operation, first_tile + blockIdx.x);
}

// A CTA owns 64 tokens throughout a feed-forward block. Intermediate published
// activations stay in shared memory instead of taking a round trip through the
// scratch arena between each GEMM. The graph's original publication/residual
// operations still define the arithmetic; no activation or rounding is removed.
template <bool HasHeadProjection>
__global__ void feed_forward_shared(phx::Operation expand, phx::Operation middle,
                                    phx::Operation contract, uint64_t first_row_tile) {
    extern __shared__ __align__(32) uint16_t temporary[];
    const uint64_t first_row = (first_row_tile + blockIdx.x) * 64;
    const unsigned live_rows = unsigned(min(uint64_t(64), uint64_t(expand.row_count) - first_row));
    const unsigned channels = contract.column_count;
    const auto original_expand = expand;
    expand.left_input =
        static_cast<uint16_t*>(expand.left_input) + first_row * expand.left_row_stride;
    expand.output = temporary;
    expand.row_count = live_rows;
    expand.output_row_stride = expand.column_count;
    expand.left_batch_stride = expand.right_batch_stride = expand.output_batch_stride = 0;
    expand.flags = 1 | phx::RightInputFp8 | phx::LeftInputFp8Grid | phx::StoreHalf |
                   (phx::PublishFusedGateFp8 << 8);
    // Reuse each activation fragment across 128 output columns. Iterating
    // 32-column tiles here would reload it four times inside the same CTA.
    for (unsigned column_tile = 0; column_tile < (expand.column_count + 127) / 128; ++column_tile)
        matrix_direct_fp8_tile<16, phx::PublishFusedGateFp8>(expand, column_tile);
    __syncthreads();
    uint16_t* contracted_input = temporary;
    if constexpr (HasHeadProjection) {
        contracted_input = temporary + 64 * original_expand.column_count;
        middle.left_input = temporary;
        middle.output = contracted_input;
        middle.row_count = live_rows;
        // The head batch advances across channels in both shared tensors.
        // Preserve the strided-batch descriptor recorded by Graph.
        for (unsigned head = 0; head < middle.batch_count; ++head)
            matrix_direct_fp8_tile<4, phx::PublishHalfThenFp8>(middle, head);
        __syncthreads();
    }
    contract.left_input = contracted_input;
    contract.left_row_stride = channels == 32 ? original_expand.column_count : channels;
    contract.row_count = live_rows;
    const unsigned output_bytes = (contract.flags & phx::StoreHalf) ? 2 : 4;
    contract.output = static_cast<uint8_t*>(contract.output) +
                      first_row * contract.output_row_stride * output_bytes;
    if (contract.flags & phx::MatrixResidual) {
        const unsigned skip_bytes = (contract.flags & phx::MatrixSkipHalf) ? 2 : 4;
        contract.auxiliary = static_cast<uint8_t*>(contract.auxiliary) +
                             first_row * contract.output_row_stride * skip_bytes;
    }
    contract.left_batch_stride = contract.right_batch_stride = contract.output_batch_stride = 0;
    matrix_direct_fp8_tile<8, phx::KeepFloat>(contract, 0);
}

// Keep the fully generic entry for other publication modes and experiments.
// This selection happens during graph capture, not on each captured replay.
template <unsigned ColumnTiles>
static void launch_matrix_direct_fp8(phx::Operation operation, uint64_t first_tile, unsigned blocks,
                                     cudaStream_t stream) {
    switch ((operation.flags >> 8u) & 15u) {
    case phx::PublishFusedGateFp8:
        matrix_direct_fp8<ColumnTiles, phx::PublishFusedGateFp8>
            <<<blocks, 128, 0, stream>>>(operation, first_tile);
        break;
    case phx::PublishHalfThenFp8:
        matrix_direct_fp8<ColumnTiles, phx::PublishHalfThenFp8>
            <<<blocks, 128, 0, stream>>>(operation, first_tile);
        break;
    case phx::PublishGateThenHalfFp8:
        matrix_direct_fp8<ColumnTiles, phx::PublishGateThenHalfFp8>
            <<<blocks, 128, 0, stream>>>(operation, first_tile);
        break;
    default:
        matrix_direct_fp8<ColumnTiles><<<blocks, 128, 0, stream>>>(operation, first_tile);
    }
}
