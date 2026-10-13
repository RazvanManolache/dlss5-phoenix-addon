#pragma once

// Shared-memory fusion for one 64-token, 32-channel window. Keep the original
// operation descriptors so every publication, learned gain and residual keeps
// its existing arithmetic. The matcher rejects other layouts rather than
// silently substituting this narrow implementation for a generic operation.
struct WindowBlockOperations {
    phx::Operation expand, contract, quantize, project_qkv;
    phx::Operation normalize_query, normalize_key, split_value, attention, project_output;
};

static bool match_window_block(const std::vector<Command>& commands, size_t begin, size_t end,
                               WindowBlockOperations& block) {
    if (end - begin < 9)
        return false;
    block = {commands[begin].operation,     commands[begin + 1].operation,
             commands[begin + 2].operation, commands[begin + 3].operation,
             commands[begin + 4].operation, commands[begin + 5].operation,
             commands[begin + 6].operation, commands[begin + 7].operation,
             commands[begin + 8].operation};
    const unsigned rows = block.expand.row_count;
    constexpr unsigned PackedMatrix = 1 | phx::RightInputFp8 | phx::LeftInputFp8Grid;
    auto matrix_matches = [rows](const phx::Operation& operation, unsigned columns,
                                 unsigned reduction, unsigned flags) {
        return operation.family == phx::MatrixMultiply && operation.batch_count == 1 &&
               operation.row_count == rows && operation.column_count == columns &&
               operation.reduction_count == reduction && operation.flags == flags &&
               operation.left_row_stride == reduction && operation.right_row_stride == reduction &&
               operation.output_row_stride == columns;
    };
    const unsigned contract_flags =
        PackedMatrix | phx::MatrixResidual | (block.contract.flags & phx::MatrixSkipHalf);
    if (!rows || rows % 64 ||
        !matrix_matches(block.expand, 128, 32,
                        PackedMatrix | phx::StoreHalf | (phx::PublishFusedGateFp8 << 8)) ||
        !matrix_matches(block.contract, 32, 128, contract_flags) ||
        !matrix_matches(block.project_qkv, 96, 32, PackedMatrix) ||
        !matrix_matches(block.project_output, 32, 32, PackedMatrix | phx::MatrixResidual))
        return false;
    const auto& query = block.normalize_query;
    const auto& key = block.normalize_key;
    return block.contract.left_input == block.expand.output &&
           block.quantize.family == phx::Elementwise &&
           block.quantize.flags ==
               (phx::CopyValues | phx::StoreHalf | (phx::PublishHalfThenFp8 << 8)) &&
           block.quantize.left_input == block.contract.output &&
           commands[begin + 2].count == uint64_t(rows) * 32 &&
           block.project_qkv.left_input == block.quantize.output &&
           query.family == phx::RowReduction && key.family == phx::RowReduction &&
           query.flags == (phx::StoreHalf | phx::InterleavedQkv) &&
           key.flags == (phx::StoreHalf | phx::InterleavedQkv | (1u << 18)) &&
           query.row_count == rows && key.row_count == rows && query.column_count == 64 &&
           key.column_count == 64 && query.batch_count == 1 && key.batch_count == 1 &&
           query.reduction_count == 1 && key.reduction_count == 0 &&
           query.left_input == block.project_qkv.output && key.left_input == query.left_input &&
           block.split_value.family == phx::Elementwise &&
           block.split_value.flags ==
               (phx::SplitQkvHeads | phx::StoreHalf | (phx::PublishHalfThenFp8 << 8)) &&
           block.split_value.left_input == query.left_input &&
           block.split_value.column_count == 32 && block.split_value.batch_count == 1 &&
           block.split_value.left_batch_stride == 64 && block.split_value.right_batch_stride == 2 &&
           block.attention.family == phx::WindowAttention && block.attention.flags == 0 &&
           block.attention.column_count == 1 && block.attention.batch_count == rows / 64 &&
           block.attention.left_input == query.output &&
           block.attention.right_input == key.output &&
           block.attention.auxiliary == block.split_value.output &&
           block.project_output.left_input == block.attention.output &&
           block.project_output.auxiliary == block.contract.output;
}

__device__ __forceinline__ phx::Operation window_matrix(phx::Operation source, unsigned columns,
                                                        unsigned reduction, unsigned flags) {
    source.family = phx::MatrixMultiply;
    source.flags = flags;
    source.row_count = 64;
    source.column_count = columns;
    source.reduction_count = reduction;
    source.batch_count = 1;
    source.left_batch_stride = source.right_batch_stride = source.output_batch_stride = 0;
    source.left_row_stride = reduction;
    source.right_row_stride = reduction;
    source.output_row_stride = columns;
    return source;
}

template <bool SkipHalf>
__global__ void window_block_shared(WindowBlockOperations block, uint64_t first_window) {
    // Projected QKV (24 KiB) reuses the MLP expansion and attention scratch.
    // Residuals remain live until the final projection. Q becomes context only
    // after Q*K has consumed it. Total shared storage is 48 KiB per CTA.
    __shared__ __align__(32) float projected[64 * 96];
    __shared__ __align__(32) float residual[64 * 32];
    __shared__ __align__(32) uint16_t quantized_residual[64 * 32];
    __shared__ __align__(32) uint16_t query[64 * 32], key[64 * 32], value[64 * 32];
    const uint64_t first_row = (first_window + blockIdx.x) * 64;

    constexpr unsigned PackedMatrix = 1 | phx::RightInputFp8 | phx::LeftInputFp8Grid;
    auto expand = window_matrix(block.expand, 128, 32,
                                PackedMatrix | phx::StoreHalf | (phx::PublishFusedGateFp8 << 8));
    expand.left_input = static_cast<uint16_t*>(expand.left_input) + first_row * 32;
    expand.output = projected;
    matrix_direct_fp8_tile<16, phx::PublishFusedGateFp8>(expand, 0);
    __syncthreads();

    auto contract =
        window_matrix(block.contract, 32, 128,
                      PackedMatrix | phx::MatrixResidual | (SkipHalf ? phx::MatrixSkipHalf : 0));
    contract.left_input = projected;
    contract.output = residual;
    constexpr unsigned skip_bytes = SkipHalf ? 2 : 4;
    contract.auxiliary = static_cast<uint8_t*>(contract.auxiliary) + first_row * 32 * skip_bytes;
    matrix_direct_fp8_tile<4, phx::KeepFloat>(contract, 0);
    __syncthreads();

    auto quantize = block.quantize;
    quantize.family = phx::Elementwise;
    quantize.flags = phx::CopyValues | phx::StoreHalf | (phx::PublishHalfThenFp8 << 8);
    quantize.left_input = residual;
    quantize.output = quantized_residual;
    for (unsigned element = threadIdx.x; element < 64 * 32; element += blockDim.x)
        phx::Kernel<GpuMath>{quantize}.unary(element);
    __syncthreads();

    auto project_qkv = window_matrix(block.project_qkv, 96, 32, PackedMatrix);
    project_qkv.left_input = quantized_residual;
    project_qkv.output = projected;
    matrix_direct_fp8_tile<12, phx::KeepFloat>(project_qkv, 0);
    __syncthreads();

    auto normalize_query = block.normalize_query;
    auto normalize_key = block.normalize_key;
    normalize_query.family = normalize_key.family = phx::RowReduction;
    normalize_query.flags = phx::StoreHalf | phx::InterleavedQkv;
    normalize_key.flags = phx::StoreHalf | phx::InterleavedQkv | (1u << 18);
    normalize_query.column_count = normalize_key.column_count = 64;
    normalize_query.batch_count = normalize_key.batch_count = 1;
    normalize_query.reduction_count = 1;
    normalize_key.reduction_count = 0;
    normalize_query.left_input = normalize_key.left_input = projected;
    normalize_query.output = query;
    normalize_key.output = key;
    for (unsigned row = threadIdx.x / 8; row < 64; row += blockDim.x / 8) {
        normalize_head_row8(normalize_query, row, threadIdx.x % 8);
        normalize_head_row8(normalize_key, row, threadIdx.x % 8);
    }
    auto split_value = block.split_value;
    split_value.family = phx::Elementwise;
    split_value.flags = phx::StoreHalf | (phx::PublishHalfThenFp8 << 8);
    split_value.left_input = projected;
    split_value.output = value;
    for (unsigned element = threadIdx.x; element < 64 * 32; element += blockDim.x)
        phx::Kernel<GpuMath>{split_value}.store(element,
                                                projected[(element / 32) * 96 + 64 + element % 32]);
    __syncthreads();

    auto attention = block.attention;
    attention.column_count = attention.batch_count = 1;
    attention.left_input = query;
    attention.right_input = key;
    attention.auxiliary = value;
    attention.output = query;
    // There is one head, so every window uses the same relative-position bias.
    window_attention_fp8_tile(attention, 0, projected,
                              reinterpret_cast<uint8_t*>(projected + 4096));
    __syncthreads();

    auto project_output =
        window_matrix(block.project_output, 32, 32, PackedMatrix | phx::MatrixResidual);
    project_output.left_input = query;
    project_output.auxiliary = residual;
    project_output.output = static_cast<float*>(project_output.output) + first_row * 32;
    matrix_direct_fp8_tile<4, phx::KeepFloat>(project_output, 0);
}
