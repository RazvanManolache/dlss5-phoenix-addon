// One block owns one 64-token window/head. Intermediate scores and probabilities
// never enter the device scratch arena. Keep the reference softmax reduction and
// FP16/E4M3 publication boundaries; this is not standard exponential softmax.
__global__ void window_attention(phx::Operation operation, uint64_t first_batch) {
    using namespace nvcuda;
    const uint64_t batch = first_batch + blockIdx.x;
    const unsigned warp = threadIdx.x / 32, first_row = warp * 16;
    const uint64_t offset = batch * 64 * 32;
    const auto* query = static_cast<const __half*>(operation.left_input) + offset;
    const auto* key = static_cast<const __half*>(operation.right_input) + offset;
    const auto* value = static_cast<const __half*>(operation.auxiliary) + offset;
    __shared__ __align__(32) float scores[4096];
    __shared__ __align__(32) __half probabilities[4096];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> left;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> key_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> sums[4];
    for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
        wmma::fill_fragment(sums[column_tile], 0.f);
    for (unsigned reduction = 0; reduction < 32; reduction += 16) {
        wmma::load_matrix_sync(left, query + first_row * 32 + reduction, 32);
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile) {
            wmma::load_matrix_sync(key_fragment, key + column_tile * 16 * 32 + reduction, 32);
            wmma::mma_sync(sums[column_tile], left, key_fragment, sums[column_tile]);
        }
    }
    for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
        wmma::store_matrix_sync(scores + first_row * 64 + column_tile * 16, sums[column_tile], 64,
                                wmma::mem_row_major);
    __syncthreads();
    if (threadIdx.x < 64) {
        phx::Operation normalize{};
        normalize.left_input = scores;
        normalize.right_input = static_cast<float*>(operation.secondary_output) +
                                (batch % operation.column_count) * 4096;
        normalize.output = probabilities;
        normalize.flags = phx::StoreHalf | phx::AddAttentionBias;
        normalize.batch_count = operation.column_count;
        phx::Kernel<GpuMath>{normalize}.softmax(threadIdx.x);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> value_fragment;
    for (unsigned column_tile = 0; column_tile < 2; ++column_tile) {
        wmma::fill_fragment(sums[column_tile], 0.f);
        for (unsigned reduction = 0; reduction < 64; reduction += 16) {
            wmma::load_matrix_sync(left, probabilities + first_row * 64 + reduction, 64);
            wmma::load_matrix_sync(value_fragment, value + reduction * 32 + column_tile * 16, 32);
            wmma::mma_sync(sums[column_tile], left, value_fragment, sums[column_tile]);
        }
        // QK scores are dead after normalization; reuse them for the context tile.
        wmma::store_matrix_sync(scores + first_row * 32 + column_tile * 16, sums[column_tile], 32,
                                wmma::mem_row_major);
    }
    __syncthreads();
    auto publish = operation;
    publish.flags = phx::StoreHalf | (5u << 8);
    phx::Kernel<GpuMath> math{publish};
    for (unsigned element = threadIdx.x; element < 2048; element += blockDim.x) {
        const uint64_t target =
            ((batch / operation.column_count) * 64 + element / 32) * operation.column_count * 32 +
            (batch % operation.column_count) * 32 + element % 32;
        math.store(target, scores[element]);
    }
}
