#pragma once

// Include after the kernel's memory helper, which supplies guarded FP16 atomics.
struct HalfPairVector {
    uint32_t pair_0;
    uint32_t pair_1;
    uint32_t pair_2;
    uint32_t pair_3;
};

// A coarsest-transformer tile holds 16 tokens, each with 1024 FP16 channels:
// 16 * 1024 / 2 = 8192 packed words. Each lane contributes four words to a
// 128-word feature group. Four groups cover this fragment's output channels.
// Accumulation stays atomic because the feed-forward projection is split across
// producer blocks. Publication fences and counters remain at the call site.
// Valid planned fragments have nonnegative tile/feature indices and a packed
// word offset below 2^30; multiplying that index by four cannot overflow 32 bits.
template <bool CheckBounds>
__device__ __forceinline__ void
accumulate_feed_forward_tile(uint64_t scratch_base, uint32_t token_tile,
                             uint32_t feature_word_offset, uint32_t lane, HalfPairVector group_0,
                             HalfPairVector group_1, HalfPairVector group_2, HalfPairVector group_3,
                             const MemoryRange* memory_ranges, uint32_t* error_counts) {
    const uint64_t first_word = uint64_t(token_tile) * 8192 + feature_word_offset + lane * 4;
    const uint64_t tile_address = scratch_base + first_word * sizeof(uint32_t);
    constexpr uint64_t feature_group_bytes = 128 * sizeof(uint32_t);

    // Keep the four atomic updates in their original group order. Ordinary
    // stores would lose contributions from other producer blocks.
    atomic_add_four_half_pairs<CheckBounds>(tile_address, group_0.pair_0, group_0.pair_1,
                                            group_0.pair_2, group_0.pair_3, memory_ranges,
                                            error_counts);
    atomic_add_four_half_pairs<CheckBounds>(tile_address + feature_group_bytes, group_1.pair_0,
                                            group_1.pair_1, group_1.pair_2, group_1.pair_3,
                                            memory_ranges, error_counts);
    atomic_add_four_half_pairs<CheckBounds>(tile_address + 2 * feature_group_bytes, group_2.pair_0,
                                            group_2.pair_1, group_2.pair_2, group_2.pair_3,
                                            memory_ranges, error_counts);
    atomic_add_four_half_pairs<CheckBounds>(tile_address + 3 * feature_group_bytes, group_3.pair_0,
                                            group_3.pair_1, group_3.pair_2, group_3.pair_3,
                                            memory_ranges, error_counts);
}
