#pragma once
#include "signed_division.cuh"
#include "warp_collectives.cuh"

// A block index is flattened as [group, tile]. Keep division separate from
// matrix operands so address setup cannot silently change their live ranges.
struct TokenTileWork {
    uint32_t group, tile;
};
__device__ __forceinline__ TokenTileWork token_tile_work(uint32_t block_index,
                                                       uint32_t tile_count) {
    const uint32_t group = signed_tile_quotient(block_index, tile_count);
    return {group, block_index - group * tile_count};
}

struct CopyWarpSchedule {
    bool without_block_leader;
    uint32_t barrier_byte_offset;
    uint32_t wait_word_low, wait_word_high;
};

// Call at a converged setup point, before any copy/barrier initialization.
// This preserves the existing wait-word encoding; it is not the opaque state
// returned later by mbarrier.arrive. All current call sites have blockDim.z=1.
__device__ __forceinline__ CopyWarpSchedule
copy_warp_schedule(bool block_leader, uint32_t block_threads, uint32_t barrier_offset) {
    const bool without_leader = vote_warp_predicate<true>(!block_leader);
    const uint32_t elected_threads = broadcast_warp_word(block_threads, block_leader, 0u);
    const uint32_t elected_offset = broadcast_warp_word(barrier_offset, block_leader, 0u);
    uint32_t wait_low = 0, wait_high = 0;
    if (!without_leader) {
        constexpr uint32_t initial_wait_encoding = 0x100000u;
        const uint32_t wait_encoding = initial_wait_encoding - elected_threads;
        wait_low = wait_encoding << 1;
        wait_high = wait_encoding << 11;
    }
    return {without_leader, elected_offset, wait_low, wait_high};
}

// Copy barriers occupy the tail of the shared tile. Derive their byte offset
// from storage geometry instead of repeating per-kernel hexadecimal offsets.
template <uint32_t SharedBytes, unsigned BarrierCount>
__device__ __forceinline__ CopyWarpSchedule
tile_copy_schedule(bool block_leader, uint32_t block_threads) {
    static_assert(BarrierCount > 0 && BarrierCount <= SharedBytes / sizeof(uint64_t));
    constexpr uint32_t barrier_offset = SharedBytes - BarrierCount * sizeof(uint64_t);
    static_assert(barrier_offset % alignof(uint64_t) == 0, "Copy barrier alignment");
    return copy_warp_schedule(block_leader, block_threads, barrier_offset);
}
