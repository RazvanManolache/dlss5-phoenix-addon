#pragma once

// Async transaction barriers: initialization publishes to the async proxy,
// copies contribute expected bytes, and callers retain arrive/wait ordering.
// This is a synchronization contract, not a replaceable arithmetic helper.

__device__ __forceinline__ void initialize_copy_barrier(uint32_t* shared_words,
                                                        uint32_t shared_offset) {

    if (threadIdx.x == __ffs(__activemask()) - 1) {
        uint32_t shared_address = uint32_t(__cvta_generic_to_shared(shared_words)) + shared_offset;
        uint32_t count = blockDim.x * blockDim.y * blockDim.z;
        asm volatile("mbarrier.init.shared.b64 [%0],%1;\n fence.proxy.async.shared::cta;" ::"r"(
                         shared_address),
                     "r"(count)
                     : "memory");
    }
}

template <bool CheckBounds>
__device__ __forceinline__ void
copy_global_to_shared(uint32_t* shared_words, uint32_t destination_offset, uint64_t source_address,
                      uint32_t byte_count, uint32_t barrier, uint32_t capacity_bytes,
                      const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(source_address, byte_count, false, memory_ranges,
                                           error_counts))
        return;
    if constexpr (CheckBounds)
        if (destination_offset > capacity_bytes ||
            byte_count > capacity_bytes - destination_offset || destination_offset & 15 ||
            byte_count & 15 || barrier > capacity_bytes - 8 || barrier & 7) {
            atomicAdd(error_counts, 1u);
            return;
        }
    uint32_t base = uint32_t(__cvta_generic_to_shared(shared_words));
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0],%1;\n"
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%2],[%3],%1,[%0];" ::"r"(
            base + barrier),
        "r"(byte_count), "r"(base + destination_offset), "l"(source_address)
        : "memory");
}

__device__ __forceinline__ uint64_t arrive_copy_barrier(uint32_t* shared_words,
                                                        uint32_t shared_offset) {
    uint64_t state;
    uint32_t shared_address = uint32_t(__cvta_generic_to_shared(shared_words)) + shared_offset;
    asm volatile("mbarrier.arrive.shared.b64 %0,[%1];"
                 : "=l"(state)
                 : "r"(shared_address)
                 : "memory");
    return state;
}

__device__ __forceinline__ bool wait_copy_barrier(uint32_t* shared_words, uint32_t shared_offset,
                                                  uint64_t state) {
    uint32_t ready;
    uint32_t shared_address = uint32_t(__cvta_generic_to_shared(shared_words)) + shared_offset;
    asm volatile("{.reg .pred p; mbarrier.try_wait.shared.b64 p,[%1],%2; selp.u32 %0,1,0,p;}"
                 : "=r"(ready)
                 : "r"(shared_address), "l"(state)
                 : "memory");
    return ready != 0;
}
