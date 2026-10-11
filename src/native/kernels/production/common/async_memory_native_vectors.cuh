// Specialized native memory-access helper for the selected kernel variant.
// Vector width, address space and half/FP8 packing distinguish this helper from its siblings.
// Do not merge variants solely because their function signatures match.

template <bool CheckBounds>
__device__ __forceinline__ bool check_vector_address(uint64_t address, uint32_t byte_count,
                                                     bool write, const MemoryRange* memory_ranges,
                                                     uint32_t* error_counts) {
    if constexpr (CheckBounds) {
        bool valid = false;
        for (int range_index = 0; range_index < 16; ++range_index)
            valid |= address >= memory_ranges[range_index].base &&
                     address - memory_ranges[range_index].base <= memory_ranges[range_index].size &&
                     byte_count <= memory_ranges[range_index].size -
                                       (address - memory_ranges[range_index].base) &&
                     (!write || memory_ranges[range_index].writable);
        if (!valid || address & 15) {
            atomicAdd(error_counts + unsigned(write), 1u);
            return false;
        }
    }
    return true;
}
template <bool CheckBounds>
__device__ __forceinline__ void
load_four_words(uint64_t address, uint32_t& address_value, uint32_t& word_y, uint32_t& word_z,
                uint32_t& weights, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, false, memory_ranges, error_counts)) {
        address_value = word_y = word_z = weights = 0;
        return;
    }
    const uint4 result_word = *reinterpret_cast<const uint4*>(address);
    address_value = result_word.x;
    word_y = result_word.y;
    word_z = result_word.z;
    weights = result_word.w;
}
template <bool CheckBounds>
__device__ __forceinline__ void
store_four_words(uint64_t address, uint32_t address_value, uint32_t word_y, uint32_t word_z,
                 uint32_t weights, const MemoryRange* memory_ranges, uint32_t* error_counts) {
    if (!check_vector_address<CheckBounds>(address, 16, true, memory_ranges, error_counts))
        return;
    *reinterpret_cast<uint4*>(address) = make_uint4(address_value, word_y, word_z, weights);
}
template <bool CheckBounds>
__device__ __forceinline__ void
load_four_shared_words(uint32_t* shared_words, uint32_t shared_offset, uint32_t& address_value,
                       uint32_t& word_y, uint32_t& word_z, uint32_t& weights,
                       uint32_t capacity_bytes, const MemoryRange*, uint32_t* error_counts) {
    if constexpr (CheckBounds)
        if (shared_offset > capacity_bytes || 16 > capacity_bytes - shared_offset ||
            shared_offset & 15) {
            atomicAdd(error_counts, 1u);
            address_value = word_y = word_z = weights = 0;
            return;
        }
    const uint4 result_word = *reinterpret_cast<const uint4*>(
        reinterpret_cast<const unsigned char*>(shared_words) + shared_offset);
    address_value = result_word.x;
    word_y = result_word.y;
    word_z = result_word.z;
    weights = result_word.w;
}
template <bool CheckBounds>
__device__ __forceinline__ void
store_four_shared_words(uint32_t* shared_words, uint32_t shared_offset, uint32_t address_value,
                        uint32_t word_y, uint32_t word_z, uint32_t weights, uint32_t capacity_bytes,
                        const MemoryRange*, uint32_t* error_counts) {
    if constexpr (CheckBounds)
        if (shared_offset > capacity_bytes || 16 > capacity_bytes - shared_offset ||
            shared_offset & 15) {
            atomicAdd(error_counts, 1u);
            return;
        }
    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(shared_words) + shared_offset) =
        make_uint4(address_value, word_y, word_z, weights);
}
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

__device__ __forceinline__ uint32_t broadcast_warp_word(uint32_t value, bool selected,
                                                        uint32_t previous) {

    uint32_t result;
    asm volatile("{ .reg .b32 mask,chosen,reversed,lane,tmp; .reg .pred pick,nonempty;\n"
                 "activemask.b32 mask; setp.ne.u32 pick,%2,0;\n"
                 "vote.sync.ballot.b32 chosen,pick,mask;\n"
                 "setp.ne.u32 nonempty,chosen,0;\n"
                 "brev.b32 reversed,chosen; bfind.u32 lane,reversed; sub.u32 lane,31,lane;\n"
                 "mov.b32 tmp,%3;\n"
                 "@nonempty shfl.sync.idx.b32 tmp,%1,lane,31,mask;\n"
                 "mov.b32 %0,tmp;\n}"
                 : "=r"(result)
                 : "r"(value), "r"(uint32_t(selected)), "r"(previous));
    return result;
}

template <bool All> __device__ __forceinline__ bool vote_warp_predicate(bool value) {
    uint32_t result;
    if constexpr (All) {
        asm volatile("{.reg .b32 mask; .reg .pred p,q; activemask.b32 mask; setp.ne.u32 p,%1,0; "
                     "vote.sync.all.pred q,p,mask; selp.u32 %0,1,0,q;}"
                     : "=r"(result)
                     : "r"(uint32_t(value)));
    } else {
        asm volatile("{.reg .b32 mask; .reg .pred p,q; activemask.b32 mask; setp.ne.u32 p,%1,0; "
                     "vote.sync.any.pred q,p,mask; selp.u32 %0,1,0,q;}"
                     : "=r"(result)
                     : "r"(uint32_t(value)));
    }
    return result != 0;
}

template <bool CheckBounds>
__device__ __forceinline__ void check_single_lane(uint32_t* error_counts) {
    if constexpr (CheckBounds) {
        uint32_t mask;
        asm volatile("activemask.b32 %0;" : "=r"(mask));
        if (!mask || (mask & (mask - 1)))
            atomicAdd(error_counts + 2, 1u);
    }
}

__device__ __forceinline__ uint32_t cosine_radians(uint32_t operand_a) {
    float result_word;
    asm("cos.approx.ftz.f32 %0,%1;" : "=f"(result_word) : "f"(__uint_as_float(operand_a)));
    return __float_as_uint(result_word);
}
__device__ __forceinline__ uint32_t sine_radians(uint32_t operand_a) {
    float result_word;
    asm("sin.approx.ftz.f32 %0,%1;" : "=f"(result_word) : "f"(__uint_as_float(operand_a)));
    return __float_as_uint(result_word);
}

template <bool CheckBounds>
__device__ __forceinline__ uint32_t load_aligned_word(uint64_t address,
                                                      const MemoryRange* memory_ranges,
                                                      uint32_t* error_counts) {
    if constexpr (CheckBounds)
        if (!is_word_address_allowed(address, memory_ranges, false) || address & 3) {
            atomicAdd(error_counts, 1u);
            return 0;
        }
    uint32_t value;
    asm volatile("ld.global.u32 %0,[%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
}

template <int Lane, bool Absolute, bool Negative>
__device__ __forceinline__ uint32_t select_half_operand(uint32_t value) {
    if constexpr (Lane < 2) {
        uint32_t packed;
        if constexpr (Lane == 0)
            asm("{ .reg .b16 lo,hi; mov.b32 {lo,hi},%1; mov.b32 %0,{lo,lo}; }"
                : "=r"(packed)
                : "r"(value));
        else
            asm("{ .reg .b16 lo,hi; mov.b32 {lo,hi},%1; mov.b32 %0,{hi,hi}; }"
                : "=r"(packed)
                : "r"(value));
        value = packed;
    }
    if constexpr (Absolute)
        asm("abs.f16x2 %0,%1;" : "=r"(value) : "r"(value));
    if constexpr (Negative)
        asm("neg.f16x2 %0,%1;" : "=r"(value) : "r"(value));
    return value;
}

__device__ __forceinline__ uint32_t pack_fp8_pair(uint32_t operand_a, uint32_t operand_c) {
    uint32_t result;
    asm("{ .reg .b16 lo,hi,unused; mov.b32 {hi,unused},%2; cvt.rn.satfinite.e4m3x2.f16x2 lo,%1; "
        "mov.b32 %0,{lo,hi}; }"
        : "=r"(result)
        : "r"(operand_a), "r"(operand_c));
    return result;
}
