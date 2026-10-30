#include "backend.h"
#include "cpu_math_avx.h"
// An AVX-512 vector holds sixteen rows of one output column. Eight
// accumulators form a 16x8 tile, doubling weight reuse without changing K order.
// Keep the ascending reduction/FMA order: changing it changes model rounding.
extern "C" void phoenix_cpu_avx512(const phoenix_cpp::Command& command, uint64_t index) {
    auto operation = command.operation;
    phx::Kernel<AvxMath> math{operation};
    unsigned column_tiles = (operation.column_count + 7) / 8,
             row_tiles = (operation.row_count + 15) / 16,
             first_column = unsigned(index % column_tiles) * 8,
             first_row = unsigned(index / column_tiles % row_tiles) * 16;
    uint64_t batch_index = index / (uint64_t(column_tiles) * row_tiles);
    __m512 column_sums[8];
    for (auto& sum : column_sums)
        sum = _mm512_setzero_ps();
    const auto& decode = fp8_table();
    const unsigned live_rows = std::min(16u, operation.row_count - first_row);
    const __mmask16 row_mask = __mmask16((1u << live_rows) - 1u);
    const bool paired_rows = operation.left_row_stride % 2 == 0 &&
                             operation.left_batch_stride % 2 == 0 &&
                             operation.left_row_stride <= unsigned(INT32_MAX / 16);
    const auto* left_base = static_cast<const uint16_t*>(operation.left_input) +
                            batch_index * operation.left_batch_stride +
                            uint64_t(first_row) * operation.left_row_stride;
    const __m512i row_pairs =
        _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                           _mm512_set1_epi32(operation.left_row_stride / 2));
    // A gather reads two adjacent halves. Retain the pair for the next K,
    // instead of issuing the same sixteen memory reads a second time.
    // This is a register cache only: weights and tensor storage stay packed.
    __m512i packed_pair = _mm512_setzero_si512();
    for (unsigned reduction_index = 0; reduction_index < operation.reduction_count;
         ++reduction_index) {
        __m512 left_values;
        if (paired_rows) {
            // Fetch one aligned pair of halves from each row, then select K.
            // The even row stride keeps the last pair inside its source row.
            if ((reduction_index & 1u) == 0u) {
                const __m512i indices =
                    _mm512_add_epi32(row_pairs, _mm512_set1_epi32(reduction_index / 2));
                packed_pair = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), row_mask, indices,
                                                         left_base, 4);
            }
            __m512i packed = packed_pair;
            if (reduction_index & 1u)
                packed = _mm512_srli_epi32(packed, 16);
            left_values = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(packed));
        } else {
            alignas(32) uint16_t gathered[16]{};
            for (unsigned row = 0; row < live_rows; ++row)
                gathered[row] =
                    left_base[uint64_t(row) * operation.left_row_stride + reduction_index];
            left_values =
                _mm512_cvtph_ps(_mm256_load_si256(reinterpret_cast<const __m256i*>(gathered)));
        }
        for (unsigned column_in_tile = 0;
             column_in_tile < 8 && first_column + column_in_tile < operation.column_count;
             ++column_in_tile) {
            uint64_t right_offset =
                batch_index * operation.right_batch_stride +
                ((operation.flags & 1)
                     ? uint64_t(first_column + column_in_tile) * operation.right_row_stride +
                           reduction_index
                     : uint64_t(reduction_index) * operation.right_row_stride + first_column +
                           column_in_tile);
            float right_value =
                (operation.flags & phx::RightInputFp8)
                    ? decode[static_cast<const uint8_t*>(operation.right_input)[right_offset]]
                    : half_to_float(
                          static_cast<const uint16_t*>(operation.right_input)[right_offset]);
            column_sums[column_in_tile] = _mm512_fmadd_ps(left_values, _mm512_set1_ps(right_value),
                                                          column_sums[column_in_tile]);
        }
    }
    for (unsigned column_in_tile = 0;
         column_in_tile < 8 && first_column + column_in_tile < operation.column_count;
         ++column_in_tile) {
        alignas(64) float output_values[16];
        _mm512_store_ps(output_values, column_sums[column_in_tile]);
        for (unsigned row_in_tile = 0;
             row_in_tile < 16 && first_row + row_in_tile < operation.row_count; ++row_in_tile)
            math.store(batch_index * operation.output_batch_stride +
                           uint64_t(first_row + row_in_tile) * operation.output_row_stride +
                           first_column + column_in_tile,
                       output_values[row_in_tile]);
    }
}
