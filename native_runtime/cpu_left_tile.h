#pragma once
#include <immintrin.h>
#include <cstdint>

namespace phoenix_cpp {
// Read eight consecutive half values per live row, then transpose in registers.
// Each result vector contains the same K from eight rows, ready for the GEMM's
// existing ascending-K FMA loop. Only call for a complete eight-element K tile;
// row tails are zero-filled and reduction tails retain the scalar loader.
inline void load_left_tile8(const uint16_t* first_row, uint32_t row_stride,
                            unsigned live_rows, __m256 (&columns)[8]) {
    __m256 rows[8];
    for (unsigned row = 0; row < 8; ++row)
        rows[row] = row < live_rows
                        ? _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(
                              first_row + uint64_t(row) * row_stride)))
                        : _mm256_setzero_ps();
    __m256 pairs[8], quarters[8];
    for (unsigned pair = 0; pair < 4; ++pair) {
        pairs[2 * pair] = _mm256_unpacklo_ps(rows[2 * pair], rows[2 * pair + 1]);
        pairs[2 * pair + 1] = _mm256_unpackhi_ps(rows[2 * pair], rows[2 * pair + 1]);
    }
    for (unsigned group = 0; group < 2; ++group) {
        const unsigned offset = group * 4;
        quarters[offset] = _mm256_shuffle_ps(pairs[offset], pairs[offset + 2], 0x44);
        quarters[offset + 1] = _mm256_shuffle_ps(pairs[offset], pairs[offset + 2], 0xee);
        quarters[offset + 2] = _mm256_shuffle_ps(pairs[offset + 1], pairs[offset + 3], 0x44);
        quarters[offset + 3] = _mm256_shuffle_ps(pairs[offset + 1], pairs[offset + 3], 0xee);
    }
    for (unsigned column = 0; column < 4; ++column) {
        columns[column] = _mm256_permute2f128_ps(quarters[column], quarters[column + 4], 0x20);
        columns[column + 4] = _mm256_permute2f128_ps(quarters[column], quarters[column + 4], 0x31);
    }
}
} // namespace phoenix_cpp
