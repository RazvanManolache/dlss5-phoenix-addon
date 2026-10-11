#include "backend.h"
#include "host_math.h"
#include <immintrin.h>
#include <array>
namespace {
float half_to_float(uint16_t value) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(value)));
}
struct AvxMath : HostMath {
    static float fma(float left, float right, float addend) {
        return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(left), _mm_set_ss(right), _mm_set_ss(addend)));
    }
};
// Decode all 256 E4M3 byte codes once. Model loading rejects NaN codes;
// this table keeps the inner GEMM loop free of scalar decoding arithmetic.
const std::array<float, 256>& fp8_table() {
    static const auto table = [] {
        std::array<float, 256> values{};
        for (unsigned code = 0; code < 256; ++code) {
            unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
            float magnitude = exponent ? std::ldexp(1.f + float(mantissa) / 8, int(exponent) - 7)
                                       : float(mantissa) / 512;
            values[code] = (code & 128) ? -magnitude : magnitude;
        }
        return values;
    }();
    return table;
}
}
// An AVX2 vector contains eight rows of one output column. Eight accumulators
// form an 8x8 output tile, reusing each right-hand weight across those rows.
// Keep the ascending reduction/FMA order: changing it changes model rounding.
extern "C" void phoenix_cpu_avx2(const phoenix_cpp::Command& command, uint64_t index) {
    auto operation = command.operation;
    phx::Kernel<AvxMath> math{operation};
    if (operation.family != phx::MatrixMultiply) {
        math(index);
        return;
    }
    unsigned column_tiles = (operation.column_count + 7) / 8,
             row_tiles = (operation.row_count + 7) / 8,
             first_column = unsigned(index % column_tiles) * 8,
             first_row = unsigned(index / column_tiles % row_tiles) * 8;
    uint64_t batch_index = index / (uint64_t(column_tiles) * row_tiles);
    __m256 column_sums[8];
    for (auto& sum : column_sums)
        sum = _mm256_setzero_ps();
    const auto& decode = fp8_table();
    for (unsigned reduction_index = 0; reduction_index < operation.reduction_count;
         ++reduction_index) {
        alignas(16) uint16_t left_half_values[8]{};
        for (unsigned row_in_tile = 0;
             row_in_tile < 8 && first_row + row_in_tile < operation.row_count; ++row_in_tile)
            left_half_values[row_in_tile] = static_cast<const uint16_t*>(
                operation
                    .left_input)[batch_index * operation.left_batch_stride +
                                 uint64_t(first_row + row_in_tile) * operation.left_row_stride +
                                 reduction_index];
        auto left_values =
            _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(left_half_values)));
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
                (operation.flags & 0x2000000)
                    ? decode[static_cast<const uint8_t*>(operation.right_input)[right_offset]]
                    : half_to_float(
                          static_cast<const uint16_t*>(operation.right_input)[right_offset]);
            column_sums[column_in_tile] = _mm256_fmadd_ps(left_values, _mm256_set1_ps(right_value),
                                                          column_sums[column_in_tile]);
        }
    }
    for (unsigned column_in_tile = 0;
         column_in_tile < 8 && first_column + column_in_tile < operation.column_count;
         ++column_in_tile) {
        alignas(32) float output_values[8];
        _mm256_store_ps(output_values, column_sums[column_in_tile]);
        for (unsigned row_in_tile = 0;
             row_in_tile < 8 && first_row + row_in_tile < operation.row_count; ++row_in_tile)
            math.store(batch_index * operation.output_batch_stride +
                           uint64_t(first_row + row_in_tile) * operation.output_row_stride +
                           first_column + column_in_tile,
                       output_values[row_in_tile]);
    }
}
