#define PHX_DEVICE __forceinline
#include "backend.h"
#include "cpu_math_avx.h"
#include "cpu_left_tile.h"
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
    const unsigned live_rows = std::min(8u, operation.row_count - first_row);
    const auto* left_base = static_cast<const uint16_t*>(operation.left_input) +
                           batch_index * operation.left_batch_stride +
                           uint64_t(first_row) * operation.left_row_stride;
    __m256 left_columns[8];
    for (unsigned reduction_index = 0; reduction_index < operation.reduction_count;
         ++reduction_index) {
        __m256 left_values;
        const unsigned tile_begin = reduction_index & ~7u;
        if (operation.reduction_count - tile_begin >= 8u) {
            if ((reduction_index & 7u) == 0u)
                phoenix_cpp::load_left_tile8(left_base + reduction_index, operation.left_row_stride,
                                             live_rows, left_columns);
            left_values = left_columns[reduction_index & 7u];
        } else {
            alignas(16) uint16_t left_half_values[8]{};
            for (unsigned row = 0; row < live_rows; ++row)
                left_half_values[row] =
                    left_base[uint64_t(row) * operation.left_row_stride + reduction_index];
            left_values =
                _mm256_cvtph_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(left_half_values)));
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

// Fix the pointwise operation once per contiguous range. This avoids copying
// and branching on the full operation packet separately for every scalar value.
template <unsigned Kind>
void cpu_pointwise_range(phx::Operation operation, uint64_t first, uint64_t end) {
    operation.flags = (operation.flags & ~255u) | Kind;
    phx::Kernel<AvxMath> math{operation};
    for (auto index = first; index < end; ++index)
        math.unary(index);
}
extern "C" void phoenix_cpu_avx2_range(const phoenix_cpp::Command& command, uint64_t first,
                                       uint64_t end) {
    if (command.operation.family == phx::Elementwise) {
        switch (command.operation.flags & 255u) {
        case phx::QuantizeFp8:
            return cpu_pointwise_range<phx::QuantizeFp8>(command.operation, first, end);
        case phx::ActivationGate:
            return cpu_pointwise_range<phx::ActivationGate>(command.operation, first, end);
        case phx::RoundHalf:
            return cpu_pointwise_range<phx::RoundHalf>(command.operation, first, end);
        case phx::FloatToHalf:
            return cpu_pointwise_range<phx::FloatToHalf>(command.operation, first, end);
        case phx::CopyValues:
            return cpu_pointwise_range<phx::CopyValues>(command.operation, first, end);
        case phx::ScaledResidual:
            return cpu_pointwise_range<phx::ScaledResidual>(command.operation, first, end);
        case phx::ScaleHalf:
            return cpu_pointwise_range<phx::ScaleHalf>(command.operation, first, end);
        case phx::GatherWindows:
            return cpu_pointwise_range<phx::GatherWindows>(command.operation, first, end);
        case phx::ScatterWindows:
            return cpu_pointwise_range<phx::ScatterWindows>(command.operation, first, end);
        case phx::AddPositionBias:
            return cpu_pointwise_range<phx::AddPositionBias>(command.operation, first, end);
        case phx::SplitQkvHeads:
            return cpu_pointwise_range<phx::SplitQkvHeads>(command.operation, first, end);
        case phx::MergeAttentionHeads:
            return cpu_pointwise_range<phx::MergeAttentionHeads>(command.operation, first, end);
        case phx::AveragePool:
            return cpu_pointwise_range<phx::AveragePool>(command.operation, first, end);
        case phx::NearestUpsample:
            return cpu_pointwise_range<phx::NearestUpsample>(command.operation, first, end);
        case phx::ChannelGain:
            return cpu_pointwise_range<phx::ChannelGain>(command.operation, first, end);
        case phx::AddInputs:
            return cpu_pointwise_range<phx::AddInputs>(command.operation, first, end);
        case phx::SpatialPad:
            return cpu_pointwise_range<phx::SpatialPad>(command.operation, first, end);
        case phx::GateFp8ToHalf:
            return cpu_pointwise_range<phx::GateFp8ToHalf>(command.operation, first, end);
        case phx::Fp8ToHalf:
            return cpu_pointwise_range<phx::Fp8ToHalf>(command.operation, first, end);
        case phx::GateToHalf:
            return cpu_pointwise_range<phx::GateToHalf>(command.operation, first, end);
        case phx::UpsampleBlendAndPublish:
            return cpu_pointwise_range<phx::UpsampleBlendAndPublish>(command.operation, first, end);
        case phx::UpsampleResidual:
            return cpu_pointwise_range<phx::UpsampleResidual>(command.operation, first, end);
        case phx::PoolAndPublish:
            return cpu_pointwise_range<phx::PoolAndPublish>(command.operation, first, end);
        case phx::GatherWindowBatch:
            return cpu_pointwise_range<phx::GatherWindowBatch>(command.operation, first, end);
        case phx::ScatterWindowBatch:
            return cpu_pointwise_range<phx::ScatterWindowBatch>(command.operation, first, end);
        case phx::ScatterWindowBatchAndPool:
            return cpu_pointwise_range<phx::ScatterWindowBatchAndPool>(command.operation, first,
                                                                       end);
        case phx::GatherUpsampleBlendBatch:
            return cpu_pointwise_range<phx::GatherUpsampleBlendBatch>(command.operation, first,
                                                                      end);
        }
    }
    if (command.operation.family == phx::RowReduction && (command.operation.flags & 255u) == 1) {
        phx::Kernel<AvxMath> math{command.operation};
        for (auto index = first; index < end; ++index)
            math.softmax(unsigned(index));
        return;
    }
    for (auto index = first; index < end; ++index)
        phoenix_cpu_avx2(command, index);
}
