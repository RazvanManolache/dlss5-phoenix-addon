#pragma once
// Compact CPU tiles limit private accumulator storage and register pressure.
// Keep each output's ascending FMA order, including partial rows and columns.
template <unsigned Rows, unsigned Columns>
sycl::event sycl_cpu_matrix(sycl::queue& queue, phx::Operation operation) {
    uint64_t column_tile_count = (operation.column_count + Columns - 1) / Columns,
             row_tile_count = (operation.row_count + Rows - 1) / Rows;
    return queue.parallel_for(
        sycl::range<1>(column_tile_count * row_tile_count * operation.batch_count),
        [=](sycl::id<1> item) {
            uint64_t tile = item[0], batch_index = tile / (column_tile_count * row_tile_count);
            unsigned first_row = unsigned(tile / column_tile_count % row_tile_count) * Rows,
                     first_column = unsigned(tile % column_tile_count) * Columns;
            phx::Kernel<SyclCpuMath> math{operation};
            float accumulators[Rows][Columns]{};
            for (unsigned reduction_index = 0; reduction_index < operation.reduction_count;
                 ++reduction_index) {
                float right_values[Columns];
                for (unsigned column_in_tile = 0; column_in_tile < Columns; ++column_in_tile) {
                    unsigned column = first_column + column_in_tile;
                    right_values[column_in_tile] =
                        column < operation.column_count
                            ? math.load_matrix_right(
                                  batch_index * operation.right_batch_stride +
                                  ((operation.flags & 1)
                                       ? uint64_t(column) * operation.right_row_stride +
                                             reduction_index
                                       : uint64_t(reduction_index) * operation.right_row_stride +
                                             column))
                            : 0;
                }
                for (unsigned row_in_tile = 0; row_in_tile < Rows; ++row_in_tile) {
                    unsigned row = first_row + row_in_tile;
                    float left_value =
                        row < operation.row_count
                            ? math.load_half(operation.left_input,
                                             batch_index * operation.left_batch_stride +
                                                 uint64_t(row) * operation.left_row_stride +
                                                 reduction_index)
                            : 0;
                    for (unsigned column_in_tile = 0; column_in_tile < Columns; ++column_in_tile)
                        accumulators[row_in_tile][column_in_tile] =
                            sycl::fma(left_value, right_values[column_in_tile],
                                      accumulators[row_in_tile][column_in_tile]);
                }
            }
            for (unsigned row_in_tile = 0;
                 row_in_tile < Rows && first_row + row_in_tile < operation.row_count; ++row_in_tile)
                for (unsigned column_in_tile = 0;
                     column_in_tile < Columns &&
                     first_column + column_in_tile < operation.column_count;
                     ++column_in_tile)
                    math.store(batch_index * operation.output_batch_stride +
                                   uint64_t(first_row + row_in_tile) * operation.output_row_stride +
                                   first_column + column_in_tile,
                               accumulators[row_in_tile][column_in_tile]);
        });
}
