// Reorder 4x4 spatial feature tiles into consecutive transformer token tiles.
// This is a bit-preserving layout conversion, with zero-filled padding to 32
// cells for the following transformer stage. No floating-point math is needed.
#include "../../common/feature_layout.cuh"

extern "C" __global__
__launch_bounds__(256) void phoenix_image_to_tokens(phoenix::FeatureLayoutArguments arguments) {
    const uint32_t cell_count = arguments.height * arguments.width;
    const uint32_t padded_cell_count = (cell_count + 31) / 32 * 32;
    const uint32_t word_count = padded_cell_count * phoenix::feature_words_per_cell;
    const uint32_t grid_stride = blockDim.x * gridDim.x;

    // Each thread moves one packed four-channel word at a time. Grid-stride iteration also
    // covers padding when the launch grid was sized for the unpadded image.
    for (uint32_t word = blockIdx.x * blockDim.x + threadIdx.x; word < word_count;
         word += grid_stride) {
        const uint32_t cell = word / phoenix::feature_words_per_cell;
        const uint32_t feature_word = word % phoenix::feature_words_per_cell;
        const uint32_t value =
            cell < cell_count
                ? arguments.input[phoenix::image_word_offset(cell, feature_word, arguments.width)]
                : 0;
        arguments.output[phoenix::token_word_offset(cell, feature_word)] = value;
    }
}
