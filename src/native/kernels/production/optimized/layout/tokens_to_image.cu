// Restore spatial feature tiles after the transformer. Padding tokens belong
// only to the token buffer; the output contains exactly height * width cells.
#include "../../common/feature_layout.cuh"

extern "C" __global__
__launch_bounds__(256) void phoenix_tokens_to_image(phoenix::FeatureLayoutArguments arguments) {
    const uint32_t word_count =
        arguments.height * arguments.width * phoenix::feature_words_per_cell;
    const uint32_t grid_stride = blockDim.x * gridDim.x;

    for (uint32_t word = blockIdx.x * blockDim.x + threadIdx.x; word < word_count;
         word += grid_stride) {
        const uint32_t cell = word / phoenix::feature_words_per_cell;
        const uint32_t feature_word = word % phoenix::feature_words_per_cell;
        arguments.output[phoenix::image_word_offset(cell, feature_word, arguments.width)] =
            arguments.input[phoenix::token_word_offset(cell, feature_word)];
    }
}
