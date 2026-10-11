#pragma once

#include <stddef.h>
#include <stdint.h>

namespace phoenix {

// The launch packet is shared with the model plan: two device pointers followed
// by the feature grid dimensions. Each grid cell contains 1024 FP8 channels.
struct alignas(8) FeatureLayoutArguments {
    const uint32_t* input;
    uint32_t* output;
    uint32_t height;
    uint32_t width;
};
static_assert(sizeof(FeatureLayoutArguments) == 24, "Feature layout launch ABI");
static_assert(offsetof(FeatureLayoutArguments, input) == 0, "Input pointer ABI");
static_assert(offsetof(FeatureLayoutArguments, output) == 8, "Output pointer ABI");
static_assert(offsetof(FeatureLayoutArguments, height) == 16, "Height ABI");
static_assert(offsetof(FeatureLayoutArguments, width) == 20, "Width ABI");

constexpr uint32_t feature_words_per_cell = 256;
constexpr uint32_t cells_per_tile = 16;
constexpr uint32_t spatial_tile_side = 4;
constexpr uint32_t words_per_tile = cells_per_tile * feature_words_per_cell;

// Both layouts use the same 16-cell tile encoding. A word carries four E4M3 FP8
// encodings; moving the word preserves every bit. Packing within the word is
// the producer/consumer matrix code's responsibility, not this conversion's.
// Within a tile, groups of 32 channels occupy 128 words. The remaining bits
// interleave the cell lane and feature word to match native matrix fragments:
//   word bits 11..7 = feature_word[7..3]
//   word bits  6..4 = cell_lane[2..0]
//   word bits  3..2 = feature_word[1..0]
//   word bit      1 = feature_word[2]
//   word bit      0 = cell_lane[3]
__device__ __forceinline__ uint32_t packed_tile_offset(uint32_t tile_index, uint32_t cell_lane,
                                                       uint32_t feature_word) {
    return tile_index * words_per_tile + (feature_word / 8) * 128 + (cell_lane % 8) * 16 +
           (feature_word % 4) * 4 + ((feature_word / 4) % 2) * 2 + cell_lane / 8;
}

// Transformer tiles group 16 consecutive cells from the raster-order sequence.
__device__ __forceinline__ uint32_t token_word_offset(uint32_t cell_index, uint32_t feature_word) {
    return packed_tile_offset(cell_index / cells_per_tile, cell_index % cells_per_tile,
                              feature_word);
}

// Image tiles group a 4x4 square. Valid model plans align both dimensions to 4;
// this rearranges cell coordinates only, keeping the inner feature encoding.
__device__ __forceinline__ uint32_t image_word_offset(uint32_t cell_index, uint32_t feature_word,
                                                      uint32_t width) {
    const uint32_t row = cell_index / width;
    const uint32_t column = cell_index % width;
    const uint32_t tile_index =
        (row / spatial_tile_side) * (width / spatial_tile_side) + column / spatial_tile_side;
    const uint32_t cell_lane =
        (row % spatial_tile_side) * spatial_tile_side + column % spatial_tile_side;
    return packed_tile_offset(tile_index, cell_lane, feature_word);
}

} // namespace phoenix
