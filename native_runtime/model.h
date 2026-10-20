#pragma once

#include "backend.h"
#include "host_math.h"
#include "../src/native/model/weight_source.h"

#include <array>
#include <fstream>
#include <map>

namespace phoenix_cpp {

#include "weight_layouts.h"
#include "models/small_fp8/weight_layouts.h"

using Json = nlohmann::json;

inline std::string file_sha(const std::filesystem::path& path) {
    return phoenix_model::mapped_file_digest(path);
}

struct TensorBytes {
    std::vector<uint8_t> data;
};

// Decodes a supplied model resource into named tensors. This class owns the
// decoded host copies; the graph uploads its weights before discarding them.
class Model {
    mutable std::map<std::string, TensorBytes> tensors_;
    phoenix_model::WeightSource source_;
    std::string resource_sha256_;
    const phoenix_model::Definition* definition_ = &phoenix_model::models::original;
    phx::Kernel<HostMath> math_{};

    // Convert a row/column coordinate into the resource's tiled FP8 layout.
    // The nested expression preserves the original lane and tile ordering.
    static size_t packed_offset(unsigned row, unsigned column, unsigned matrix_rows,
                                unsigned matrix_columns, unsigned inner_tiles) {
        unsigned outer_tile = row / (16 * inner_tiles);
        unsigned inner_tile = (row / 16) % inner_tiles;
        unsigned row_half = row / 8 % 2;
        unsigned row_lane = row % 8;
        unsigned column_group = column / 32;
        unsigned column_quarter = column / 8 % 4;
        unsigned column_pair = column / 2 % 4;
        unsigned column_parity = column % 2;

        return (((((((size_t(outer_tile) * (matrix_columns / 32) + column_group) * inner_tiles +
                     inner_tile) *
                        8 +
                    row_lane) *
                       4 +
                   column_pair) *
                      2 +
                  row_half) *
                     4 +
                 column_quarter) *
                    2 +
                column_parity);
    }

#include "models/common/decoder.inc"
#include "models/original/decoder.inc"
#include "models/small_fp8/decoder.inc"

    using Decoder = void (Model::*)(const uint8_t*, size_t);
    static Decoder decoder_for(const phoenix_model::Definition& model) {
        phoenix_model::require_implemented(model);
        struct Entry { std::string_view architecture; Decoder decode; };
        static constexpr std::array handlers{
            Entry{phoenix_model::models::original.architecture, &Model::decode_original},
            Entry{phoenix_model::models::small_fp8.architecture, &Model::decode_small_fp8}};
        for (const auto& handler : handlers)
            if (handler.architecture == model.architecture)
                return handler.decode;
        throw std::runtime_error("No Phoenix DLSS5 tensor decoder for architecture: " + std::string(model.architecture));
    }

  public:
    static void require_supported(const phoenix_model::Definition& model) {
        (void)decoder_for(model);
    }
    const phoenix_model::Definition& definition() const { return *definition_; }
    const std::string& resource_sha256() const { return resource_sha256_; }
    bool decoded() const {
        return !tensors_.empty();
    }

    void discard_decoded() {
        tensors_.clear();
    }

    template <class Visitor> void visit_tensors(Visitor visitor) const {
        for (const auto& [name, tensor] : tensors_)
            visitor(name, tensor.data);
    }

    explicit Model(const phoenix_model::WeightSource& source) : source_(source), definition_(&source.model()) {
        const auto decode = decoder_for(definition());
        phoenix_model::with_model_resource(
            source, file_sha, [&](const uint8_t* resource_bytes, size_t resource_size) {
                resource_sha256_ = phoenix_model::weight_digest(resource_bytes, resource_size);
                (this->*decode)(resource_bytes, resource_size);
                return 0;
            });
    }

    const std::vector<uint8_t>& raw(int block, int layer, const std::string& field) const {
        if (!decoded()) {
            Model replacement(source_);
            demand(replacement.resource_sha256() == resource_sha256_,
                   "Model resource changed during session; create a new session");
            tensors_ = std::move(replacement.tensors_);
        }
        return tensors_
            .at("block" + std::to_string(block) + ".layer" + std::to_string(layer) + '.' + field)
            .data;
    }

    // Accessors retain their existing names for callers of the runtime API.
    std::vector<float> half(int block, int layer, const std::string& field) const {
        auto& bytes = raw(block, layer, field);
        demand(bytes.size() % 2 == 0, "Half tensor alignment");
        std::vector<float> values(bytes.size() / 2);
        for (size_t index = 0; index < values.size(); ++index) {
            uint16_t half_bits;
            std::memcpy(&half_bits, bytes.data() + index * 2, 2);
            values[index] = math_.half_value(half_bits);
        }
        return values;
    }

    float history_gain() const {
        const auto values = half(int(definition().history_gain_block), 0, "blend_scale");
        demand(values.size() == 1, "Model blend_scale must contain one scalar");
        return values.front();
    }

    std::vector<float> floats(int block, int layer, const std::string& field) const {
        auto& bytes = raw(block, layer, field);
        demand(bytes.size() % 4 == 0, "Float tensor alignment");
        std::vector<float> values(bytes.size() / 4);
        std::memcpy(values.data(), bytes.data(), bytes.size());
        return values;
    }

    std::vector<float> fp8(int block, int layer, const std::string& field) const {
        auto& bytes = raw(block, layer, field);
        std::vector<float> values(bytes.size());
        for (size_t index = 0; index < bytes.size(); ++index) {
            auto code = bytes[index];
            demand((code & 127) != 127, "NaN FP8 weight");
            auto exponent = (code >> 3) & 15;
            auto mantissa = code & 7;
            float magnitude = exponent ? std::ldexp(1.f + float(mantissa) / 8, int(exponent) - 7)
                                       : float(mantissa) / 512;
            values[index] = (code & 128) ? -magnitude : magnitude;
        }
        return values;
    }

    std::vector<float> lift() const {
        auto packed_values = half(0, 0, "input_lift");
        std::vector<float> matrix(32 * 16);
        for (unsigned channel = 0; channel < 32; ++channel) {
            for (unsigned feature = 0; feature < 16; ++feature) {
                matrix[channel * 16 + feature] =
                    packed_values[(channel / 16) * 256 + 32 * (channel % 8) +
                                  8 * ((feature % 8) / 2) + 4 * ((channel / 8) % 2) + feature % 2 +
                                  2 * (feature / 8)];
            }
        }
        return matrix;
    }

    std::vector<float> bias(int block, int layer, unsigned head_count) const {
        auto packed_values = half(block, layer, "attn_pos_bias");
        demand(packed_values.size() == head_count * 4096, "Attention bias shape");
        std::vector<float> bias_matrix(packed_values.size());

        // Each head has a 64x64 relative-position bias matrix, stored in
        // 16x16 tiles with a lane/slot permutation inside each tile.
        for (unsigned head = 0; head < head_count; ++head) {
            for (unsigned query_token = 0; query_token < 64; ++query_token) {
                for (unsigned key_token = 0; key_token < 64; ++key_token) {
                    unsigned lane = 4 * (query_token % 8) + (key_token % 8) / 2;
                    unsigned slot = (key_token % 2) | ((query_token % 16 / 8) << 1) |
                                    ((key_token % 16 / 8) << 2);
                    // A head contains sixteen 16x16 tiles. Each tile holds
                    // eight half values per lane across 32 lanes. Reconstruct
                    // ordinary [head][query][key] order for the softmax bias.
                    unsigned head_offset = head * 4096;
                    unsigned tile_index = 4 * (query_token / 16) + key_token / 16;
                    unsigned packed_offset = head_offset + tile_index * 256 + lane * 8 + slot;
                    unsigned row_major_offset = head_offset + query_token * 64 + key_token;
                    bias_matrix[row_major_offset] = packed_values[packed_offset];
                }
            }
        }
        return bias_matrix;
    }

    std::vector<float> gain(int block) const {
        auto values = half(block, 0, "residual_scale");
        // Some records prefix the channel gains with eight zero values.
        size_t first_gain =
            values.size() >= 8 && std::all_of(values.begin(), values.begin() + 8,
                                              [](float value) { return value == 0; })
                ? 8
                : 0;
        return {values.begin() + first_gain, values.end()};
    }

    std::array<std::vector<uint8_t>, 3> split(int block) const {
        auto first_matrix = raw(block, 0, "ffwd_a");
        auto second_matrix = raw(block, 0, "ffwd_b");
        std::vector<uint8_t> packed_record(524288);

        // Rebuild the two packed halves before selecting the three grouped
        // projections. These offsets describe storage, not new arithmetic.
        for (unsigned row = 0; row < 512; ++row) {
            for (unsigned column = 0; column < 512; ++column) {
                auto offset = packed_offset(row, column, 512, 512, 32);
                packed_record[offset] = first_matrix[row * 512 + column];
                packed_record[262144 + offset] = second_matrix[row * 512 + column];
            }
        }

        auto move_bit = [](unsigned value, unsigned source_bit, unsigned target_bit) {
            return ((value >> source_bit) & 1) << target_bit;
        };
        auto decode = [](uint8_t code) {
            unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
            float magnitude = exponent ? std::ldexp(1.f + float(mantissa) / 8, int(exponent) - 7)
                                       : float(mantissa) / 512;
            return code & 128 ? -magnitude : magnitude;
        };

        std::array<std::vector<uint8_t>, 3> projections;
        projections[0].resize(8 * 64 * 512);
        projections[1].resize(8 * 256 * 64);
        projections[2].resize(8 * 64 * 256);

        for (unsigned group = 0; group < 8; ++group) {
            for (unsigned row = 0; row < 64; ++row) {
                for (unsigned column = 0; column < 512; ++column) {
                    unsigned offset =
                        move_bit(column, 0, 0) | move_bit(column, 1, 4) | move_bit(column, 2, 5) |
                        move_bit(column, 3, 1) | move_bit(column, 4, 2) | move_bit(column, 5, 14) |
                        move_bit(column, 6, 15) | move_bit(column, 7, 16) |
                        move_bit(column, 8, 17) | move_bit(row, 0, 3) | move_bit(row, 1, 6) |
                        move_bit(row, 2, 7) | move_bit(row, 3, 8) | move_bit(row, 4, 9) |
                        move_bit(row, 5, 10) | move_bit(group, 0, 11) | move_bit(group, 1, 12) |
                        move_bit(group, 2, 13);
                    projections[0][(group * 64 + row) * 512 + column] = packed_record[offset];
                }
            }
            for (unsigned hidden_channel = 0; hidden_channel < 256; ++hidden_channel) {
                for (unsigned row = 0; row < 64; ++row) {
                    unsigned expand_offset =
                        262144 + group * 16384 +
                        (move_bit(row, 0, 1) | move_bit(row, 1, 0) | move_bit(row, 2, 4) |
                         move_bit(row, 3, 5) | move_bit(row, 4, 2) | move_bit(row, 5, 13) |
                         move_bit(hidden_channel, 0, 6) | move_bit(hidden_channel, 1, 3) |
                         move_bit(hidden_channel, 2, 9) | move_bit(hidden_channel, 3, 7) |
                         move_bit(hidden_channel, 4, 8) | move_bit(hidden_channel, 5, 10) |
                         move_bit(hidden_channel, 6, 11) | move_bit(hidden_channel, 7, 12));
                    projections[1][(group * 256 + hidden_channel) * 64 + row] =
                        packed_record[expand_offset];

                    unsigned contract_offset =
                        393216 + group * 16384 +
                        (move_bit(hidden_channel, 0, 0) | move_bit(hidden_channel, 1, 1) |
                         move_bit(hidden_channel, 2, 2) | move_bit(hidden_channel, 3, 4) |
                         move_bit(hidden_channel, 4, 5) | move_bit(hidden_channel, 5, 11) |
                         move_bit(hidden_channel, 6, 12) | move_bit(hidden_channel, 7, 13) |
                         move_bit(row, 0, 6) | move_bit(row, 1, 7) | move_bit(row, 2, 8) |
                         move_bit(row, 3, 3) | move_bit(row, 4, 9) | move_bit(row, 5, 10));
                    projections[2][(group * 64 + row) * 256 + hidden_channel] =
                        packed_record[contract_offset];
                }
            }
        }
        return projections;
    }
};

} // namespace phoenix_cpp
