#pragma once

#include "backend.h"
#include "host_math.h"
#include "../src/native/model/weight_source.h"

#include <array>
#include <fstream>
#include <map>

namespace phoenix_cpp {

#include "weight_layouts.h"

using Json = nlohmann::json;

inline std::string file_sha(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    demand(bool(file), "Model DLL open failed");
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
    return phoenix_model::weight_digest(bytes.data(), bytes.size());
}

struct TensorBytes {
    std::vector<uint8_t> data;
};

// Decodes a supplied model resource into named tensors. This class owns the
// decoded host copies; the graph uploads its weights before discarding them.
class Model {
    std::map<std::string, TensorBytes> tensors_;
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

  public:
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

    explicit Model(const phoenix_model::WeightSource& source) {
        phoenix_model::with_model_resource(
            source, file_sha, [&](const uint8_t* resource_bytes, size_t resource_size) {
                auto records = phoenix_model::resource_records(resource_bytes, resource_size);

                for (auto& [record_name, record] : records) {
                    auto separator = record_name.rfind('.');
                    demand(separator != std::string::npos, "Malformed tensor name");
                    auto layer_prefix = record_name.substr(0, separator);
                    auto record_bytes = resource_bytes + record.byte_offset;

                    auto store_tensor = [&](std::string field, std::vector<uint8_t> bytes) {
                        demand(
                            tensors_
                                .emplace(layer_prefix + '.' + field, TensorBytes{std::move(bytes)})
                                .second,
                            "Duplicate tensor");
                    };

                    // Standalone fields already have their final byte layout.
                    if (record_name.substr(separator + 1) != "layer") {
                        store_tensor(record_name.substr(separator + 1),
                                     {record_bytes, record_bytes + record.bytes});
                        continue;
                    }
                    if (record.bytes == 2) {
                        store_tensor("scale", {record_bytes, record_bytes + 2});
                        continue;
                    }

                    // Larger layer records contain several tensors. The
                    // generated table supplies each tensor's extent and kind.
                    visit_layout(record.bytes, [&](TensorPartLayout part) {
                        demand(part.byte_end <= record.bytes, "Tensor part out of bounds");
                        std::vector<uint8_t> decoded_bytes(part.byte_end - part.byte_begin);
                        auto packed_bytes = record_bytes + part.byte_begin;

                        if (part.encoding == RawBytes)
                            std::memcpy(decoded_bytes.data(), packed_bytes, decoded_bytes.size());

                        if (part.encoding == PackedFp8) {
                            // Restore ordinary row-major FP8 matrices without
                            // changing their encoded numerical values.
                            for (unsigned row = 0; row < part.rows; ++row) {
                                for (unsigned column = 0; column < part.columns; ++column) {
                                    auto code = packed_bytes[packed_offset(
                                        row, column, part.rows, part.columns, part.inner_tiles)];
                                    demand((code & 127) != 127, "NaN FP8 weight");
                                    decoded_bytes[size_t(row) * part.columns + column] = code;
                                }
                            }
                        }

                        if (part.encoding == InterleavedQkv) {
                            // The global-attention Q/K/V matrices share tiled
                            // storage. Separate them into three 1024x1024 arrays.
                            for (unsigned matrix_index = 0; matrix_index < 3; ++matrix_index) {
                                std::vector<uint8_t> matrix(1024 * 1024);
                                for (unsigned row = 0; row < 1024; ++row) {
                                    for (unsigned column = 0; column < 1024; ++column) {
                                        unsigned lane_offset =
                                            (column & 1) | ((column >> 1 & 1) << 4) |
                                            ((column >> 2 & 1) << 5) | ((column >> 3 & 1) << 1) |
                                            ((column >> 4 & 1) << 2) | ((row & 1) << 6) |
                                            ((row >> 1 & 1) << 7) | ((row >> 2 & 1) << 8) |
                                            ((row >> 3 & 1) << 3) | ((row >> 4 & 1) << 9);
                                        unsigned tile_index = (row >> 5) | ((column >> 5) << 5);
                                        matrix[size_t(row) * 1024 + column] =
                                            packed_bytes[(3 * tile_index + matrix_index) * 1024 +
                                                         lane_offset];
                                    }
                                }
                                store_tensor(std::string(1, "qkv"[matrix_index]),
                                             std::move(matrix));
                            }
                            return;
                        }

                        if (part.encoding == PackedProjection) {
                            // The small projection is a 16x32 FP16 matrix;
                            // copy both bytes of each value after unswizzling.
                            for (unsigned row = 0; row < 16; ++row) {
                                for (unsigned column = 0; column < 32; ++column) {
                                    unsigned packed_index = (column / 16) * 256 + 32 * (row % 8) +
                                                            8 * ((column % 8) / 2) + 4 * (row / 8) +
                                                            column % 2 + 2 * ((column % 16) / 8);
                                    std::memcpy(decoded_bytes.data() + 2 * (row * 32 + column),
                                                packed_bytes + 2 * packed_index, 2);
                                }
                            }
                        }
                        store_tensor(part.name, std::move(decoded_bytes));
                    });
                }
                demand(tensors_.size() == 658, "Incomplete model tensor catalogue");
                return 0;
            });
    }

    const std::vector<uint8_t>& raw(int block, int layer, const std::string& field) const {
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
