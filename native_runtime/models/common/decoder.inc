// Shared packed-family tensor decoding; each architecture supplies its own layout visitor.
    template <class Records, class ReadRecord, class VisitLayout>
    void decode_record_layout(const Records& records, ReadRecord read_record, VisitLayout visit_layout) {
        for (auto& [record_name, record] : records) {
            auto separator = record_name.rfind('.');
            demand(separator != std::string::npos, "Malformed tensor name");
            auto layer_prefix = record_name.substr(0, separator);
            const auto storage = read_record(record);
            demand(storage.size() == record.bytes, "Incomplete model record read");
            const auto* record_bytes = storage.data();

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
        demand(tensors_.size() == definition().decoded_tensor_count, "Incomplete model tensor catalogue");
    }

    template <class VisitLayout>
    void decode_layout(const uint8_t* resource_bytes, size_t resource_size, VisitLayout visit_layout) {
        const auto records = phoenix_model::resource_records(resource_bytes, resource_size, definition());
        decode_record_layout(records, [&](const phoenix_model::WeightRecord& record) {
            return std::span<const uint8_t>(resource_bytes + record.byte_offset, record.bytes);
        }, visit_layout);
    }
