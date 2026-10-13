#pragma once

// Compare the nine-operation shared-memory window block with nine separate
// launches on the same backend. Independent allocations deliberately keep the
// entry operands immutable. Exercise multiple CTAs and both residual formats.
inline void test_window_block_contract(phoenix_cpp::Backend& backend) {
    using namespace phoenix_cpp;
    phx::Kernel<HostMath> math{};
    for (unsigned windows : {1u, 3u, 17u}) {
        for (bool half_skip : {false, true}) {
            std::vector<Allocation> owned;
            struct Release {
                Backend& backend;
                std::vector<Allocation>& owned;
                ~Release() {
                    try {
                        backend.wait();
                        for (auto allocation : owned)
                            backend.release(allocation);
                    } catch (...) {
                        // Preserve the original test failure during unwinding.
                    }
                }
            } release{backend, owned};
            auto allocate = [&](size_t bytes) {
                auto allocation = backend.allocate(bytes);
                owned.push_back(allocation);
                return allocation;
            };
            const unsigned rows = windows * 64;
            auto input = allocate(size_t(rows) * 32 * 2);
            auto skip = allocate(size_t(rows) * 32 * (half_skip ? 2 : 4));
            auto hidden = allocate(size_t(rows) * 128 * 2);
            auto residual = allocate(size_t(rows) * 32 * 4);
            auto quantized = allocate(size_t(rows) * 32 * 2);
            auto projected = allocate(size_t(rows) * 96 * 4);
            auto query = allocate(size_t(rows) * 32 * 2);
            auto key = allocate(size_t(rows) * 32 * 2);
            auto value = allocate(size_t(rows) * 32 * 2);
            auto context = allocate(size_t(rows) * 32 * 2);
            auto output = allocate(size_t(rows) * 32 * 4);
            auto expand_weights = allocate(128 * 32), contract_weights = allocate(32 * 128);
            auto qkv_weights = allocate(96 * 32), output_weights = allocate(32 * 32);
            auto gains = allocate(32 * 4), head_scale = allocate(4), bias = allocate(4096 * 4);
            std::vector<uint16_t> input_values(size_t(rows) * 32),
                half_skip_values(input_values.size());
            std::vector<float> skip_values(input_values.size());
            for (size_t index = 0; index < input_values.size(); ++index) {
                input_values[index] =
                    uint16_t(math.half_bits(float(int(index * 13 % 31) - 15) * .0625f));
                skip_values[index] = float(int(index * 17 % 43) - 21) * .007123f;
                half_skip_values[index] = uint16_t(math.half_bits(skip_values[index]));
            }
            backend.upload(input, input_values.data(), input.bytes);
            backend.upload(skip,
                           half_skip ? static_cast<void*>(half_skip_values.data())
                                     : static_cast<void*>(skip_values.data()),
                           skip.bytes);
            for (auto weights : {expand_weights, contract_weights, qkv_weights, output_weights}) {
                std::vector<uint8_t> values(size_t(weights.bytes));
                for (size_t index = 0; index < values.size(); ++index)
                    values[index] = uint8_t((24 + index * 17 % 33) | ((index & 1) ? 128 : 0));
                backend.upload(weights, values.data(), values.size());
            }
            std::vector<float> gain_values(32), bias_values(4096);
            for (unsigned channel = 0; channel < 32; ++channel)
                gain_values[channel] = .731f + float(channel) * .00123f;
            for (unsigned index = 0; index < 4096; ++index)
                bias_values[index] = float(int(index * 17 % 43) - 21) * .007123f;
            const float learned_head_scale = .7134f;
            backend.upload(gains, gain_values.data(), gains.bytes);
            backend.upload(head_scale, &learned_head_scale, 4);
            backend.upload(bias, bias_values.data(), bias.bytes);

            constexpr unsigned PackedMatrix = 1 | phx::RightInputFp8 | phx::LeftInputFp8Grid;
            auto matrix = [&](Allocation left, Allocation right, Allocation destination,
                              unsigned columns, unsigned reduction, unsigned flags) {
                phx::Operation operation{};
                operation.family = phx::MatrixMultiply;
                operation.flags = PackedMatrix | flags;
                operation.left_input = left.address;
                operation.right_input = right.address;
                operation.output = destination.address;
                operation.row_count = rows;
                operation.column_count = columns;
                operation.reduction_count = reduction;
                operation.batch_count = 1;
                operation.left_row_stride = reduction;
                operation.right_row_stride = reduction;
                operation.output_row_stride = columns;
                return operation;
            };
            auto expand = matrix(input, expand_weights, hidden, 128, 32,
                                 phx::StoreHalf | (phx::PublishFusedGateFp8 << 8));
            auto contract = matrix(hidden, contract_weights, residual, 32, 128,
                                   phx::MatrixResidual | (half_skip ? phx::MatrixSkipHalf : 0));
            contract.auxiliary = skip.address;
            contract.secondary_output = gains.address;
            phx::Operation quantize{};
            quantize.flags = phx::CopyValues | phx::StoreHalf | (phx::PublishHalfThenFp8 << 8);
            quantize.left_input = residual.address;
            quantize.output = quantized.address;
            auto project_qkv = matrix(quantized, qkv_weights, projected, 96, 32, 0);
            phx::Operation normalize_query{};
            normalize_query.family = phx::RowReduction;
            normalize_query.flags = phx::StoreHalf | phx::InterleavedQkv;
            normalize_query.left_input = projected.address;
            normalize_query.output = query.address;
            normalize_query.auxiliary = head_scale.address;
            normalize_query.row_count = rows;
            normalize_query.column_count = 64;
            normalize_query.batch_count = normalize_query.reduction_count = 1;
            auto normalize_key = normalize_query;
            normalize_key.flags |= 1u << 18;
            normalize_key.output = key.address;
            normalize_key.reduction_count = 0;
            phx::Operation split_value{};
            split_value.flags =
                phx::SplitQkvHeads | phx::StoreHalf | (phx::PublishHalfThenFp8 << 8);
            split_value.left_input = projected.address;
            split_value.output = value.address;
            split_value.column_count = 32;
            split_value.batch_count = 1;
            split_value.left_batch_stride = 64;
            split_value.right_batch_stride = 2;
            phx::Operation attention{};
            attention.family = phx::WindowAttention;
            attention.left_input = query.address;
            attention.right_input = key.address;
            attention.auxiliary = value.address;
            attention.secondary_output = bias.address;
            attention.output = context.address;
            attention.column_count = 1;
            attention.batch_count = windows;
            auto project_output =
                matrix(context, output_weights, output, 32, 32, phx::MatrixResidual);
            project_output.auxiliary = residual.address;
            project_output.secondary_output = gains.address;
            std::vector<Command> commands = {{expand, uint64_t(rows) * 128, 0, 0},
                                             {contract, uint64_t(rows) * 32, 0, 0},
                                             {quantize, uint64_t(rows) * 32, 0, 0},
                                             {project_qkv, uint64_t(rows) * 96, 0, 0},
                                             {normalize_query, rows, 0, 0},
                                             {normalize_key, rows, 0, 0},
                                             {split_value, uint64_t(rows) * 32, 0, 0},
                                             {attention, windows, 0, 0},
                                             {project_output, uint64_t(rows) * 32, 0, 0}};
            std::vector<Event> events;
            for (auto command : commands)
                backend.execute({command}, events);
            std::vector<float> expected(size_t(rows) * 32), actual(expected.size());
            backend.download(output, expected.data(), output.bytes);
            for (unsigned replay = 0; replay < 3; ++replay) {
                backend.execute(commands, events);
                backend.download(output, actual.data(), output.bytes);
                demand(std::memcmp(expected.data(), actual.data(), output.bytes) == 0,
                       "Shared window block differs from nine separate operations");
            }
        }
    }
    std::cout << "Shared window block: six shapes/residual formats, three replays each exact\n";
}
