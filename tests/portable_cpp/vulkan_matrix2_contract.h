#pragma once
#include <array>
#include <cstring>
#include <iostream>
#include <vector>

// Compare grouped dispatches against individually submitted original operations.
// Partial rows and legacy +/-480 weight codes exercise routing, not tolerances.
template <class Allocate>
void test_vulkan_matrix2_contract(phoenix_cpp::Backend& backend, Allocate& allocate) {
    using namespace phoenix_cpp;
    phx::Kernel<HostMath> math{};
    unsigned checks = 0;
    for (unsigned rows : {1u, 17u, 31u, 32u, 33u, 63u, 64u, 129u}) {
        std::cout << "Workgroup fusion rows=" << rows << std::endl;
        std::vector<uint16_t> input(rows * 32), skip_half(rows * 32);
        std::vector<float> skip(rows * 32), gains(32, .07113f);
        std::vector<uint8_t> expand_weights(128 * 32), contract_weights(32 * 128),
            project_weights(96 * 32);
        for (size_t index = 0; index < input.size(); ++index) {
            input[index] = uint16_t(math.half_bits(float(int(index * 13 % 31) - 15) * .125f));
            skip[index] = float(int(index * 19 % 41) - 20) * .03117f;
            skip_half[index] = uint16_t(math.half_bits(skip[index]));
        }
        for (auto* weights : {&expand_weights, &contract_weights, &project_weights})
            for (size_t index = 0; index < weights->size(); ++index)
                (*weights)[index] = uint8_t((32 + index * 17 % 57) | ((index & 1) ? 128 : 0));
        auto input_buffer = allocate(input.size() * 2), expansion = allocate(expand_weights.size()),
             contraction = allocate(contract_weights.size()),
             projection = allocate(project_weights.size()), hidden = allocate(rows * 128 * 2),
             output = allocate(rows * 32 * 4), skip_buffer = allocate(rows * 32 * 4),
             gain = allocate(gains.size() * 4);
        backend.upload(input_buffer, input.data(), input_buffer.bytes);
        backend.upload(contraction, contract_weights.data(), contraction.bytes);
        backend.upload(gain, gains.data(), gain.bytes);
        phx::Operation expand{};
        expand.family = phx::MatrixMultiply;
        expand.flags = 1 | phx::StoreHalf | (phx::PublishFusedGateFp8 << 8) | phx::RightInputFp8 |
                       phx::LeftInputFp8Grid;
        expand.left_input = input_buffer.address;
        expand.right_input = expansion.address;
        expand.output = hidden.address;
        expand.row_count = rows;
        expand.column_count = 128;
        expand.reduction_count = 32;
        expand.batch_count = 1;
        expand.left_row_stride = expand.right_row_stride = 32;
        expand.output_row_stride = 128;
        phx::Operation contract{};
        contract.family = phx::MatrixMultiply;
        contract.flags = 1 | phx::RightInputFp8 | phx::LeftInputFp8Grid | phx::MatrixResidual;
        contract.left_input = hidden.address;
        contract.right_input = contraction.address;
        contract.output = output.address;
        contract.auxiliary = skip_buffer.address;
        contract.secondary_output = gain.address;
        contract.row_count = rows;
        contract.column_count = 32;
        contract.reduction_count = 128;
        contract.batch_count = 1;
        contract.left_row_stride = contract.right_row_stride = 128;
        contract.output_row_stride = 32;
        std::vector<Event> events;
        for (bool half_skip : {false, true}) {
            contract.flags = 1 | phx::RightInputFp8 | phx::LeftInputFp8Grid | phx::MatrixResidual |
                             (half_skip ? phx::MatrixSkipHalf : 0);
            backend.upload(skip_buffer,
                           half_skip ? static_cast<const void*>(skip_half.data()) : skip.data(),
                           rows * 32 * (half_skip ? 2 : 4));
            std::vector<Command> pair{{expand, rows * 128, 0, 0}, {contract, rows * 32, 0, 0}};
            for (unsigned encoding : {0u, 1u, 2u}) {
                auto weights = expand_weights;
                if (encoding == 1)
                    weights[0] = 0x7f; // Legacy +480, hardware NaN.
                backend.upload(expansion, weights.data(), weights.size());
                if (encoding == 2) {
                    uint8_t code = 0xff; // Invalidate a previously compatible allocation.
                    auto view = expansion;
                    view.address = reinterpret_cast<void*>(uintptr_t(view.address) + 1);
                    view.bytes = 1;
                    backend.upload(view, &code, 1);
                }
                backend.execute(pair, events);
                std::vector<float> fused(rows * 32), separate(fused.size());
                backend.download(output, fused.data(), output.bytes);
                backend.execute({pair[0]}, events);
                backend.execute({pair[1]}, events);
                backend.download(output, separate.data(), output.bytes);
                if (std::memcmp(fused.data(), separate.data(), output.bytes) != 0)
                    for (size_t index = 0; index < fused.size(); ++index)
                        if (std::memcmp(&fused[index], &separate[index], 4)) {
                            std::cerr << "FF mismatch row=" << rows << " half=" << half_skip
                                      << " encoding=" << encoding << " index=" << index
                                      << " fused=" << fused[index]
                                      << " separate=" << separate[index] << std::endl;
                            break;
                        }
                demand(std::memcmp(fused.data(), separate.data(), output.bytes) == 0,
                       "Workgroup feed-forward differs from separate passes");
                ++checks;
            }
        }
        auto projected = allocate(rows * 96 * 4), q = allocate(rows * 32 * 2),
             k = allocate(rows * 32 * 2), v = allocate(rows * 32 * 2);
        phx::Operation project = expand;
        project.flags = 1 | phx::RightInputFp8 | phx::LeftInputFp8Grid;
        project.right_input = projection.address;
        project.output = projected.address;
        project.column_count = project.output_row_stride = 96;
        std::vector<Command> group{{project, rows * 96, 0, 0}};
        for (unsigned part = 0; part < 3; ++part) {
            phx::Operation prepare{};
            prepare.left_input = projected.address;
            prepare.auxiliary = gain.address;
            prepare.output = part == 0 ? q.address : part == 1 ? k.address : v.address;
            prepare.row_count = rows;
            prepare.batch_count = 1;
            prepare.column_count = 64;
            if (part < 2) {
                prepare.family = phx::RowReduction;
                prepare.flags = phx::StoreHalf | phx::InterleavedQkv | (part << 18);
                prepare.reduction_count = part == 0;
            } else {
                prepare.family = phx::Elementwise;
                prepare.flags = phx::SplitQkvHeads | phx::StoreHalf | (5u << 8);
                prepare.column_count = 32;
                prepare.left_batch_stride = 64;
                prepare.right_batch_stride = 2;
            }
            group.push_back({prepare, rows * (part == 2 ? 32 : 1), 0, 0});
        }
        for (unsigned encoding : {0u, 1u, 2u}) {
            auto weights = project_weights;
            if (encoding == 1)
                weights[0] = 0x7f;
            backend.upload(projection, weights.data(), weights.size());
            if (encoding == 2) {
                uint8_t code = 0xff;
                auto view = projection;
                view.address = reinterpret_cast<void*>(uintptr_t(view.address) + 1);
                view.bytes = 1;
                backend.upload(view, &code, 1);
            }
            backend.execute(group, events);
            std::array<std::vector<uint16_t>, 3> fused, separate;
            const std::array<Allocation, 3> buffers{q, k, v};
            for (unsigned part = 0; part < 3; ++part) {
                fused[part].resize(rows * 32);
                backend.download(buffers[part], fused[part].data(), buffers[part].bytes);
            }
            for (const auto& command : group)
                backend.execute({command}, events);
            for (unsigned part = 0; part < 3; ++part) {
                separate[part].resize(rows * 32);
                backend.download(buffers[part], separate[part].data(), buffers[part].bytes);
            }
            demand(fused == separate, "Projected QKV differs from separate passes");
            ++checks;
        }
    }
    std::cout << checks << " workgroup fusion/legacy encoding/tail contracts exact\n";
    for (unsigned rows : {513u, 529u})
        for (unsigned columns : {64u, 129u, 256u}) {
            constexpr unsigned reduction = 256, left_stride = 264, right_stride = 272, batches = 2;
            unsigned output_stride = columns + 3;
            size_t left_batch = size_t(rows) * left_stride,
                   right_batch = size_t(columns) * right_stride,
                   output_batch = size_t(rows) * output_stride + 16;
            std::vector<uint16_t> inputs(batches * left_batch);
            std::vector<uint8_t> weights(batches * right_batch);
            std::vector<float> sentinel(batches * output_batch, 71315.5f),
                expected(sentinel.size()), actual(sentinel.size());
            for (size_t index = 0; index < inputs.size(); ++index)
                inputs[index] = uint16_t(math.half_bits(float(int(index * 13 % 31) - 15) * .0625f));
            for (size_t index = 0; index < weights.size(); ++index)
                weights[index] = uint8_t((24 + index * 17 % 33) | ((index & 1) ? 128 : 0));
            auto input = allocate(inputs.size() * 2), weight = allocate(weights.size()),
                 output = allocate(sentinel.size() * 4);
            backend.upload(input, inputs.data(), input.bytes);
            backend.upload(weight, weights.data(), weight.bytes);
            phx::Operation matrix{};
            matrix.family = phx::MatrixMultiply;
            matrix.flags = 1 | phx::RightInputFp8;
            matrix.left_input = input.address;
            matrix.right_input = weight.address;
            matrix.output = output.address;
            matrix.row_count = rows;
            matrix.column_count = columns;
            matrix.reduction_count = reduction;
            matrix.batch_count = batches;
            matrix.left_row_stride = left_stride;
            matrix.right_row_stride = right_stride;
            matrix.output_row_stride = output_stride;
            matrix.left_batch_stride = left_batch;
            matrix.right_batch_stride = right_batch;
            matrix.output_batch_stride = output_batch;
            std::vector<Event> events;
            backend.upload(output, sentinel.data(), output.bytes);
            backend.execute({{matrix, uint64_t(rows) * columns * batches, 0, 0}}, events);
            backend.download(output, expected.data(), output.bytes);
            matrix.flags |= phx::LeftInputFp8Grid;
            backend.upload(output, sentinel.data(), output.bytes);
            backend.execute({{matrix, uint64_t(rows) * columns * batches, 0, 0}}, events);
            backend.download(output, actual.data(), output.bytes);
            demand(std::memcmp(expected.data(), actual.data(), output.bytes) == 0,
                   "Shape-selected workgroup matrix changed output or padding");
        }
    std::cout << "32/64-row workgroup matrices exact for row/column tails and strided batches\n";
}
