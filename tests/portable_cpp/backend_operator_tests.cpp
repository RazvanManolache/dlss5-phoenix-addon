// Device tests compare fused execution with the same backend's separate passes.
// No model, captured data or numerical tolerance is needed for these contracts.
#include "../../native_runtime/backend.h"
#include "../../native_runtime/backend_options.h"
#include "../../native_runtime/host_math.h"
#include "../../native_runtime/frame.h"
#include <windows.h>
#include <filesystem>
#include <iostream>
#include <cstring>
using namespace phoenix_cpp;
int wmain(int argc, wchar_t** argv) {
    try {
        demand(argc == 4, "Usage: backend_operator_tests RUNTIME BACKEND DEVICE");
        auto root = std::filesystem::absolute(argv[1]);
        std::string kind = std::filesystem::path(argv[2]).string();
        std::string device = std::filesystem::path(argv[3]).string();
        auto module =
            LoadLibraryExW((root / ("phoenix_" + kind + "_cpp.dll")).c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        demand(module, "Backend DLL missing");
        using Create = Backend* (*)(const char*, const wchar_t*, uint32_t, char*, size_t);
        auto create =
            reinterpret_cast<Create>(GetProcAddress(module, "phoenix_backend_create_configured"));
        auto destroy =
            reinterpret_cast<DestroyBackend>(GetProcAddress(module, "phoenix_backend_destroy"));
        demand(create && destroy, "Configured backend ABI missing");
        char error[2048]{};
        auto backend = std::unique_ptr<Backend, DestroyBackend>(
            create(device.c_str(), (root / "vulkan").c_str(), DispatchProfiling | OrderedAttention,
                   error, sizeof(error)),
            destroy);
        demand(bool(backend), error);
        phx::Kernel<HostMath> math{};
        std::vector<Allocation> owned;
        auto allocate = [&](size_t bytes) {
            auto a = backend->allocate(bytes);
            owned.push_back(a);
            return a;
        };
        constexpr unsigned rows = 96, channels = 32, reduction = 128;
        auto left = allocate(rows * reduction * 2), right = allocate(channels * reduction);
        auto skip = allocate(rows * channels * 4), gains = allocate(channels * 4);
        auto projection = allocate(rows * channels * 4), reference = allocate(rows * channels * 4);
        auto fused = allocate(rows * channels * 4);
        std::vector<uint16_t> activations(rows * reduction);
        std::vector<uint8_t> weights(channels * reduction);
        std::vector<float> skip_values(rows * channels), scale(channels);
        for (size_t i = 0; i < activations.size(); ++i)
            activations[i] = uint16_t(math.half_bits(float(int(i * 7 % 19) - 9) * .125f));
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] = uint8_t(32 + i % 57);
        for (size_t i = 0; i < skip_values.size(); ++i)
            skip_values[i] = float(int(i % 13) - 6) * .134567f;
        for (size_t i = 0; i < scale.size(); ++i)
            scale[i] = float(i % 7 + 1) * .123456f;
        backend->upload(left, activations.data(), activations.size() * 2);
        backend->upload(right, weights.data(), weights.size());
        backend->upload(gains, scale.data(), scale.size() * 4);
        for (bool half_skip : {false, true})
            for (unsigned publication : {0u, 4u, 5u}) {
                if (half_skip) {
                    std::vector<uint16_t> packed(skip_values.size());
                    for (size_t i = 0; i < packed.size(); ++i)
                        packed[i] = uint16_t(math.half_bits(skip_values[i]));
                    backend->upload(skip, packed.data(), packed.size() * 2);
                } else
                    backend->upload(skip, skip_values.data(), skip_values.size() * 4);
                phx::Operation matrix{};
                matrix.family = phx::MatrixMultiply;
                matrix.flags = 1 | phx::RightInputFp8;
                matrix.left_input = left.address;
                matrix.right_input = right.address;
                matrix.output = projection.address;
                matrix.row_count = rows;
                matrix.column_count = channels;
                matrix.reduction_count = reduction;
                matrix.batch_count = 1;
                matrix.left_row_stride = matrix.right_row_stride = reduction;
                matrix.output_row_stride = channels;
                phx::Operation residual{};
                residual.left_input = projection.address;
                residual.right_input = skip.address;
                residual.auxiliary = gains.address;
                residual.output = reference.address;
                residual.flags = phx::ScaledResidual | (publication << 8) |
                                 (half_skip ? phx::RightInputHalf : 0);
                residual.column_count = channels;
                auto fused_matrix = matrix;
                fused_matrix.output = fused.address;
                fused_matrix.auxiliary = skip.address;
                fused_matrix.secondary_output = gains.address;
                fused_matrix.flags |= phx::MatrixResidual | (publication << 8) |
                                      (half_skip ? phx::MatrixSkipHalf : 0);
                std::vector<Event> events;
                backend->execute(
                    {{matrix, rows * channels, 0, 0}, {residual, rows * channels, 0, 0}}, events);
                backend->execute({{fused_matrix, rows * channels, 0, 0}}, events);
                std::vector<float> expected(rows * channels), actual(rows * channels);
                backend->download(reference, expected.data(), expected.size() * 4);
                backend->download(fused, actual.data(), actual.size() * 4);
                size_t differences = 0;
                for (size_t i = 0; i < actual.size(); ++i)
                    if (std::memcmp(&actual[i], &expected[i], 4)) {
                        if (differences++ < 4)
                            std::cerr << "element " << i << ": " << expected[i] << " vs "
                                      << actual[i] << '\n';
                    }
                std::cout << "skip_half=" << half_skip << " publication=" << publication
                          << " differences=" << differences << '\n';
                demand(!differences, "Fused residual differs from separate execution");
            }

        for (unsigned heads : {1u, 2u, 16u}) {
            unsigned batches = 2 * heads;
            auto query = allocate(batches * 2048 * 2), key = allocate(batches * 2048 * 2);
            auto value = allocate(batches * 2048 * 2), bias = allocate(heads * 4096 * 4);
            auto scores = allocate(batches * 4096 * 4),
                 probabilities = allocate(batches * 4096 * 2);
            auto context = allocate(batches * 2048 * 4),
                 expected_output = allocate(batches * 2048 * 2);
            auto fused_output = allocate(batches * 2048 * 2);
            std::vector<uint16_t> qkv(batches * 2048);
            std::vector<float> position(heads * 4096);
            for (size_t i = 0; i < qkv.size(); ++i)
                qkv[i] = uint16_t(math.half_bits(float(int(i * 13 % 23) - 11) * .0625f));
            for (size_t i = 0; i < position.size(); ++i)
                position[i] = float(int(i * 17 % 43) - 21) * .007123f;
            backend->upload(query, qkv.data(), qkv.size() * 2);
            std::reverse(qkv.begin(), qkv.end());
            backend->upload(key, qkv.data(), qkv.size() * 2);
            std::rotate(qkv.begin(), qkv.begin() + 97, qkv.end());
            backend->upload(value, qkv.data(), qkv.size() * 2);
            backend->upload(bias, position.data(), position.size() * 4);
            phx::Operation product{};
            product.family = phx::MatrixMultiply;
            product.flags = 1;
            product.left_input = query.address;
            product.right_input = key.address;
            product.output = scores.address;
            product.row_count = product.column_count = 64;
            product.reduction_count = 32;
            product.batch_count = batches;
            product.left_batch_stride = product.right_batch_stride = 2048;
            product.output_batch_stride = 4096;
            product.left_row_stride = product.right_row_stride = 32;
            product.output_row_stride = 64;
            phx::Operation softmax{};
            softmax.family = phx::RowReduction;
            softmax.flags = phx::StoreHalf | phx::AddAttentionBias | 1;
            softmax.left_input = scores.address;
            softmax.right_input = bias.address;
            softmax.output = probabilities.address;
            softmax.auxiliary = scores.address;
            softmax.row_count = batches * 64;
            softmax.column_count = softmax.left_batch_stride = 64;
            softmax.batch_count = heads;
            auto context_product = product;
            context_product.flags = 0;
            context_product.left_input = probabilities.address;
            context_product.right_input = value.address;
            context_product.output = context.address;
            context_product.column_count = 32;
            context_product.reduction_count = 64;
            context_product.left_batch_stride = 4096;
            context_product.output_batch_stride = 2048;
            context_product.left_row_stride = 64;
            context_product.output_row_stride = 32;
            phx::Operation merge{};
            merge.left_input = context.address;
            merge.output = expected_output.address;
            merge.flags = phx::MergeAttentionHeads | phx::StoreHalf | (5 << 8);
            merge.column_count = heads * 32;
            merge.batch_count = heads;
            merge.left_batch_stride = 64;
            phx::Operation attention{};
            attention.family = phx::WindowAttention;
            attention.left_input = query.address;
            attention.right_input = key.address;
            attention.auxiliary = value.address;
            attention.secondary_output = bias.address;
            attention.output = fused_output.address;
            attention.batch_count = batches;
            attention.column_count = heads;
            std::vector<Event> events;
            backend->execute({{product, batches * 4096, 0, 0},
                              {softmax, batches * 64, 0, 0},
                              {context_product, batches * 2048, 0, 0},
                              {merge, batches * 2048, 0, 0}},
                             events);
            backend->execute({{attention, batches, 0, 0}}, events);
            std::vector<uint16_t> expected(batches * 2048), actual(expected.size());
            backend->download(expected_output, expected.data(), expected.size() * 2);
            backend->download(fused_output, actual.data(), actual.size() * 2);
            demand(expected == actual,
                   "Fused window attention differs from ordered separate passes");
            std::cout << "Fused attention heads=" << heads << " exact\n";
        }
        FrameState frame(17, 11, 32, 20, 1);
        std::vector<float> color(17 * 11 * 3), prior(color.size());
        std::vector<uint8_t> validity(17 * 11);
        for (size_t i = 0; i < color.size(); ++i) {
            color[i] = float(i % 37) / 39.f;
            prior[i] = float(i % 19) / 21.f;
        }
        for (size_t i = 0; i < validity.size(); ++i)
            validity[i] = (i % 3) != 0;
        frame.prepare_noise(0);
        auto color_plane = allocate(color.size() * 4), prior_plane = allocate(prior.size() * 4);
        auto noise_plane = allocate(frame.noise.size() * 2),
             valid_plane = allocate(validity.size() + 4);
        auto packed_features = allocate(32 * 20 * 16 * 2);
        backend->upload(color_plane, color.data(), color.size() * 4);
        backend->upload(prior_plane, prior.data(), prior.size() * 4);
        backend->upload(noise_plane, frame.noise.data(), frame.noise.size() * 2);
        backend->upload(valid_plane, validity.data(), validity.size());
        for (bool auto_mask : {false, true}) {
            Controls controls;
            controls.style = 2;
            controls.tone = .731f;
            controls.structure = .813f;
            controls.skin = -1;
            controls.auto_mask = auto_mask;
            auto expected = frame.features(color.data(), controls, prior, validity);
            phx::Operation pack{};
            pack.family = phx::FrameFeatures;
            pack.left_input = color_plane.address;
            pack.right_input = prior_plane.address;
            pack.auxiliary = noise_plane.address;
            pack.secondary_output = valid_plane.address;
            pack.output = packed_features.address;
            pack.row_count = 11;
            pack.column_count = 17;
            pack.reduction_count = 32;
            pack.batch_count = 20;
            pack.left_batch_stride = controls.style;
            pack.right_batch_stride = std::bit_cast<uint32_t>(controls.tone);
            pack.output_batch_stride = std::bit_cast<uint32_t>(controls.structure);
            pack.left_row_stride = std::bit_cast<uint32_t>(controls.skin);
            pack.flags = auto_mask;
            std::vector<Event> ignored;
            backend->execute({{pack, 32 * 20, 0, 0}}, ignored);
            std::vector<uint16_t> actual(expected.size());
            backend->download(packed_features, actual.data(), actual.size() * 2);
            demand(expected == actual,
                   "GPU feature packing differs from host half rounding/mirror padding");
        }
        std::cout << "Device feature packing exact, including odd extents, masks and controls\n";
        for (auto allocation : owned)
            backend->release(allocation);
        backend.reset();
        FreeLibrary(module);
        std::cout << "Backend fused operator contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
