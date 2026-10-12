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
        demand(argc >= 4 && argc <= 6,
               "Usage: backend_operator_tests RUNTIME BACKEND DEVICE [OPTION_BITS] [fp8-only]");
        const auto extra_options = argc >= 5 ? uint32_t(std::stoul(argv[4])) : 0u;
        const bool fp8_only = argc == 6 && std::wstring(argv[5]) == L"fp8-only";
        demand(argc != 6 || fp8_only, "Unknown conversion subset");
        auto root = std::filesystem::absolute(argv[1]);
        std::string kind = std::filesystem::path(argv[2]).string();
        std::string device = std::filesystem::path(argv[3]).string();
        // Match the broker's process-local CPU ICD setup for SYCL tests.
        // No driver is registered or installed on the machine.
        struct PrivateCpuOpenCl {
            DLL_DIRECTORY_COOKIE directory = nullptr;
            HMODULE module = nullptr;
            ~PrivateCpuOpenCl() {
                if (module)
                    FreeLibrary(module);
                if (directory)
                    RemoveDllDirectory(directory);
            }
        } private_cpu;
        if (device == "sycl:cpu" && std::filesystem::exists(root / "opencl_cpu/OpenCL.dll")) {
            private_cpu.directory = AddDllDirectory((root / "opencl_cpu").c_str());
            demand(private_cpu.directory, "Private CPU DLL directory unavailable");
            private_cpu.module =
                LoadLibraryExW((root / "opencl_cpu/OpenCL.dll").c_str(), nullptr,
                               LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            demand(private_cpu.module, "Private CPU OpenCL forwarding library unavailable");
            SetEnvironmentVariableW(L"CL_CONFIG_CPU_TBB_NUM_WORKERS", L"8");
        }
        auto module =
            LoadLibraryExW((root / ("phoenix_" + kind + "_cpp.dll")).c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        demand(module, "Backend DLL missing");
        using Create = Backend* (*)(const char*, const wchar_t*, uint32_t, char*, size_t);
        auto create =
            reinterpret_cast<Create>(GetProcAddress(module, "phoenix_backend_create_configured"));
        auto destroy =
            reinterpret_cast<DestroyBackend>(GetProcAddress(module, "phoenix_backend_destroy"));
        using BasicCreate = Backend* (*)(const char*, const wchar_t*, char*, size_t);
        auto basic_create =
            reinterpret_cast<BasicCreate>(GetProcAddress(module, "phoenix_backend_create"));
        using Features = uint32_t (*)(Backend*);
        auto features =
            reinterpret_cast<Features>(GetProcAddress(module, "phoenix_backend_features"));
        demand((create || basic_create) && destroy, "Backend factory ABI missing");
        char error[2048]{};
        auto backend = std::unique_ptr<Backend, DestroyBackend>(
            create
                ? create(device.c_str(), (root / "vulkan").c_str(),
                         DispatchProfiling | OrderedAttention | extra_options, error, sizeof(error))
                : basic_create(device.c_str(), (root / "vulkan").c_str(), error, sizeof(error)),
            destroy);
        demand(bool(backend), error);
        phx::Kernel<HostMath> math{};
        std::vector<Allocation> owned;
        auto allocate = [&](size_t bytes) {
            auto a = backend->allocate(bytes);
            owned.push_back(a);
            return a;
        };
        // Exercise tile tails, both weight layouts and strided batches. Even K
        // checks masked paired-half gathers; odd K checks the CPU's safe fallback.
        for (bool transposed_weights : {false, true})
            for (unsigned matrix_reduction : {128u, 256u, 259u})
                for (unsigned padding : {0u, 3u}) {
                    constexpr unsigned matrix_rows = 33, matrix_columns = 129, batches = 2;
                    const unsigned left_stride = matrix_reduction + padding,
                                   right_stride =
                                       (transposed_weights ? matrix_reduction : matrix_columns) +
                                       padding,
                                   output_stride = matrix_columns + 5;
                    const size_t left_batch = size_t(matrix_rows) * left_stride,
                                 right_batch = size_t(transposed_weights ? matrix_columns
                                                                         : matrix_reduction) *
                                               right_stride,
                                 output_batch = size_t(matrix_rows) * output_stride;
                    std::vector<uint16_t> left_values(batches * left_batch),
                        expected(batches * output_batch, 0x3555), actual(expected.size());
                    std::vector<uint8_t> right_values(batches * right_batch);
                    for (size_t index = 0; index < left_values.size(); ++index)
                        left_values[index] =
                            uint16_t(math.half_bits(float(int(index * 13 % 23) - 11) * .0625f));
                    for (size_t index = 0; index < right_values.size(); ++index)
                        right_values[index] =
                            uint8_t((32 + index * 17 % 57) | ((index & 1) ? 128 : 0));
                    phx::Operation matrix{};
                    matrix.family = phx::MatrixMultiply;
                    matrix.flags =
                        (transposed_weights ? 1u : 0u) | phx::RightInputFp8 | phx::StoreHalf;
                    matrix.row_count = matrix_rows;
                    matrix.column_count = matrix_columns;
                    matrix.reduction_count = matrix_reduction;
                    matrix.batch_count = batches;
                    matrix.left_row_stride = left_stride;
                    matrix.right_row_stride = right_stride;
                    matrix.output_row_stride = output_stride;
                    matrix.left_batch_stride = left_batch;
                    matrix.right_batch_stride = right_batch;
                    matrix.output_batch_stride = output_batch;
                    matrix.left_input = left_values.data();
                    matrix.right_input = right_values.data();
                    matrix.output = expected.data();
                    phx::Kernel<HostMath> reference_matrix{matrix};
                    for (uint64_t index = 0;
                         index < uint64_t(batches) * matrix_rows * matrix_columns; ++index)
                        reference_matrix.gemm(index);
                    auto a = allocate(left_values.size() * 2), b = allocate(right_values.size()),
                         c = allocate(expected.size() * 2);
                    std::vector<uint16_t> sentinel(expected.size(), 0x3555);
                    backend->upload(a, left_values.data(), left_values.size() * 2);
                    backend->upload(b, right_values.data(), right_values.size());
                    backend->upload(c, sentinel.data(), sentinel.size() * 2);
                    matrix.left_input = a.address;
                    matrix.right_input = b.address;
                    matrix.output = c.address;
                    std::vector<Event> events;
                    backend->execute(
                        {{matrix, uint64_t(batches) * matrix_rows * matrix_columns, 0, 0}}, events);
                    backend->download(c, actual.data(), actual.size() * 2);
                    demand(actual == expected,
                           "Matrix tile tails, batch strides or output padding differ");
                }
        std::cout << "Matrix tile tails, both weight layouts and batch strides exact\n";
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
                matrix.flags =
                    1 | phx::RightInputFp8 |
                    ((extra_options & (CudaDirectFp8 | LosslessFp8Matrices)) ? phx::LeftInputFp8Grid
                                                                             : 0);
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

        // Reference-only backends have no fused attention operator. Exercise
        // their normalization, GEMM, packing and conversion contracts below.
        if (features && (features(backend.get()) & FusedWindowAttention))
            for (unsigned heads : {1u, 2u, 4u, 8u, 16u}) {
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
        // Tail rows and both interleaved Q/K views must retain the reference
        // cosine reduction tree, even when eight lanes now cooperate on a row.
        for (unsigned heads : {1u, 2u, 4u, 8u})
            for (unsigned part : {0u, 1u}) {
                constexpr unsigned row_count = 129;
                const unsigned windows = (row_count + heads * 64 - 1) / (heads * 64);
                std::vector<float> projection(windows * heads * 64 * 96);
                std::vector<float> gains(heads);
                for (size_t index = 0; index < projection.size(); ++index)
                    projection[index] = float(int(index * 17 % 311) - 155) * .001131f;
                for (unsigned index = 0; index < heads; ++index)
                    gains[index] = .72341f + index * .023411f;
                auto input = allocate(projection.size() * 4), gain = allocate(gains.size() * 4);
                auto output = allocate(row_count * 32 * 2);
                backend->upload(input, projection.data(), projection.size() * 4);
                backend->upload(gain, gains.data(), gains.size() * 4);
                phx::Operation normalize{};
                normalize.family = phx::RowReduction;
                normalize.flags = phx::StoreHalf | phx::InterleavedQkv | (part << 18);
                normalize.left_input = input.address;
                normalize.auxiliary = gain.address;
                normalize.output = output.address;
                normalize.row_count = row_count;
                normalize.column_count = 64;
                normalize.batch_count = heads;
                normalize.reduction_count = part == 0;
                std::vector<Event> events;
                backend->execute({{normalize, row_count, 0, 0}}, events);
                std::vector<uint16_t> expected(row_count * 32), actual(expected.size());
                backend->download(output, actual.data(), actual.size() * 2);
                normalize.left_input = projection.data();
                normalize.auxiliary = gains.data();
                normalize.output = expected.data();
                for (unsigned row = 0; row < row_count; ++row)
                    phx::Kernel<HostMath>{normalize}.norm(row);
                demand(actual == expected,
                       "Cooperative cosine normalization differs from reference");
            }
        std::cout << "Cosine normalization exact for Q/K, head counts and partial rows\n";
        // The streamed local normalizer trades an extra read for fewer private
        // temporaries. Compare every published half bit with the original tree.
        for (unsigned heads : {1u, 2u, 4u, 8u, 16u}) {
            constexpr unsigned rows = 129;
            std::vector<float> scores(rows * 64), bias(heads * 4096);
            for (size_t index = 0; index < scores.size(); ++index)
                scores[index] = float(int(index * 17 % 311) - 155) * .132413f;
            for (size_t index = 0; index < bias.size(); ++index)
                bias[index] = float(int(index * 19 % 47) - 23) * .006712f;
            auto input = allocate(scores.size() * 4), position = allocate(bias.size() * 4);
            auto output = allocate(scores.size() * 2);
            backend->upload(input, scores.data(), scores.size() * 4);
            backend->upload(position, bias.data(), bias.size() * 4);
            phx::Operation normalize{};
            normalize.family = phx::RowReduction;
            normalize.flags = 1 | phx::StoreHalf | phx::AddAttentionBias;
            normalize.left_input = input.address;
            normalize.right_input = position.address;
            normalize.output = output.address;
            normalize.row_count = rows;
            normalize.column_count = 64;
            normalize.batch_count = heads;
            std::vector<Event> events;
            backend->execute({{normalize, rows, 0, 0}}, events);
            std::vector<uint16_t> actual(scores.size()), expected(scores.size());
            backend->download(output, actual.data(), actual.size() * 2);
            normalize.left_input = scores.data();
            normalize.right_input = bias.data();
            normalize.output = expected.data();
            for (unsigned row = 0; row < rows; ++row)
                phx::Kernel<HostMath>{normalize}.softmax(row);
            demand(actual == expected, "Local row normalization differs from reference");
        }
        std::cout << "Local row normalization exact for all head counts and row tails\n";
        // Global rows use a half-rounded reduction tree and publish the inverse
        // sum separately. Check both partial 64-key tiles and padded row strides.
        for (unsigned width : {1u, 63u, 64u, 65u, 640u})
            for (unsigned padding : {0u, 7u, 72u}) {
                constexpr unsigned row_count = 129;
                const unsigned stride = width + padding;
                std::vector<float> scores(row_count * stride);
                for (size_t index = 0; index < scores.size(); ++index)
                    scores[index] = float(int(index * 19 % 701) - 350) * .007123f;
                auto input = allocate(scores.size() * 4), output = allocate(scores.size() * 2);
                auto inverse = allocate(row_count * 4);
                backend->upload(input, scores.data(), scores.size() * 4);
                phx::Operation normalize{};
                normalize.family = phx::RowReduction;
                normalize.flags = 6u | phx::StoreHalf;
                normalize.left_input = input.address;
                normalize.output = output.address;
                normalize.auxiliary = inverse.address;
                normalize.row_count = row_count;
                normalize.column_count = width;
                normalize.left_batch_stride = stride;
                std::vector<Event> events;
                backend->execute({{normalize, row_count, 0, 0}}, events);
                std::vector<uint16_t> actual(scores.size()), expected(scores.size());
                std::vector<float> actual_inverse(row_count), expected_inverse(row_count);
                backend->download(output, actual.data(), actual.size() * 2);
                backend->download(inverse, actual_inverse.data(), actual_inverse.size() * 4);
                normalize.left_input = scores.data();
                normalize.output = expected.data();
                normalize.auxiliary = expected_inverse.data();
                for (unsigned row = 0; row < row_count; ++row)
                    phx::Kernel<HostMath>{normalize}.global_attention_softmax(row);
                demand(actual == expected &&
                           std::memcmp(actual_inverse.data(), expected_inverse.data(),
                                       row_count * 4) == 0,
                       "Global row normalization differs from reference or overwrites padding");
            }
        std::cout
            << "Global row normalization exact for tile boundaries, row tails and padded strides\n";
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
        // Conversion tests exercise every finite half value, rounding midpoints
        // and arbitrary finite FP32 patterns. Compare bits, including signed zero.
        std::vector<float> rounding_inputs;
        for (unsigned code = 0; code < 65536; ++code) {
            float value = math.half_value(code);
            if (std::isfinite(value))
                rounding_inputs.push_back(value);
            if (code < 0x7bffu) {
                float midpoint = (value + math.half_value(code + 1)) * .5f;
                rounding_inputs.push_back(midpoint);
                rounding_inputs.push_back(-midpoint);
                rounding_inputs.push_back(std::nextafter(midpoint, 0.f));
                rounding_inputs.push_back(std::nextafter(midpoint, 65536.f));
            }
        }
        uint32_t random_bits = 1059;
        for (unsigned index = 0; index < 262144; ++index) {
            random_bits ^= random_bits << 13;
            random_bits ^= random_bits >> 17;
            random_bits ^= random_bits << 5;
            float value = HostMath::value(random_bits);
            if (std::isfinite(value))
                rounding_inputs.push_back(value);
        }
        auto rounding_source = allocate(rounding_inputs.size() * 4);
        auto rounding_output = allocate(rounding_inputs.size() * 4);
        backend->upload(rounding_source, rounding_inputs.data(), rounding_inputs.size() * 4);
        const std::vector<unsigned> conversion_kinds =
            fp8_only ? std::vector<unsigned>{phx::QuantizeFp8}
                     : std::vector<unsigned>{phx::RoundHalf, phx::FloatToHalf, phx::QuantizeFp8};
        for (unsigned kind : conversion_kinds) {
            phx::Operation convert{};
            convert.flags = kind;
            convert.left_input = rounding_source.address;
            convert.output = rounding_output.address;
            convert.row_count = unsigned(rounding_inputs.size());
            std::vector<Event> events;
            backend->execute({{convert, rounding_inputs.size(), 0, 0}}, events);
            const unsigned bytes = kind == phx::FloatToHalf ? 2 : 4;
            std::vector<uint8_t> actual(rounding_inputs.size() * bytes);
            backend->download(rounding_output, actual.data(), actual.size());
            for (size_t index = 0; index < rounding_inputs.size(); ++index) {
                const float value = rounding_inputs[index];
                uint32_t expected =
                    kind == phx::FloatToHalf
                        ? math.half_bits(value)
                        : HostMath::bits(kind == phx::RoundHalf ? math.half(value)
                                                                : math.quantize_e4m3(value));
                if (std::memcmp(actual.data() + index * bytes, &expected, bytes)) {
                    uint32_t actual_bits = 0;
                    std::memcpy(&actual_bits, actual.data() + index * bytes, bytes);
                    std::cerr << "conversion kind=" << kind << " index=" << index
                              << " input_bits=" << std::hex << HostMath::bits(value)
                              << " expected=" << expected << " actual=" << actual_bits << std::dec
                              << '\n';
                    demand(false, "Device conversion differs from portable rounding reference");
                }
            }
        }
        std::cout << rounding_inputs.size() * conversion_kinds.size()
                  << " finite conversion checks exact\n";
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
