// Offline profiling of individual portable commands. Model and input stay external.
#include "../../native_runtime/graph.h"
#include "../../native_runtime/frame.h"
#include "../../native_runtime/backend_options.h"
#include <iostream>
#include <algorithm>
#include <chrono>

using namespace phoenix_cpp;
int wmain(int argc, wchar_t** argv) {
    try {
        demand(argc == 3, "Usage: graph_profile JOB.json NEW_REPORT.json");
        demand(!std::filesystem::exists(argv[2]), "Fresh report required");
        Json job;
        std::ifstream(argv[1]) >> job;
        const auto& config = job.at("config");
        auto root = std::filesystem::u8path(config.at("runtime_root").get<std::string>());
        const auto kind = config.at("backend").get<std::string>();
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
        if (config.at("device") == "sycl:cpu") {
            private_cpu.directory = AddDllDirectory((root / "opencl_cpu").c_str());
            private_cpu.module =
                LoadLibraryExW((root / "opencl_cpu/OpenCL.dll").c_str(), nullptr,
                               LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            demand(private_cpu.directory && private_cpu.module, "Private CPU ICD unavailable");
            SetEnvironmentVariableW(L"CL_CONFIG_CPU_TBB_NUM_WORKERS", L"32");
        }
        auto module =
            LoadLibraryExW((root / ("phoenix_" + kind + "_cpp.dll")).c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        demand(module, "Backend DLL unavailable");
        using Create = Backend* (*)(const char*, const wchar_t*, uint32_t, char*, size_t);
        auto create =
            reinterpret_cast<Create>(GetProcAddress(module, "phoenix_backend_create_configured"));
        auto destroy =
            reinterpret_cast<DestroyBackend>(GetProcAddress(module, "phoenix_backend_destroy"));
        using Features = uint32_t (*)(Backend*);
        auto features =
            reinterpret_cast<Features>(GetProcAddress(module, "phoenix_backend_features"));
        auto basic_create =
            reinterpret_cast<CreateBackend>(GetProcAddress(module, "phoenix_backend_create"));
        demand((create || basic_create) && destroy, "Backend factory ABI missing");
        char error[2048]{};
        const bool benchmark_pieces = job.contains("pieces");
        uint32_t options = benchmark_pieces ? 0u : DispatchProfiling;
        if (config.value("scalar_matrices", false))
            options |= ScalarMatrices;
        if (kind == "vulkan" && config.value("vulkan_tiled_fallback", true))
            options |= VulkanTiledFallback;
        if (kind == "vulkan" && config.value("vulkan_coalesced_fallback", true))
            options |= VulkanCoalescedFallback;
        if (kind == "vulkan" && config.value("vulkan_scalar_attention", false))
            options |= VulkanScalarAttention;
        if (config.value("vulkan_fp8_matrices", false))
            options |= LosslessFp8Matrices;
        if (config.value("vulkan_native_half", false))
            options |= VulkanHardwareHalf;
        if (config.value("vulkan_shape_matrices", false))
            options |= VulkanShapeMatrices;
        if (kind == "vulkan" && config.value("vulkan_fused_qkv", true))
            options |= VulkanFusedQkv;
        if (!config.value("vulkan_specialize_wide", true))
            options |= VulkanGenericWide;
        if (config.value("vulkan_matrix_columns", 16u) == 32u)
            options |= VulkanWideColumns32;
        if (config.value("vulkan_matrix_columns", 16u) == 64u)
            options |= VulkanWideColumns64;
        if (config.value("cuda_direct_fp8", true))
            options |= CudaDirectFp8;
        if (config.value("cuda_fused_feed_forward", false))
            options |= CudaFusedFeedForward;
        if (config.value("cuda_fused_window_block", false))
            options |= CudaFusedWindowBlock;
        if (config.value("cuda_compact_window_block", false))
            options |= CudaCompactWindowBlock;
        if (config.value("cuda_fused_wide_feed_forward", false))
            options |= CudaFusedWideFeedForward;
        if (config.value("cuda_shape_matrices", false))
            options |= CudaShapeMatrices;
        if (config.value("cuda_fp8_columns", 32u) == 64u)
            options |= Fp8Columns64;
        if (config.value("cuda_fp8_columns", 32u) == 128u)
            options |= Fp8Columns128;
        if (config.value("vulkan_matrix_warps", 0u) == 1u)
            options |= NarrowMatrixWorkgroups;
        if (config.value("vulkan_matrix_warps", 0u) == 4u)
            options |= WideMatrixWorkgroups;
        if (config.value("cuda_matrix_columns", 16u) == 32u)
            options |= MatrixColumns32;
        if (config.value("cuda_matrix_columns", 16u) == 64u)
            options |= MatrixColumns64;
        auto backend = std::unique_ptr<Backend, DestroyBackend>(
            create ? create(config.at("device").get<std::string>().c_str(),
                            (root / "vulkan").c_str(), options, error, sizeof(error))
                   : basic_create(config.at("device").get<std::string>().c_str(),
                                  (root / "vulkan").c_str(), error, sizeof(error)),
            destroy);
        demand(bool(backend), error);
        Model model(phoenix_model::WeightSource::config(config.at("weight_source")));
        // Use the runtime's recorded padded extent from a completed frame.
        unsigned width = job.at("width"), height = job.at("height");
        unsigned padded_width = job.at("padded_width"), padded_height = job.at("padded_height");
        FrameState frame(width, height, padded_width, padded_height);
        std::vector<float> color(size_t(width) * height * 3);
        std::ifstream input(
            std::filesystem::u8path(job.at("cases")[0].at("color").get<std::string>()),
            std::ios::binary);
        input.read(reinterpret_cast<char*>(color.data()), std::streamsize(color.size() * 4));
        demand(bool(input), "Input unavailable");
        std::vector<float> prior(color.size());
        std::vector<uint8_t> validity(size_t(width) * height);
        auto input_features = frame.features(color.data(), Controls{}, prior, validity);
        const auto supported = features ? features(backend.get()) : 0;
        Graph graph(
            *backend, model,
            (supported & FusedWindowAttention) &&
                config.value("fused_attention", !(supported & PreferUnfusedWindowAttention)),
            config.value("window_batch_limit", 512u), false, supported & FusedMatrixResidual);
        graph.build(padded_width, padded_height, {1, 1, 1, 1, 1, 1, 1, 1});
        model.discard_decoded();
        if (config.value("plan_only", false)) {
            Json report = {{"passed", true},
                           {"resident_bytes", graph.memory_bytes()},
                           {"weight_bytes", graph.weight_bytes()},
                           {"commands", graph.commands.size()},
                           {"scope", "Allocation plan only; no inference or timing validation"}};
            std::ofstream(argv[2]) << report.dump(2) << '\n';
            return 0;
        }
        if (benchmark_pieces) {
            // A prefix initializes the real operands before an isolated piece.
            // Replay one piece many times under a single timing span, so event
            // overhead is amortized rather than charged once per tiny kernel.
            const auto original_commands = graph.commands;
            const unsigned repetitions = job.value("piece_repetitions", 32u);
            demand(repetitions >= 2 && repetitions <= 256, "Piece repetitions must be 2..256");
            // Scratch lifetimes are planned for one forward pass. A later
            // output may reuse an entry input's address, making a repeated
            // piece consume its own output on subsequent replays. Preserve
            // external scratch inputs in a benchmark-only immutable snapshot.
            // This allocation is not part of the runtime's resident footprint.
            const auto arena = *graph.input.memory;
            const auto snapshot = backend->allocate(arena.bytes);
            struct ReleaseSnapshot {
                Backend& backend;
                Allocation allocation;
                ~ReleaseSnapshot() {
                    backend.release(allocation);
                }
            } release_snapshot{*backend, snapshot};
            std::vector<uint8_t> snapshot_bytes(size_t(arena.bytes));
            Json rows = Json::array();
            for (const auto& selection : job.at("pieces")) {
                size_t index = 0;
                for (; index < original_commands.size(); ++index) {
                    const auto& operation = original_commands[index].operation;
                    if (operation.family == selection.at("family").get<unsigned>() &&
                        operation.row_count == selection.value("m", operation.row_count) &&
                        operation.column_count == selection.value("n", operation.column_count) &&
                        operation.reduction_count ==
                            selection.value("k", operation.reduction_count) &&
                        operation.batch_count == selection.value("batch", operation.batch_count) &&
                        operation.flags == selection.value("flags", operation.flags) &&
                        (operation.flags & 255u) == selection.value("kind", operation.flags & 255u))
                        break;
                }
                demand(index < original_commands.size(), "Requested piece absent from graph");
                const auto command = original_commands[index];
                const unsigned piece_span = selection.value("span", 1u);
                demand(piece_span >= 1 && index + piece_span <= original_commands.size(),
                       "Piece span outside graph");
                demand(command.operation.output != command.operation.left_input &&
                           command.operation.output != command.operation.right_input &&
                           command.operation.output != command.operation.auxiliary,
                       "Repeated piece must have a separate output");
                graph.commands.assign(original_commands.begin(), original_commands.begin() + index);
                std::vector<float> unused_head;
                std::vector<Event> events;
                auto features_copy = input_features;
                graph.run(features_copy, unused_head, events);
                backend->download(arena, snapshot_bytes.data(), arena.bytes);
                backend->upload(snapshot, snapshot_bytes.data(), arena.bytes);
                std::vector<Command> piece(original_commands.begin() + index,
                                           original_commands.begin() + index + piece_span);
                std::vector<std::pair<uintptr_t, uint64_t>> written;
                auto preserve_entry = [&](void*& pointer) {
                    const uintptr_t address = uintptr_t(pointer);
                    if (address < uintptr_t(arena.address) ||
                        address - uintptr_t(arena.address) >= arena.bytes)
                        return;
                    for (const auto& [base, bytes] : written)
                        if (address >= base && address - base < bytes)
                            return;
                    pointer = reinterpret_cast<void*>(uintptr_t(snapshot.address) + address -
                                                      uintptr_t(arena.address));
                };
                for (auto& item : piece) {
                    auto& operation = item.operation;
                    preserve_entry(operation.left_input);
                    preserve_entry(operation.right_input);
                    const bool global_scores =
                        operation.family == phx::RowReduction && (operation.flags & 255u) == 6;
                    if (!global_scores)
                        preserve_entry(operation.auxiliary);
                    preserve_entry(operation.secondary_output);
                    uint64_t elements = item.count;
                    if (operation.family == phx::MatrixMultiply) {
                        elements =
                            uint64_t(operation.batch_count - 1) * operation.output_batch_stride +
                            uint64_t(operation.row_count - 1) * operation.output_row_stride +
                            operation.column_count;
                    } else if (operation.family == phx::RowReduction) {
                        const unsigned kind = operation.flags & 255u;
                        demand(kind == 0 || kind == 1 || kind == 6,
                               "Piece snapshot does not support this row operation");
                        const unsigned width = kind == 0 ? 32 : operation.column_count;
                        elements *= global_scores && operation.left_batch_stride
                                        ? operation.left_batch_stride
                                        : width;
                    } else if (operation.family == phx::WindowAttention) {
                        elements *= 64 * 32;
                    } else {
                        const unsigned kind = operation.flags & 255u;
                        demand(operation.family == phx::Elementwise &&
                                   (kind == phx::CopyValues || kind == phx::SplitQkvHeads ||
                                    kind == phx::QuantizeFp8 || kind == phx::ActivationGate ||
                                    kind == phx::RoundHalf),
                               "Piece snapshot requires a contiguous supported output");
                    }
                    const unsigned bytes = (operation.flags & phx::StoreHalf) ||
                                                   operation.family == phx::WindowAttention
                                               ? 2
                                               : 4;
                    written.push_back({uintptr_t(operation.output), elements * bytes});
                    if (global_scores)
                        written.push_back(
                            {uintptr_t(operation.auxiliary), uint64_t(operation.row_count) * 4});
                }
                std::vector<Command> repeated;
                for (unsigned repeat = 0; repeat < repetitions; ++repeat)
                    repeated.insert(repeated.end(), piece.begin(), piece.end());
                std::vector<double> gpu_samples, host_samples;
                for (unsigned sample = 0; sample < 5; ++sample) {
                    events.clear();
                    const auto started = std::chrono::steady_clock::now();
                    backend->execute(repeated, events);
                    backend->wait();
                    const double host_ms = std::chrono::duration<double, std::milli>(
                                               std::chrono::steady_clock::now() - started)
                                               .count();
                    double gpu_ms = 0;
                    for (const auto& event : events) {
                        demand(event.gpu, "Piece benchmark requires GPU timing");
                        gpu_ms += event.milliseconds;
                    }
                    if (sample >= 2) {
                        gpu_samples.push_back(gpu_ms / repetitions);
                        host_samples.push_back(host_ms / repetitions);
                    }
                }
                auto sorted = gpu_samples;
                std::sort(sorted.begin(), sorted.end());
                // Read only the regions produced by this isolated piece, after
                // timing finishes. Matching full-frame output remains a separate
                // gate, but these hashes catch local rounding/store regressions.
                Json output_hashes = Json::array();
                for (const auto& [address, byte_count] : written) {
                    demand(address >= uintptr_t(arena.address) &&
                               address - uintptr_t(arena.address) <= arena.bytes &&
                               byte_count <= arena.bytes - (address - uintptr_t(arena.address)),
                           "Piece output escapes scratch allocation");
                    std::vector<uint8_t> bytes(size_t(byte_count), uint8_t(0));
                    backend->download({reinterpret_cast<void*>(address), byte_count, arena.token},
                                      bytes.data(), byte_count);
                    output_hashes.push_back({{"offset", address - uintptr_t(arena.address)},
                                             {"bytes", byte_count},
                                             {"sha256", phoenix_model::weight_digest(bytes.data(), bytes.size())}});
                }
                const auto& operation = command.operation;
                rows.push_back({{"selection", selection},
                                {"index", index},
                                {"block", command.block},
                                {"flags", operation.flags},
                                {"gpu_median_ms", sorted[1]},
                                {"gpu_samples_ms", gpu_samples},
                                {"host_samples_ms", host_samples},
                                {"output_hashes", output_hashes}});
                std::cout << "Piece " << index << ": " << sorted[1] << " ms\n" << std::flush;
            }
            Json report = {
                {"passed", true},
                {"device", config.at("device")},
                {"repetitions", repetitions},
                {"rows", rows},
                {"resident_bytes", graph.memory_bytes()},
                {"weight_bytes", graph.weight_bytes()},
                {"benchmark_snapshot_bytes", arena.bytes},
                {"command_count", original_commands.size()},
                {"scope",
                 "Real-input isolated repeated GPU operators; two warmups and three samples, prefix/capture/transfers excluded. Temporal correctness requires the separate full-frame regression."}};
            std::ofstream(argv[2]) << report.dump(2) << '\n';
            return 0;
        }
        std::vector<float> head;
        std::vector<Event> events;
        auto upload = [&] {
            auto copy = input_features;
            graph.run(copy, head, events);
        };
        upload(); // Exclude graph capture and first-use pipelines.
        events.clear();
        upload();
        demand(events.size() == graph.commands.size(),
               "Backend did not supply per-command timings");
        Json rows = Json::array();
        for (size_t index = 0; index < graph.commands.size(); ++index) {
            const auto& command = graph.commands[index];
            const auto& op = command.operation;
            rows.push_back({{"index", index},
                            {"block", command.block},
                            {"layer", command.layer},
                            {"family", op.family},
                            {"flags", op.flags},
                            {"count", command.count},
                            {"m", op.row_count},
                            {"n", op.column_count},
                            {"k", op.reduction_count},
                            {"batch", op.batch_count},
                            {"ms", events[index].milliseconds}});
        }
        Json report = {
            {"device", config.at("device")},
            {"resolution", {width, height}},
            {"padded", {padded_width, padded_height}},
            {"commands", rows},
            {"resident_bytes", graph.memory_bytes()},
            {"passed", true},
            {"scope",
             "Warm per-command GPU timings; detailed profiling has overhead; no game/host frame-time claim"}};
        std::ofstream(argv[2]) << report.dump(2) << '\n';
        std::cout << "Profiled " << rows.size() << " commands\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
