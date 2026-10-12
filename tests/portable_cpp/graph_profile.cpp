// Offline profiling of individual portable commands. Model and input stay external.
#include "../../native_runtime/graph.h"
#include "../../native_runtime/frame.h"
#include "../../native_runtime/backend_options.h"
#include <iostream>

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
        demand(create && destroy && features, "Configured backend ABI missing");
        char error[2048]{};
        uint32_t options = DispatchProfiling;
        if (config.value("vulkan_matrix_warps", 0u) == 1u)
            options |= NarrowMatrixWorkgroups;
        if (config.value("vulkan_matrix_warps", 0u) == 4u)
            options |= WideMatrixWorkgroups;
        if (config.value("cuda_matrix_columns", 16u) == 32u)
            options |= MatrixColumns32;
        if (config.value("cuda_matrix_columns", 16u) == 64u)
            options |= MatrixColumns64;
        auto backend = std::unique_ptr<Backend, DestroyBackend>(
            create(config.at("device").get<std::string>().c_str(), (root / "vulkan").c_str(),
                   options, error, sizeof(error)),
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
        const auto supported = features(backend.get());
        Graph graph(*backend, model, supported & FusedWindowAttention,
                    config.value("window_batch_limit", 512u), false,
                    supported & FusedMatrixResidual);
        graph.build(padded_width, padded_height, {1, 1, 1, 1, 1, 1, 1, 1});
        model.discard_decoded();
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
