#include "../../native_runtime/graph.h"
#include "../../native_runtime/frame.h"
using namespace phoenix_cpp;
using Callback = void (*)(const char*, const void*, size_t);
thread_local std::string error;
extern "C" __declspec(dllexport) const char* probe_error() {
    return error.c_str();
}
extern "C" __declspec(dllexport) int probe_warp(unsigned width, unsigned height,
                                                const float* history, const float* motion,
                                                const float* depth, const float* validity,
                                                Callback callback) {
    try {
        FrameState frame(width, height, width, height);
        frame.history.assign(history, history + size_t(width) * height * 3);
        std::vector<float> pre, post;
        std::vector<uint8_t> prior_validity, post_validity;
        frame.warp(motion, depth, validity, false, pre, post, prior_validity, post_validity);
        callback("pre", pre.data(), pre.size() * 4);
        callback("post", post.data(), post.size() * 4);
        callback("pre_valid", prior_validity.data(), prior_validity.size());
        callback("post_valid", post_validity.data(), post_validity.size());
        return 0;
    } catch (const std::exception& exception) {
        error = exception.what();
        return -1;
    }
}
extern "C" __declspec(dllexport) int probe_model(const char* config, Callback callback) {
    try {
        Model model(phoenix_model::WeightSource::config(Json::parse(config)));
        model.visit_tensors([&](const std::string& name, const auto& bytes) {
            callback(name.c_str(), bytes.data(), bytes.size());
        });
        return 0;
    } catch (const std::exception& exception) {
        error = exception.what();
        return -1;
    }
}
extern "C" __declspec(dllexport) int probe_features(unsigned width, unsigned height,
                                                    unsigned padded_width, unsigned padded_height,
                                                    const float* color, const char* controls,
                                                    Callback callback) {
    try {
        FrameState frame(width, height, padded_width, padded_height);
        std::vector<float> prior, post;
        std::vector<uint8_t> valid, post_validity;
        frame.warp(nullptr, nullptr, nullptr, false, prior, post, valid, post_validity);
        auto result = frame.features(color, Controls::from(Json::parse(controls)), prior, valid);
        callback("features", result.data(), result.size() * 2);
        return 0;
    } catch (const std::exception& exception) {
        error = exception.what();
        return -1;
    }
}
extern "C" __declspec(dllexport) int probe_graph(const char* config, unsigned width,
                                                 unsigned height, const uint16_t* features,
                                                 Callback callback) {
    try {
        auto configuration = Json::parse(config);
        auto root = std::filesystem::u8path(configuration.at("runtime_root").get<std::string>());
        std::string kind = configuration.at("backend");

        auto dll =
            LoadLibraryExW((root / ("phoenix_" + kind + "_cpp.dll")).c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        demand(dll, "Probe backend unavailable");
        struct Module {
            HMODULE library_handle;
            ~Module() {
                FreeLibrary(library_handle);
            }
        } module{dll};
        auto create =
            reinterpret_cast<CreateBackend>(GetProcAddress(dll, "phoenix_backend_create"));
        auto destroy =
            reinterpret_cast<DestroyBackend>(GetProcAddress(dll, "phoenix_backend_destroy"));
        char message[2048]{};
        std::unique_ptr<Backend, DestroyBackend> backend(
            create(configuration.at("device").get<std::string>().c_str(), (root / "vulkan").c_str(),
                   message, sizeof(message)),
            destroy);
        demand(bool(backend), message);
        Model model(phoenix_model::WeightSource::config(configuration.at("weight_source")));
        Graph graph(*backend, model);
        graph.build(
            width, height,
            configuration.value("repeats", std::array<unsigned, 8>{1, 1, 1, 1, 1, 1, 1, 1}));
        Allocation input{graph.input.ptr(), graph.input.size() * 2, graph.input.memory->token};
        backend->upload(input, features, graph.input.size() * 2);
        size_t command_offset = 0;
        std::vector<Event> events;
        for (const auto& point : graph.checkpoints) {
            std::vector<Command> part(graph.commands.begin() + command_offset,
                                      graph.commands.begin() + point.after);
            backend->execute(part, events);
            command_offset = point.after;
            std::vector<uint8_t> bytes(point.bytes);
            Allocation view{
                reinterpret_cast<void*>(uintptr_t(point.allocation.address) + point.offset),
                point.bytes, point.allocation.token};
            backend->download(view, bytes.data(), bytes.size());
            Json desc = {{"name", point.name},
                         {"shape", {point.height, point.width, point.channels}},
                         {"element", point.element}};
            callback(desc.dump().c_str(), bytes.data(), point.bytes);
        }
        return 0;
    } catch (const std::exception& exception) {
        error = exception.what();
        return -1;
    }
}
