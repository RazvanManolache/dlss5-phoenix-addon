// Full-model C-API validation on synthetic frames. No game or capture hook.
#include "../../native_runtime/platform.h"
#include "../../third_party/nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
using Json = nlohmann::json;
using namespace phoenix_cpp;

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Expected runtime root, weight-source JSON, device and fresh output");
        const auto root = std::filesystem::absolute(argv[1]);
        if (std::filesystem::exists(argv[4])) throw std::runtime_error("Fresh output required");
        std::ifstream source(argv[2]);
        Json weights; source >> weights;
        auto library = open_library(root / L"phoenix_runtime_cpp.dll");
        if (!library) throw std::runtime_error("Runtime load failed");
        std::unique_ptr<std::remove_pointer_t<LibraryHandle>, decltype(&close_library)> owned_library(library, close_library);
        auto create = reinterpret_cast<void* (*)(const char*, unsigned, unsigned)>(library_symbol(library, "phoenix_runtime_create"));
        auto destroy = reinterpret_cast<void (*)(void*)>(library_symbol(library, "phoenix_runtime_destroy"));
        auto frame = reinterpret_cast<int (*)(void*, const float*, const float*, const float*, const float*, const char*, float*)>(library_symbol(library, "phoenix_runtime_frame"));
        auto error = reinterpret_cast<const char* (*)()>(library_symbol(library, "phoenix_runtime_error"));
        auto status = reinterpret_cast<size_t (*)(void*, char*, size_t)>(library_symbol(library, "phoenix_runtime_status"));
        if (!create || !destroy || !frame || !error || !status) throw std::runtime_error("Public C ABI exports missing");
        const auto device = std::filesystem::path(argv[3]).string();
        Json config{{"runtime_root", root.string()}, {"backend", "vulkan"}, {"device", device},
                    {"allow_unvalidated", true}, {"weight_source", weights}, {"cpu_threads", 2},
                    {"validation_stage_hashes", true}};
        std::ofstream output(argv[4], std::ios::binary);
        if (!output) throw std::runtime_error("Output open failed");
        const auto configuration = config.dump();
        for (auto [width, height] : {std::pair{64u, 64u}, std::pair{129u, 97u}, std::pair{1920u, 1080u}}) {
            std::unique_ptr<void, decltype(destroy)> session(create(configuration.c_str(), width, height), destroy);
            if (!session) throw std::runtime_error(error());
            const size_t pixels = size_t(width) * height;
            std::vector<float> color(pixels * 3), motion(pixels * 2), depth(pixels), validity(pixels), result(pixels * 3);
            for (unsigned index = 0; index < 3; ++index) {
                for (size_t pixel = 0; pixel < pixels; ++pixel) {
                    for (size_t channel = 0; channel < 3; ++channel)
                        color[pixel * 3 + channel] = float((pixel * (channel * 4 + 3) + index * 17) % 251) / 256.f;
                    depth[pixel] = float(pixel % 127 + 1) / 128.f;
                    motion[pixel * 2] = index ? .25f : 0.f;
                    motion[pixel * 2 + 1] = index ? -.125f : 0.f;
                    validity[pixel] = pixel % 11 ? 1.f : 0.f;
                }
                Json metadata{{"reset", index == 0}, {"has_motion", true}, {"has_depth", true},
                              {"has_history_validity", true}, {"motion_from_frame", index ? index - 1 : 0},
                              {"controls", {{"seed", 13}, {"intensity", 1.f}}}};
                const auto text = metadata.dump();
                if (frame(session.get(), color.data(), motion.data(), depth.data(), validity.data(), text.c_str(), result.data()))
                    throw std::runtime_error(error());
                bool changed = false;
                for (size_t i = 0; i < result.size(); ++i) {
                    if (!std::isfinite(result[i])) throw std::runtime_error("Nonfinite output");
                    changed |= result[i] != color[i];
                }
                std::vector<char> state(status(session.get(), nullptr, 0));
                status(session.get(), state.data(), state.size());
                auto report = Json::parse(state.data());
                if (report.value("state", "") != "completed" || report.value("backend", "") != "vulkan-cpp" ||
                    report.value("command_count", 0u) == 0 || !changed)
                    throw std::runtime_error("No proven full-graph processed output");
                output.write(reinterpret_cast<const char*>(result.data()), std::streamsize(result.size() * sizeof(float)));
                if (!output) throw std::runtime_error("Output write failed");
                std::cout << Json{{"width", width}, {"height", height}, {"frame", index},
                    {"pointer_bits", sizeof(void*) * 8}, {"command_count", report["command_count"]},
                    {"history_used", report["history_used"]}, {"resident_bytes", report["resident_bytes"]},
                    {"output_changed", changed}, {"backend", report["backend"]},
                    {"validation_stages", report.at("validation_stages")}}.dump() << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
