#include "../../src/native/support/model_tools.h"
#include "../../src/native/support/routing.h"
#include <iostream>
using Json = nlohmann::json;
static void check(bool success) {
    if (!success)
        throw std::runtime_error("Model tools contract failed");
}
template <class F> static void rejects(F work) {
    bool failed = false;
    try {
        work();
    } catch (const std::exception&) {
        failed = true;
    }
    check(failed);
}
int main(int argc, char** argv) {
    try {
        for (const auto* path : {"../escape", "C:/absolute", "a/../b", "a:file", "", "/absolute"})
            rejects([&] { phoenix_embedded::relative_path(path); });
        check(phoenix_embedded::relative_path("runtime/opencl_cpu/Intel.dll").is_relative());
        using phoenix_model_tools::selection;
        phoenix_model::WeightSource configured;
        check(!selection("configured", "", "", configured).external());
        rejects([&] { selection("unknown", "", "", configured); });
        rejects([&] { selection("model_folder", "", "", configured); });
        rejects([&] { selection("nvidia_dll", "", "", configured); });
        phoenix_addon::phoenix::Settings settings;
        settings.model_source = "nvidia_dll";
        settings.model_dll = "C:/model.dll";
        check(settings.valid());
        bool visited = false;
        phoenix_addon::phoenix::visit_settings(settings, [&](const char* key, auto&) {
            if (std::string(key) == "ModelDll")
                visited = true;
        });
        check(visited);
        settings.model_dll = std::string(4096, 'a');
        check(!settings.valid());
        Json result{{"passed", true},
                    {"cpu_only", true},
                    {"path_contracts", true},
                    {"model_selection_contracts", true}};
        if (argc == 3) {
            const auto path = std::filesystem::u8path(argv[1]);
            const auto module =
                LoadLibraryExW(path.c_str(), nullptr,
                               LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
            check(module != nullptr);
            struct Close {
                HMODULE module;
                ~Close() {
                    FreeLibrary(module);
                }
            } close{module};
            auto installation = phoenix_embedded::install(
                module, std::filesystem::absolute(std::filesystem::u8path(argv[2])));
            check(installation.files > 41);
            check(!std::filesystem::exists(installation.runtime / "amd_comgr.dll"));
            check(!std::filesystem::exists(installation.runtime / "opencl_cpu/intelocl64.dll"));
            // A damaged cache entry must be repaired from the addon payload.
            const auto victim = installation.native / "native_graph.json";
            const auto expected = phoenix_embedded::file_digest(victim);
            std::ofstream(victim, std::ios::binary | std::ios::trunc) << "corrupted";
            auto repaired = phoenix_embedded::install(
                module, std::filesystem::absolute(std::filesystem::u8path(argv[2])));
            check(repaired.identity == installation.identity &&
                  phoenix_embedded::file_digest(victim) == expected);
            result["installation"] = {{"root", installation.root.u8string()},
                                      {"files", installation.files},
                                      {"identity", installation.identity},
                                      {"tamper_repair", true},
                                      {"optional_compilers_deferred", true}};
        }
        std::cout << result.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
