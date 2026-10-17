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
        check(phoenix_embedded::relative_path("native/optimized/kernel.bin").native().find(L'/') == std::wstring::npos);
        const auto cache = phoenix_embedded::default_cache();
        check(cache.is_absolute() && std::filesystem::is_directory(cache));
        check(cache.wstring().rfind(L"\\\\?\\", 0) == 0);
        // Discovering the writable root must be stable in packaged launchers too.
        check(phoenix_embedded::default_cache() == cache);
        const auto long_directory = cache / (L".long-path-test-" + std::to_wstring(GetCurrentProcessId())) /
            std::wstring(100, L'a') / std::wstring(100, L'b');
        const auto long_root = long_directory.parent_path().parent_path();
        struct CleanupLongPath {
            std::filesystem::path path;
            ~CleanupLongPath() { std::error_code error; std::filesystem::remove_all(path, error); }
        } cleanup_long_path{long_root};
        std::filesystem::create_directories(long_directory);
        { std::ofstream test_file(long_directory / "test.bin", std::ios::binary); test_file << "test"; check(bool(test_file)); }
        check(phoenix_embedded::file_digest(long_directory / "test.bin").size() == 64);
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
