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
        configured.model_id = "small-fp8";
        check(selection("configured", "", "", configured).model_id == "small-fp8");
        check(selection("configured", "", "", configured, "original").model_id == "original");
        check(selection("model_folder", "C:/fixture", "", configured).model_id == "small-fp8");
        configured.model_id = "original";
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
        settings.nvidia_dll_folder = "C:/shared-model-library";
        check(settings.valid());
        bool library_visited = false;
        phoenix_addon::phoenix::visit_settings(settings, [&](const char* key, auto&) {
            if (std::string_view(key) == "NvidiaDllFolder") library_visited = true;
        });
        check(library_visited);
        settings.nvidia_dll_folder = std::string(4096, 'a');
        check(!settings.valid());
        settings.nvidia_dll_folder = std::string("bad\0folder", 10);
        check(!settings.valid());
        settings.nvidia_dll_folder.clear();
        settings.model_dll = std::string(4096, 'a');
        check(!settings.valid());
        // Tiny fake DLLs exercise location precedence only; no vendor binary is
        // loaded, copied, hashed as a valid model or used for inference here.
        const auto local = long_directory / "game", shared = long_directory / "shared";
        std::filesystem::create_directories(local);
        std::filesystem::create_directories(shared);
        const auto name = std::filesystem::path("nvngx_dlssnr.dll");
        using phoenix_model::find_nvidia_dll;
        check(find_nvidia_dll({}, local, shared).empty());
        std::ofstream(shared / name) << "shared fixture";
        check(find_nvidia_dll({}, local, shared) == shared / name);
        check(find_nvidia_dll(local / name, local, shared) == shared / name);
        check(find_nvidia_dll(name, local, "../shared") == shared / name);
        std::ofstream(local / name) << "local fixture";
        check(find_nvidia_dll({}, local, shared) == local / name);
        check(find_nvidia_dll({}, local) == local / name);
        check(find_nvidia_dll(local / name, local, shared) == local / name);
        check(find_nvidia_dll(shared / name, local, "missing") == shared / name);
        std::filesystem::create_directory(shared / "directory.dll");
        check(find_nvidia_dll("directory.dll", local, shared).empty());
        rejects([&] { find_nvidia_dll({}, "relative", shared); });
        rejects([&] { find_nvidia_dll("C:ambiguous.dll", local, shared); });
        rejects([&] { find_nvidia_dll(std::filesystem::path(std::wstring(L"bad\0.dll", 8)), local, shared); });
        rejects([&] { find_nvidia_dll({}, local, std::filesystem::path(std::wstring(L"bad\0folder", 10))); });
        // A local incompatible file must not be silently replaced by shared.
        rejects([&] { selection("configured", "", "", {}, "original", phoenix_model::source_path_string(shared)); });
        Json result{{"passed", true},
                    {"cpu_only", true},
                    {"path_contracts", true},
                    {"model_selection_contracts", true}};
        result["shared_dll_search_contracts"] = true;
        if (argc == 4 && std::string_view(argv[1]) == "--validate-model") {
            Json configuration;
            std::ifstream(argv[2]) >> configuration;
            auto selected_source = phoenix_model::WeightSource::config(configuration.at("weight_source"));
            check(selected_source.model_id == "small-fp8");
            phoenix_model_tools::Task validation;
            validation.validate(selected_source, std::filesystem::absolute(std::filesystem::u8path(argv[3])));
            const auto deadline = GetTickCount64() + 90000;
            auto state = validation.poll();
            while (state.busy && GetTickCount64() < deadline) {
                Sleep(50);
                state = validation.poll();
            }
            if (!state.passed) throw std::runtime_error(state.message);
            check(state.message.find("77 records") != std::string::npos);
            result["bits"] = sizeof(void*) * 8;
            result["small_model_cpu_validator"] = state.message;
        }
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
