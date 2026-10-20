#pragma once
#include <windows.h>
#include <shobjidl.h>
#include <thread>
#include <mutex>
#include <memory>
#include <cmath>
#include <algorithm>
#include "embedded_runtime.h"
#include "../model/resource_identity.h"
#include "../model/dll_search.h"
#include "../../../native_runtime/broker_client.h"

namespace phoenix_model_tools {
using Json = nlohmann::json;
inline std::string selected_dll_digest(const std::filesystem::path& path,
                                        const phoenix_model::Definition& model = phoenix_model::models::original) {
    phoenix_model::require_implemented(model);
    if (path.empty() || !path.is_absolute())
        throw std::runtime_error("Choose an absolute NVIDIA DLSS5 DLL path");
    // Poll only file metadata during drawing. Hash once per actual selection
    // or file change; the loader independently hashes the file again on use.
    struct Cache {
        std::filesystem::path path;
        std::filesystem::file_time_type time;
        uintmax_t bytes = 0;
        std::string digest;
        std::string model_id;
        std::string error;
        std::mutex mutex;
    };
    static Cache cache;
    std::lock_guard lock(cache.mutex);
    const auto time = std::filesystem::last_write_time(path);
    const auto bytes = std::filesystem::file_size(path);
    if (cache.path != path || cache.time != time || cache.bytes != bytes || cache.model_id != model.id) {
        cache.digest.clear();
        cache.error.clear();
        try {
            const auto digest = phoenix_embedded::file_digest(path);
            if (!phoenix_model::known_model_container(digest))
                throw std::runtime_error("Unknown NVIDIA DLSS5 DLL container; use a verified edition or a compatible disk-model folder");
            phoenix_model::WeightSource source{path, digest};
            source.model_id = std::string(model.id);
            phoenix_model::with_dll_resource(source,
                [](const std::filesystem::path& file) { return phoenix_embedded::file_digest(file); },
                [&](const uint8_t* data, size_t count) {
                    const auto identity = phoenix_model::resource_identity(data, count, model);
                    phoenix_model::require_resource_identity(model, identity, model.resource, model.reference_resource_sha256);
                    return 0;
                });
            cache.digest = digest;
        } catch (const std::exception& error) {
            // Remember rejection too: drawing must not rehash a bad DLL every frame.
            cache.error = error.what();
        }
        cache.model_id = std::string(model.id);
        cache.path = path;
        cache.time = time;
        cache.bytes = bytes;
    }
    if (!cache.error.empty())
        throw std::runtime_error(cache.error);
    return cache.digest;
}
inline std::filesystem::path local_library_directory() {
    wchar_t path[32768]{};
    const auto length = GetModuleFileNameW(nullptr, path, 32768);
    if (!length || length >= 32768)
        throw std::runtime_error("Cannot identify the local NVIDIA DLSS5 library directory");
    return std::filesystem::path(path).parent_path();
}
inline phoenix_model::WeightSource selection(const std::string& kind, const std::string& folder,
                                             const std::string& dll,
                                             const phoenix_model::WeightSource& configured,
                                             std::string_view model_id = {},
                                             const std::string& search_folder = {}) {
    const auto& model = phoenix_model::definition(model_id.empty() ? configured.model_id : model_id);
    if (kind == "configured") {
        auto selected = configured;
        selected.model_id = std::string(model.id);
        if (!search_folder.empty() && !selected.disk_model()) {
            const auto resolved = phoenix_model::find_nvidia_dll(selected.dll, local_library_directory(),
                                                                std::filesystem::u8path(search_folder));
            if (resolved.empty())
                throw std::runtime_error("NVIDIA DLSS5 DLL was not found locally or in the configured search folder");
            selected.dll = resolved;
            selected.dll_sha256 = selected_dll_digest(resolved, model);
        }
        return selected;
    }
    if (kind == "model_folder")
        return phoenix_model::WeightSource::config({{"kind", kind}, {"path", folder}, {"model", model.id}});
    if (kind == "nvidia_dll") {
        auto path = std::filesystem::u8path(dll);
        if (!search_folder.empty()) {
            // An explicit empty file selection is not an automatic source.
            if (path.empty()) throw std::runtime_error("Choose a NVIDIA DLSS5 DLL filename");
            path = phoenix_model::find_nvidia_dll(path, local_library_directory(),
                                                  std::filesystem::u8path(search_folder));
            if (path.empty()) throw std::runtime_error("NVIDIA DLSS5 DLL was not found locally or in the configured search folder");
        }
        return phoenix_model::WeightSource::config(
            {{"kind", kind},
             {"path", phoenix_model::source_path_string(path)},
             {"model", model.id},
             {"sha256", selected_dll_digest(path, model)}});
    }
    throw std::runtime_error("Unknown model selection");
}
template <class Settings>
inline phoenix_model::WeightSource selection(const Settings& settings,
                                             const phoenix_model::WeightSource& configured) {
    return selection(settings.model_source, settings.model_folder, settings.model_dll,
                     configured, settings.model_id, settings.nvidia_dll_folder);
}
struct Result {
    bool busy = false, passed = false, cancelled = false;
    std::string key, message, picked;
};
inline size_t selected_record_count(const phoenix_model::WeightSource& source) {
    return phoenix_model::with_model_resource(source,
        [](const std::filesystem::path& path) { return phoenix_embedded::file_digest(path); },
        [&](const uint8_t* data, size_t bytes) {
            return phoenix_model::resource_records(data, bytes, source.model()).size();
        });
}
class Task {
    struct State {
        std::mutex mutex;
        Result result;
    };
    std::shared_ptr<State> state = std::make_shared<State>();
    template <class Work> void start(std::string key, Work work) {
        {
            std::lock_guard lock(state->mutex);
            if (state->result.busy)
                return;
            state->result = {true, false, false, std::move(key), "Working...", {}};
        }
        // Both the picker and validation run away from the render thread.
        HMODULE pin{};
        if (!GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                reinterpret_cast<LPCWSTR>(&phoenix_embedded::default_cache), &pin)) {
            std::lock_guard lock(state->mutex);
            state->result.busy = false;
            state->result.message = "Cannot keep model tools available";
            return;
        }
        try {
            std::thread([shared = state, work = std::move(work)]() mutable {
                Result result;
                {
                    std::lock_guard lock(shared->mutex);
                    result.key = shared->result.key;
                }
                try {
                    work(result);
                } catch (const std::exception& error) {
                    result.passed = false;
                    result.picked.clear();
                    result.message = error.what();
                }
                std::lock_guard lock(shared->mutex);
                shared->result = std::move(result);
            }).detach();
        } catch (...) {
            std::lock_guard lock(state->mutex);
            state->result.busy = false;
            throw;
        }
    }

  public:
    Result poll() const {
        std::lock_guard lock(state->mutex);
        return state->result;
    }
    void clear_pick() {
        std::lock_guard lock(state->mutex);
        state->result.picked.clear();
    }
    void browse(bool folder, HWND owner, const std::filesystem::path& initial,
                const std::string& purpose = {}) {
        start(purpose.empty() ? (folder ? "folder" : "dll") : purpose, [folder, owner, initial, purpose](Result& result) {
            const auto initialized =
                CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
            if (FAILED(initialized))
                throw std::runtime_error("Windows file browser unavailable");
            struct Apartment {
                ~Apartment() {
                    CoUninitialize();
                }
            } apartment;
            IFileOpenDialog* dialog = nullptr;
            if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                        IID_PPV_ARGS(&dialog))))
                throw std::runtime_error("Cannot open the Windows file browser");
            struct Close {
                IFileOpenDialog* dialog;
                ~Close() {
                    dialog->Release();
                }
            } close{dialog};
            DWORD options = 0;
            dialog->GetOptions(&options);
            dialog->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | FOS_NOCHANGEDIR |
                               (folder ? FOS_PICKFOLDERS : FOS_FILEMUSTEXIST));
            dialog->SetTitle(purpose == "dll_search_folder" ? L"Choose shared NVIDIA DLSS5 DLL folder"
                                   : folder ? L"Choose Phoenix DLSS5 model folder"
                                    : L"Choose NVIDIA DLSS5 model DLL");
            if (!folder) {
                COMDLG_FILTERSPEC filters[]{{L"NVIDIA DLSS5 DLL", L"*.dll"}};
                dialog->SetFileTypes(1, filters);
            }
            if (!initial.empty()) {
                IShellItem* item = nullptr;
                const auto directory = folder ? initial : initial.parent_path();
                if (SUCCEEDED(SHCreateItemFromParsingName(directory.c_str(), nullptr,
                                                          IID_PPV_ARGS(&item)))) {
                    dialog->SetFolder(item);
                    item->Release();
                }
            }
            const auto shown = dialog->Show(owner);
            if (shown == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
                result.cancelled = true;
                result.message.clear();
                return;
            }
            if (FAILED(shown))
                throw std::runtime_error("Model browsing failed");
            IShellItem* item = nullptr;
            if (FAILED(dialog->GetResult(&item)))
                throw std::runtime_error("No model selected");
            PWSTR path = nullptr;
            const auto named = item->GetDisplayName(SIGDN_FILESYSPATH, &path);
            item->Release();
            if (FAILED(named))
                throw std::runtime_error("Selected model has no filesystem path");
            result.picked = phoenix_model::source_path_string(std::filesystem::path(path));
            CoTaskMemFree(path);
            if (result.picked.size() >= 4096)
                throw std::runtime_error("Selected model path exceeds 4095 UTF-8 bytes");
            result.passed = true;
        });
    }
    void validate(const phoenix_model::WeightSource& source, const std::filesystem::path& runtime,
                  HMODULE embedded_addon = nullptr) {
        const auto key = source.configuration().dump();
        start(key, [source, runtime, embedded_addon](Result& result) {
            if (!source.external())
                throw std::runtime_error("Select a model folder or NVIDIA DLSS5 DLL first");
            const size_t records = selected_record_count(source);
            // A small real CPU inference checks decoding, loading, and finite
            // temporal output without competing with the game's GPU.
            auto runtime_root = runtime;
            if (runtime_root.empty() && embedded_addon) {
                // Validation owns a separate CPU broker. It must not activate
                // the candidate or replace the game's working graphics helper.
#ifdef _WIN64
                runtime_root = phoenix_embedded::install(embedded_addon).runtime;
#else
                const auto frontend = phoenix_embedded::install(embedded_addon, {}, "compatibility");
                const auto core_path = frontend.root / "compatibility" / "Phoenix.addon64";
                HMODULE core = LoadLibraryExW(core_path.c_str(), nullptr,
                    LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE);
                if (!core) throw std::runtime_error("Embedded model validator is unavailable");
                struct Unload { HMODULE module; ~Unload() { FreeLibrary(module); } } unload{core};
                runtime_root = phoenix_embedded::install(core).runtime;
#endif
            }
            std::ifstream file(runtime_root / "phoenix_native_cpp.json");
            if (!file)
                throw std::runtime_error(
                    "Model integrity passed; inference test needs the C++ runtime");
            Json config;
            file >> config;
            config["backend"] = "cpu";
            config["device"] = "cpu";
            config["execution"] = "broker";
            config["cpu_threads"] = 2;
            config["weight_source"] = source.configuration();
            config["runtime_root"] = phoenix_model::source_path_string(runtime_root);
            constexpr unsigned width = 96, height = 64;
            std::vector<float> color(width * height * 3), motion(width * height * 2, 0),
                output(color.size());
            for (size_t iteration_index = 0; iteration_index < color.size(); ++iteration_index)
                color[iteration_index] = float((iteration_index * 17) % 251) / 250.f;
            phoenix_cpp::NativeSession session;
            session.open(runtime_root, config, width, height);
            for (unsigned frame = 0; frame < 3; ++frame) {
                Json metadata{
                    {"reset", frame == 0},
                    {"has_motion", true},
                    {"controls",
                     {{"intensity", 1.0}, {"local_tone", 1.0}, {"local_structure", 1.0}}}};
                if (session.frame(color.data(), motion.data(), nullptr, nullptr, metadata.dump(),
                                  output.data()))
                    throw std::runtime_error(std::string("Model inference test failed: ") +
                                             session.error());
                if (!std::all_of(output.begin(), output.end(),
                                 [](float value) { return std::isfinite(value); }))
                    throw std::runtime_error("Model inference produced nonfinite values");
            }
            result.passed = true;
            result.message = "Model passed: checksum, compatible architecture, " +
                             std::to_string(records) +
                             " records and 3 real CPU test frames. Visual quality is not assessed.";
        });
    }
};
} // namespace phoenix_model_tools
