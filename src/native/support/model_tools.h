#pragma once
#include <windows.h>
#include <shobjidl.h>
#include <thread>
#include <mutex>
#include <memory>
#include <cmath>
#include <algorithm>
#include "embedded_runtime.h"
#include "../../../native_runtime/broker_client.h"

namespace phoenix_model_tools {
using Json = nlohmann::json;
inline std::string selected_dll_digest(const std::filesystem::path& path) {
    if (path.empty() || !path.is_absolute())
        throw std::runtime_error("Choose an absolute NVIDIA DLSS5 DLL path");
    // Poll only file metadata during drawing. Hash once per actual selection
    // or file change; the loader independently hashes the file again on use.
    struct Cache {
        std::filesystem::path path;
        std::filesystem::file_time_type time;
        uintmax_t bytes = 0;
        std::string digest;
        std::mutex mutex;
    };
    static Cache cache;
    std::lock_guard lock(cache.mutex);
    const auto time = std::filesystem::last_write_time(path);
    const auto bytes = std::filesystem::file_size(path);
    if (cache.path != path || cache.time != time || cache.bytes != bytes) {
        cache.digest = phoenix_embedded::file_digest(path);
        cache.path = path;
        cache.time = time;
        cache.bytes = bytes;
    }
    const std::string reference =
        "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e";
    if (cache.digest != reference && !phoenix_model::equivalent_model_dll(cache.digest, reference))
        throw std::runtime_error(
            "Unsupported NVIDIA DLSS5 DLL edition; choose a supported DLL or compatible model folder");
    return cache.digest;
}
inline phoenix_model::WeightSource selection(const std::string& kind, const std::string& folder,
                                             const std::string& dll,
                                             const phoenix_model::WeightSource& configured) {
    if (kind == "configured")
        return configured;
    if (kind == "model_folder")
        return phoenix_model::WeightSource::config({{"kind", kind}, {"path", folder}});
    if (kind == "nvidia_dll")
        return phoenix_model::WeightSource::config(
            {{"kind", kind},
             {"path", dll},
             {"sha256", selected_dll_digest(std::filesystem::u8path(dll))}});
    throw std::runtime_error("Unknown model selection");
}
struct Result {
    bool busy = false, passed = false, cancelled = false;
    std::string key, message, picked;
};
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
    void browse(bool folder, HWND owner, const std::filesystem::path& initial) {
        start(folder ? "folder" : "dll", [folder, owner, initial](Result& result) {
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
            dialog->SetTitle(folder ? L"Choose Phoenix DLSS5 model folder"
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
            size_t records = 0;
            phoenix_model::with_model_resource(
                source, [](const std::filesystem::path& path) {
                    return phoenix_embedded::file_digest(path);
                }, [&](const uint8_t* data, size_t bytes) {
                    phoenix_model::validate_model_layout(data, bytes);
                    records = phoenix_model::resource_records(data, bytes).size();
                    return 0;
                });
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
