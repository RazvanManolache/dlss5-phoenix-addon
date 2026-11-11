#pragma once
#include "legacy_depth_api.h"
#include <array>
#include <filesystem>
#include <iterator>
#include <vector>
#include <windows.h>

class PhoenixLegacyDepth {
    HMODULE module_ = nullptr;
    PhoenixDepthOpenV1Proc open_ = nullptr;
    PhoenixDepthEstimateV1Proc estimate_ = nullptr;
    PhoenixDepthCloseV1Proc close_ = nullptr;
    void* session_ = nullptr;
    int selection_ = -1;
    int active_ = 0;
    struct Attempt { int model; PhoenixDepthBackend backend; };
    std::array<Attempt, 5> attempts_{};
    unsigned next_ = 0, attempt_count_ = 0;
    uint64_t adapter_luid_ = 0;
    std::filesystem::path directory_;
    std::string status_ = "Waiting for a captured frame.";
    PhoenixDepthBackend active_backend_ = PhoenixDepthBackend::Cpu;

    void close_session() {
        if (session_) close_(session_);
        session_ = nullptr;
        active_ = 0;
    }

    bool load_provider() {
        wchar_t path[32768]{};
        const auto length = GetEnvironmentVariableW(L"PHOENIX_LEGACY_DEPTH_DIR", path, DWORD(std::size(path)));
        if (length && length < std::size(path) && std::filesystem::path(path).is_absolute())
            directory_ = path;
        if (!directory_.is_absolute()) {
            status_ = "Set LegacyDepthDirectory or PHOENIX_LEGACY_DEPTH_DIR to the absolute depth provider directory.";
            return false;
        }
        const auto library = directory_ / L"phoenix_legacy_depth.dll";
        module_ = LoadLibraryExW(library.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (module_) {
            open_ = reinterpret_cast<PhoenixDepthOpenV1Proc>(GetProcAddress(module_, "PhoenixDepthOpenV1"));
            estimate_ = reinterpret_cast<PhoenixDepthEstimateV1Proc>(GetProcAddress(module_, "PhoenixDepthEstimateV1"));
            close_ = reinterpret_cast<PhoenixDepthCloseV1Proc>(GetProcAddress(module_, "PhoenixDepthCloseV1"));
            if (open_ && estimate_ && close_) return true;
            FreeLibrary(module_);
            module_ = nullptr;
        }
        status_ = "Depth provider or its ONNX Runtime dependencies are missing or incompatible.";
        return false;
    }

  public:
    explicit PhoenixLegacyDepth(const std::filesystem::path& directory = {}) : directory_(directory) {}
    PhoenixLegacyDepth(const PhoenixLegacyDepth&) = delete;
    PhoenixLegacyDepth& operator=(const PhoenixLegacyDepth&) = delete;
    ~PhoenixLegacyDepth() {
        close_session();
        if (module_) FreeLibrary(module_);
    }

    const std::string& status() const { return status_; }

    int estimate(int selection, const uint8_t* rgba, unsigned width, unsigned height,
                 uint64_t adapter_luid, std::vector<float>& output) {
        if (selection_ != selection || adapter_luid_ != adapter_luid) {
            close_session();
            selection_ = selection;
            adapter_luid_ = adapter_luid;
            next_ = attempt_count_ = 0;
            status_ = selection == 3 ? "Depth source disabled." : "Preparing depth estimation.";
            if (selection != 3 && (module_ || load_provider())) {
                auto add_gpu = [&](int model) {
                    attempts_[attempt_count_++] = {model, PhoenixDepthBackend::Cuda};
                    attempts_[attempt_count_++] = {model, PhoenixDepthBackend::DirectML};
                };
                if (selection == 0 || selection == 2) add_gpu(2);
                if (selection == 0 || selection == 1) add_gpu(1);
                // Auto avoids the Large model's multi-second CPU latency.
                attempts_[attempt_count_++] = {selection == 2 ? 2 : 1, PhoenixDepthBackend::Cpu};
            }
        }
        while (session_ || next_ < attempt_count_) {
            char error[512]{};
            if (!session_) {
                const auto [method, backend] = attempts_[next_++];
                const auto file = method == 2 ? L"nvdepthanythingv2.onnx" : L"depth-anything-v2-small.onnx";
                const auto path = directory_ / file;
                if (!std::filesystem::is_regular_file(path, file_error_)) {
                    status_ = method == 2 ? "NVIDIA depth model is missing." : "Depth Anything V2 Small model is missing.";
                    continue;
                }
                session_ = open_(path.c_str(), backend, adapter_luid, error, uint32_t(sizeof(error)));
                if (!session_) {
                    status_ = error[0] ? error : "Could not load the depth model.";
                    continue;
                }
                active_ = method;
                active_backend_ = backend;
            }
            output.resize(size_t(width) * height);
            if (estimate_(session_, rgba, width, height, output.data(), error, uint32_t(sizeof(error)))) {
                status_ = active_ == 2 ? "NVIDIA NvDepthAnythingV2" : "Depth Anything V2 Small";
                switch (active_backend_) {
                case PhoenixDepthBackend::Cuda: status_ += " relative depth (CUDA)."; break;
                case PhoenixDepthBackend::DirectML: status_ += " relative depth (DirectML)."; break;
                default: status_ += " relative depth (CPU)."; break;
                }
                return active_;
            }
            status_ = error[0] ? error : "Depth estimation failed.";
            close_session();
        }
        return 0;
    }

  private:
    std::error_code file_error_;
};
