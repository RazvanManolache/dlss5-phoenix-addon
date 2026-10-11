#pragma once
#include "compute_pass.h"
#include "resolution.h"
#include "../../../third_party/xess/xess_d3d12.h"
#include <filesystem>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>

namespace phoenix_addon::phoenix::direct {
// Public vendor APIs only. NR inference and temporal reconstruction are
// independent choices; both run on the rendering device at this boundary.
class TemporalUpscaler {
    using Create = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*, NVSDK_NGX_Feature,
                                                 const NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
    using Evaluate = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*,
                                                   const NVSDK_NGX_Handle*,
                                                   const NVSDK_NGX_Parameter*,
                                                   PFN_NVSDK_NGX_ProgressCallback);
    using Release = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
    using Parameters = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);
    using DestroyParameters = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*);
    ComPtr<ID3D12Device> device_;
    Resolution geometry_;
    int method_;
    std::filesystem::path directory_;
    std::shared_ptr<phoenix_native::memory::Generation> accounting_;
    HMODULE module_ = nullptr;
    NVSDK_NGX_Handle* dlss_ = nullptr;
    NVSDK_NGX_Parameter* parameters_ = nullptr;
    Create create_ = nullptr;
    Evaluate evaluate_ = nullptr;
    Release release_ = nullptr;
    DestroyParameters destroy_parameters_ = nullptr;
    xess_context_handle_t xess_ = nullptr;
    decltype(&xessDestroyContext) destroy_xess_ = nullptr;
    decltype(&xessD3D12Execute) execute_xess_ = nullptr;
    ComPtr<ID3D12Resource> output_;
    bool attempted_ = false, initialized_ = false, inverted_ = true, history_gap_ = true;
    uint64_t frames_ = 0;
    std::string status_ = "Waiting for observed depth and motion guides";
    template <class Function> Function function(const char* name) {
        auto address = GetProcAddress(module_, name);
        if (!address)
            throw std::runtime_error(std::string("Upscaler runtime lacks ") + name);
        return reinterpret_cast<Function>(address);
    }
    static NVSDK_NGX_Result create_feature(Create function, ID3D12GraphicsCommandList* command,
                                           const NVSDK_NGX_Parameter* parameters,
                                           NVSDK_NGX_Handle** feature) {
        __try {
            return function(command, NVSDK_NGX_Feature_SuperSampling, parameters, feature);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            return NVSDK_NGX_Result_Fail;
        }
    }
    static NVSDK_NGX_Result evaluate_feature(Evaluate function, ID3D12GraphicsCommandList* command,
                                             const NVSDK_NGX_Handle* feature,
                                             const NVSDK_NGX_Parameter* parameters) {
        __try {
            return function(command, feature, parameters, nullptr);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            return NVSDK_NGX_Result_Fail;
        }
    }
    HMODULE load_core() {
        if (auto loaded = GetModuleHandleW(L"_nvngx.dll")) {
            wchar_t path[32768]{};
            check(GetModuleFileNameW(loaded, path, 32768) > 0, "NGX module path unavailable");
            return LoadLibraryExW(
                path, nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        }
        wchar_t windows[32768]{};
        check(GetWindowsDirectoryW(windows, 32768) > 0, "Windows directory unavailable");
        auto repository = std::filesystem::path(windows) / L"System32/DriverStore/FileRepository";
        std::error_code error;
        for (const auto& folder : std::filesystem::directory_iterator(repository, error)) {
            if (folder.path().filename().wstring().find(L"nv_dispi.inf_") != 0)
                continue;
            auto path = folder.path() / L"_nvngx.dll";
            if (std::filesystem::is_regular_file(path, error))
                if (auto loaded = LoadLibraryExW(path.c_str(), nullptr,
                                                 LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                                     LOAD_LIBRARY_SEARCH_DEFAULT_DIRS))
                    return loaded;
        }
        return nullptr;
    }
    void initialize_dlss(ID3D12GraphicsCommandList* command, bool inverted) {
        check(std::filesystem::is_regular_file(directory_ / L"nvngx_dlss.dll"),
              "NVIDIA DLSS SR requires nvngx_dlss.dll beside the game");
        // The addon runs at an observed game NGX boundary. Reuse that device
        // context; initializing or shutting it down here would affect the game.
        module_ = load_core();
        check(module_ != nullptr, "NVIDIA NGX runtime unavailable on this device");
        auto allocate = function<Parameters>("NVSDK_NGX_D3D12_AllocateParameters");
        auto capabilities = function<Parameters>("NVSDK_NGX_D3D12_GetCapabilityParameters");
        destroy_parameters_ = function<DestroyParameters>("NVSDK_NGX_D3D12_DestroyParameters");
        create_ = function<Create>("NVSDK_NGX_D3D12_CreateFeature");
        evaluate_ = function<Evaluate>("NVSDK_NGX_D3D12_EvaluateFeature");
        release_ = function<Release>("NVSDK_NGX_D3D12_ReleaseFeature");
        NVSDK_NGX_Parameter* caps = nullptr;
        const auto queried = capabilities(&caps);
        int available = 0;
        if (NVSDK_NGX_SUCCEED(queried) && caps)
            caps->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &available);
        if (caps)
            destroy_parameters_(caps);
        check(NVSDK_NGX_SUCCEED(queried), "DLSS SR requires the game's initialized NGX context");
        check(available != 0, "DLSS SR is unavailable on this rendering GPU/runtime");
        check(NVSDK_NGX_SUCCEED(allocate(&parameters_)) && parameters_,
              "DLSS parameter allocation failed");
        parameters_->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u);
        parameters_->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
        parameters_->Set(NVSDK_NGX_Parameter_Width, geometry_.image_width);
        parameters_->Set(NVSDK_NGX_Parameter_Height, geometry_.image_height);
        parameters_->Set(NVSDK_NGX_Parameter_OutWidth, geometry_.output_width);
        parameters_->Set(NVSDK_NGX_Parameter_OutHeight, geometry_.output_height);
        const float scale = geometry_.motion_x();
        const auto quality = scale >= .6f    ? NVSDK_NGX_PerfQuality_Value_MaxQuality
                             : scale >= .45f ? NVSDK_NGX_PerfQuality_Value_MaxPerf
                                             : NVSDK_NGX_PerfQuality_Value_UltraPerformance;
        parameters_->Set(NVSDK_NGX_Parameter_PerfQualityValue, int(quality));
        parameters_->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                         int(NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
                             NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                             NVSDK_NGX_DLSS_Feature_Flags_AutoExposure |
                             (inverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0)));
        parameters_->Set(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 1);
        NVSDK_NGX_Handle* created = nullptr;
        const auto result = create_feature(create_, command, parameters_, &created);
        if (!NVSDK_NGX_SUCCEED(result) || !created) {
            char message[160]{};
            std::snprintf(message, sizeof(message),
                          "NGX DLSS SR creation failed (0x%08X, command list type %u)",
                          unsigned(result), unsigned(command->GetType()));
            throw std::runtime_error(message);
        }
        // Failed creation must never adopt or release a host-owned handle.
        dlss_ = created;
    }
    void initialize_xess(bool inverted) {
        auto path = directory_ / L"libxess.dll";
        module_ =
            LoadLibraryExW(path.c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        check(module_ != nullptr,
              "Intel XeSS requires libxess.dll and its dependencies beside the game");
        auto create = function<decltype(&xessD3D12CreateContext)>("xessD3D12CreateContext");
        destroy_xess_ = function<decltype(&xessDestroyContext)>("xessDestroyContext");
        execute_xess_ = function<decltype(&xessD3D12Execute)>("xessD3D12Execute");
        check(create(device_.Get(), &xess_) >= XESS_RESULT_SUCCESS && xess_,
              "XeSS does not support this rendering GPU/driver");
        auto query =
            function<decltype(&xessGetOptimalInputResolution)>("xessGetOptimalInputResolution");
        auto initialize = function<decltype(&xessD3D12Init)>("xessD3D12Init");
        xess_2d_t target{geometry_.output_width, geometry_.output_height};
        xess_quality_settings_t best = XESS_QUALITY_SETTING_QUALITY;
        double score = 1e30;
        bool found = false;
        for (int value = XESS_QUALITY_SETTING_ULTRA_PERFORMANCE; value <= XESS_QUALITY_SETTING_AA;
             ++value) {
            xess_2d_t optimal{}, minimum{}, maximum{};
            auto quality = static_cast<xess_quality_settings_t>(value);
            if (query(xess_, &target, quality, &optimal, &minimum, &maximum) < XESS_RESULT_SUCCESS)
                continue;
            if (geometry_.image_width < minimum.x || geometry_.image_width > maximum.x ||
                geometry_.image_height < minimum.y || geometry_.image_height > maximum.y)
                continue;
            double distance = std::abs(double(optimal.x) - geometry_.image_width) +
                              std::abs(double(optimal.y) - geometry_.image_height);
            if (distance < score) {
                found = true;
                score = distance;
                best = quality;
            }
        }
        check(found,
              "XeSS does not accept this active resolution; choose a different quality/aspect");
        xess_d3d12_init_params_t settings{};
        settings.outputResolution = target;
        settings.qualitySetting = best;
        settings.initFlags =
            XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE | (inverted ? XESS_INIT_FLAG_INVERTED_DEPTH : 0);
        check(initialize(xess_, &settings) >= XESS_RESULT_SUCCESS, "XeSS initialization failed");
    }

  public:
    TemporalUpscaler(ID3D12Device* device, Resolution geometry, int method,
                     std::shared_ptr<phoenix_native::memory::Generation> accounting = {},
                     std::filesystem::path directory = {})
        : device_(device), geometry_(geometry), method_(method), directory_(std::move(directory)),
          accounting_(std::move(accounting)) {
        if (directory_.empty()) {
            wchar_t path[32768]{};
            GetModuleFileNameW(nullptr, path, 32768);
            directory_ = std::filesystem::path(path).parent_path();
        }
    }
    ~TemporalUpscaler() {
        // Runner generations are destroyed only after their submission fence.
        if (dlss_ && release_)
            release_(dlss_);
        if (parameters_ && destroy_parameters_)
            destroy_parameters_(parameters_);
        if (xess_ && destroy_xess_)
            destroy_xess_(xess_);
        // Never shut down the game's NGX context or its other features.
        if (module_)
            FreeLibrary(module_);
    }
    const std::string& status() const {
        return status_;
    }
    uint64_t frames() const {
        return frames_;
    }
    ID3D12Resource* record(ID3D12GraphicsCommandList* command, ID3D12Resource* color,
                           ID3D12Resource* motion, ID3D12Resource* depth, bool reset, bool inverted,
                           float motion_x, float motion_y, float jitter_x, float jitter_y) {
        if (!depth || !motion) {
            history_gap_ = true;
            status_ = "Spatial fallback: observed depth and motion are required";
            return nullptr;
        }
        if (!attempted_) {
            attempted_ = true;
            inverted_ = inverted;
            try {
                if (method_ == 1)
                    initialize_dlss(command, inverted);
                else
                    initialize_xess(inverted);
                output_ = phoenix_native::image(
                    device_.Get(), geometry_.output_width, geometry_.output_height,
                    DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                phoenix_native::memory::resource(device_.Get(), output_.Get(), accounting_,
                                                 phoenix_native::memory::Kind::bridge);
                initialized_ = true;
            } catch (const std::exception& error) {
                status_ = std::string("Spatial fallback: ") + error.what();
                return nullptr;
            }
        }
        if (!initialized_)
            return nullptr;
        // A changed depth convention changes the create-time contract. Rebuild
        // at a fence boundary rather than silently using the wrong history.
        if (inverted_ != inverted) {
            history_gap_ = true;
            status_ =
                "Spatial fallback: depth convention changed; reselect the upscaler to recreate it";
            return nullptr;
        }
        if (method_ == 1) {
            parameters_->Set(NVSDK_NGX_Parameter_Color, color);
            parameters_->Set(NVSDK_NGX_Parameter_Output, output_.Get());
            parameters_->Set(NVSDK_NGX_Parameter_MotionVectors, motion);
            parameters_->Set(NVSDK_NGX_Parameter_Depth, depth);
            parameters_->Set(NVSDK_NGX_Parameter_Reset, int(reset || history_gap_ || !frames_));
            parameters_->Set(NVSDK_NGX_Parameter_MV_Scale_X, motion_x);
            parameters_->Set(NVSDK_NGX_Parameter_MV_Scale_Y, motion_y);
            parameters_->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, jitter_x);
            parameters_->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, jitter_y);
            parameters_->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.f);
            parameters_->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.f);
            parameters_->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, geometry_.left);
            parameters_->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, geometry_.top);
            parameters_->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, geometry_.left);
            parameters_->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, geometry_.top);
            parameters_->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, geometry_.left);
            parameters_->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, geometry_.top);
            parameters_->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
                             geometry_.image_width);
            parameters_->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
                             geometry_.image_height);
            check(NVSDK_NGX_SUCCEED(evaluate_feature(evaluate_, command, dlss_, parameters_)),
                  "DLSS SR evaluation failed");
        } else {
            auto scale = function<decltype(&xessSetVelocityScale)>("xessSetVelocityScale");
            check(scale(xess_, motion_x, motion_y) >= XESS_RESULT_SUCCESS,
                  "XeSS motion scale rejected");
            xess_d3d12_execute_params_t frame{};
            frame.pColorTexture = color;
            frame.pVelocityTexture = motion;
            frame.pDepthTexture = depth;
            frame.pOutputTexture = output_.Get();
            frame.jitterOffsetX = jitter_x;
            frame.jitterOffsetY = jitter_y;
            frame.exposureScale = 1;
            frame.resetHistory = reset || history_gap_ || !frames_;
            frame.inputWidth = geometry_.image_width;
            frame.inputHeight = geometry_.image_height;
            frame.inputColorBase = frame.inputMotionVectorBase =
                frame.inputDepthBase = {geometry_.left, geometry_.top};
            check(execute_xess_(xess_, command, &frame) >= XESS_RESULT_SUCCESS,
                  "XeSS evaluation failed");
        }
        phoenix_native::uav(command);
        ++frames_;
        history_gap_ = false;
        status_ = method_ == 1 ? "NVIDIA DLSS Super Resolution active" : "Intel XeSS active";
        return output_.Get();
    }
};
} // namespace phoenix_addon::phoenix::direct
