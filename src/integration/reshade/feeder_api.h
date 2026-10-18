#pragma once
#include <d3d12.h>
#include <cstdint>
#include <cmath>

// Borrowed D3D12 objects; the caller submits the observed list and waits for its
// fence. No NGX feature or NVIDIA SR evaluation is involved in this interface.
inline constexpr uint32_t PhoenixFeederAbiV1 = 1;
enum class PhoenixFeederStatusV1 : uint32_t { Deferred = 0, Recorded = 1, Bypass = 2 };
struct PhoenixFeederFrameV1 {
    uint32_t size = sizeof(PhoenixFeederFrameV1);
    uint32_t version = PhoenixFeederAbiV1;
    ID3D12GraphicsCommandList* command_list = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    // The host seeds output with the original frame. Phoenix processes it in
    // place. All resources enter/leave in COMMON with fresh observed barriers.
    ID3D12Resource* output = nullptr;
    ID3D12Resource* motion = nullptr;
    ID3D12Resource* depth = nullptr;
    uint32_t reset = 0;
    uint32_t stream_id = 0; // 0 mono, 1 left eye, 2 right eye
    float motion_scale_x = 1.0f;
    float motion_scale_y = 1.0f;
    uint32_t depth_inverted = 0;
    uint32_t reserved = 0;
};
struct PhoenixFeederResultV1 {
    uint32_t size = sizeof(PhoenixFeederResultV1);
    uint32_t version = PhoenixFeederAbiV1;
    PhoenixFeederStatusV1 status = PhoenixFeederStatusV1::Deferred;
    uint32_t reserved = 0;
    uint64_t recorded_frames = 0;
};
inline bool PhoenixFeederContractV1(const PhoenixFeederFrameV1& frame,
                                   const PhoenixFeederResultV1& result) noexcept {
    return frame.size == sizeof(frame) && frame.version == PhoenixFeederAbiV1 &&
           result.size == sizeof(result) && result.version == PhoenixFeederAbiV1 &&
           frame.command_list && frame.queue && frame.output && frame.motion &&
           frame.output != frame.motion && frame.output != frame.depth &&
           (!frame.depth || frame.depth != frame.motion) &&
           frame.stream_id <= 2 && frame.reset <= 1 && frame.depth_inverted <= 1 &&
           frame.reserved == 0 && result.reserved == 0 &&
           std::isfinite(frame.motion_scale_x) && std::isfinite(frame.motion_scale_y);
}
using PhoenixFeederVersionProc = uint32_t(WINAPI*)();
using PhoenixFeederEvaluateProc = HRESULT(WINAPI*)(const PhoenixFeederFrameV1*, PhoenixFeederResultV1*);
// Optional additive entry point. A nonzero token identifies an immutable input
// held across retries; callers must change it before supplying different pixels.
using PhoenixFeederEvaluateLockedProc = HRESULT(WINAPI*)(const PhoenixFeederFrameV1*, PhoenixFeederResultV1*, uint64_t);
// Optional read-only progress export: buffer size includes the trailing NUL.
// Completion still requires the caller's submitted GPU fence to retire.
using PhoenixFeederProgressJsonProc = uint32_t(WINAPI*)(char*, uint32_t);
// Optional live-only tuning transaction (bounded JSON, no model/resource edits).
using PhoenixFeederLiveSettingsProc = BOOL(WINAPI*)(const char*, uint32_t);
// Additive experimental interface. V1 and its version export remain unchanged.
// Optional resources cover the full frame, enter/leave COMMON, and must have
// fresh observed barriers on the supplied list. RGB values are SDR model proxy;
// maps/UI/distortion use RGBA32F, alpha R32F, backbuffer RGBA16F. UI premultiplied.
struct PhoenixFeederFrameV2 {
    uint32_t size=sizeof(PhoenixFeederFrameV2),version=2;
    PhoenixFeederFrameV1 frame;
    float intensity=1,local_tone=1,local_structure=1,skin_structure=-1;
    uint32_t style=0,auto_mask=1;
    ID3D12Resource *control_mask=nullptr,*ui=nullptr,*ui_alpha=nullptr,*backbuffer=nullptr,*distortion=nullptr;
    uint32_t appearance_generation=0,reserved=0;
};
using PhoenixFeederEvaluateProcV2=HRESULT(WINAPI*)(const PhoenixFeederFrameV2*,PhoenixFeederResultV1*);
inline bool PhoenixFeederContractV2(const PhoenixFeederFrameV2& frame,const PhoenixFeederResultV1& result) noexcept {
    return frame.size==sizeof(frame)&&frame.version==2&&!frame.reserved&&PhoenixFeederContractV1(frame.frame,result)&&
        std::isfinite(frame.intensity)&&frame.intensity>=0&&frame.intensity<=2&&frame.style<=2&&frame.auto_mask<=1&&
        std::isfinite(frame.local_tone)&&frame.local_tone>=0&&frame.local_tone<=2&&
        std::isfinite(frame.local_structure)&&frame.local_structure>=0&&frame.local_structure<=2&&
        std::isfinite(frame.skin_structure)&&frame.skin_structure>=-1&&frame.skin_structure<=2&&!(frame.ui&&frame.ui_alpha);
}
