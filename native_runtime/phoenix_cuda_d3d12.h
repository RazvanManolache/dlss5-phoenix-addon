// Optional Windows GPU texture transport for the native CUDA neural stage.
#ifndef PHOENIX_CUDA_D3D12_H
#define PHOENIX_CUDA_D3D12_H
#ifdef _WIN32
#include <d3d12.h>
#include <stdint.h>
#if defined(PHOENIX_RUNTIME_IMPLEMENTATION)
#define PHOENIX_CUDA_D3D12_API __declspec(dllexport)
#else
#define PHOENIX_CUDA_D3D12_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
// Synchronous, version 1, reserved=0. Canonical SDR color/output are RGBA16F;
// optional pixel-space motion is RG16F. Resources/queue must belong to one
// DIRECT-queue device matching the selected CUDA GPU LUID. All caller states
// are restored. Textures are borrowed for the call; queue/device are AddRef'd
// for the session lifetime. RGB input is clamped as in the
// shared SDK's host adapter; output retains source alpha. No optional V3 planes.
// Metadata is the native frame contract, with exactly one network pass.
// Keep the same DIRECT queue for a session; model replacement retires imports.
// 0=success, 1=unavailable before evaluation (caller may use host transport),
// -1=invalid request/evaluation failure. Rejected inputs do not write output or
// advance history. GPU/device failures require recreating the owning session.
typedef struct phoenix_cuda_d3d12_frame {
    uint32_t struct_bytes, version, reserved[2];
    ID3D12CommandQueue* queue;
    ID3D12Resource* color;
    ID3D12Resource* motion;
    ID3D12Resource* output;
    D3D12_RESOURCE_STATES color_state, motion_state, output_state;
    float motion_scale_x, motion_scale_y;
    const char* metadata_json;
} phoenix_cuda_d3d12_frame;
PHOENIX_CUDA_D3D12_API int phoenix_runtime_frame_d3d12(void*, const phoenix_cuda_d3d12_frame*);
#ifdef __cplusplus
}
#endif
#endif
#endif
