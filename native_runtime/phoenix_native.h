// Windows x64 extension. Reuses Phoenix's optimized D3D12/NVAPI executor.
#ifndef DLSS5_PHOENIX_NATIVE_API_H
#define DLSS5_PHOENIX_NATIVE_API_H
#include <stddef.h>
#include <stdint.h>
#include <d3d12.h>
#ifdef PHOENIX_NATIVE_BUILD
#define PHOENIX_NATIVE_API __declspec(dllexport)
#else
#define PHOENIX_NATIVE_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct PhoenixNativeSession PhoenixNativeSession;
// Structures are versioned by byte size. Initialize with the defaults function.
typedef struct PhoenixNativeControls {
    uint32_t struct_bytes;
    uint32_t reset;
    float motion_scale_x, motion_scale_y;
    float local_tone, local_structure, skin_structure;
    uint32_t auto_mask, inverted_depth;
} PhoenixNativeControls;
PHOENIX_NATIVE_API void phoenix_native_default_controls(PhoenixNativeControls* controls);
// UTF-8 config: width/height are source extent. Optional output_width/height
// default to source extent; neural_width/height select a fitted inference canvas.
// device/queue must both be NULL (owned device) or both supplied (DIRECT queue).
// Supplied interfaces are AddRef'd. Calls on a session stay on its creator thread.
PHOENIX_NATIVE_API PhoenixNativeSession*
phoenix_native_create(const char* config_json, ID3D12Device* device, ID3D12CommandQueue* queue);
// Synchronous: records on its own command list, submits on the selected queue,
// and waits for completion. Caller must order prior producers on that queue and
// externally serialize queue submissions during this call. Inputs/output cannot
// alias. Formats: color/output RGBA16F, optional motion RG16F. Full 2D textures,
// one mip/sample/slice, same device; all states are restored before return.
PHOENIX_NATIVE_API int phoenix_native_frame_d3d12(
    PhoenixNativeSession* session, ID3D12Resource* color, D3D12_RESOURCE_STATES color_state,
    ID3D12Resource* motion, D3D12_RESOURCE_STATES motion_state, ID3D12Resource* output,
    D3D12_RESOURCE_STATES output_state, const PhoenixNativeControls* controls);
// Host alternative: IEEE half bits, RGBA color/output and optional RG motion.
// Strides/capacities are bytes, including only accessible bytes of the final row.
// Input values use the native model's SDR proxy domain; no implicit tonemapping.
PHOENIX_NATIVE_API int phoenix_native_frame_host(PhoenixNativeSession* session, const void* color,
                                                 size_t color_bytes, size_t color_stride,
                                                 const void* motion, size_t motion_bytes,
                                                 size_t motion_stride, void* output,
                                                 size_t output_bytes, size_t output_stride,
                                                 const PhoenixNativeControls* controls);
// Returns AddRef'd interfaces; caller releases them. Either output may be NULL.
PHOENIX_NATIVE_API int phoenix_native_device(PhoenixNativeSession* session, ID3D12Device** device,
                                             ID3D12CommandQueue** queue);
// Wait/drain before destruction. On failure returns -1 and RETAINS the session;
// retry destruction after resolving the device/queue failure. Never free it yourself.
PHOENIX_NATIVE_API int phoenix_native_destroy(PhoenixNativeSession* session);
// JSON reports actual native/spatial path, GPU processing (excludes transfers),
// complete call time and owned allocation ledger. Size includes terminating NUL.
PHOENIX_NATIVE_API size_t phoenix_native_status(PhoenixNativeSession* session, char* destination,
                                                size_t capacity);
PHOENIX_NATIVE_API const char* phoenix_native_error(void);
#ifdef __cplusplus
}
#endif
#endif
