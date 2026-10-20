// Stable host-plane C interface for applications that do not use ReShade.
// Declarations intentionally contain no Windows, C++, Metal or vendor SDK types.
#ifndef DLSS5_PHOENIX_C_API_H
#define DLSS5_PHOENIX_C_API_H
#include <stddef.h>
#include <stdint.h>
#if defined(PHOENIX_RUNTIME_IMPLEMENTATION) && defined(_WIN32)
#define PHOENIX_C_API __declspec(dllexport)
#elif defined(PHOENIX_RUNTIME_IMPLEMENTATION)
#define PHOENIX_C_API __attribute__((visibility("default")))
#else
#define PHOENIX_C_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
// config_json: UTF-8 JSON, documented in docs/LIBRARY.md. Returns NULL on error.
// Each session owns its model, scratch arena and temporal history. Call destroy
// only after all calls using the handle have returned. One frame at a time per
// session; caller memory must remain valid until frame() returns.
PHOENIX_C_API void *phoenix_runtime_create(const char *config_json, unsigned width, unsigned height);
PHOENIX_C_API void phoenix_runtime_destroy(void *session);
// Packed row-major float32: color/output=RGB, motion=RG (pixel units), depth and
// validity=R. Optional planes may be NULL; metadata must declare their presence.
// Output has width*height*3 floats. This ABI does not infer buffer capacities.
// Returns 0 on success, -1 on failure. No pointers are retained after return.
// Metadata also accepts transformer_blocks (1..8), repeats (eight 0..4 counts),
// passes (1..10), and adaptive_budget. GPU-time budgets consume backend GPU
// events; FPS budgets require caller application presentation timestamps.
// These controls share one scheduler across all portable backends/models.
PHOENIX_C_API int phoenix_runtime_frame(void *session, const float *color, const float *motion,
                          const float *depth, const float *validity, const char *metadata_json,
                          float *output);
// Versioned bounded host planes. Zero stride means tightly packed; capacity is
// measured in bytes from data to the end of the last available row. All planes
// cover the session's full extent; no resampling or color transfer is applied.
typedef struct phoenix_plane_f32 {
    const float *data;
    uint64_t capacity_bytes, row_stride_bytes;
    uint32_t width, height, channels, reserved;
} phoenix_plane_f32;
typedef struct phoenix_output_f32 {
    float *data;
    uint64_t capacity_bytes, row_stride_bytes;
    uint32_t width, height, channels, reserved;
} phoenix_output_f32;
enum {
    PHOENIX_FRAME_VERSION_2 = 2,
    PHOENIX_CONTROL_NONE = 0,
    // Experimental Phoenix contract, NOT certified NVIDIA mask semantics:
    // RGBA in 0..1; R multiplies display intensity, G tone, B structure;
    // A must be 1 (reserved). Map mode disables automatic skin conditioning.
    PHOENIX_CONTROL_EXPERIMENTAL_RGB_1 = 1
};
typedef struct phoenix_frame_v2 {
    uint32_t struct_bytes, version, control_profile, reserved;
    phoenix_plane_f32 color, motion, depth, validity, control_mask;
    phoenix_output_f32 output;
    const char *metadata_json;
} phoenix_frame_v2;
// Optional absent planes must be entirely zero-initialized. Plane declarations
// determine has_motion/depth/history_validity; conflicting metadata is rejected.
// All input planes are copied before execution, so input/output aliasing is safe.
// Failure during validation leaves output and temporal state untouched.
PHOENIX_C_API int phoenix_runtime_frame_v2(void *session, const phoenix_frame_v2 *frame);
// Advanced experimental host contract. base.struct_bytes=sizeof(v3), version=3.
// UI is premultiplied RGBA in the same SDR proxy domain; UIAlpha is coverage R;
// Backbuffer is RGB with UI already present. Distortion is RGBA normalized UV
// displacement: RG distorted->canonical, BA canonical->distorted. Full extents.
typedef struct phoenix_frame_v3 {
    phoenix_frame_v2 base;
    phoenix_plane_f32 ui, ui_alpha, backbuffer, distortion;
} phoenix_frame_v3;
enum { PHOENIX_FRAME_VERSION_3 = 3 };
PHOENIX_C_API int phoenix_runtime_frame_v3(void *session, const phoenix_frame_v3 *frame);
// Thread-local error; copy it before making another call on the same thread.
PHOENIX_C_API const char *phoenix_runtime_error(void);
// UTF-8 JSON status. Return includes terminating NUL. If capacity is insufficient,
// nothing is written; query size with (session,NULL,0), then allocate and repeat.
PHOENIX_C_API size_t phoenix_runtime_status(void *session, char *destination, size_t capacity);
// Model-free portable API discovery. Same size/NUL/capacity rules as status().
// Describes compiled contracts, not installed backend/device validation.
PHOENIX_C_API size_t phoenix_runtime_capabilities(char *destination, size_t capacity);
// Transactionally loads a compatible weight_source JSON object. An invalid
// source leaves the session intact. Success retires graph/caches and history.
PHOENIX_C_API int phoenix_runtime_replace_model(void *session, const char *weight_source_json);
#ifdef __cplusplus
}
#endif
#undef PHOENIX_C_API
#endif
