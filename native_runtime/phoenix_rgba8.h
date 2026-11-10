// Optional bounded SDR byte-plane extension to the Phoenix C API.
#ifndef PHOENIX_RGBA8_H
#define PHOENIX_RGBA8_H
#include <stddef.h>
#include <stdint.h>
#if defined(PHOENIX_RUNTIME_IMPLEMENTATION) && defined(_WIN32)
#define PHOENIX_RGBA8_API __declspec(dllexport)
#elif defined(PHOENIX_RUNTIME_IMPLEMENTATION)
#define PHOENIX_RGBA8_API __attribute__((visibility("default")))
#else
#define PHOENIX_RGBA8_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
// Version 1, flags=1 (reset), format 0=RGBA8 or 1=BGRA8. Zero stride means
// width*4. Alpha and row padding are preserved. Failure never writes output.
// No caller pointer survives the synchronous call.
typedef struct phoenix_rgba8_frame {
    uint32_t struct_bytes, version, width, height;
    void* data;
    uint64_t capacity_bytes, row_stride_bytes;
    uint32_t format, flags;
} phoenix_rgba8_frame;
PHOENIX_RGBA8_API int phoenix_runtime_frame_rgba8(void* session, const phoenix_rgba8_frame* frame);
#ifdef __cplusplus
}
#endif
#endif
