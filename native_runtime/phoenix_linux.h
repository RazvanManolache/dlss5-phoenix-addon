#ifndef PHOENIX_LINUX_H
#define PHOENIX_LINUX_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Optional native-CUDA frontend extension. All rows are bounded. The session
// owns staging storage; caller memory is read before any result is written.
// Format 0=RGBA8, 1=BGRA8. flags must be 1 (reset); alpha is preserved.
// struct_bytes=sizeof(phoenix_rgba8_frame), version=1. Returns 0 or -1;
// phoenix_runtime_error reports failure. No caller pointers survive this call.
typedef struct phoenix_rgba8_frame {
    uint32_t struct_bytes, version, width, height;
    void* data;
    uint64_t capacity_bytes, row_stride_bytes;
    uint32_t format, flags;
} phoenix_rgba8_frame;
int phoenix_linux_frame_rgba8(void* session, const phoenix_rgba8_frame* frame);
#ifdef __cplusplus
}
#endif
#endif
