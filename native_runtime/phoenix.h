// Stable host-plane C interface for applications that do not use ReShade.
// Declarations intentionally contain no Windows, C++, Metal or vendor SDK types.
#ifndef DLSS5_PHOENIX_C_API_H
#define DLSS5_PHOENIX_C_API_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// config_json: UTF-8 JSON, documented in docs/LIBRARY.md. Returns NULL on error.
// Each session owns its model, scratch arena and temporal history. Call destroy
// only after all calls using the handle have returned. One frame at a time per
// session; caller memory must remain valid until frame() returns.
void *phoenix_runtime_create(const char *config_json, unsigned width, unsigned height);
void phoenix_runtime_destroy(void *session);
// Packed row-major float32: color/output=RGB, motion=RG (pixel units), depth and
// validity=R. Optional planes may be NULL; metadata must declare their presence.
// Output has width*height*3 floats. This ABI does not infer buffer capacities.
// Returns 0 on success, -1 on failure. No pointers are retained after return.
int phoenix_runtime_frame(void *session, const float *color, const float *motion,
                          const float *depth, const float *validity, const char *metadata_json,
                          float *output);
// Thread-local error; copy it before making another call on the same thread.
const char *phoenix_runtime_error(void);
// UTF-8 JSON status. Return includes terminating NUL. If capacity is insufficient,
// nothing is written; query size with (session,NULL,0), then allocate and repeat.
size_t phoenix_runtime_status(void *session, char *destination, size_t capacity);
#ifdef __cplusplus
}
#endif
#endif
