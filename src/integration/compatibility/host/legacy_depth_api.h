#pragma once
#include <cstdint>

// Optional x64 provider ABI. The caller owns the full-size RGBA8 and
// conventional depth buffers (near=0, far=1). The provider owns its session.
enum class PhoenixDepthBackend : int { Cpu, Cuda, DirectML };
using PhoenixDepthOpenV1Proc = void* (*)(const wchar_t* model_path, PhoenixDepthBackend backend,
                                        uint64_t adapter_luid, char* error, uint32_t error_size);
using PhoenixDepthEstimateV1Proc = int (*)(void* session, const uint8_t* rgba, uint32_t width, uint32_t height,
                                       float* depth, char* error, uint32_t error_size);
using PhoenixDepthCloseV1Proc = void (*)(void* session);
