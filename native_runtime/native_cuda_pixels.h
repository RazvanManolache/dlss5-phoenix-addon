#pragma once
#include <cuda_runtime.h>
namespace phoenix_native_cuda {
cudaError_t unpack_rgba8(const uint8_t* source, cudaSurfaceObject_t target, unsigned width,
                         unsigned height, bool bgra, cudaStream_t stream);
cudaError_t pack_rgba8(cudaSurfaceObject_t source, uint8_t* target, uint32_t* invalid,
                       unsigned width, unsigned height, bool bgra, cudaStream_t stream);
} // namespace phoenix_native_cuda
