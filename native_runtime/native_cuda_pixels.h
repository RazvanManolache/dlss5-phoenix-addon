#pragma once
#include <cuda_runtime.h>
namespace phoenix_native_cuda {
cudaError_t unpack_rgba8(const uint8_t* source, cudaSurfaceObject_t target, unsigned width,
                         unsigned height, bool bgra, cudaStream_t stream);
cudaError_t pack_rgba8(cudaSurfaceObject_t source, uint8_t* target, uint32_t* invalid,
                       unsigned width, unsigned height, bool bgra, cudaStream_t stream);
cudaError_t unpack_external(cudaTextureObject_t source, cudaTextureObject_t motion,
    cudaSurfaceObject_t target,cudaSurfaceObject_t motion_target,uint32_t* invalid,unsigned width,unsigned height,
    float motion_x,float motion_y,cudaStream_t stream);
cudaError_t validate_external(cudaSurfaceObject_t source,uint32_t* invalid,
    unsigned width,unsigned height,cudaStream_t stream);
cudaError_t finish_external(cudaTextureObject_t original,cudaSurfaceObject_t alpha,
    cudaSurfaceObject_t output,cudaSurfaceObject_t history,const uint32_t* invalid,
    unsigned width,unsigned height,float intensity,unsigned style,
    float tone,cudaStream_t stream);
} // namespace phoenix_native_cuda
