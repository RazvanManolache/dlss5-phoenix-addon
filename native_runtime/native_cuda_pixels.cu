#include <cstdint>
#include <cuda_fp16.h>
#include "native_cuda_pixels.h"
namespace phoenix_native_cuda {
__global__ void unpack_pixels(const uchar4* source, cudaSurfaceObject_t target, unsigned width,
                              unsigned height, bool bgra) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    const auto pixel = source[size_t(y) * width + x];
    const auto value = make_float4(__fdiv_rn(float(bgra ? pixel.z : pixel.x), 255.f),
                                   __fdiv_rn(float(pixel.y), 255.f),
                                   __fdiv_rn(float(bgra ? pixel.x : pixel.z), 255.f), 1.f);
    surf2Dwrite(value, target, x * sizeof(float4), y);
}
__device__ uint8_t quantize(float value, uint32_t* invalid) {
    if (!isfinite(value)) {
        atomicExch(invalid, 1u);
        return 0;
    }
    const float scaled = fminf(1.f, fmaxf(0.f, value)) * 255.f;
    const unsigned whole = __float2uint_rz(scaled);
    return uint8_t(whole + unsigned(scaled - float(whole) >= 0.5f));
}
__global__ void pack_pixels(cudaSurfaceObject_t source, uchar4* target, uint32_t* invalid,
                            unsigned width, unsigned height, bool bgra) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    const auto half = surf2Dread<ushort4>(source, x * sizeof(ushort4), y);
    const uint8_t r = quantize(__half2float(__ushort_as_half(half.x)), invalid);
    const uint8_t g = quantize(__half2float(__ushort_as_half(half.y)), invalid);
    const uint8_t b = quantize(__half2float(__ushort_as_half(half.z)), invalid);
    const auto alpha = target[size_t(y) * width + x].w;
    target[size_t(y) * width + x] = make_uchar4(bgra ? b : r, g, bgra ? r : b, alpha);
}
cudaError_t unpack_rgba8(const uint8_t* source, cudaSurfaceObject_t target, unsigned width,
                         unsigned height, bool bgra, cudaStream_t stream) {
    unpack_pixels<<<dim3((width + 31) / 32, (height + 7) / 8), dim3(32, 8), 0, stream>>>(
        reinterpret_cast<const uchar4*>(source), target, width, height, bgra);
    return cudaGetLastError();
}
cudaError_t pack_rgba8(cudaSurfaceObject_t source, uint8_t* target, uint32_t* invalid,
                       unsigned width, unsigned height, bool bgra, cudaStream_t stream) {
    pack_pixels<<<dim3((width + 31) / 32, (height + 7) / 8), dim3(32, 8), 0, stream>>>(
        source, reinterpret_cast<uchar4*>(target), invalid, width, height, bgra);
    return cudaGetLastError();
}
} // namespace phoenix_native_cuda
