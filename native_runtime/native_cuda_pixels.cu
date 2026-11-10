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
__device__ float clamp_unit(float v){return fminf(1.f,fmaxf(0.f,v));}
__global__ void unpack_external_pixels(cudaTextureObject_t source,cudaTextureObject_t motion,
    cudaSurfaceObject_t target,cudaSurfaceObject_t motion_target,uint32_t* invalid,unsigned width,unsigned height,float motion_x,float motion_y){
    const unsigned x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=width||y>=height)return;
    const float u=(float(x)+.5f)/float(width),v=(float(y)+.5f)/float(height);
    const auto rgb=tex2D<float4>(source,u,v);
    if(!isfinite(rgb.x)||!isfinite(rgb.y)||!isfinite(rgb.z))atomicOr(invalid,1u);
    if(motion){auto mv=tex2D<float2>(motion,u,v);mv.x*=motion_x;mv.y*=motion_y;
        if(!isfinite(mv.x)||!isfinite(mv.y)){
            atomicOr(invalid,1u);mv=make_float2(0.f,0.f);
        }
        surf2Dwrite(mv,motion_target,x*sizeof(float2),y);}
    surf2Dwrite(make_float4(clamp_unit(rgb.x),clamp_unit(rgb.y),clamp_unit(rgb.z),1.f),target,x*sizeof(float4),y);
}
__global__ void validate_external_pixels(cudaSurfaceObject_t source,uint32_t* invalid,unsigned width,unsigned height){
    const unsigned x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=width||y>=height)return;
    const auto h=surf2Dread<ushort4>(source,x*sizeof(ushort4),y);
    if(!isfinite(__half2float(__ushort_as_half(h.x)))||!isfinite(__half2float(__ushort_as_half(h.y)))||
       !isfinite(__half2float(__ushort_as_half(h.z))))atomicOr(invalid,2u);
}
__global__ void finish_external_pixels(cudaTextureObject_t original,cudaSurfaceObject_t alpha,
    cudaSurfaceObject_t output,cudaSurfaceObject_t history,const uint32_t* invalid,
    unsigned width,unsigned height,float intensity,unsigned style,float tone){
    const unsigned x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=width||y>=height||*invalid)return;
    auto h=surf2Dread<ushort4>(output,x*sizeof(ushort4),y);
    // Validation has completed on this stream. Rejected frames write neither
    // history nor presentation. Commit raw half bits before grading the output.
    surf2Dwrite(h,history,x*sizeof(ushort4),y);
    float rgb[3]={__half2float(__ushort_as_half(h.x)),__half2float(__ushort_as_half(h.y)),__half2float(__ushort_as_half(h.z))};
    if(intensity<1.f||(style&&tone>0)){
        const auto source=tex2D<float4>(original,(float(x)+.5f)/float(width),(float(y)+.5f)/float(height));
        const float base[3]={clamp_unit(source.x),clamp_unit(source.y),clamp_unit(source.z)};
        const float amount=clamp_unit(tone);
        if(style==1){
            const float exposure=exp2f(-.1f*amount),contrast=-.25f*amount;
            for(auto& channel:rgb){channel=clamp_unit(channel*exposure);channel=clamp_unit(channel+contrast*(channel*channel*(3.f-2.f*channel)-channel));}
        }
        if(style){
            const float low=fminf(rgb[0],fminf(rgb[1],rgb[2])),high=fmaxf(rgb[0],fmaxf(rgb[1],rgb[2]));
            const float light=(high+low)*.5f,chroma=high-low;
            if(chroma>0){const float denominator=1.f-fabsf(2.f*light-1.f),old=denominator>0?chroma/denominator:0;
                const float scale=1.f-(style==1?.1f:.15f)*amount;
                const float factor=old>0?clamp_unit(old*scale)/old:0;
                for(auto& channel:rgb)channel=clamp_unit(light+(channel-light)*factor);}
        }
        for(unsigned c=0;c<3;++c)rgb[c]=clamp_unit(base[c]+intensity*(rgb[c]-base[c]));
        h.x=__half_as_ushort(__float2half_rn(rgb[0]));h.y=__half_as_ushort(__float2half_rn(rgb[1]));h.z=__half_as_ushort(__float2half_rn(rgb[2]));
    }
    h.w=surf2Dread<ushort4>(alpha,x*sizeof(ushort4),y).w;
    surf2Dwrite(h,output,x*sizeof(ushort4),y);
}
cudaError_t unpack_external(cudaTextureObject_t source,cudaTextureObject_t motion,cudaSurfaceObject_t target,cudaSurfaceObject_t motion_target,uint32_t* invalid,unsigned width,unsigned height,float motion_x,float motion_y,cudaStream_t stream){
    unpack_external_pixels<<<dim3((width+31)/32,(height+7)/8),dim3(32,8),0,stream>>>(source,motion,target,motion_target,invalid,width,height,motion_x,motion_y);return cudaGetLastError();
}
cudaError_t validate_external(cudaSurfaceObject_t source,uint32_t* invalid,unsigned width,unsigned height,cudaStream_t stream){
    validate_external_pixels<<<dim3((width+31)/32,(height+7)/8),dim3(32,8),0,stream>>>(source,invalid,width,height);return cudaGetLastError();
}
cudaError_t finish_external(cudaTextureObject_t original,cudaSurfaceObject_t alpha,cudaSurfaceObject_t output,cudaSurfaceObject_t history,const uint32_t* invalid,unsigned width,unsigned height,float intensity,unsigned style,float tone,cudaStream_t stream){
    finish_external_pixels<<<dim3((width+31)/32,(height+7)/8),dim3(32,8),0,stream>>>(original,alpha,output,history,invalid,width,height,intensity,style,tone);return cudaGetLastError();
}
} // namespace phoenix_native_cuda
