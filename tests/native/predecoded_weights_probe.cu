// Operator-only experiment: expand the B/weight fragments once, then reuse
// their FP16 representation. This is not a full-model performance estimate.
#define PHOENIX_ARCHITECTURE_TEST
#include "../../src/native/kernels/production/common/architecture_math.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

struct DecodedWord { uint32_t even,odd; };
__global__ void expand_weights(const uint32_t* packed,DecodedWord* decoded,unsigned count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)phoenix_architecture::decode_fp8_fragment(packed[i],decoded[i].even,decoded[i].odd);
}
template<bool Cached> __global__ void operator_probe(
    const uint32_t* activation,const uint32_t* weight,const DecodedWord* decoded,
    uint32_t* output,unsigned count) {
    const unsigned lane=blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t checksum=0;
#pragma unroll 1
    for(unsigned step=0;step<32;++step) {
        const unsigned index=(lane+step*32)&(count-1);
        const auto* a=activation+index*4;
        uint32_t d0,d1;
        if constexpr(!Cached)
            phoenix_architecture::fp8_matrix(d0,d1,a[0],a[1],a[2],a[3],weight[index*2],weight[index*2+1],0,0);
        else {
            uint32_t x0,y0,x1,y1,x2,y2,x3,y3;
            phoenix_architecture::decode_fp8_fragment(a[0],x0,y0);
            phoenix_architecture::decode_fp8_fragment(a[1],x1,y1);
            phoenix_architecture::decode_fp8_fragment(a[2],x2,y2);
            phoenix_architecture::decode_fp8_fragment(a[3],x3,y3);
            const auto b0=decoded[index*2],b1=decoded[index*2+1];
            phoenix_architecture::half_matrix(d0,d1,x0,x1,y0,y1,b0.even,b0.odd,0,0);
            phoenix_architecture::half_matrix(d0,d1,x2,x3,y2,y3,b1.even,b1.odd,d0,d1);
        }
        checksum=(checksum<<1 | checksum>>31)^d0^d1;
    }
    output[lane]=checksum;
}
static void checked(cudaError_t value) {
    if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));
}
int main() {
    uint32_t *a=nullptr,*b=nullptr,*out=nullptr;DecodedWord* expanded=nullptr;
    cudaEvent_t start=nullptr,end=nullptr;
    try {
        constexpr unsigned count=16384;
        constexpr std::array<unsigned,7> codes{0,0x20,0x28,0x30,0xa0,0xa8,0xb0};
        std::vector<uint32_t> inputs(count*4),weights(count*2),reference(count),actual(count);
        uint32_t random=17;
        auto word=[&]{uint32_t value=0;for(unsigned byte=0;byte<4;++byte){random=random*1664525+1013904223;value|=codes[(random>>16)%codes.size()]<<(byte*8);}return value;};
        for(auto& v:inputs)v=word();for(auto& v:weights)v=word();
        checked(cudaMalloc(&a,inputs.size()*4));checked(cudaMalloc(&b,weights.size()*4));
        checked(cudaMalloc(&out,count*4));checked(cudaMalloc(&expanded,weights.size()*sizeof(DecodedWord)));
        checked(cudaMemcpy(a,inputs.data(),inputs.size()*4,cudaMemcpyHostToDevice));
        checked(cudaMemcpy(b,weights.data(),weights.size()*4,cudaMemcpyHostToDevice));
        checked(cudaEventCreate(&start));checked(cudaEventCreate(&end));
        checked(cudaEventRecord(start));expand_weights<<<weights.size()/128,128>>>(b,expanded,unsigned(weights.size()));
        checked(cudaEventRecord(end));checked(cudaEventSynchronize(end));float preparation=0;
        checked(cudaEventElapsedTime(&preparation,start,end));
        operator_probe<false><<<count/128,128>>>(a,b,expanded,out,count);
        checked(cudaMemcpy(reference.data(),out,count*4,cudaMemcpyDeviceToHost));
        operator_probe<true><<<count/128,128>>>(a,b,expanded,out,count);
        checked(cudaMemcpy(actual.data(),out,count*4,cudaMemcpyDeviceToHost));
        if(reference!=actual)throw std::runtime_error("Predecoded weight operator differs");
        std::array<std::vector<float>,2> samples;
        for(unsigned round=0;round<30;++round)
            for(unsigned order=0;order<2;++order) {
                const unsigned candidate=round%2?1-order:order;
                checked(cudaEventRecord(start));
                if(candidate)operator_probe<true><<<count/128,128>>>(a,b,expanded,out,count);
                else operator_probe<false><<<count/128,128>>>(a,b,expanded,out,count);
                checked(cudaEventRecord(end));checked(cudaEventSynchronize(end));float elapsed=0;
                checked(cudaEventElapsedTime(&elapsed,start,end));
                if(round>=5)samples[candidate].push_back(elapsed);
            }
        for(auto& values:samples)std::sort(values.begin(),values.end());
        std::cout<<"{\"passed\":true,\"operator_only\":true,\"gpu_used\":true,\"exact\":true,"
                 <<"\"baseline_ms\":"<<samples[0][samples[0].size()/2]<<",\"cached_ms\":"<<samples[1][samples[1].size()/2]
                 <<",\"predecode_ms\":"<<preparation<<",\"packed_weight_bytes\":"<<weights.size()*4
                 <<",\"expanded_weight_bytes\":"<<weights.size()*sizeof(DecodedWord)
                 <<",\"whole_model_validated\":false,\"older_device_validated\":false}\n";
        checked(cudaEventDestroy(start));start=nullptr;checked(cudaEventDestroy(end));end=nullptr;
        checked(cudaFree(a));a=nullptr;checked(cudaFree(b));b=nullptr;checked(cudaFree(out));out=nullptr;
        checked(cudaFree(expanded));expanded=nullptr;
    }catch(const std::exception& error) {
        if(start)cudaEventDestroy(start);if(end)cudaEventDestroy(end);
        if(a)cudaFree(a);if(b)cudaFree(b);if(out)cudaFree(out);if(expanded)cudaFree(expanded);
        std::cerr<<error.what()<<'\n';return 1;
    }
}
