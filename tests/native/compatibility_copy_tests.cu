// Isolated GPU tests for leader-only compatibility copies and FP8 decoding.
// The host decoder remains an independent scalar reference for all byte codes.
#define PHOENIX_ARCHITECTURE_TEST
#include "../../src/native/kernels/production/common/pipeline_memory.cuh"
#include <cuda_runtime.h>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static void checked(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
template<bool Debug> __global__ void copy_case(
    const uint32_t* input, uint32_t* output, unsigned* errors,
    unsigned source_offset, unsigned destination_offset, unsigned bytes, unsigned leader) {
    __shared__ __align__(16) uint32_t tile[512];
    for (unsigned i = threadIdx.x; i < 512; i += blockDim.x) tile[i] = 0xdeadbeefu;
    phoenix_architecture::finish_compatibility_copies();
    if (threadIdx.x == leader) {
        MemoryRange ranges[16]{};
        ranges[0] = {uint64_t(input), 2048, 0};
        bulk_copy<Debug>(tile, destination_offset, uint64_t(input) + source_offset,
                         bytes, 2048, ranges, errors);
    }
    phoenix_architecture::finish_compatibility_copies();
    for (unsigned i = threadIdx.x; i < 512; i += blockDim.x) output[i] = tile[i];
}
__global__ void decode_case(uint16_t* output) {
    output[threadIdx.x] = phoenix_architecture::decode_fp8_value(uint8_t(threadIdx.x));
}
__global__ void decode_pair_case(uint32_t* output) {
    const unsigned pair=blockIdx.x*blockDim.x+threadIdx.x;
    output[pair]=phoenix_architecture::decode_fp8_pair(uint16_t(pair));
}
__global__ void decode_weight_case(uint32_t* output) {
    const unsigned pair=blockIdx.x*blockDim.x+threadIdx.x;
    const uint32_t word=(pair&255u)|((pair&0xff00u)<<8)|
        (((~pair)&255u)<<8)|(((~pair)&0xff00u)<<16);
    phoenix_architecture::decode_fp8_weight_fragment(word,output[pair*2],output[pair*2+1]);
}
int main() {
    uint32_t *input = nullptr, *output = nullptr;
    unsigned* errors = nullptr;
    uint16_t* decoded = nullptr;
    uint32_t* decoded_pairs = nullptr;
    uint32_t* decoded_weights = nullptr;
    try {
        checked(cudaMalloc(&input, 2048)); checked(cudaMalloc(&output, 2048));
        checked(cudaMalloc(&errors, 4*sizeof(unsigned))); checked(cudaMemset(errors, 0, 4*sizeof(unsigned)));
        checked(cudaMalloc(&decoded, 256*sizeof(uint16_t)));
        checked(cudaMalloc(&decoded_pairs, 65536*sizeof(uint32_t)));
        checked(cudaMalloc(&decoded_weights, 131072*sizeof(uint32_t)));
        std::array<uint32_t,512> source{}, actual{};
        for (unsigned i=0;i<source.size();++i) source[i]=i*0x9e3779b9u;
        checked(cudaMemcpy(input,source.data(),2048,cudaMemcpyHostToDevice));
        unsigned cases=0;
        for (bool debug : {false,true})
            for (unsigned leader : {0u,17u})
                for (unsigned src : {0u,4u,16u})
                    for (unsigned dst : {0u,4u,16u})
                        for (unsigned bytes : {0u,4u,12u,16u,32u,128u,512u,1024u}) {
                            if(debug)copy_case<true><<<1,64>>>(input,output,errors,src,dst,bytes,leader);
                            else copy_case<false><<<1,64>>>(input,output,errors,src,dst,bytes,leader);
                            checked(cudaGetLastError());
                            checked(cudaMemcpy(actual.data(),output,2048,cudaMemcpyDeviceToHost));
                            for(unsigned i=0;i<512;++i) {
                                const auto expected=i>=dst/4&&i<(dst+bytes)/4?source[src/4+i-dst/4]:0xdeadbeefu;
                                require(actual[i]==expected,"Compatibility copy or untouched sentinel differs");
                            }
                            ++cases;
                        }
        std::array<unsigned,4> counts{};
        checked(cudaMemcpy(counts.data(),errors,sizeof(counts),cudaMemcpyDeviceToHost));
        for(auto count:counts)require(count==0,"Unexpected copy bounds error");
        decode_case<<<1,256>>>(decoded); checked(cudaGetLastError());
        std::array<uint16_t,256> values{};
        checked(cudaMemcpy(values.data(),decoded,sizeof(values),cudaMemcpyDeviceToHost));
        for(unsigned i=0;i<256;++i)
            require(values[i]==phoenix_architecture::decode_fp8_value(uint8_t(i)),"GPU FP8 decoder differs from scalar reference");
        decode_pair_case<<<256,256>>>(decoded_pairs);checked(cudaGetLastError());
        std::vector<uint32_t> pairs(65536);
        checked(cudaMemcpy(pairs.data(),decoded_pairs,pairs.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        for(unsigned i=0;i<pairs.size();++i) {
            const uint32_t expected=uint32_t(phoenix_architecture::decode_fp8_value(uint8_t(i))) |
                (uint32_t(phoenix_architecture::decode_fp8_value(uint8_t(i>>8)))<<16);
            require(pairs[i]==expected,"Packed GPU FP8 decoder differs from independent scalar lanes");
        }
        checked(cudaFree(decoded_pairs));decoded_pairs=nullptr;
        decode_weight_case<<<256,256>>>(decoded_weights);checked(cudaGetLastError());
        std::vector<uint32_t> weights(131072);
        checked(cudaMemcpy(weights.data(),decoded_weights,weights.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        for(unsigned i=0;i<65536;++i) {
            const auto reference=[](unsigned pair) {
                return uint32_t(phoenix_architecture::decode_fp8_value(uint8_t(pair))) |
                    (uint32_t(phoenix_architecture::decode_fp8_value(uint8_t(pair>>8)))<<16);
            };
            require(weights[i*2]==reference(i) && weights[i*2+1]==reference(i^65535u),
                    "Cached GPU weight fragment differs from scalar reference");
        }
        checked(cudaFree(decoded_weights));decoded_weights=nullptr;
        checked(cudaFree(decoded));decoded=nullptr;checked(cudaFree(errors));errors=nullptr;
        checked(cudaFree(output));output=nullptr;checked(cudaFree(input));input=nullptr;
        std::cout<<"{\"passed\":true,\"gpu_used\":true,\"copy_cases\":"<<cases
                 <<",\"fp8_codes\":256,\"fp8_pairs\":65536,\"weight_pairs\":131072,\"older_device_validated\":false}\n";
    } catch(const std::exception& error) {
        if(decoded_pairs)cudaFree(decoded_pairs);
        if(decoded_weights)cudaFree(decoded_weights);
        if(decoded)cudaFree(decoded);if(errors)cudaFree(errors);if(output)cudaFree(output);if(input)cudaFree(input);
        std::cerr<<error.what()<<'\n';return 1;
    }
}
