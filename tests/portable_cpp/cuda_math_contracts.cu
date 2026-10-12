// Compare native conversions with the unchanged portable integer reference.
// All binary16 codes, neighbors of half midpoints and random FP32 bit patterns.
#include <cstdint>
#include <vector>
#include <iostream>
#include <limits>
#include <cmath>
#define PHX_DEVICE
#include "../../src/backends/hip_sycl/compute_ops.h"
#include "../../native_runtime/host_math.h"
#include "../../native_runtime/cuda_narrow_math.cuh"
__global__ void convert(const float* input, uint32_t* output, unsigned count) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    output[index * 4] = __float_as_uint(CudaNarrowMath::round_half(input[index]));
    output[index * 4 + 1] = CudaNarrowMath::encode_half(input[index]);
    output[index * 4 + 2] = __float_as_uint(CudaNarrowMath::round_e4m3(input[index]));
    output[index * 4 + 3] = __float_as_uint(CudaNarrowMath::decode_half(index & 65535u));
}
int main() {
    phx::Kernel<HostMath> reference{};
    std::vector<float> values;
    for (unsigned code = 0; code < 65536; ++code) {
        float value = reference.half_value(code);
        values.push_back(value);
        if (code < 0x7bffu) {
            float midpoint = (value + reference.half_value(code + 1)) * .5f;
            values.push_back(midpoint);
            values.push_back(std::nextafter(midpoint, -std::numeric_limits<float>::infinity()));
            values.push_back(std::nextafter(midpoint, std::numeric_limits<float>::infinity()));
            values.push_back(-midpoint);
        }
    }
    uint32_t bits = 1059;
    for (unsigned index = 0; index < 1048576; ++index) {
        bits ^= bits << 13;
        bits ^= bits >> 17;
        bits ^= bits << 5;
        values.push_back(HostMath::value(bits));
    }
    float* device_input = nullptr;
    uint32_t* device_output = nullptr;
    auto check = [](cudaError_t result) {
        if (result != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(result));
    };
    try {
        check(cudaMalloc(&device_input, values.size() * 4));
        check(cudaMalloc(&device_output, values.size() * 16));
        check(cudaMemcpy(device_input, values.data(), values.size() * 4, cudaMemcpyHostToDevice));
        convert<<<unsigned((values.size() + 255) / 256), 256>>>(device_input, device_output,
                                                                unsigned(values.size()));
        check(cudaGetLastError());
        std::vector<uint32_t> actual(values.size() * 4);
        check(cudaMemcpy(actual.data(), device_output, actual.size() * 4, cudaMemcpyDeviceToHost));
        size_t differences = 0;
        for (size_t index = 0; index < values.size(); ++index) {
            const uint32_t expected[] = {
                HostMath::bits(reference.half(values[index])), reference.half_bits(values[index]),
                HostMath::bits(reference.quantize_e4m3(values[index])),
                HostMath::bits(reference.half_value(uint32_t(index) & 65535u))};
            for (unsigned operation = 0; operation < 4; ++operation)
                if (actual[index * 4 + operation] != expected[operation]) {
                    if (differences++ < 8)
                        std::cerr << std::hex << "input=" << HostMath::bits(values[index])
                                  << " op=" << operation << " expected=" << expected[operation]
                                  << " actual=" << actual[index * 4 + operation] << '\n';
                }
        }
        cudaFree(device_input);
        cudaFree(device_output);
        std::cout << values.size() * 4 << " conversion checks, " << differences << " differences\n";
        return differences ? 1 : 0;
    } catch (const std::exception& error) {
        cudaFree(device_input);
        cudaFree(device_output);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
