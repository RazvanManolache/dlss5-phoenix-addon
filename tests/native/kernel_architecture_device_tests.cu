// Compare the compatibility fragment mapping with native FP8 MMA on a GPU that
// supports both instruction forms. This does not stand in for an older GPU test.
#define PHOENIX_ARCHITECTURE_TEST
#include "../../src/native/kernels/production/common/architecture_math.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
__device__ uint32_t fragment(unsigned lane, unsigned index, unsigned seed) {
    constexpr unsigned codes[] = {0, 0x20, 0x28, 0x30, 0xa0, 0xa8, 0xb0};
    uint32_t result = 0;
    for (unsigned byte = 0; byte < 4; ++byte) {
        uint32_t mixed = (lane + 1) * 0x9e3779b9u ^ (index + 3) * 0x85ebca6bu ^
                         (seed + 5) * 0xc2b2ae35u ^ (byte + 7) * 0x27d4eb2fu;
        mixed ^= mixed >> 16; mixed *= 0x85ebca6bu; mixed ^= mixed >> 13;
        result |= codes[mixed % 7] << (byte * 8);
    }
    return result;
}
__global__ void compare(unsigned* errors) {
    const unsigned lane = threadIdx.x;
    for (unsigned seed = 0; seed < 49; ++seed) {
        const uint32_t a0 = fragment(lane, 0, seed), a1 = fragment(lane, 1, seed);
        const uint32_t a2 = fragment(lane, 2, seed), a3 = fragment(lane, 3, seed);
        const uint32_t b0 = fragment(lane, 7, seed), b1 = fragment(lane, 8, seed);
        uint32_t expected0, expected1, actual0, actual1;
        // Inputs are multiples of 1/8 with magnitude <= 1/2. Products and all
        // partial sums are exactly representable in FP16, isolating lane order
        // from the expected rounding differences between MMA reduction shapes.
        asm("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};"
            : "=r"(expected0), "=r"(expected1)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0u), "r"(0u));
        phoenix_architecture::fp8_matrix(actual0, actual1, a0, a1, a2, a3,
#ifdef PHOENIX_TEST_CORRUPT_OPERAND
            b0 ^ 0x00008000u,
#else
            b0,
#endif
            b1, 0, 0);
        if (expected0 != actual0 || expected1 != actual1) atomicAdd(errors, 1u);
    }
}
int main() {
    unsigned* errors = nullptr;
    try {
        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, 0));
        if (device.major < 9 && !(device.major == 8 && device.minor >= 9))
            throw std::runtime_error("Native FP8 comparison requires an Ada or newer GPU");
        check(cudaMalloc(&errors, sizeof(unsigned)));
        check(cudaMemset(errors, 0, sizeof(unsigned)));
        compare<<<1, 32>>>(errors);
        check(cudaGetLastError());
        unsigned mismatches = 0;
        check(cudaMemcpy(&mismatches, errors, sizeof(unsigned), cudaMemcpyDeviceToHost));
        check(cudaFree(errors)); errors = nullptr;
        if (mismatches) throw std::runtime_error("Compatibility MMA fragment order differs from native FP8");
        std::cout << "{\"passed\":true,\"gpu_used\":true,\"warp_cases\":49,\"lane_results\":1568,\"mismatches\":0,\"turing_steps\":"
#ifdef PHOENIX_TEST_TURING_MMA
                  << "true"
#else
                  << "false"
#endif
                  << ",\"device_sm\":" << device.major * 10 + device.minor
                  << ",\"older_device_validated\":false}\n";
    } catch (const std::exception& error) {
        if (errors) cudaFree(errors);
        std::cerr << error.what() << '\n'; return 1;
    }
}
