// Explicit model-free GPU checks of production arithmetic. No game or inference.
#include <cuda_runtime.h>
#define DLSS5_SAMPLING_HELPERS_ONLY
#include "../../src/native/kernels/production/common/texture_sampling.cuh"
#include "../../src/native/kernels/production/common/device_memory_scalar.cuh"
#include "../../src/native/kernels/production/common/attention_tail.cuh"
#define PHX_DEVICE __device__
#include "../../src/backends/hip_sycl/compute_ops.h"
#include "../portable_cpp/fp8_reference.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct DeviceMath {
    __device__ static uint32_t bits(float value) { return __float_as_uint(value); }
    __device__ static float value(uint32_t bits) { return __uint_as_float(bits); }
    __device__ static float fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
    __device__ static float sqrt(float a) { return sqrtf(a); }
    __device__ static float abs(float a) { return fabsf(a); }
    __device__ static float min(float a, float b) { return fminf(a, b); }
    __device__ static float max(float a, float b) { return fmaxf(a, b); }
    __device__ static bool inf(float a) { return isinf(a); }
};
__device__ uint32_t previous_attention_tail(uint32_t sum, uint32_t count) {
    uint32_t coefficient = 0x3ed63ed6u;
    coefficient = maximum_half_pairs(coefficient, 0x3dc23dc2u);
    coefficient = minimum_half_pairs(coefficient, 0x3fe93fe9u);
    coefficient = (coefficient << 4) + 0x4000u;
    coefficient = half_to_float_bits((coefficient & 65535u) * 65537u);
    float correction;
    asm("mul.rn.ftz.f32 %0,%1,%2;" : "=f"(correction)
        : "f"(__uint_as_float(coefficient)), "f"(__uint2float_rn(count)));
    return add_half_pairs(sum, select_half_operand<0, false, true>(pack_half(0u, __float_as_uint(correction))));
}
struct Result { uint32_t fp8, tail_old, tail_new; };
__global__ void arithmetic(const uint32_t* inputs, Result* outputs, unsigned count) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const uint32_t bits = inputs[index];
        constexpr uint32_t tail_edges[] = {0, 1, 2, 31, 32, 65535, 0xffffffu,
            0x1000000u, 0x1000001u, 0x7fffffffu, 0x80000000u, 0xffffffffu};
        const uint32_t tail_count = index < 12 ? tail_edges[index] : index * 1664525u + 1013904223u;
        const phx::Kernel<DeviceMath> math{};
        outputs[index] = {__float_as_uint(math.quantize_e4m3(__uint_as_float(bits))),
            previous_attention_tail(bits, tail_count), subtract_attention_tail(bits, tail_count)};
    }
}
__global__ void residuals(uint32_t* output) {
    const unsigned index = threadIdx.x;
    constexpr unsigned widths[] = {32, 64, 128, 256, 512, 1024};
    if (index >= 6) return;
    float projected = -(1.f + 0x1p-9f), skip = 1.f + 0x1p-10f, gain = skip, value;
    uint16_t published;
    phx::Operation operation{};
    operation.flags = phx::UpsampleResidual;
    operation.column_count = widths[index];
    operation.left_batch_stride = operation.right_batch_stride = 1;
    operation.left_input = &projected; operation.right_input = &skip;
    operation.auxiliary = &gain; operation.output = &value; operation.secondary_output = &published;
    phx::Kernel<DeviceMath>{operation}.unary(0);
    output[index] = __float_as_uint(value);
    output[index + 6] = published;
}
static void checked(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template <class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { checked(cudaMalloc(&data, count * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
};
static uint32_t bits(float input) { uint32_t result; memcpy(&result, &input, 4); return result; }
static float value(uint32_t input) { float result; memcpy(&result, &input, 4); return result; }
int main() {
    try {
        std::vector<uint32_t> inputs;
        for (unsigned half = 0; half < 65536; ++half) {
            const unsigned exponent = (half >> 10) & 31, fraction = half & 1023;
            float decoded = exponent == 31
                ? (fraction ? NAN : INFINITY)
                : exponent ? std::ldexp(float(1024 + fraction), int(exponent) - 25)
                           : std::ldexp(float(fraction), -24);
            inputs.push_back(bits(half & 32768 ? -decoded : decoded));
        }
        uint32_t random = 0x94f2e175u;
        for (unsigned i = 0; i < 100'000; ++i) {
            random = random * 1664525u + 1013904223u;
            inputs.push_back(random);
        }
        DeviceBuffer<uint32_t> input(inputs.size()), residual_output(12);
        DeviceBuffer<Result> output(inputs.size());
        checked(cudaMemcpy(input.data, inputs.data(), inputs.size() * 4, cudaMemcpyHostToDevice));
        arithmetic<<<(unsigned(inputs.size()) + 255) / 256, 256>>>(input.data, output.data, unsigned(inputs.size()));
        checked(cudaGetLastError());
        std::vector<Result> results(inputs.size());
        checked(cudaMemcpy(results.data(), output.data, results.size() * sizeof(Result), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < results.size(); ++i) {
            if (results[i].fp8 != bits(reference_fp8(value(inputs[i]))))
                throw std::runtime_error("GPU FP8 oracle mismatch at " + std::to_string(i));
            if (results[i].tail_new != results[i].tail_old)
                throw std::runtime_error("Attention tail differential mismatch at " + std::to_string(i));
        }
        residuals<<<1, 32>>>(residual_output.data);
        checked(cudaGetLastError());
        uint32_t observed[12];
        checked(cudaMemcpy(observed, residual_output.data, sizeof(observed), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < 6; ++i)
            if (observed[i] != bits(i == 4 ? 0.f : 0x1p-20f) || observed[i + 6])
                throw std::runtime_error("GPU residual rounding mismatch");
        std::cout << "{\"passed\":true,\"gpu_fp8_oracle_cases\":" << results.size()
                  << ",\"attention_tail_differential_cases\":" << results.size()
                  << ",\"residual_dispatch_cases\":6,\"gpu_executed\":true}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
