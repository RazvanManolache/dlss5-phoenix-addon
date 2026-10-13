// Independent cuBLASLt experiment: compare the same FP8 matrices and explicit
// half/FP8 publication as matrix_benchmark.cpp. This is not a runtime dependency.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include "../../native_runtime/backend.h"
#include "../../native_runtime/host_math.h"
#include "../../src/native/model/weight_source.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

static void cuda_check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
static void blas_check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLASLt status " + std::to_string(int(status)));
}
__global__ void encode_activations(const __half* input, uint8_t* output, size_t count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = __nv_cvt_halfraw_to_fp8(static_cast<__half_raw>(input[index]),
                                                __NV_SATFINITE, __NV_E4M3);
}
__global__ void publish_matrix(const float* input, uint16_t* output, size_t count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const __half rounded = __float2half_rn(input[index]);
        const auto code =
            __nv_cvt_halfraw_to_fp8(static_cast<__half_raw>(rounded), __NV_SATFINITE, __NV_E4M3);
        output[index] = __half_as_ushort(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3));
    }
}
int main(int argc, char** argv) {
    try {
        if (argc != 2 || std::filesystem::exists(argv[1]))
            throw std::runtime_error("Usage: vendor_matrix_benchmark NEW_REPORT.json");
        cublasLtHandle_t handle{};
        blas_check(cublasLtCreate(&handle));
        constexpr size_t workspace_bytes = 32 * 1024 * 1024;
        void* workspace{};
        cuda_check(cudaMalloc(&workspace, workspace_bytes));
        float* unit_scale{};
        const float unit = 1;
        cuda_check(cudaMalloc(&unit_scale, sizeof(float)));
        cuda_check(cudaMemcpy(unit_scale, &unit, sizeof(float), cudaMemcpyHostToDevice));
        nlohmann::json report = {
            {"passed", false},
            {"rows", nlohmann::json::array()},
            {"scope",
             "Synthetic resident cuBLASLt FP8 matrix, activation encoding and explicit half/FP8 epilogue included; not frame time"}};
        int device = 0, driver_version = 0, runtime_version = 0;
        cudaDeviceProp properties{};
        cuda_check(cudaGetDevice(&device));
        cuda_check(cudaGetDeviceProperties(&properties, device));
        cuda_check(cudaDriverGetVersion(&driver_version));
        cuda_check(cudaRuntimeGetVersion(&runtime_version));
        report["hardware"] = {{"cuda_device", device},
                              {"gpu", properties.name},
                              {"compute_major", properties.major},
                              {"compute_minor", properties.minor},
                              {"driver_api_version", driver_version},
                              {"runtime_api_version", runtime_version}};
        report["workspace_bytes"] = workspace_bytes;
        phx::Kernel<HostMath> converter{};
        struct Shape {
            unsigned rows, columns, reduction;
        };
        for (const auto shape :
             {Shape{8192, 128, 32}, Shape{8192, 32, 128}, Shape{32768, 512, 128},
              Shape{130560, 512, 128}, Shape{32640, 1024, 256}, Shape{32640, 256, 1024},
              Shape{2048, 1024, 256}, Shape{640, 1536, 512}, Shape{640, 4096, 1024},
              Shape{640, 1024, 4096}}) {
            const size_t input_count = size_t(shape.rows) * shape.reduction;
            const size_t weight_count = size_t(shape.columns) * shape.reduction;
            const size_t output_count = size_t(shape.rows) * shape.columns;
            std::vector<uint16_t> activations(input_count), output(output_count);
            std::vector<uint8_t> weights(weight_count);
            for (size_t index = 0; index < input_count; ++index)
                activations[index] =
                    uint16_t(converter.half_bits(float(int(index * 13 % 23) - 11) * .0625f));
            for (size_t index = 0; index < weight_count; ++index)
                weights[index] = uint8_t((32 + index * 17 % 57) | ((index & 1) ? 128 : 0));
            __half* half_input{};
            uint8_t *packed_input{}, *packed_weights{};
            float* accumulator{};
            uint16_t* published{};
            cuda_check(cudaMalloc(&half_input, input_count * 2));
            cuda_check(cudaMalloc(&packed_input, input_count));
            cuda_check(cudaMalloc(&packed_weights, weight_count));
            cuda_check(cudaMalloc(&accumulator, output_count * 4));
            cuda_check(cudaMalloc(&published, output_count * 2));
            cuda_check(cudaMemcpy(half_input, activations.data(), input_count * 2,
                                  cudaMemcpyHostToDevice));
            cuda_check(
                cudaMemcpy(packed_weights, weights.data(), weight_count, cudaMemcpyHostToDevice));
            cublasLtMatmulDesc_t descriptor{};
            blas_check(cublasLtMatmulDescCreate(&descriptor, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            const cublasOperation_t transpose = CUBLAS_OP_T;
            blas_check(cublasLtMatmulDescSetAttribute(descriptor, CUBLASLT_MATMUL_DESC_TRANSA,
                                                      &transpose, sizeof transpose));
            blas_check(cublasLtMatmulDescSetAttribute(
                descriptor, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &unit_scale, sizeof unit_scale));
            blas_check(cublasLtMatmulDescSetAttribute(
                descriptor, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &unit_scale, sizeof unit_scale));
            // C(row,column) in row-major is C^T(column,row) in column-major.
            // Swap operands and transpose the column-major view of packed weights.
            cublasLtMatrixLayout_t weight_layout{}, input_layout{}, output_layout{};
            blas_check(cublasLtMatrixLayoutCreate(&weight_layout, CUDA_R_8F_E4M3, shape.reduction,
                                                  shape.columns, shape.reduction));
            blas_check(cublasLtMatrixLayoutCreate(&input_layout, CUDA_R_8F_E4M3, shape.reduction,
                                                  shape.rows, shape.reduction));
            blas_check(cublasLtMatrixLayoutCreate(&output_layout, CUDA_R_32F, shape.columns,
                                                  shape.rows, shape.columns));
            cublasLtMatmulPreference_t preference{};
            blas_check(cublasLtMatmulPreferenceCreate(&preference));
            blas_check(cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
                sizeof workspace_bytes));
            cublasLtMatmulHeuristicResult_t algorithm{};
            int found = 0;
            blas_check(cublasLtMatmulAlgoGetHeuristic(handle, descriptor, weight_layout,
                                                      input_layout, output_layout, output_layout,
                                                      preference, 1, &algorithm, &found));
            if (!found)
                throw std::runtime_error("No algorithm for shape");
            const float alpha = 1, beta = 0;
            std::vector<double> samples;
            cudaEvent_t start{}, end{};
            cuda_check(cudaEventCreate(&start));
            cuda_check(cudaEventCreate(&end));
            for (unsigned iteration = 0; iteration < 7; ++iteration) {
                cuda_check(cudaEventRecord(start));
                encode_activations<<<unsigned((input_count + 255) / 256), 256>>>(
                    half_input, packed_input, input_count);
                blas_check(cublasLtMatmul(handle, descriptor, &alpha, packed_weights, weight_layout,
                                          packed_input, input_layout, &beta, accumulator,
                                          output_layout, accumulator, output_layout,
                                          &algorithm.algo, workspace, workspace_bytes, 0));
                publish_matrix<<<unsigned((output_count + 255) / 256), 256>>>(
                    accumulator, published, output_count);
                cuda_check(cudaEventRecord(end));
                cuda_check(cudaEventSynchronize(end));
                float milliseconds = 0;
                cuda_check(cudaEventElapsedTime(&milliseconds, start, end));
                if (iteration >= 2)
                    samples.push_back(milliseconds);
            }
            cuda_check(
                cudaMemcpy(output.data(), published, output_count * 2, cudaMemcpyDeviceToHost));
            auto sorted = samples;
            std::sort(sorted.begin(), sorted.end());
            const auto digest = phoenix_model::weight_digest(
                reinterpret_cast<const uint8_t*>(output.data()), output_count * 2);
            report["rows"].push_back(
                {{"m", shape.rows},
                 {"n", shape.columns},
                 {"k", shape.reduction},
                 {"median_ms", sorted[2]},
                 {"samples_ms", samples},
                 {"output_sha256", digest},
                 {"owned_bytes", input_count * 3 + weight_count + output_count * 6 +
                                     workspace_bytes + sizeof(float)}});
            std::ofstream(argv[1]) << report.dump(2) << '\n';
            std::cout << shape.rows << 'x' << shape.columns << 'x' << shape.reduction << ": "
                      << sorted[2] << " ms\n"
                      << std::flush;
            cuda_check(cudaEventDestroy(start));
            cuda_check(cudaEventDestroy(end));
            blas_check(cublasLtMatmulPreferenceDestroy(preference));
            blas_check(cublasLtMatrixLayoutDestroy(weight_layout));
            blas_check(cublasLtMatrixLayoutDestroy(input_layout));
            blas_check(cublasLtMatrixLayoutDestroy(output_layout));
            blas_check(cublasLtMatmulDescDestroy(descriptor));
            cuda_check(cudaFree(half_input));
            cuda_check(cudaFree(packed_input));
            cuda_check(cudaFree(packed_weights));
            cuda_check(cudaFree(accumulator));
            cuda_check(cudaFree(published));
        }
        report["passed"] = true;
        std::ofstream(argv[1]) << report.dump(2) << '\n';
        cuda_check(cudaFree(unit_scale));
        cuda_check(cudaFree(workspace));
        blas_check(cublasLtDestroy(handle));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
