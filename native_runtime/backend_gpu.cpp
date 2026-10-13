// CUDA/HIP/SYCL execution of the portable graph, including graph capture and grouped timing.
// This is separate from the native RTX instruction kernels under src/native/kernels/production/.

#include <cstring>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <algorithm>
#if defined(PHX_SYCL)
#include <sycl/sycl.hpp>
#define PHX_DEVICE
#elif defined(PHX_CUDA)
#include <cuda_runtime.h>
#define PHX_DEVICE __device__
#else
#include <hip/hip_runtime.h>
#define PHX_DEVICE __device__
#endif
#include "backend.h"
#include "command_cache.h"
#include "backend_options.h"
#if defined(PHX_SYCL)
#include "fp8_rounding.h"
#endif
using namespace phoenix_cpp;
#define PHX_CHECK(expression) check((expression), __LINE__)
#if defined(PHX_CUDA)
#include "cuda_narrow_math.cuh"
struct GpuMath : CudaNarrowMath {
#elif !defined(PHX_SYCL)
#include "hip_narrow_math.cuh"
struct GpuMath : HipNarrowMath {
#else
struct GpuMath {
#endif
    PHX_DEVICE static uint32_t bits(float value_or_bits) {
#if defined(PHX_SYCL)
        return sycl::bit_cast<uint32_t>(value_or_bits);
#else
        return __float_as_uint(value_or_bits);
#endif
    }
    PHX_DEVICE static float value(uint32_t value_or_bits) {
#if defined(PHX_SYCL)
        return sycl::bit_cast<float>(value_or_bits);
#else
        return __uint_as_float(value_or_bits);
#endif
    }
#if defined(PHX_SYCL)
    static float round_half(float input) {
        return sycl::isfinite(input) ? float(sycl::half(input)) : input;
    }
    static uint32_t encode_half(float input) {
        const uint32_t raw = bits(input);
        if ((raw & 0x7fffffffu) > 0x7f800000u)
            return ((raw >> 16u) & 0x8000u) | 0x7e00u;
        return sycl::bit_cast<uint16_t>(sycl::half(input));
    }
    static float decode_half(uint32_t raw) {
        if ((raw & 0x7c00u) == 0x7c00u)
            return value(((raw & 0x8000u) << 16u) | 0x7f800000u | ((raw & 1023u) << 13u));
        return float(sycl::bit_cast<sycl::half>(uint16_t(raw)));
    }
    static float fma(float left, float right, float addend) {
        return sycl::fma(left, right, addend);
    }
    static float sqrt(float value_or_bits) {
        return sycl::sqrt(value_or_bits);
    }
    static float abs(float value_or_bits) {
        return sycl::fabs(value_or_bits);
    }
    static float min(float left, float right) {
        return sycl::fmin(left, right);
    }
    static float max(float left, float right) {
        return sycl::fmax(left, right);
    }
    static float even(float value_or_bits) {
        return sycl::rint(value_or_bits);
    }
    static bool inf(float value_or_bits) {
        return sycl::isinf(value_or_bits);
    }
#else
    __device__ static float fma(float left, float right, float addend) {
        return fmaf(left, right, addend);
    }
    __device__ static float sqrt(float value_or_bits) {
        return sqrtf(value_or_bits);
    }
    __device__ static float abs(float value_or_bits) {
        return fabsf(value_or_bits);
    }
    __device__ static float min(float left, float right) {
        return fminf(left, right);
    }
    __device__ static float max(float left, float right) {
        return fmaxf(left, right);
    }
    __device__ static float even(float value_or_bits) {
        return rintf(value_or_bits);
    }
    __device__ static bool inf(float value_or_bits) {
        return isinf(value_or_bits);
    }
#endif
};
#if defined(PHX_SYCL)
// Select the measured CPU quantizer without changing untested GPU defaults.
struct SyclCpuMath : GpuMath {
    static float round_e4m3(float input) {
        return phoenix_cpp::round_e4m3_bits<SyclCpuMath>(input);
    }
};
#endif
#if !defined(PHX_SYCL)
__global__ void run_kernel(phx::Operation operation, uint64_t count, uint64_t offset) {
    uint64_t element_index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element_index < count)
        phx::Kernel<GpuMath>{operation}(offset + element_index);
}
// A 32x32 output tile is shared by 256 threads, four rows per thread.
// Cooperative loads decode each weight once per tile. The two barriers protect
// shared operands before use and before the next reduction tile overwrites them.
__global__ void run_gemm(phx::Operation operation, uint64_t start, uint64_t tiles) {
    uint64_t tile = start + blockIdx.x;
    if (blockIdx.x >= tiles)
        return;
    constexpr unsigned edge = 32, rows_per_thread = 4;
    unsigned column_tile_count = (operation.column_count + edge - 1) / edge,
             row_tile_count = (operation.row_count + edge - 1) / edge;
    unsigned thread_column = threadIdx.x % edge, thread_row = threadIdx.x / edge;
    unsigned first_row = unsigned(tile / column_tile_count % row_tile_count) * edge + thread_row,
             column = unsigned(tile % column_tile_count) * edge + thread_column;
    uint64_t batch_index = tile / (uint64_t(column_tile_count) * row_tile_count);
    phx::Kernel<GpuMath> math{operation};
    // A padded right tile avoids bank conflicts when transposing coalesced
    // K-contiguous weight loads into the row-major compute layout.
    __shared__ float left_tile[edge][edge], right_tile[edge][edge + 1];
    float accumulators[rows_per_thread]{};
    for (unsigned reduction_tile_start = 0; reduction_tile_start < operation.reduction_count;
         reduction_tile_start += edge) {
        for (unsigned lane = threadIdx.x; lane < edge * edge; lane += 256) {
            unsigned tile_y = lane / edge, tile_x = lane % edge,
                     row = unsigned(tile / column_tile_count % row_tile_count) * edge + tile_y;
            left_tile[tile_y][tile_x] =
                row < operation.row_count &&
                        reduction_tile_start + tile_x < operation.reduction_count
                    ? math.load_half(operation.left_input,
                                     batch_index * operation.left_batch_stride +
                                         uint64_t(row) * operation.left_row_stride +
                                         reduction_tile_start + tile_x)
                    : 0;
            const unsigned right_reduction = (operation.flags & 1) ? tile_x : tile_y,
                           right_column = (operation.flags & 1) ? tile_y : tile_x;
            const unsigned global_column = unsigned(tile % column_tile_count) * edge + right_column;
            right_tile[right_reduction][right_column] =
                global_column < operation.column_count &&
                        reduction_tile_start + right_reduction < operation.reduction_count
                    ? math.load_matrix_right(
                          batch_index * operation.right_batch_stride +
                          ((operation.flags & 1)
                               ? uint64_t(global_column) * operation.right_row_stride +
                                     reduction_tile_start + right_reduction
                               : uint64_t(reduction_tile_start + right_reduction) *
                                         operation.right_row_stride +
                                     global_column))
                    : 0;
        }
        __syncthreads();
        for (unsigned reduction_in_tile = 0;
             reduction_in_tile < edge &&
             reduction_tile_start + reduction_in_tile < operation.reduction_count;
             ++reduction_in_tile) {
            float value = right_tile[reduction_in_tile][thread_column];
#pragma unroll
            for (unsigned row_in_thread = 0; row_in_thread < rows_per_thread; ++row_in_thread)
                accumulators[row_in_thread] =
                    fmaf(left_tile[thread_row + row_in_thread * 8][reduction_in_tile], value,
                         accumulators[row_in_thread]);
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned row_in_thread = 0; row_in_thread < rows_per_thread; ++row_in_thread) {
        unsigned row = first_row + row_in_thread * 8;
        if (row < operation.row_count && column < operation.column_count)
            math.store(batch_index * operation.output_batch_stride +
                           uint64_t(row) * operation.output_row_stride + column,
                       accumulators[row_in_thread]);
    }
}
// The output head has at most four columns. One thread handles all four
// channels of a pixel, reusing each input feature and avoiding a general GEMM tile.
__global__ void run_projection(phx::Operation operation, uint64_t start, uint64_t count) {
    uint64_t index = start + uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= start + count)
        return;
    uint64_t batch_index = index / operation.row_count;
    unsigned row = unsigned(index % operation.row_count);
    phx::Kernel<GpuMath> math{operation};
    float accumulators[4]{};
    for (unsigned reduction_index = 0; reduction_index < operation.reduction_count;
         ++reduction_index) {
        float value = math.load_half(
            operation.left_input, batch_index * operation.left_batch_stride +
                                      uint64_t(row) * operation.left_row_stride + reduction_index);
#pragma unroll
        for (unsigned output_channel = 0; output_channel < 4; ++output_channel)
            if (output_channel < operation.column_count)
                accumulators[output_channel] =
                    fmaf(value,
                         math.load_matrix_right(
                             batch_index * operation.right_batch_stride +
                             ((operation.flags & 1)
                                  ? uint64_t(output_channel) * operation.right_row_stride +
                                        reduction_index
                                  : uint64_t(reduction_index) * operation.right_row_stride +
                                        output_channel)),
                         accumulators[output_channel]);
    }
#pragma unroll
    for (unsigned output_channel = 0; output_channel < 4; ++output_channel)
        if (output_channel < operation.column_count)
            math.store(batch_index * operation.output_batch_stride +
                           uint64_t(row) * operation.output_row_stride + output_channel,
                       accumulators[output_channel]);
}

#endif
#if !defined(PHX_SYCL)
#include "gpu_rows.cuh"
#endif
#if defined(PHX_CUDA)
#include <mma.h>
#include <cuda_fp16.h>
#include "cuda_rows.cuh"
// Four warps compute a 64x16 output tile using 16x16x16 tensor-core fragments.
// Packed FP8 weights decode to half in shared memory; accumulators stay FP32.
// The fragment types and synchronization are part of the existing numerical path.
template <unsigned ColumnTiles>
__global__ void matrix_tensor_f16(phx::Operation operation, uint64_t first_tile) {
    using namespace nvcuda;
    uint64_t tile_index = first_tile + blockIdx.x,
             column_tile_count =
                 (operation.column_count + 16 * ColumnTiles - 1) / (16 * ColumnTiles),
             row_tile_count = (operation.row_count + 63) / 64,
             batch_index = tile_index / (column_tile_count * row_tile_count);
    unsigned first_row = unsigned(tile_index / column_tile_count % row_tile_count) * 64,
             first_column = unsigned(tile_index % column_tile_count) * 16 * ColumnTiles,
             warp_index = threadIdx.x / 32;
    phx::Kernel<GpuMath> math{operation};
    // WMMA accepts aligned half operands directly. Interior tiles need no
    // float decode/re-encode or shared-memory copy of the activation matrix.
    const auto* left_data = static_cast<const __half*>(operation.left_input) +
                            batch_index * operation.left_batch_stride +
                            uint64_t(first_row) * operation.left_row_stride;
    const bool direct_left = first_row + 64 <= operation.row_count &&
                             operation.left_row_stride % 8 == 0 &&
                             (reinterpret_cast<uintptr_t>(left_data) & 31u) == 0;
    __shared__ __align__(32) __half left_tiles[4][256], right_tile[ColumnTiles][256];
    __shared__ __align__(32) float output_tiles[4][ColumnTiles][256];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> left_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> right_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulators[ColumnTiles];
#pragma unroll
    for (unsigned column_tile = 0; column_tile < ColumnTiles; ++column_tile)
        wmma::fill_fragment(accumulators[column_tile], 0.f);
    for (unsigned first_reduction = 0; first_reduction < operation.reduction_count;
         first_reduction += 16) {
        if (!direct_left) {
            for (unsigned tile_element = threadIdx.x; tile_element < 1024; tile_element += 128) {
                unsigned row = first_row + tile_element / 16, reduction_in_tile = tile_element % 16;
                left_tiles[tile_element / 256][tile_element % 256] = __float2half_rn(
                    row < operation.row_count
                        ? math.load_half(operation.left_input,
                                         batch_index * operation.left_batch_stride +
                                             uint64_t(row) * operation.left_row_stride +
                                             first_reduction + reduction_in_tile)
                        : 0);
            }
        }
        for (unsigned tile_element = threadIdx.x; tile_element < ColumnTiles * 256;
             tile_element += 128) {
            unsigned column = first_column + tile_element / 16,
                     reduction_in_tile = tile_element % 16;
            right_tile[tile_element / 256][tile_element % 256] = __float2half_rn(
                column < operation.column_count
                    ? math.load_matrix_right(batch_index * operation.right_batch_stride +
                                             ((operation.flags & 1)
                                                  ? uint64_t(column) * operation.right_row_stride +
                                                        first_reduction + reduction_in_tile
                                                  : uint64_t(first_reduction + reduction_in_tile) *
                                                            operation.right_row_stride +
                                                        column))
                    : 0);
        }
        __syncthreads();
        if (direct_left)
            wmma::load_matrix_sync(left_fragment,
                                   left_data + warp_index * 16 * operation.left_row_stride +
                                       first_reduction,
                                   operation.left_row_stride);
        else
            wmma::load_matrix_sync(left_fragment, left_tiles[warp_index], 16);
#pragma unroll
        for (unsigned column_tile = 0; column_tile < ColumnTiles; ++column_tile) {
            wmma::load_matrix_sync(right_fragment, right_tile[column_tile], 16);
            wmma::mma_sync(accumulators[column_tile], left_fragment, right_fragment,
                           accumulators[column_tile]);
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned column_tile = 0; column_tile < ColumnTiles; ++column_tile)
        wmma::store_matrix_sync(output_tiles[warp_index][column_tile], accumulators[column_tile],
                                16, wmma::mem_row_major);
    __syncthreads();
    for (unsigned tile_element = threadIdx.x; tile_element < ColumnTiles * 1024;
         tile_element += 128) {
        unsigned tile_warp = tile_element / (ColumnTiles * 256);
        unsigned column_tile = (tile_element / 256) % ColumnTiles;
        unsigned element = tile_element % 256;
        unsigned row = first_row + tile_warp * 16 + element / 16;
        unsigned column = first_column + column_tile * 16 + element % 16;
        if (row < operation.row_count && column < operation.column_count)
            math.store(batch_index * operation.output_batch_stride +
                           uint64_t(row) * operation.output_row_stride + column,
                       output_tiles[tile_warp][column_tile][element]);
    }
}

#include "cuda_fp8_matrix.cuh"
#include "cuda_window_attention.cuh"
#include "cuda_fp8_attention.cuh"
#include "cuda_window_block.cuh"

#endif
#if !defined(PHX_SYCL)
// Fix the operation before compilation. Simple copies/gathers no longer inherit
// register and stack requirements from unrelated row reductions and softmax.
template <unsigned Kind>
__global__ void run_elementwise(phx::Operation operation, uint64_t count, uint64_t first) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        operation.family = phx::Elementwise;
        operation.flags = (operation.flags & ~255u) | Kind;
        phx::Kernel<GpuMath>{operation}.unary(first + index);
    }
}
__global__ void run_frame_features(phx::Operation operation, uint64_t count, uint64_t first) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        phx::Kernel<GpuMath>{operation}.frame_features(first + index);
}
static void launch_elementwise(phx::Operation operation, uint64_t count, uint64_t first,
                               hipStream_t stream) {
    const unsigned groups = unsigned((count + 255) / 256);
    // The named cases follow ElementwiseKind; unsupported future kinds retain
    // the generic implementation rather than silently selecting another operation.
#define PHX_POINTWISE_CASE(Kind)                                                                   \
    case phx::Kind:                                                                                \
        run_elementwise<phx::Kind><<<groups, 256, 0, stream>>>(operation, count, first);           \
        break
    switch (operation.flags & 255u) {
        PHX_POINTWISE_CASE(QuantizeFp8);
        PHX_POINTWISE_CASE(ActivationGate);
        PHX_POINTWISE_CASE(RoundHalf);
        PHX_POINTWISE_CASE(FloatToHalf);
        PHX_POINTWISE_CASE(CopyValues);
        PHX_POINTWISE_CASE(ScaledResidual);
        PHX_POINTWISE_CASE(ScaleHalf);
        PHX_POINTWISE_CASE(GatherWindows);
        PHX_POINTWISE_CASE(ScatterWindows);
        PHX_POINTWISE_CASE(AddPositionBias);
        PHX_POINTWISE_CASE(SplitQkvHeads);
        PHX_POINTWISE_CASE(MergeAttentionHeads);
        PHX_POINTWISE_CASE(AveragePool);
        PHX_POINTWISE_CASE(NearestUpsample);
        PHX_POINTWISE_CASE(ChannelGain);
        PHX_POINTWISE_CASE(AddInputs);
        PHX_POINTWISE_CASE(SpatialPad);
        PHX_POINTWISE_CASE(GateFp8ToHalf);
        PHX_POINTWISE_CASE(Fp8ToHalf);
        PHX_POINTWISE_CASE(GateToHalf);
        PHX_POINTWISE_CASE(UpsampleBlendAndPublish);
        PHX_POINTWISE_CASE(UpsampleResidual);
        PHX_POINTWISE_CASE(PoolAndPublish);
        PHX_POINTWISE_CASE(GatherWindowBatch);
        PHX_POINTWISE_CASE(ScatterWindowBatch);
        PHX_POINTWISE_CASE(ScatterWindowBatchAndPool);
        PHX_POINTWISE_CASE(GatherUpsampleBlendBatch);
    default:
        run_kernel<<<groups, 256, 0, stream>>>(operation, count, first);
        break;
    }
#undef PHX_POINTWISE_CASE
}
#endif
#if defined(PHX_SYCL)
#include "sycl_elementwise.h"
#include "sycl_rows.h"
#include "sycl_matrix.h"
#endif
class GpuBackend final : public Backend {
    bool tensor_attention = true;
    bool detailed_profile = false;
    unsigned tensor_column_tiles = 1;
    bool direct_fp8 = false;
    bool fused_feed_forward = false;
    bool fused_window_block = false;
    bool compact_window_block = false;
    bool fused_wide_feed_forward = false;
    bool shape_matrices = false;
    unsigned fp8_columns = 32;
#if defined(PHX_SYCL)
    std::unique_ptr<sycl::queue> queue_;
#else
    // Prefixes below are translated to the CUDA equivalents for the CUDA build.
    hipStream_t stream{};
#if defined(PHX_CUDA)
    hipGraph_t graph{};
    hipGraphExec_t executable{};
    struct Timer {
        hipEvent_t started{}, finished{};
    };
    std::vector<Timer> graph_timers;
    std::vector<Command> cached_commands;
    std::vector<CommandSpan> graph_spans;
    void clear_graph() {
        if (executable) {
            PHX_CHECK(hipGraphExecDestroy(executable));
            executable = nullptr;
        }
        if (graph) {
            PHX_CHECK(hipGraphDestroy(graph));
            graph = nullptr;
        }
        for (auto timer : graph_timers) {
            if (timer.started)
                hipEventDestroy(timer.started);
            if (timer.finished)
                hipEventDestroy(timer.finished);
        }
        graph_timers.clear();
        cached_commands.clear();
    }
#else
    struct Timer {
        hipEvent_t started{}, finished{};
    };
    std::vector<Timer> hip_timers;
    void clear_hip_timers() {
        for (auto timer : hip_timers) {
            if (timer.started)
                (void)hipEventDestroy(timer.started);
            if (timer.finished)
                (void)hipEventDestroy(timer.finished);
        }
        hip_timers.clear();
    }
#endif
    static void check(hipError_t error_code, int line) {
        if (error_code != hipSuccess)
            throw std::runtime_error(std::string(hipGetErrorString(error_code)) +
                                     " at backend_gpu.cpp:" + std::to_string(line));
    }
#endif
  public:
    explicit GpuBackend(const std::string& device_id, uint32_t options = 0) {
        tensor_attention = !(options & OrderedAttention);
        detailed_profile = (options & DispatchProfiling) != 0;
        direct_fp8 = (options & CudaDirectFp8) != 0;
        fused_feed_forward = (options & CudaFusedFeedForward) != 0;
        fused_window_block = (options & CudaFusedWindowBlock) != 0;
        compact_window_block = (options & CudaCompactWindowBlock) != 0;
        fused_wide_feed_forward = (options & CudaFusedWideFeedForward) != 0;
        shape_matrices = (options & CudaShapeMatrices) != 0;
        fp8_columns = (options & Fp8Columns128) ? 128 : (options & Fp8Columns64) ? 64 : 32;
        tensor_column_tiles = (options & MatrixColumns64) ? 4 : (options & MatrixColumns32) ? 2 : 1;
#if defined(PHX_SYCL)
        std::vector<sycl::device> matches;
        for (auto& candidate_device : sycl::device::get_devices()) {
            std::string key;
            if (candidate_device.is_cpu())
                key = "sycl:cpu";
            else if (candidate_device.has(sycl::aspect::ext_intel_device_info_uuid)) {
                auto uuid = candidate_device.get_info<sycl::ext::intel::info::device::uuid>();
                key = "sycl:";
                for (auto uuid_byte : uuid) {
                    char hex_byte[3];
                    std::snprintf(hex_byte, 3, "%02x", uuid_byte);
                    key += hex_byte;
                }
            }
            if (key == device_id)
                matches.push_back(candidate_device);
        }
        demand(matches.size() == 1, "SYCL device missing or ambiguous");
        queue_ = std::make_unique<sycl::queue>(
            matches[0], sycl::property_list{sycl::property::queue::in_order{},
                                            sycl::property::queue::enable_profiling {}});
#else
        std::string requested = device_id;
        if (requested.rfind("cuda:GPU-", 0) == 0)
            requested.erase(5, 4);
        requested.erase(std::remove(requested.begin(), requested.end(), '-'), requested.end());
        int count = 0, selected = -1;
        PHX_CHECK(hipGetDeviceCount(&count));
        for (int device_index = 0; device_index < count; ++device_index) {
#if defined(PHX_CUDA)
            cudaDeviceProp properties{};
            PHX_CHECK(cudaGetDeviceProperties(&properties, device_index));
            auto uuid = properties.uuid;
#else
            hipUUID uuid{};
            PHX_CHECK(hipDeviceGetUuid(&uuid, device_index));
#endif
            std::string key = DEVICE_PREFIX;
            for (unsigned uuid_index = 0; uuid_index < 16; ++uuid_index) {
                char hex_byte[3];
                std::snprintf(hex_byte, 3, "%02x",
                              static_cast<unsigned char>(uuid.bytes[uuid_index]));
                key += hex_byte;
            }
            if (key == requested) {
                demand(selected < 0, "Ambiguous GPU UUID");
                selected = device_index;
            }
        }
        demand(selected >= 0, "Requested GPU UUID not found");
        PHX_CHECK(hipSetDevice(selected));
#if defined(PHX_CUDA)
        cudaDeviceProp selected_properties{};
        PHX_CHECK(cudaGetDeviceProperties(&selected_properties, selected));
        // A build targeting an older virtual architecture may be JIT-loaded on
        // a newer GPU. It still lacks the FP8 instruction body; gate both sides.
        cudaFuncAttributes fp8_attributes{};
        PHX_CHECK(cudaFuncGetAttributes(&fp8_attributes, matrix_direct_fp8<4>));
        direct_fp8 = direct_fp8 &&
                     selected_properties.major * 10 + selected_properties.minor >= 89 &&
                     fp8_attributes.ptxVersion >= 89;
        // This routing was measured on Blackwell GeForce. Other architectures
        // retain their existing column choice until independently benchmarked.
        shape_matrices =
            shape_matrices && selected_properties.major == 12 && selected_properties.minor == 0;
#endif
        PHX_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
#endif
    }
    ~GpuBackend() {
        try {
            wait();
        } catch (...) {
        }
#if !defined(PHX_SYCL)
#if defined(PHX_CUDA)
        try {
            clear_graph();
        } catch (...) {
        }
#else
        clear_hip_timers();
#endif
        if (stream)
            hipStreamDestroy(stream);
#endif
    }
    const char* name() const override {
        return DEVICE_NAME;
    }
    Allocation allocate(uint64_t byte_count) override {
        demand(byte_count > 0 && byte_count <= size_t(-1), "Device allocation size");
        void* address = nullptr;
#if defined(PHX_SYCL)
        address = sycl::malloc_device(size_t(byte_count), *queue_);
        if (!address)
            throw std::bad_alloc();
        queue_->memset(address, 0, size_t(byte_count)).wait_and_throw();
#else
        PHX_CHECK(hipMalloc(&address, size_t(byte_count)));
        PHX_CHECK(hipMemsetAsync(address, 0, size_t(byte_count), stream));
#endif
        return {address, byte_count, 0};
    }
    void release(Allocation allocation) override {
        wait();
#if defined(PHX_CUDA)
        clear_graph();
#endif
#if defined(PHX_SYCL)
        sycl::free(allocation.address, *queue_);
#else
        PHX_CHECK(hipFree(allocation.address));
#endif
    }
    void upload(Allocation allocation, const void* source, uint64_t byte_count) override {
        demand(byte_count <= allocation.bytes, "GPU upload bounds");
#if defined(PHX_SYCL)
        queue_->memcpy(allocation.address, source, size_t(byte_count)).wait_and_throw();
#else
        PHX_CHECK(hipMemcpyAsync(allocation.address, source, size_t(byte_count),
                                 hipMemcpyHostToDevice, stream));
        PHX_CHECK(hipStreamSynchronize(stream));
#endif
    }
    void download(Allocation allocation, void* destination, uint64_t byte_count) override {
        demand(byte_count <= allocation.bytes, "GPU download bounds");
#if defined(PHX_SYCL)
        queue_->memcpy(destination, allocation.address, size_t(byte_count)).wait_and_throw();
#else
        PHX_CHECK(hipMemcpyAsync(destination, allocation.address, size_t(byte_count),
                                 hipMemcpyDeviceToHost, stream));
        PHX_CHECK(hipStreamSynchronize(stream));
#endif
    }
    void wait() override {
#if defined(PHX_SYCL)
        if (queue_)
            queue_->wait_and_throw();
#else
        if (stream)
            PHX_CHECK(hipStreamSynchronize(stream));
#endif
    }
    void execute(const std::vector<Command>& commands, std::vector<Event>& events) override {
#if defined(PHX_CUDA)
        // Dynamic frame packing is outside the immutable neural graph. Recording
        // it through the graph cache would evict/rebuild the network every frame.
        if (commands.size() == 1 && commands.front().operation.family == phx::FrameFeatures) {
            const auto& command = commands.front();
            for (uint64_t first = 0; first < command.count; first += uint64_t(65535) * 256) {
                auto count = std::min(uint64_t(65535) * 256, command.count - first);
                run_frame_features<<<unsigned((count + 255) / 256), 256, 0, stream>>>(
                    command.operation, count, first);
            }
            PHX_CHECK(hipGetLastError());
            wait();
            return;
        }
#endif

        if (commands.empty())
            return;
#if defined(PHX_SYCL)
        const bool cpu_device = queue_->get_device().is_cpu();
        auto spans = command_spans(commands);
        if (detailed_profile) {
            spans.clear();
            for (size_t index = 0; index < commands.size(); ++index)
                spans.push_back({index, index + 1, commands[index].block, commands[index].layer});
        }
        struct Timing {
            sycl::event first, last;
        };
        std::vector<Timing> timers(spans.size());
        size_t span = 0;
        for (size_t index = 0; index < commands.size(); ++index) {
            auto command = commands[index];
            auto operation = command.operation;
            auto count = command.count;
            sycl::event event;
            if (cpu_device && operation.family == phx::MatrixMultiply) {
                event = sycl_cpu_matrix<4, 4>(*queue_, operation);
            } else if (operation.family == phx::Elementwise)
                event = cpu_device ? sycl_elementwise<SyclCpuMath>(*queue_, operation, count)
                                   : sycl_elementwise<GpuMath>(*queue_, operation, count);
            else if (operation.family == phx::RowReduction &&
                     ((operation.flags & 255u) <= 1 || (operation.flags & 255u) == 6))
                event = cpu_device ? sycl_row<SyclCpuMath>(*queue_, operation, count)
                                   : sycl_row<GpuMath>(*queue_, operation, count);
            else
                event = queue_->parallel_for(sycl::range<1>(count), [=](sycl::id<1> work_index) {
                    phx::Kernel<GpuMath>{operation}(uint64_t(work_index[0]));
                });
            if (index == spans[span].begin)
                timers[span].first = event;
            if (index + 1 == spans[span].end) {
                timers[span].last = event;
                ++span;
            }
        }

        wait();
        for (size_t work_index = 0; work_index < timers.size(); ++work_index) {
            auto start =
                timers[work_index]
                    .first.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end = timers[work_index]
                           .last.get_profiling_info<sycl::info::event_profiling::command_end>();
            events.push_back({spans[work_index].block, UINT32_MAX, double(end - start) / 1e6, true,
                              spans[work_index].layer});
        }
#else
#if defined(PHX_CUDA)
        if (!same_commands(commands, cached_commands)) {
            wait();
            clear_graph();
            graph_spans = command_spans(commands);
            if (detailed_profile) {
                graph_spans.clear();
                for (size_t index = 0; index < commands.size(); ++index)
                    graph_spans.push_back(
                        {index, index + 1, commands[index].block, commands[index].layer});
            }
            graph_timers.resize(graph_spans.size());
            bool capturing = false;
            try {
                for (auto& timer : graph_timers) {
                    PHX_CHECK(hipEventCreate(&timer.started));
                    PHX_CHECK(hipEventCreate(&timer.finished));
                }
                PHX_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
                capturing = true;
                size_t span = 0;
                for (size_t work_index = 0; work_index < commands.size(); ++work_index) {
                    auto command = commands[work_index];
                    auto& timer = graph_timers[span];
                    if (work_index == graph_spans[span].begin)
                        PHX_CHECK(
                            hipEventRecordWithFlags(timer.started, stream, hipEventRecordExternal));
                    // Fuse only a complete, proven feed-forward chain. Detailed
                    // dispatch profiling keeps the separate operations so its
                    // event-to-command mapping remains truthful.
                    unsigned fused_count = 0;
                    WindowBlockOperations window_block;
                    if (direct_fp8 && fused_window_block && !detailed_profile &&
                        match_window_block(commands, work_index, graph_spans[span].end,
                                           window_block)) {
                        const uint64_t windows = window_block.expand.row_count / 64;
                        for (uint64_t first = 0; first < windows; first += 65535) {
                            const unsigned blocks =
                                unsigned(std::min(uint64_t(65535), windows - first));
                            if (compact_window_block &&
                                (window_block.contract.flags & phx::MatrixSkipHalf))
                                window_block_shared<true, true>
                                    <<<blocks, 128, 0, stream>>>(window_block, first);
                            else if (compact_window_block)
                                window_block_shared<false, true>
                                    <<<blocks, 128, 0, stream>>>(window_block, first);
                            else if (window_block.contract.flags & phx::MatrixSkipHalf)
                                window_block_shared<true>
                                    <<<blocks, 128, 0, stream>>>(window_block, first);
                            else
                                window_block_shared<false>
                                    <<<blocks, 128, 0, stream>>>(window_block, first);
                        }
                        fused_count = 9;
                    }
                    if (!fused_count && direct_fp8 && fused_feed_forward && !detailed_profile &&
                        command.operation.family == phx::MatrixMultiply &&
                        ((command.operation.flags >> 8) & 15u) == phx::PublishFusedGateFp8 &&
                        (command.operation.flags & phx::LeftInputFp8Grid) &&
                        command.operation.batch_count == 1 &&
                        (command.operation.reduction_count == 32 ||
                         (fused_wide_feed_forward && command.operation.reduction_count == 64)) &&
                        command.operation.column_count == 4 * command.operation.reduction_count &&
                        work_index + 1 < graph_spans[span].end) {
                        const auto& next = commands[work_index + 1].operation;
                        const bool has_middle = command.operation.reduction_count == 64;
                        const size_t last = work_index + (has_middle ? 2 : 1);
                        if (last < graph_spans[span].end) {
                            const auto& contract = commands[last].operation;
                            const bool middle_matches =
                                !has_middle ||
                                (next.family == phx::MatrixMultiply &&
                                 next.left_input == command.operation.output &&
                                 next.batch_count == 2 && next.reduction_count == 128 &&
                                 next.column_count == 32 && next.left_batch_stride == 128 &&
                                 next.right_batch_stride == 4096 && next.output_batch_stride == 32);
                            if (middle_matches && contract.family == phx::MatrixMultiply &&
                                contract.left_input ==
                                    (has_middle ? next.output : command.operation.output) &&
                                contract.row_count == command.operation.row_count &&
                                contract.batch_count == 1 &&
                                contract.column_count == command.operation.reduction_count &&
                                ((contract.flags >> 8) & 15u) == phx::KeepFloat &&
                                (contract.flags & phx::MatrixResidual) &&
                                (contract.flags & phx::LeftInputFp8Grid) &&
                                (contract.flags & phx::RightInputFp8)) {
                                const uint64_t tiles =
                                    (uint64_t(command.operation.row_count) + 63) / 64;
                                const unsigned shared_bytes =
                                    64 * (command.operation.column_count + (has_middle ? 64 : 0)) *
                                    2;
                                for (uint64_t first = 0; first < tiles; first += 65535) {
                                    const unsigned blocks =
                                        unsigned(std::min(uint64_t(65535), tiles - first));
                                    if (has_middle)
                                        feed_forward_shared<true>
                                            <<<blocks, 128, shared_bytes, stream>>>(
                                                command.operation, next, contract, first);
                                    else
                                        feed_forward_shared<false>
                                            <<<blocks, 128, shared_bytes, stream>>>(
                                                command.operation, {}, contract, first);
                                }
                                fused_count = has_middle ? 3 : 2;
                            }
                        }
                    }
                    if (fused_count) {
                        work_index += fused_count - 1;
                    } else if (command.operation.family == phx::WindowAttention) {
                        for (uint64_t first = 0; first < command.operation.batch_count;
                             first += 65535) {
                            const auto blocks = unsigned(std::min(
                                uint64_t(65535), uint64_t(command.operation.batch_count) - first));
                            if (direct_fp8)
                                window_attention_fp8<<<blocks, 128, 0, stream>>>(command.operation,
                                                                                 first);
                            else
                                window_attention<<<blocks, 128, 0, stream>>>(command.operation,
                                                                             first);
                        }
                        // Short reductions spend more time encoding activations than
                        // multiplying them. Keep their measured FP16 tensor path.
                    } else if (direct_fp8 && command.operation.family == phx::MatrixMultiply &&
                               (command.operation.flags & phx::RightInputFp8) &&
                               (command.operation.flags & phx::LeftInputFp8Grid) &&
                               (command.operation.flags & 1) &&
                               command.operation.reduction_count >= 128 &&
                               command.operation.reduction_count % 32 == 0 &&
                               command.operation.left_row_stride % 4 == 0) {
                        unsigned columns =
                            fp8_columns >= 128 && command.operation.column_count >= 128 ? 128
                            : fp8_columns >= 64 && command.operation.column_count >= 64 ? 64
                                                                                        : 32;
                        if (shape_matrices) {
                            const auto& operation = command.operation;
                            const unsigned publication = (operation.flags >> 8) & 15u;
                            // QKV projects C channels to 3*C. Sharing the left
                            // fragment across wider output tiles pays here,
                            // while the gated spatial MLP stays on 32 columns.
                            if (!(operation.flags & phx::MatrixResidual) &&
                                publication == phx::KeepFloat && operation.column_count >= 384 &&
                                operation.column_count == 3 * operation.reduction_count)
                                columns = 128;
                            else if (!(operation.flags & phx::MatrixResidual) &&
                                     publication == phx::PublishGateThenHalfFp8 &&
                                     operation.column_count >= 2048 &&
                                     operation.column_count == 4 * operation.reduction_count)
                                columns = 128;
                            else if ((operation.flags & phx::MatrixResidual) &&
                                     publication == phx::KeepFloat &&
                                     operation.column_count == 256 &&
                                     operation.reduction_count == 256 && operation.row_count >= 512)
                                columns = 64;
                        }
                        const uint64_t tiles =
                            uint64_t((command.operation.row_count + 63) / 64) *
                            ((command.operation.column_count + columns - 1) / columns) *
                            command.operation.batch_count;
                        for (uint64_t first = 0; first < tiles; first += 65535) {
                            const auto blocks = unsigned(std::min(uint64_t(65535), tiles - first));
                            if (columns == 128)
                                launch_matrix_direct_fp8<16>(command.operation, first, blocks,
                                                             stream);
                            else if (columns == 64)
                                launch_matrix_direct_fp8<8>(command.operation, first, blocks,
                                                            stream);
                            else
                                launch_matrix_direct_fp8<4>(command.operation, first, blocks,
                                                            stream);
                        }
                    } else if (command.operation.family == phx::MatrixMultiply &&
                               command.operation.column_count <= 4) {
                        uint64_t rows =
                            uint64_t(command.operation.row_count) * command.operation.batch_count;
                        for (uint64_t first = 0; first < rows; first += uint64_t(65535) * 256) {
                            auto launch_count = std::min(uint64_t(65535) * 256, rows - first);
                            run_projection<<<unsigned((launch_count + 255) / 256), 256, 0,
                                             stream>>>(command.operation, first, launch_count);
                        }
                    } else if (command.operation.family == phx::MatrixMultiply &&
                               command.operation.reduction_count % 16 == 0 &&
                               ((command.operation.flags & phx::RightInputFp8) ||
                                (tensor_attention && command.operation.batch_count > 1 &&
                                 command.operation.row_count == 64 &&
                                 (command.operation.column_count == 32 ||
                                  command.operation.column_count == 64) &&
                                 command.operation.flags <= 1))) {
                        // Reuse each activation fragment across adjacent output tiles.
                        // Keep 16/32-column alternatives for narrow and ragged shapes.
                        unsigned columns =
                            tensor_column_tiles >= 4 && command.operation.column_count % 64 == 0 ? 4
                            : tensor_column_tiles >= 2 && command.operation.column_count >= 32   ? 2
                                                                                               : 1;
                        uint64_t tiles =
                            uint64_t((command.operation.row_count + 63) / 64) *
                            ((command.operation.column_count + columns * 16 - 1) / (columns * 16)) *
                            command.operation.batch_count;
                        for (uint64_t first = 0; first < tiles; first += 65535) {
                            auto launch_count = std::min(uint64_t(65535), tiles - first);
                            if (columns == 4)
                                matrix_tensor_f16<4><<<unsigned(launch_count), 128, 0, stream>>>(
                                    command.operation, first);
                            else if (columns == 2)
                                matrix_tensor_f16<2><<<unsigned(launch_count), 128, 0, stream>>>(
                                    command.operation, first);
                            else
                                matrix_tensor_f16<1><<<unsigned(launch_count), 128, 0, stream>>>(
                                    command.operation, first);
                        }
                    } else if (command.operation.family == phx::MatrixMultiply) {
                        uint64_t tiles = uint64_t((command.operation.row_count + 31) / 32) *
                                         ((command.operation.column_count + 31) / 32) *
                                         command.operation.batch_count;
                        for (uint64_t first = 0; first < tiles; first += 65535) {
                            auto launch_count = std::min(uint64_t(65535), tiles - first);
                            run_gemm<<<unsigned(launch_count), 256, 0, stream>>>(
                                command.operation, first, launch_count);
                        }
                    } else if (command.operation.family == phx::RowReduction &&
                               (command.operation.flags & 255u) == 0 &&
                               command.operation.batch_count != 16 &&
                               !(command.operation.flags & phx::GlobalAttention)) {
                        for (uint64_t first = 0; first < command.count;
                             first += uint64_t(65535) * 32) {
                            auto count = std::min(uint64_t(65535) * 32, command.count - first);
                            normalize_heads8<<<unsigned((count + 31) / 32), 256, 0, stream>>>(
                                command.operation, count, first);
                        }
                    } else if (command.operation.family == phx::RowReduction &&
                               (command.operation.flags & 255u) == 6) {
                        for (uint64_t first = 0; first < command.count;
                             first += uint64_t(65535) * 32) {
                            auto count = std::min(uint64_t(65535) * 32, command.count - first);
                            normalize_global_scores8<<<unsigned((count + 31) / 32), 256, 0,
                                                       stream>>>(command.operation, count, first);
                        }
                    } else
                        for (uint64_t first = 0; first < command.count;
                             first += uint64_t(65535) * 256) {
                            auto launch_count =
                                std::min(uint64_t(65535) * 256, command.count - first);
                            if (command.operation.family == phx::Elementwise)
                                launch_elementwise(command.operation, launch_count, first, stream);
                            else
                                run_kernel<<<unsigned((launch_count + 255) / 256), 256, 0,
                                             stream>>>(command.operation, launch_count, first);
                        }
                    PHX_CHECK(hipGetLastError());
                    if (work_index + 1 == graph_spans[span].end) {
                        PHX_CHECK(hipEventRecordWithFlags(timer.finished, stream,
                                                          hipEventRecordExternal));
                        ++span;
                    }
                }
                auto ended = hipStreamEndCapture(stream, &graph);
                capturing = false;
                PHX_CHECK(ended);
                PHX_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                cached_commands = commands;
            } catch (...) {
                if (capturing) {
                    hipGraph_t abandoned{};
                    hipStreamEndCapture(stream, &abandoned);
                    if (abandoned)
                        hipGraphDestroy(abandoned);
                }
                clear_graph();
                throw;
            }
        }
        PHX_CHECK(hipGraphLaunch(executable, stream));
        wait();
        for (size_t work_index = 0; work_index < graph_timers.size(); ++work_index) {
            float milliseconds = 0;
            PHX_CHECK(hipEventElapsedTime(&milliseconds, graph_timers[work_index].started,
                                          graph_timers[work_index].finished));
            events.push_back({graph_spans[work_index].block, UINT32_MAX, milliseconds, true,
                              graph_spans[work_index].layer});
        }
#else
        auto spans = command_spans(commands);
        if (detailed_profile) {
            spans.clear();
            for (size_t index = 0; index < commands.size(); ++index)
                spans.push_back({index, index + 1, commands[index].block, commands[index].layer});
        }
        // Reuse timestamp events between frames, as the CUDA graph path does.
        // Creating/destroying hundreds of driver objects per frame is avoidable.
        if (hip_timers.size() < spans.size()) {
            const auto previous = hip_timers.size();
            hip_timers.resize(spans.size());
            try {
                for (size_t index = previous; index < hip_timers.size(); ++index) {
                    PHX_CHECK(hipEventCreate(&hip_timers[index].started));
                    PHX_CHECK(hipEventCreate(&hip_timers[index].finished));
                }
            } catch (...) {
                clear_hip_timers();
                throw;
            }
        }
        auto& timers = hip_timers;
        try {
            size_t span = 0;
            for (size_t work_index = 0; work_index < commands.size(); ++work_index) {
                auto command = commands[work_index];
                auto& timer = timers[span];
                if (work_index == spans[span].begin) {
                    PHX_CHECK(hipEventRecord(timer.started, stream));
                }
                if (command.operation.family == phx::MatrixMultiply &&
                    command.operation.column_count <= 4) {
                    uint64_t rows =
                        uint64_t(command.operation.row_count) * command.operation.batch_count;
                    for (uint64_t first = 0; first < rows; first += uint64_t(65535) * 256) {
                        auto launch_count = std::min(uint64_t(65535) * 256, rows - first);
                        run_projection<<<unsigned((launch_count + 255) / 256), 256, 0, stream>>>(
                            command.operation, first, launch_count);
                    }
                } else if (command.operation.family == phx::MatrixMultiply) {
                    uint64_t tiles = uint64_t((command.operation.row_count + 31) / 32) *
                                     ((command.operation.column_count + 31) / 32) *
                                     command.operation.batch_count;
                    for (uint64_t first = 0; first < tiles; first += 65535) {
                        auto launch_count = std::min(uint64_t(65535), tiles - first);
                        run_gemm<<<unsigned(launch_count), 256, 0, stream>>>(command.operation,
                                                                             first, launch_count);
                    }
                } else if (command.operation.family == phx::RowReduction &&
                           (command.operation.flags & 255u) == 0 &&
                           command.operation.batch_count != 16 &&
                           !(command.operation.flags & phx::GlobalAttention)) {
                    for (uint64_t first = 0; first < command.count; first += uint64_t(65535) * 32) {
                        const auto count = std::min(uint64_t(65535) * 32, command.count - first);
                        normalize_heads8<<<unsigned((count + 31) / 32), 256, 0, stream>>>(
                            command.operation, count, first);
                    }
                } else if (command.operation.family == phx::RowReduction &&
                           (command.operation.flags & 255u) == 6) {
                    for (uint64_t first = 0; first < command.count; first += uint64_t(65535) * 32) {
                        const auto count = std::min(uint64_t(65535) * 32, command.count - first);
                        normalize_global_scores8<<<unsigned((count + 31) / 32), 256, 0, stream>>>(
                            command.operation, count, first);
                    }
                } else if (command.operation.family == phx::RowReduction &&
                           (command.operation.flags & 255u) == 1) {
                    for (uint64_t first = 0; first < command.count; first += uint64_t(65535) * 32) {
                        const auto count = std::min(uint64_t(65535) * 32, command.count - first);
                        normalize_window_scores8<<<unsigned((count + 31) / 32), 256, 0, stream>>>(
                            command.operation, count, first);
                    }
                } else
                    for (uint64_t first = 0; first < command.count;
                         first += uint64_t(65535) * 256) {
                        auto launch_count = std::min(uint64_t(65535) * 256, command.count - first);
                        if (command.operation.family == phx::Elementwise)
                            launch_elementwise(command.operation, launch_count, first, stream);
                        else
                            run_kernel<<<unsigned((launch_count + 255) / 256), 256, 0, stream>>>(
                                command.operation, launch_count, first);
                    }
                PHX_CHECK(hipGetLastError());
                if (work_index + 1 == spans[span].end) {
                    PHX_CHECK(hipEventRecord(timer.finished, stream));
                    ++span;
                }
            }
            wait();
            for (size_t work_index = 0; work_index < spans.size(); ++work_index) {
                float milliseconds = 0;
                PHX_CHECK(hipEventElapsedTime(&milliseconds, timers[work_index].started,
                                              timers[work_index].finished));
                events.push_back({spans[work_index].block, UINT32_MAX, milliseconds, true,
                                  spans[work_index].layer});
            }
        } catch (...) {
            try {
                wait();
            } catch (...) {
            }
            clear_hip_timers();
            throw;
        }
#endif
#endif
    }
};
extern "C" __declspec(dllexport) Backend*
phoenix_backend_create(const char* device_id, const wchar_t*, char* error, size_t size) {
    try {
        return new GpuBackend(device_id ? device_id : "");
    } catch (const std::exception& exception) {
        strncpy_s(error, size, exception.what(), _TRUNCATE);
        return nullptr;
    }
}
extern "C" __declspec(dllexport) void phoenix_backend_destroy(Backend* backend) {
    delete backend;
}
extern "C" __declspec(dllexport) int phoenix_backend_devices(void (*emit)(const char*, const char*),
                                                             char* error, size_t size) {
    try {
        int found = 0;
#if defined(PHX_SYCL)
        for (const auto& candidate_device : sycl::device::get_devices()) {
            std::string key;
            if (candidate_device.is_cpu())
                key = "sycl:cpu";
            else if (candidate_device.has(sycl::aspect::ext_intel_device_info_uuid)) {
                key = "sycl:";
                for (auto uuid_byte :
                     candidate_device.get_info<sycl::ext::intel::info::device::uuid>()) {
                    char hex_byte[3];
                    std::snprintf(hex_byte, 3, "%02x", uuid_byte);
                    key += hex_byte;
                }
            }
            if (!key.empty()) {
                auto label = candidate_device.get_info<sycl::info::device::name>();
                emit(key.c_str(), label.c_str());
                ++found;
            }
        }
#else
        int count = 0;
        auto result = hipGetDeviceCount(&count);
        if (result != hipSuccess)
            throw std::runtime_error(hipGetErrorString(result));
        for (int device_index = 0; device_index < count; ++device_index) {
#if defined(PHX_CUDA)
            cudaDeviceProp properties{};
            auto code = cudaGetDeviceProperties(&properties, device_index);
            auto uuid = properties.uuid;
#else
            hipDeviceProp_t properties{};
            auto code = hipGetDeviceProperties(&properties, device_index);
            hipUUID uuid{};
            if (code == hipSuccess)
                code = hipDeviceGetUuid(&uuid, device_index);
#endif
            if (code != hipSuccess)
                throw std::runtime_error(hipGetErrorString(code));
            std::string key = DEVICE_PREFIX;
            for (unsigned uuid_index = 0; uuid_index < 16; ++uuid_index) {
                char hex_byte[3];
                std::snprintf(hex_byte, 3, "%02x",
                              static_cast<unsigned char>(uuid.bytes[uuid_index]));
                key += hex_byte;
            }
            emit(key.c_str(), properties.name);
            ++found;
        }
#endif
        return found;
    } catch (const std::exception& exception) {
        strncpy_s(error, size, exception.what(), _TRUNCATE);
        return -1;
    }
}

extern "C" __declspec(dllexport) Backend*
phoenix_backend_create_configured(const char* device_id, const wchar_t*, uint32_t options,
                                  char* error, size_t size) {
    try {
        return new GpuBackend(device_id ? device_id : "", options);
    } catch (const std::exception& exception) {
        strncpy_s(error, size, exception.what(), _TRUNCATE);
        return nullptr;
    }
}
extern "C" __declspec(dllexport) uint32_t phoenix_backend_features(Backend*) {
#if defined(PHX_CUDA)
    return FusedWindowAttention | DeviceFeaturePacking | FusedMatrixResidual;
#else
    return 0;
#endif
}
