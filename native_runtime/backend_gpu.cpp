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
using namespace phoenix_cpp;
#define PHX_CHECK(expression) check((expression), __LINE__)
struct GpuMath {
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
#if !defined(PHX_SYCL)
__global__ void run_kernel(phx::Operation operation, uint64_t count, uint64_t offset) {
    uint64_t element_index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element_index < count)
        phx::Kernel<GpuMath>{operation}(offset + element_index);
}
// Four output rows per thread reuse each B value. Each output still reduces
// k in ascending order using the same scalar FMA and publication operation.
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
    __shared__ float left_tile[edge][edge], right_tile[edge][edge];
    float accumulators[rows_per_thread]{};
    for (unsigned reduction_tile_start = 0; reduction_tile_start < operation.reduction_count;
         reduction_tile_start += edge) {
        for (unsigned lane = threadIdx.x; lane < edge * edge; lane += 256) {
            unsigned tile_y = lane / edge, tile_x = lane % edge,
                     row = unsigned(tile / column_tile_count % row_tile_count) * edge + tile_y,
                     tile_column = unsigned(tile % column_tile_count) * edge + tile_x;
            left_tile[tile_y][tile_x] =
                row < operation.row_count &&
                        reduction_tile_start + tile_x < operation.reduction_count
                    ? math.load_half(operation.left_input,
                                     batch_index * operation.left_batch_stride +
                                         uint64_t(row) * operation.left_row_stride +
                                         reduction_tile_start + tile_x)
                    : 0;
            right_tile[tile_y][tile_x] =
                tile_column < operation.column_count &&
                        reduction_tile_start + tile_y < operation.reduction_count
                    ? math.load_matrix_right(
                          batch_index * operation.right_batch_stride +
                          ((operation.flags & 1)
                               ? uint64_t(tile_column) * operation.right_row_stride +
                                     reduction_tile_start + tile_y
                               : uint64_t(reduction_tile_start + tile_y) *
                                         operation.right_row_stride +
                                     tile_column))
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
#if defined(PHX_CUDA)
#include <mma.h>
#include <cuda_fp16.h>
// Four warps compute a 64x16 output tile using 16x16x16 tensor-core fragments.
// Packed FP8 weights decode to half in shared memory; accumulators stay FP32.
// The fragment types and synchronization are part of the existing numerical path.
__global__ void matrix_tensor_f16(phx::Operation operation, uint64_t first_tile) {
    using namespace nvcuda;
    uint64_t tile_index = first_tile + blockIdx.x,
             column_tile_count = (operation.column_count + 15) / 16,
             row_tile_count = (operation.row_count + 63) / 64,
             batch_index = tile_index / (column_tile_count * row_tile_count);
    unsigned first_row = unsigned(tile_index / column_tile_count % row_tile_count) * 64,
             first_column = unsigned(tile_index % column_tile_count) * 16,
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
    __shared__ __align__(32) __half left_tiles[4][256], right_tile[256];
    __shared__ __align__(32) float output_tiles[4][256];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> left_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> right_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.f);
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
        for (unsigned tile_element = threadIdx.x; tile_element < 256; tile_element += 128) {
            unsigned column = first_column + tile_element / 16,
                     reduction_in_tile = tile_element % 16;
            right_tile[tile_element] = __float2half_rn(
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
        wmma::load_matrix_sync(right_fragment, right_tile, 16);
        wmma::mma_sync(accumulator, left_fragment, right_fragment, accumulator);
        __syncthreads();
    }
    wmma::store_matrix_sync(output_tiles[warp_index], accumulator, 16, wmma::mem_row_major);
    __syncthreads();
    for (unsigned tile_element = threadIdx.x; tile_element < 1024; tile_element += 128) {
        unsigned row = first_row + tile_element / 16, column = first_column + tile_element % 16;
        if (row < operation.row_count && column < operation.column_count)
            math.store(batch_index * operation.output_batch_stride +
                           uint64_t(row) * operation.output_row_stride + column,
                       output_tiles[tile_element / 256][tile_element % 256]);
    }
}

#include "cuda_window_attention.cuh"
#endif
class GpuBackend final : public Backend {
    bool tensor_attention = true;
    bool detailed_profile = false;
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
                run_kernel<<<unsigned((count + 255) / 256), 256, 0, stream>>>(command.operation,
                                                                              count, first);
            }
            PHX_CHECK(hipGetLastError());
            wait();
            return;
        }
#endif

        if (commands.empty())
            return;
#if defined(PHX_SYCL)
        auto spans = command_spans(commands);
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
            if (queue_->get_device().is_cpu() && operation.family == phx::MatrixMultiply) {
                uint64_t column_tile_count = (operation.column_count + 7) / 8,
                         row_tile_count = (operation.row_count + 7) / 8;
                event = queue_->parallel_for(
                    sycl::range<1>(column_tile_count * row_tile_count * operation.batch_count),
                    [=](sycl::id<1> item) {
                        uint64_t tile = item[0],
                                 batch_index = tile / (column_tile_count * row_tile_count);
                        unsigned first_row =
                                     unsigned(tile / column_tile_count % row_tile_count) * 8,
                                 first_column = unsigned(tile % column_tile_count) * 8;
                        phx::Kernel<GpuMath> math{operation};
                        float accumulators[8][8]{};
                        for (unsigned reduction_index = 0;
                             reduction_index < operation.reduction_count; ++reduction_index) {
                            float right_values[8];
                            for (unsigned column_in_tile = 0; column_in_tile < 8;
                                 ++column_in_tile) {
                                unsigned column = first_column + column_in_tile;
                                right_values[column_in_tile] =
                                    column < operation.column_count
                                        ? math.load_matrix_right(
                                              batch_index * operation.right_batch_stride +
                                              ((operation.flags & 1)
                                                   ? uint64_t(column) * operation.right_row_stride +
                                                         reduction_index
                                                   : uint64_t(reduction_index) *
                                                             operation.right_row_stride +
                                                         column))
                                        : 0;
                            }
                            for (unsigned work_index = 0; work_index < 8; ++work_index) {
                                unsigned row = first_row + work_index;
                                float left_value =
                                    row < operation.row_count
                                        ? math.load_half(operation.left_input,
                                                         batch_index * operation.left_batch_stride +
                                                             uint64_t(row) *
                                                                 operation.left_row_stride +
                                                             reduction_index)
                                        : 0;
                                for (unsigned column_in_tile = 0; column_in_tile < 8;
                                     ++column_in_tile)
                                    accumulators[work_index][column_in_tile] =
                                        sycl::fma(left_value, right_values[column_in_tile],
                                                  accumulators[work_index][column_in_tile]);
                            }
                        }
                        for (unsigned work_index = 0;
                             work_index < 8 && first_row + work_index < operation.row_count;
                             ++work_index)
                            for (unsigned column_in_tile = 0;
                                 column_in_tile < 8 &&
                                 first_column + column_in_tile < operation.column_count;
                                 ++column_in_tile)
                                math.store(batch_index * operation.output_batch_stride +
                                               uint64_t(first_row + work_index) *
                                                   operation.output_row_stride +
                                               first_column + column_in_tile,
                                           accumulators[work_index][column_in_tile]);
                    });
            } else
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
                    if (command.operation.family == phx::WindowAttention) {
                        for (uint64_t first = 0; first < command.operation.batch_count;
                             first += 65535)
                            window_attention<<<
                                unsigned(std::min(uint64_t(65535),
                                                  uint64_t(command.operation.batch_count) - first)),
                                128, 0, stream>>>(command.operation, first);
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
                        uint64_t tiles = uint64_t((command.operation.row_count + 63) / 64) *
                                         ((command.operation.column_count + 15) / 16) *
                                         command.operation.batch_count;
                        for (uint64_t first = 0; first < tiles; first += 65535) {
                            auto launch_count = std::min(uint64_t(65535), tiles - first);
                            matrix_tensor_f16<<<unsigned(launch_count), 128, 0, stream>>>(
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
                    } else
                        for (uint64_t first = 0; first < command.count;
                             first += uint64_t(65535) * 256) {
                            auto launch_count =
                                std::min(uint64_t(65535) * 256, command.count - first);
                            run_kernel<<<unsigned((launch_count + 255) / 256), 256, 0, stream>>>(
                                command.operation, launch_count, first);
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
        struct Timer {
            hipEvent_t started{}, finished{};
        };
        std::vector<Timer> timers(spans.size());
        try {
            size_t span = 0;
            for (size_t work_index = 0; work_index < commands.size(); ++work_index) {
                auto command = commands[work_index];
                auto& timer = timers[span];
                if (work_index == spans[span].begin) {
                    PHX_CHECK(hipEventCreate(&timer.started));
                    PHX_CHECK(hipEventCreate(&timer.finished));
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
                } else
                    for (uint64_t first = 0; first < command.count;
                         first += uint64_t(65535) * 256) {
                        auto launch_count = std::min(uint64_t(65535) * 256, command.count - first);
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
            for (size_t work_index = 0; work_index < timers.size(); ++work_index) {
                float milliseconds = 0;
                PHX_CHECK(hipEventElapsedTime(&milliseconds, timers[work_index].started,
                                              timers[work_index].finished));
                events.push_back({spans[work_index].block, UINT32_MAX, milliseconds, true,
                                  spans[work_index].layer});
            }
        } catch (...) {
            for (auto timer : timers) {
                if (timer.started)
                    hipEventDestroy(timer.started);
                if (timer.finished)
                    hipEventDestroy(timer.finished);
            }
            throw;
        }
        for (auto timer : timers) {
            hipEventDestroy(timer.started);
            hipEventDestroy(timer.finished);
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

#if defined(PHX_CUDA)
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
    return FusedWindowAttention | DeviceFeaturePacking | FusedMatrixResidual;
}
#endif
