// Isolated integer scheduling probe: no models, inference, games or file writes.
#include <cuda_runtime.h>
#include "../../src/native/kernels/production/common/window_math.cuh"
#include "../../src/native/kernels/production/common/signed_division.cuh"
#include "../../src/native/kernels/production/common/token_dispatch.cuh"
#include "../../src/native/kernels/production/common/boundary_schedule.cuh"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct DivisionCase { uint32_t numerator, divisor; };
struct BoundaryCase { uint32_t next, end, stride; };
struct BoundaryResult { uint32_t signed_steps, unsigned_steps; };
__global__ void unsigned_divide_cases(const DivisionCase* cases, uint32_t* output, unsigned count) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = unsigned_word_quotient(cases[index].numerator, cases[index].divisor);
}
__global__ void boundary_cases(const BoundaryCase* cases, BoundaryResult* output, unsigned count) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const auto item = cases[index];
        output[index] = {boundary_tail_steps<true>(item.next, item.end, item.stride),
                         boundary_tail_steps<false>(item.next, item.end, item.stride)};
    }
}
__global__ void divide_cases(const DivisionCase* cases, uint32_t* output, unsigned count) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = signed_tile_quotient(cases[index].numerator, cases[index].divisor);
}
static void checked(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template <class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { checked(cudaMalloc(&data, count * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
};
struct ScheduleResult {
    uint32_t group, tile, without_leader, barrier, wait_low, wait_high;
};
__global__ void schedule_cases(ScheduleResult* output, uint32_t tile_count, bool exclude_leader,
                               unsigned layout) {
    const unsigned index = threadIdx.y * blockDim.x + threadIdx.x;
    if (exclude_leader && index == 0)
        return;
    const auto work = token_tile_work(index, tile_count);
    const unsigned threads = blockDim.x * blockDim.y;
    CopyWarpSchedule copy;
    switch (layout) {
    case 0: copy = tile_copy_schedule<9232, 2>(index == 0, threads); break;
    case 1: copy = tile_copy_schedule<25624, 3>(index == 0, threads); break;
    case 2: copy = tile_copy_schedule<17424, 2>(index == 0, threads); break;
    default: copy = tile_copy_schedule<3088, 2>(index == 0, threads); break;
    }
    output[index] = {work.group, work.tile, uint32_t(copy.without_block_leader),
                     copy.barrier_byte_offset, copy.wait_word_low, copy.wait_word_high};
}
static unsigned check_schedules() {
    unsigned checked_cases = 0;
    for (dim3 block : {dim3(32, 2), dim3(32, 8), dim3(17, 3), dim3(48, 2)}) {
        const unsigned threads = block.x * block.y;
        DeviceBuffer<ScheduleResult> device(threads);
        std::vector<ScheduleResult> result(threads);
        constexpr uint32_t offsets[] = {0x2400u, 0x6400u, 0x4400u, 0xc00u};
        for (unsigned layout = 0; layout < 4; ++layout) {
            for (uint32_t tiles : {0u, 1u, 2u, 17u, 129u}) {
                for (bool exclude : {false, true}) {
                    schedule_cases<<<1, block>>>(device.data, tiles, exclude, layout);
                    checked(cudaGetLastError());
                    checked(cudaMemcpy(result.data(), device.data, threads * sizeof(ScheduleResult), cudaMemcpyDeviceToHost));
                    for (unsigned index = exclude ? 1 : 0; index < threads; ++index) {
                        const bool leader_warp = index < 32 && !exclude;
                        const uint32_t wait = 0x100000u - threads;
                        const auto actual = result[index];
                        const auto group = tiles ? index / tiles : UINT32_MAX;
                        if (actual.group != group || actual.tile != index - group * tiles ||
                            actual.without_leader != uint32_t(!leader_warp) ||
                            actual.barrier != (leader_warp ? offsets[layout] : 0u) ||
                            actual.wait_low != (leader_warp ? wait << 1 : 0u) ||
                            actual.wait_high != (leader_warp ? wait << 11 : 0u))
                            throw std::runtime_error("Token/copy scheduling contract changed");
                        ++checked_cases;
                    }
                }
            }
        }
    }
    return checked_cases;
}
static unsigned check_unsigned_division(const std::vector<DivisionCase>& cases) {
    DeviceBuffer<DivisionCase> input(cases.size());
    DeviceBuffer<uint32_t> output(cases.size());
    checked(cudaMemcpy(input.data, cases.data(), cases.size() * sizeof(DivisionCase), cudaMemcpyHostToDevice));
    unsigned_divide_cases<<<(unsigned(cases.size()) + 255) / 256, 256>>>(input.data, output.data, unsigned(cases.size()));
    checked(cudaGetLastError());
    std::vector<uint32_t> actual(cases.size());
    checked(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto item = cases[i];
        const uint32_t expected = item.divisor ? uint32_t(uint64_t(item.numerator) / item.divisor) : UINT32_MAX;
        if (actual[i] != expected)
            throw std::runtime_error("Unsigned quotient mismatch at case " + std::to_string(i));
    }
    return unsigned(cases.size());
}
static unsigned check_boundary_counts(const std::vector<BoundaryCase>& cases) {
    DeviceBuffer<BoundaryCase> input(cases.size());
    DeviceBuffer<BoundaryResult> output(cases.size());
    checked(cudaMemcpy(input.data, cases.data(), cases.size() * sizeof(BoundaryCase), cudaMemcpyHostToDevice));
    boundary_cases<<<(unsigned(cases.size()) + 255) / 256, 256>>>(input.data, output.data, unsigned(cases.size()));
    checked(cudaGetLastError());
    std::vector<BoundaryResult> actual(cases.size());
    checked(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(BoundaryResult), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto item = cases[i];
        for (bool signed_indices : {false, true}) {
            const bool has_next = signed_indices ? int32_t(item.next) < int32_t(item.end) : item.next < item.end;
            // Independent 64-bit integer reference for the original max/subtract
            // expression, including uint32 wrap before division and zero stride.
            const uint32_t clipped_end = has_next ? item.end : item.next;
            const uint32_t numerator = clipped_end - item.next - uint32_t(has_next);
            const uint32_t quotient = item.stride ? uint32_t(uint64_t(numerator) / item.stride) : UINT32_MAX;
            const uint32_t expected = uint32_t(has_next) + quotient;
            const uint32_t observed = signed_indices ? actual[i].signed_steps : actual[i].unsigned_steps;
            if (observed != expected)
                throw std::runtime_error("Boundary trip count mismatch at case " + std::to_string(i));
        }
    }
    return unsigned(cases.size()) * 2;
}
int main() {
    try {
        std::vector<DivisionCase> cases;
        const uint32_t boundaries[] = {0, 1, 2, 31, 32, 127, 128, 129, 65535,
                                       0x7fffffffu, 0x80000000u, 0xffffffffu};
        // All signed bit patterns, including INT_MIN divisors, zero and
        // INT_MIN/-1's defined unsigned bit result.
        for (auto numerator : boundaries)
            for (auto divisor : boundaries)
                cases.push_back({numerator, divisor});
        uint32_t random = 0x4a85c9f1u;
        for (unsigned i = 0; i < 1'000'000; ++i) {
            random = random * 1664525u + 1013904223u;
            const uint32_t numerator = random;
            random = random * 1664525u + 1013904223u;
            uint32_t divisor = random & 0x7fffffffu;
            if (i & 1) divisor = 0u - divisor;
            cases.push_back({numerator, divisor});
        }
        DeviceBuffer<DivisionCase> input(cases.size());
        DeviceBuffer<uint32_t> output(cases.size());
        checked(cudaMemcpy(input.data, cases.data(), cases.size() * sizeof(DivisionCase), cudaMemcpyHostToDevice));
        divide_cases<<<(unsigned(cases.size()) + 255) / 256, 256>>>(input.data, output.data, unsigned(cases.size()));
        checked(cudaGetLastError());
        std::vector<uint32_t> actual(cases.size());
        checked(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto item = cases[i];
            const uint32_t expected = item.divisor ?
                uint32_t(int64_t(int32_t(item.numerator)) / int64_t(int32_t(item.divisor))) : UINT32_MAX;
            if (actual[i] != expected)
                throw std::runtime_error("Signed tile quotient mismatch at case " + std::to_string(i));
        }
        const unsigned schedules = check_schedules();
        std::vector<DivisionCase> unsigned_cases;
        std::vector<BoundaryCase> boundaries_to_check;
        for (auto numerator : boundaries)
            for (auto divisor : boundaries) {
                unsigned_cases.push_back({numerator, divisor});
                for (auto end : boundaries)
                    boundaries_to_check.push_back({numerator, end, divisor});
            }
        for (unsigned i = 0; i < 1'000'000; ++i) {
            random = random * 1664525u + 1013904223u;
            const uint32_t numerator = random;
            random = random * 1664525u + 1013904223u;
            const uint32_t divisor = random;
            unsigned_cases.push_back({numerator, divisor});
            random = random * 1664525u + 1013904223u;
            boundaries_to_check.push_back({numerator, random, divisor});
        }
        // Exact tile/flattened extents and every participating thread: exercise
        // the 0..3-item tail and four-item groups at the five production widths.
        for (uint32_t stride : {32u, 64u, 128u, 256u})
            for (uint32_t end : {1u, 63u, 64u, 65u, 127u, 128u, 129u, 255u, 256u, 257u,
                                 511u, 512u, 513u, 1023u, 1024u, 1025u, 1920u, 801u})
                for (uint32_t thread = 0; thread < stride && thread < end; ++thread)
                    boundaries_to_check.push_back({thread + stride, end, stride});
        const unsigned unsigned_count = check_unsigned_division(unsigned_cases);
        const unsigned boundary_count = check_boundary_counts(boundaries_to_check);
        std::cout << "{\"passed\":true,\"integer_cases\":" << cases.size()
                  << ",\"scheduling_cases\":" << schedules
                  << ",\"unsigned_division_cases\":" << unsigned_count
                  << ",\"boundary_count_cases\":" << boundary_count << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
