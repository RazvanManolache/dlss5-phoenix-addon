// Model-free address/order replay. Deliberately never dereference the computed
// tensor addresses: signed-stride and carry fixtures may name unmapped memory.
#include <cuda_runtime.h>
#include "../../src/native/kernels/production/common/window_math.cuh"
#include "../../src/native/kernels/production/common/boundary_clear.cuh"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void checked(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template <class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { checked(cudaMalloc(&data, count * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
};

constexpr unsigned address_capacity = 128;
struct ClearCase {
    BoundaryVectorLayout layout;
    uint32_t width, first, extent, stride;
};
struct ClearResult {
    uint64_t addresses[address_capacity];
    uint32_t count, overflow;
};
struct RecordStore {
    ClearResult* result;
    __device__ __forceinline__ void operator()(uint64_t address) const {
        const unsigned index = result->count++;
        if (index < address_capacity)
            result->addresses[index] = address;
        else
            ++result->overflow;
    }
};
template <unsigned ComponentBits>
__global__ void clear_cases(const ClearCase* cases, ClearResult* output, unsigned count) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const auto item = cases[index];
        output[index].count = output[index].overflow = 0;
        clear_boundary_rows<ComponentBits>(item.layout, item.width, item.first,
                                          item.extent, item.stride, RecordStore{output + index});
    }
}

// Independent byte-domain reference. Preserve uint32 origin wrap, and scale
// each signed term before accumulating instead of calling production helpers.
static uint64_t reference_address(const ClearCase& item, uint32_t index, unsigned bits) {
    const uint32_t spatial = index / (1u << bits);
    const uint32_t row = spatial / item.width;
    const uint32_t column = item.layout.column_origin + spatial % item.width;
    const uint32_t component = index % (1u << bits);
    return item.layout.base_address +
        uint64_t(int64_t(int32_t(component)) * int64_t(int32_t(item.layout.component_stride_words))) * 4u +
        uint64_t(int64_t(int32_t(row)) * int64_t(int32_t(item.layout.row_stride_words))) * 4u +
        uint64_t(int64_t(int32_t(column))) * 16u;
}

template <unsigned ComponentBits>
static unsigned check_cases(const std::vector<ClearCase>& cases) {
    DeviceBuffer<ClearCase> input(cases.size());
    DeviceBuffer<ClearResult> output(cases.size());
    checked(cudaMemcpy(input.data, cases.data(), cases.size() * sizeof(ClearCase), cudaMemcpyHostToDevice));
    clear_cases<ComponentBits><<<(unsigned(cases.size()) + 127) / 128, 128>>>(
        input.data, output.data, unsigned(cases.size()));
    checked(cudaGetLastError());
    std::vector<ClearResult> actual(cases.size());
    checked(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(ClearResult), cudaMemcpyDeviceToHost));
    unsigned stores = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto item = cases[i];
        unsigned count = 0;
        // Enumerate scalar work directly, without the production tail/group
        // decomposition. Compare every word, including per-thread store order.
        for (uint32_t index = item.first; index < item.extent; index += item.stride) {
            const uint64_t address = reference_address(item, index, ComponentBits);
            for (unsigned word = 0; word < 4; ++word) {
                if (count >= address_capacity || actual[i].addresses[count] != address + 4u * word)
                    throw std::runtime_error("Boundary address/order mismatch at case " + std::to_string(i));
                ++count;
            }
        }
        if (actual[i].overflow || actual[i].count != count)
            throw std::runtime_error("Boundary store count/guard changed");
        stores += count;
    }
    return stores;
}

int main() {
    try {
        std::vector<ClearCase> cases;
        for (uint32_t stride : {1u, 17u, 32u, 64u, 128u, 256u})
            for (uint32_t iterations = 1; iterations <= 13; ++iterations)
                for (uint32_t width : {1u, 3u, 7u, 17u, 801u})
                    for (uint32_t first : {0u, stride / 2, stride - 1}) {
                        const uint32_t extent = stride * iterations + (iterations % 3) - 1u;
                        if (first < extent)
                            cases.push_back({{0xfffffff0ull, 0x40000000u, 4096u, 3u},
                                             width, first, extent, stride});
                    }
        uint32_t random = 0xb31a496du;
        for (unsigned i = 0; i < 10'000; ++i) {
            auto next = [&]() { return random = random * 1664525u + 1013904223u; };
            const uint32_t stride = next() % 256u + 1u;
            const uint32_t first = next() % stride;
            const uint32_t extent = first + stride * (next() % 12u) + 1u;
            const uint32_t width = next() % 1024u + 1u;
            const uint32_t low = next(), high = next();
            const uint32_t component_stride = next(), row_stride = next(), origin = next();
            cases.push_back({{combine_words(low, high), component_stride, row_stride, origin},
                             width, first, extent, stride});
        }
        const unsigned stores = check_cases<2>(cases) + check_cases<3>(cases) +
            check_cases<4>(cases) + check_cases<5>(cases) + check_cases<6>(cases);
        std::cout << "{\"passed\":true,\"boundary_clear_cases\":" << cases.size() * 5
                  << ",\"boundary_store_checks\":" << stores << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
