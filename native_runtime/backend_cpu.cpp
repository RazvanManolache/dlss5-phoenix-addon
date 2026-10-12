#include "backend.h"
#include "host_math.h"
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <intrin.h>
#include "../src/native/support/cpu_threads.h"
using namespace phoenix_cpp;
extern "C" void phoenix_cpu_avx2(const Command&, uint64_t);
extern "C" void phoenix_cpu_avx2_range(const Command&, uint64_t, uint64_t);
extern "C" void phoenix_cpu_avx512(const Command&, uint64_t);
bool has_avx2() {
    int cpuid_registers[4];
    __cpuid(cpuid_registers, 1);
    if ((cpuid_registers[2] & ((1 << 27) | (1 << 28) | (1 << 12) | (1 << 29))) !=
        ((1 << 27) | (1 << 28) | (1 << 12) | (1 << 29)))
        return false;
    if ((_xgetbv(0) & 6) != 6)
        return false;
    __cpuidex(cpuid_registers, 7, 0);
    return (cpuid_registers[1] & (1 << 5)) != 0;
}
bool has_avx512() {
    if (!has_avx2() || (_xgetbv(0) & 0xe6) != 0xe6)
        return false;
    int capabilities[4];
    __cpuidex(capabilities, 7, 0);
    constexpr unsigned required = (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
    return (unsigned(capabilities[1]) & required) == required;
}
class CpuBackend final : public Backend {
    std::mutex dispatch_mutex_;
    std::condition_variable work_ready_, work_finished_;
    std::vector<std::thread> worker_threads_;
    bool stopping_ = false, use_avx2_ = has_avx2(), use_avx512_ = has_avx512();
    uint64_t dispatch_generation_ = 0;
    size_t unfinished_workers_ = 0;
    std::atomic<uint64_t> next_work_item_{0};
    Command pending_command_{};
    bool use_wide_matrix(const phx::Operation& operation) const {
        // Wider tiles help the decoder MLPs, including their K=128 reductions.
        // Keep small attention tiles and pointwise work on the AVX2 path.
        return use_avx512_ && operation.family == phx::MatrixMultiply &&
               operation.reduction_count >= 128 && operation.column_count >= 128 &&
               operation.left_row_stride == operation.reduction_count;
    }
    // Sleep between graph commands; workers claim contiguous ranges to amortize
    // scheduling and operation dispatch without assigning fixed rows to a thread.
    void worker() {
        uint64_t observed_generation = 0;
        std::unique_lock lock(dispatch_mutex_);
        for (;;) {
            work_ready_.wait(
                lock, [&] { return stopping_ || observed_generation != dispatch_generation_; });
            if (stopping_)
                return;
            observed_generation = dispatch_generation_;
            auto command = pending_command_;
            lock.unlock();
            const bool wide_matrix = use_wide_matrix(command.operation);
            const uint64_t maximum_chunk = command.operation.family == phx::Elementwise ? 2048 : 32;
            const uint64_t minimum_chunk = command.operation.family == phx::MatrixMultiply ? 1 : 32;
            const uint64_t chunk = std::clamp<uint64_t>(
                (command.count + worker_threads_.size() * 4 - 1) / (worker_threads_.size() * 4),
                minimum_chunk, maximum_chunk);
            for (uint64_t first; (first = next_work_item_.fetch_add(chunk)) < command.count;) {
                const auto end = std::min(first + chunk, command.count);
                if (wide_matrix)
                    for (auto item = first; item < end; ++item)
                        phoenix_cpu_avx512(command, item);
                else if (use_avx2_)
                    phoenix_cpu_avx2_range(command, first, end);
                else
                    for (auto item = first; item < end; ++item)
                        phx::Kernel<HostMath>{command.operation}(item);
            }
            lock.lock();
            if (--unfinished_workers_ == 0)
                work_finished_.notify_one();
        }
    }

  public:
    explicit CpuBackend(unsigned count = 0) {
        auto thread_count = phoenix_threads::resolve(count);
        try {
            for (unsigned item_index = 0; item_index < thread_count; ++item_index)
                worker_threads_.emplace_back([this] { worker(); });
        } catch (...) {
            {
                std::lock_guard lock(dispatch_mutex_);
                stopping_ = true;
            }
            work_ready_.notify_all();
            for (auto& worker : worker_threads_)
                worker.join();
            throw;
        }
    }
    ~CpuBackend() {
        {
            std::lock_guard lock(dispatch_mutex_);
            stopping_ = true;
        }
        work_ready_.notify_all();
        for (auto& worker : worker_threads_)
            worker.join();
    }
    const char* name() const override {
        return "cpu-cpp";
    }
    Allocation allocate(uint64_t byte_count) override {
        demand(byte_count && byte_count <= size_t(-1), "CPU allocation extent");
        auto address = _aligned_malloc(size_t(byte_count), 64);
        if (!address)
            throw std::bad_alloc();
        std::memset(address, 0, size_t(byte_count));
        return {address, byte_count, 0};
    }
    void release(Allocation allocation) override {
        _aligned_free(allocation.address);
    }
    void upload(Allocation allocation, const void* source, uint64_t byte_count) override {
        demand(byte_count <= allocation.bytes, "Upload bounds");
        std::memcpy(allocation.address, source, size_t(byte_count));
    }
    void download(Allocation allocation, void* destination, uint64_t byte_count) override {
        demand(byte_count <= allocation.bytes, "Download bounds");
        std::memcpy(destination, allocation.address, size_t(byte_count));
    }
    // Workers claim disjoint output tiles. A command completes on every worker
    // before the next command is published, preserving graph dependencies.
    void execute(const std::vector<Command>& commands, std::vector<Event>& events) override {
        for (auto& command : commands) {
            auto started_at = std::chrono::steady_clock::now();
            {
                std::unique_lock lock(dispatch_mutex_);
                pending_command_ = command;
                if (use_avx2_ && command.operation.family == phx::MatrixMultiply) {
                    const bool wide_matrix = use_wide_matrix(command.operation);
                    const unsigned rows_per_tile = wide_matrix ? 16 : 8;
                    pending_command_.count =
                        uint64_t((command.operation.row_count + rows_per_tile - 1) /
                                 rows_per_tile) *
                        ((command.operation.column_count + 7) / 8) * command.operation.batch_count;
                }
                next_work_item_ = 0;
                unfinished_workers_ = worker_threads_.size();
                ++dispatch_generation_;
                work_ready_.notify_all();
                work_finished_.wait(lock, [&] { return unfinished_workers_ == 0; });
            }
            events.push_back({command.block, command.operation.family,
                              std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started_at)
                                  .count(),
                              false, command.layer});
        }
    }
    void wait() override {}
};
extern "C" __declspec(dllexport) Backend*
phoenix_backend_create(const char* device_id, const wchar_t*, char* error, size_t size) {
    try {
        demand(device_id && std::string(device_id) == "cpu", "Explicit CPU identity required");
        return new CpuBackend;
    } catch (const std::exception& exception) {
        strncpy_s(error, size, exception.what(), _TRUNCATE);
        return nullptr;
    }
}
extern "C" __declspec(dllexport) Backend*
phoenix_backend_create_threads(const char* device_id, unsigned count, char* error, size_t size) {
    try {
        demand(device_id && std::string(device_id) == "cpu", "Explicit CPU identity required");
        return new CpuBackend(count);
    } catch (const std::exception& exception) {
        strncpy_s(error, size, exception.what(), _TRUNCATE);
        return nullptr;
    }
}
extern "C" __declspec(dllexport) void phoenix_backend_destroy(Backend* backend) {
    delete backend;
}
