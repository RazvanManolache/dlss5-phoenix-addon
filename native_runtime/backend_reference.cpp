// Portable correctness baseline for ARM/Unix hosts. No AVX or vendor SDK needed.
#include "../src/native/support/cpu_threads.h"
#include "backend.h"
#include "host_math.h"
#include "parallel_rows.h"
#include "platform.h"
#include <chrono>
#include <cstdlib>
using namespace phoenix_cpp;
class ReferenceBackend final : public Backend {
    RowExecutor rows_;

  public:
    explicit ReferenceBackend(unsigned threads) : rows_(phoenix_threads::resolve(threads)) {}
    const char *name() const override { return "cpu-reference-cpp"; }
    Allocation allocate(uint64_t bytes) override {
        demand(bytes && bytes <= SIZE_MAX - 63, "CPU allocation extent");
        auto *address = ::operator new(size_t(bytes), std::align_val_t(64));
        std::memset(address, 0, size_t(bytes));
        return {address, bytes, 0};
    }
    void release(Allocation allocation) override {
        ::operator delete(allocation.address, std::align_val_t(64));
    }
    void upload(Allocation allocation, const void *source, uint64_t bytes) override {
        demand(bytes <= allocation.bytes, "CPU upload bounds");
        std::memcpy(allocation.address, source, size_t(bytes));
    }
    void download(Allocation allocation, void *destination, uint64_t bytes) override {
        demand(bytes <= allocation.bytes, "CPU download bounds");
        std::memcpy(destination, allocation.address, size_t(bytes));
    }
    void execute(const std::vector<Command> &commands, std::vector<Event> &events) override {
        for (const auto &command : commands) {
            demand(command.operation.family != phx::WindowAttention,
                   "Reference backend requires unfused attention");
            demand(command.count <= UINT32_MAX, "CPU dispatch extent");
            auto started = std::chrono::steady_clock::now();
            rows_.run(unsigned(command.count),
                      [&](unsigned index) { phx::Kernel<HostMath>{command.operation}(index); });
            events.push_back({command.block, command.operation.family,
                              std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count(),
                              false, command.layer});
        }
    }
    void wait() override {}
};
extern "C" PHOENIX_EXPORT Backend *
phoenix_backend_create_threads(const char *device, unsigned threads, char *error, size_t capacity) {
    try {
        demand(device && std::string(device) == "cpu", "CPU identity required");
        return new ReferenceBackend(threads);
    } catch (const std::exception &exception) {
        copy_error(error, capacity, exception.what());
        return nullptr;
    }
}
extern "C" PHOENIX_EXPORT Backend *phoenix_backend_create(const char *device, const wchar_t *,
                                                          char *error, size_t capacity) {
    return phoenix_backend_create_threads(device, 0, error, capacity);
}
extern "C" PHOENIX_EXPORT void phoenix_backend_destroy(Backend *backend) { delete backend; }
extern "C" PHOENIX_EXPORT uint32_t phoenix_backend_features(Backend *) { return 0; }
