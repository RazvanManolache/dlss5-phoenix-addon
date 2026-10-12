// Adapts portable descriptors to the Vulkan resident-command ABI.
// Logical addresses encode buffer identity in high bits and byte offset in low bits; they are never dereferenced.

#include "backend.h"
#include "command_cache.h"
#include "backend_options.h"
#include <windows.h>
#include <filesystem>
#include <cstring>
#include <mutex>
using namespace phoenix_cpp;
class VulkanBackend final : public Backend {
    HMODULE module = nullptr;
    std::vector<int> live;
    std::unique_lock<std::mutex> ownership;
    std::vector<int> graphs;
    std::vector<Command> cached_commands;
    size_t submission_size = 256;
    bool detailed_profile = false;
    unsigned matrix_rows = 0;
    bool prefer_unfused_attention = false;
    static std::mutex& global() {
        static std::mutex m;
        return m;
    }
    template <class R, class... A> R call(const char* name, A... args) {
        auto callback = reinterpret_cast<R (*)(A...)>(GetProcAddress(module, name));
        demand(callback != nullptr, name);
        return callback(args...);
    }
    void check(int code) {
        if (code < 0)
            throw std::runtime_error(call<const char*>("xmx_error"));
    }
    static int buffer_id(void* logical_address) {
        return int(uintptr_t(logical_address) >> 40);
    }
    static uint32_t offset(void* logical_address, unsigned element_bytes) {
        auto byte_offset = uintptr_t(logical_address) & ((uint64_t(1) << 40) - 1);
        demand(byte_offset % element_bytes == 0 && byte_offset / element_bytes <= UINT32_MAX,
               "Vulkan operand offset");
        return uint32_t(byte_offset / element_bytes);
    }
    void clear_graphs() {
        for (int graph : graphs)
            check(call<int>("xmx_graph_destroy", graph));
        graphs.clear();
        cached_commands.clear();
    }

  public:
    VulkanBackend(const std::string& device, const std::filesystem::path& folder,
                  uint32_t options = 0)
        : ownership(global(), std::try_to_lock) {
        demand(ownership.owns_lock(), "Vulkan provider is single-session until context migration");
        demand(device.starts_with("vulkan:") && device.size() == 39, "Vulkan UUID required");
        struct Rollback {
            HMODULE& library_handle;
            bool done = false;
            ~Rollback() {
                if (library_handle && !done) {
                    auto stop = reinterpret_cast<void (*)()>(
                        GetProcAddress(library_handle, "xmx_shutdown"));
                    if (stop)
                        stop();
                    FreeLibrary(library_handle);
                    library_handle = nullptr;
                }
            }
        } rollback{module};
        module =
            LoadLibraryExW((folder / "phoenix_vulkan.dll").c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        demand(module != nullptr, "Vulkan provider load failed");
        auto shader_path = [&](const char* name) { return (folder / name).string(); };
        auto matrix_shader = shader_path("gemm_portable.spv");
        auto elementwise_shader = shader_path("resident.spv");
        auto attention_shader = shader_path("attention.spv");
        auto history_shader = shader_path("history.spv");
        auto matrix8 = shader_path("gemm_matrix8.spv"), matrix16 = shader_path("gemm_matrix16.spv");
        auto fp8_matrix = shader_path("gemm_fp832.spv");
        auto wide_matrix = shader_path("gemm_matrix16x4.spv");
        check(call<int>("phoenix_native_init_configured", device.substr(7).c_str(),
                        matrix_shader.c_str(), matrix8.c_str(), matrix16.c_str(),
                        elementwise_shader.c_str(), attention_shader.c_str(),
                        history_shader.c_str(), options, fp8_matrix.c_str(), wide_matrix.c_str()));
        matrix_rows = call<unsigned>("phoenix_native_matrix");
        // This tested RDNA2 iGPU advertises cooperative matrices, but its
        // fused attention shader is slower than separate ordered operators.
        prefer_unfused_attention = call<unsigned>("phoenix_native_vendor") == 0x1002 &&
                                   call<unsigned>("phoenix_native_device_id") == 0x13c0;
        if (matrix_rows) {
            auto window = shader_path(matrix_rows == 8 ? "phoenix_window_attention8.spv"
                                                       : "phoenix_window_attention.spv");
            check(call<int>("xmx_window_init", window.c_str(), 1u));
        }
        auto features_shader = shader_path("frame_features.spv");
        check(call<int>("phoenix_native_features_init", features_shader.c_str()));
        detailed_profile = (options & DispatchProfiling) != 0;
        check(call<int>("xmx_profile", detailed_profile ? 1 : 2));
        if (call<unsigned>("phoenix_native_vendor") == 0x10de)
            submission_size = 1024;
        rollback.done = true;
    }
    ~VulkanBackend() {
        if (module) {
            try {
                clear_graphs();
            } catch (...) {
            }
            auto stop = reinterpret_cast<void (*)()>(GetProcAddress(module, "xmx_shutdown"));
            if (stop)
                stop();
            FreeLibrary(module);
        }
    }
    uint32_t features() const {
        return DeviceFeaturePacking |
               (matrix_rows ? FusedWindowAttention | FusedMatrixResidual : 0) |
               (prefer_unfused_attention ? PreferUnfusedWindowAttention : 0);
    }
    const char* name() const override {
        return "vulkan-cpp";
    }
    Allocation allocate(uint64_t byte_count) override {
        demand(byte_count < (uint64_t(1) << 40), "Vulkan allocation extent");
        int key = call<int>("xmx_buf_create_kind", byte_count, 0);
        check(key);
        live.push_back(key);
        check(call<int>("xmx_buf_zero", key));
        return {reinterpret_cast<void*>(uint64_t(key) << 40), byte_count, uint64_t(key)};
    }
    void release(Allocation allocation) override {
        clear_graphs();
        check(call<int>("xmx_buf_destroy", int(allocation.token)));
    }
    void upload(Allocation allocation, const void* source, uint64_t byte_count) override {
        demand(byte_count <= allocation.bytes, "Vulkan upload bounds");
        check(call<int>("xmx_buf_upload", int(allocation.token), source,
                        uint64_t(offset(allocation.address, 1)), byte_count));
    }
    void download(Allocation allocation, void* destination, uint64_t byte_count) override {
        demand(byte_count <= allocation.bytes, "Vulkan download bounds");
        check(call<int>("xmx_buf_download", int(allocation.token), destination,
                        uint64_t(offset(allocation.address, 1)), byte_count));
    }
    void wait() override {} // xmx_submit is fenced; there is no outstanding work on return.
    void execute(const std::vector<Command>& commands, std::vector<Event>& events) override {
        if (commands.empty())
            return;
        if (commands.size() == 1 && commands.front().operation.family == phx::FrameFeatures) {
            const auto& operation = commands.front().operation;
            check(call<int>("xmx_begin"));
            try {
                call<void>(
                    "phoenix_native_operand_offsets", uint64_t(offset(operation.left_input, 1)),
                    uint64_t(offset(operation.right_input, 1)),
                    uint64_t(offset(operation.output, 1)), uint64_t(offset(operation.auxiliary, 1)),
                    uint64_t(offset(operation.secondary_output, 1)));
                check(call<int>("phoenix_native_features", buffer_id(operation.left_input),
                                buffer_id(operation.right_input), buffer_id(operation.auxiliary),
                                buffer_id(operation.secondary_output), buffer_id(operation.output),
                                operation.row_count, operation.column_count,
                                operation.reduction_count, operation.batch_count,
                                operation.left_batch_stride, operation.right_batch_stride,
                                operation.output_batch_stride, operation.left_row_stride,
                                operation.flags));
                check(call<int>("xmx_submit"));
            } catch (...) {
                call<int>("xmx_abort");
                throw;
            }
            return;
        }
        call<void>("xmx_profile_reset");
        try {
            if (!same_commands(commands, cached_commands)) {
                clear_graphs();
                demand((commands.size() + submission_size - 1) / submission_size <= 128,
                       "Vulkan graph capacity");
                check(call<int>("xmx_begin"));
                for (auto& command : commands) {
                    auto operation = command.operation;
                    if (operation.family == phx::MatrixMultiply) {
                        if (operation.flags & phx::MatrixResidual)
                            call<void>("phoenix_native_matrix_residual",
                                       buffer_id(operation.auxiliary),
                                       buffer_id(operation.secondary_output),
                                       uint64_t(offset(operation.auxiliary, 1)),
                                       uint64_t(offset(operation.secondary_output, 1)));
                        check(call<int>(
                            "xmx_rec_gemm", buffer_id(operation.left_input),
                            buffer_id(operation.right_input), buffer_id(operation.output),
                            operation.row_count, operation.column_count, operation.reduction_count,
                            operation.batch_count, operation.left_batch_stride,
                            operation.right_batch_stride, operation.output_batch_stride,
                            operation.flags, operation.left_row_stride, operation.right_row_stride,
                            operation.output_row_stride, offset(operation.left_input, 2),
                            offset(operation.right_input, (operation.flags & 0x2000000) ? 1 : 2),
                            offset(operation.output, (operation.flags & 0x1000) ? 2 : 4)));
                    } else {
                        call<void>("phoenix_native_operand_offsets",
                                   uint64_t(offset(operation.left_input, 1)),
                                   uint64_t(offset(operation.right_input, 1)),
                                   uint64_t(offset(operation.output, 1)),
                                   uint64_t(offset(operation.auxiliary, 1)),
                                   uint64_t(offset(operation.secondary_output, 1)));
                        if (operation.family == phx::WindowAttention)
                            check(call<int>(
                                "phoenix_native_window_attention", buffer_id(operation.left_input),
                                buffer_id(operation.right_input), buffer_id(operation.auxiliary),
                                buffer_id(operation.secondary_output), buffer_id(operation.output),
                                operation.batch_count, operation.column_count));
                        else if (operation.family == phx::RowReduction)
                            check(call<int>(
                                "xmx_rec_row", operation.flags, buffer_id(operation.left_input),
                                buffer_id(operation.right_input), buffer_id(operation.output),
                                buffer_id(operation.auxiliary), operation.row_count,
                                operation.column_count, operation.batch_count,
                                operation.reduction_count, operation.left_batch_stride, 0.f));
                        else if (operation.secondary_output)
                            check(call<int>(
                                "xmx_rec_unary2", operation.flags, buffer_id(operation.left_input),
                                buffer_id(operation.right_input), buffer_id(operation.output),
                                buffer_id(operation.auxiliary),
                                buffer_id(operation.secondary_output), uint32_t(command.count),
                                operation.column_count, operation.scale, operation.batch_count,
                                operation.left_batch_stride, operation.right_batch_stride,
                                operation.output_batch_stride, operation.reduction_count));
                        else
                            check(call<int>(
                                "xmx_rec_unary", operation.flags, buffer_id(operation.left_input),
                                buffer_id(operation.right_input), buffer_id(operation.output),
                                buffer_id(operation.auxiliary), uint32_t(command.count),
                                operation.column_count, operation.scale, operation.batch_count,
                                operation.left_batch_stride, operation.right_batch_stride,
                                operation.output_batch_stride, operation.reduction_count));
                    }
                    // Keep submission bounds below the Windows watchdog risk; all dependencies
                    // cross a fenced boundary. Later device validation can increase this batch.
                    if (&command == &commands.back() ||
                        ((&command - commands.data() + 1) % submission_size) == 0) {
                        auto graph = call<int>("xmx_graph_capture");
                        check(graph);
                        graphs.push_back(graph);
                        if (&command != &commands.back())
                            check(call<int>("xmx_begin"));
                    }
                }
                cached_commands = commands;
            }
            for (int graph : graphs)
                check(call<int>("xmx_graph_run", graph));
            auto event_count = call<unsigned>("xmx_profile_each_count");
            demand(event_count == (detailed_profile ? commands.size() : graphs.size()),
                   "Incomplete Vulkan profiling");
            auto kind = reinterpret_cast<unsigned (*)(unsigned)>(
                GetProcAddress(module, "xmx_profile_each_kind"));
            auto milliseconds = reinterpret_cast<double (*)(unsigned)>(
                GetProcAddress(module, "xmx_profile_each_ms"));
            demand(kind && milliseconds, "Vulkan profiling ABI");
            for (unsigned event_index = 0; event_index < event_count; ++event_index)
                events.push_back({detailed_profile ? commands[event_index].block : UINT32_MAX,
                                  kind(event_index), milliseconds(event_index), true,
                                  detailed_profile ? commands[event_index].layer : UINT32_MAX});
        } catch (...) {
            call<int>("xmx_abort");
            clear_graphs();
            throw;
        }
    }
};
extern "C" __declspec(dllexport) Backend*
phoenix_backend_create(const char* device_id, const wchar_t* root, char* error, size_t size) {
    try {
        return new VulkanBackend(device_id ? device_id : "", root);
    } catch (const std::exception& exception) {
        strncpy_s(error, size, exception.what(), _TRUNCATE);
        return nullptr;
    }
}
extern "C" __declspec(dllexport) void phoenix_backend_destroy(Backend* backend) {
    delete backend;
}

extern "C" __declspec(dllexport) Backend*
phoenix_backend_create_configured(const char* device_id, const wchar_t* root, uint32_t options,
                                  char* error, size_t size) {
    try {
        return new VulkanBackend(device_id ? device_id : "", root, options);
    } catch (const std::exception& exception) {
        strncpy_s(error, size, exception.what(), _TRUNCATE);
        return nullptr;
    }
}
extern "C" __declspec(dllexport) uint32_t phoenix_backend_features(Backend* backend) {
    return static_cast<VulkanBackend*>(backend)->features();
}
