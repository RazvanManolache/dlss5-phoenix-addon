// Adapts portable descriptors to the Vulkan resident-command ABI.
// Logical addresses encode buffer identity in high bits and byte offset in low bits; they are never dereferenced.

#include "backend.h"
#include "command_cache.h"
#include "backend_options.h"
#include "platform.h"
#include <filesystem>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
using namespace phoenix_cpp;
class VulkanBackend final : public Backend {
    LibraryHandle module = nullptr;
    std::vector<int> live;
    std::unique_lock<std::mutex> ownership;
    std::vector<int> graphs;
    std::vector<Command> cached_commands;
    size_t submission_size = 256;
    bool detailed_profile = false;
    unsigned matrix_rows = 0;
    bool scalar_attention = false;
    bool prefer_unfused_attention = false;
    bool fused_qkv = false;
    bool cache_noise = false;
    bool workgroup_fp8 = false, fused_feed_forward = false, projected_qkv = false;
    bool shape_tiles = false;
    bool workgroup_compatibility = false;
    bool fast_math = false;
    std::unordered_map<int, uint64_t> allocation_extents;
    std::unordered_set<int> native_fp8_weights;
    bool packed_weights_supported(void* address) const {
        return workgroup_fp8 && native_fp8_weights.contains(buffer_id(address));
    }
    static std::mutex& global() {
        static std::mutex m;
        return m;
    }
    template <class R, class... A> R call(const char* name, A... args) {
        auto callback = reinterpret_cast<R (*)(A...)>(library_symbol(module, name));
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
            LibraryHandle& library_handle;
            bool done = false;
            ~Rollback() {
                if (library_handle && !done) {
                    auto stop = reinterpret_cast<void (*)()>(
                        library_symbol(library_handle, "xmx_shutdown"));
                    if (stop)
                        stop();
                    close_library(library_handle);
                    library_handle = nullptr;
                }
            }
        } rollback{module};
        module = open_library(folder / library_name("phoenix_vulkan"));
        demand(module != nullptr, "Vulkan provider load failed");
        auto shader_path = [&](const char* name) { return (folder / name).string(); };
        fast_math = (options & VulkanFastMath) != 0;
        workgroup_compatibility = fast_math || (options & VulkanWorkgroupCompatibility) != 0;
        if (workgroup_compatibility)
            options = (options & (DispatchProfiling | OrderedAttention | VulkanTiledFallback |
                                  VulkanCoalescedFallback)) |
                      ScalarMatrices | VulkanWorkgroupCompatibility;
        auto matrix_shader = shader_path(fast_math
            ? (options & VulkanTiledFallback ? "gemm_portable_tiled_fast_workgroup.spv"
                                            : "gemm_portable_fast_workgroup.spv")
            : workgroup_compatibility
            ? (options & VulkanTiledFallback ? "gemm_portable_tiled_workgroup.spv"
                                            : "gemm_portable_workgroup.spv")
            : (options & VulkanTiledFallback ? "gemm_portable_tiled.spv" : "gemm_portable.spv"));
        auto elementwise_shader = shader_path(fast_math ? "resident_fast_workgroup.spv"
                                            : workgroup_compatibility ? "resident_workgroup.spv"
                                                                    : "resident.spv");
        auto attention_shader = shader_path(fast_math ? "attention_fast_workgroup.spv"
                                          : workgroup_compatibility ? "attention_workgroup.spv"
                                                                  : "attention.spv");
        auto history_shader = shader_path("history.spv");
        auto matrix8 = shader_path("gemm_matrix8.spv"), matrix16 = shader_path("gemm_matrix16.spv");
        auto fp8_matrix = shader_path("gemm_fp832.spv");
        auto wide_matrix = shader_path("gemm_matrix16x4.spv");
        check(call<int>("phoenix_native_init_configured", device.substr(7).c_str(),
                        matrix_shader.c_str(), matrix8.c_str(), matrix16.c_str(),
                        elementwise_shader.c_str(), attention_shader.c_str(),
                        history_shader.c_str(), options, fp8_matrix.c_str(), wide_matrix.c_str()));
        // The provider routes cooperative row reductions only after this
        // optional pipeline is initialized. Merely shipping the SPIR-V leaves
        // g.rrows null and silently selects the 32-lane scalar fallback.
        auto parallel_rows = shader_path(fast_math ? "attention_parallel_fast_workgroup.spv"
                                       : workgroup_compatibility ? "attention_parallel_workgroup.spv"
                                                               : "attention_rows.spv");
        check(call<int>("xmx_rows_init", parallel_rows.c_str()));
        matrix_rows = call<unsigned>("phoenix_native_matrix");
        scalar_attention = !matrix_rows && (options & VulkanScalarAttention);
        if (!matrix_rows && (options & VulkanTiledFallback) &&
            (options & VulkanCoalescedFallback)) {
            auto coalesced_shader = shader_path(fast_math
                ? "gemm_portable_coalesced_fast_workgroup.spv" : workgroup_compatibility
                ? "gemm_portable_coalesced_workgroup.spv" : "gemm_portable_coalesced.spv");
            check(call<int>("phoenix_native_coalesced_init", coalesced_shader.c_str()));
        }
        const bool shape_matrices =
            (options & VulkanShapeMatrices) && call<unsigned>("phoenix_native_vendor") == 0x10de;
        if (shape_matrices || (options & (VulkanWideColumns32 | VulkanWideColumns64))) {
            const unsigned columns = options & VulkanWideColumns64 ? 64 : 32;
            const auto wide = shader_path(columns == 64 ? "gemm_wide64.spv" : "gemm_wide32.spv");
            check(call<int>("phoenix_native_wide_init", wide.c_str(), columns));
        }
        // This tested RDNA2 iGPU advertises cooperative matrices, but its
        // fused attention shader is slower than separate ordered operators.
        prefer_unfused_attention = call<unsigned>("phoenix_native_vendor") == 0x1002 &&
                                   call<unsigned>("phoenix_native_device_id") == 0x13c0;
        if (matrix_rows || scalar_attention) {
            auto window = shader_path(scalar_attention   ? "phoenix_window_attention_scalar.spv"
                                      : matrix_rows == 8 ? "phoenix_window_attention8.spv"
                                                         : "phoenix_window_attention.spv");
            check(call<int>("xmx_window_init", window.c_str(), 1u));
        }
        auto features_shader = shader_path(fast_math ? "frame_features_fast_workgroup.spv"
                                         : workgroup_compatibility ? "frame_features_workgroup.spv"
                                                                 : "frame_features.spv");
        check(call<int>("phoenix_native_features_init", features_shader.c_str()));
        // Other vendors retain their measured scheduling and memory policy.
        const bool tested_nvidia = call<unsigned>("phoenix_native_vendor") == 0x10de;
        fused_qkv = tested_nvidia && (options & VulkanFusedQkv);
        cache_noise = tested_nvidia && (options & VulkanCacheNoise);
        if (fused_qkv)
            check(call<int>("xmx_qkv_init", shader_path("phoenix_qkv_prepare.spv").c_str()));
        detailed_profile = (options & DispatchProfiling) != 0;
        check(call<int>("xmx_profile", detailed_profile ? 1 : 2));
        if (call<unsigned>("phoenix_native_vendor") == 0x10de)
            submission_size = 1024;
        workgroup_fp8 = call<unsigned>("phoenix_native_matrix2") != 0;
        shape_tiles = workgroup_fp8 && (options & VulkanWorkgroupShapeTiles);
        fused_feed_forward = workgroup_fp8 && (options & VulkanFusedFeedForward);
        projected_qkv = workgroup_fp8 && (options & VulkanProjectedQkv);
        if (workgroup_fp8)
            check(call<int>("xmx_matrix2_init", 0u, shader_path("phoenix_matrix2.spv").c_str()));
        if (fused_feed_forward)
            check(call<int>("xmx_matrix2_init", 1u,
                            shader_path("phoenix_feed_forward_matrix2.spv").c_str()));
        if (projected_qkv)
            check(
                call<int>("xmx_matrix2_init", 2u, shader_path("phoenix_project_qkv.spv").c_str()));
        rollback.done = true;
    }
    ~VulkanBackend() {
        if (module) {
            try {
                clear_graphs();
            } catch (...) {
            }
            auto stop = reinterpret_cast<void (*)()>(library_symbol(module, "xmx_shutdown"));
            if (stop)
                stop();
            close_library(module);
        }
    }
    uint32_t features() const {
        return DeviceFeaturePacking | (matrix_rows || scalar_attention ? FusedWindowAttention : 0) |
               (matrix_rows ? FusedMatrixResidual : 0) |
               (prefer_unfused_attention ? PreferUnfusedWindowAttention : 0) |
               (cache_noise ? DeviceNoiseCache : 0) |
               (fused_qkv && !detailed_profile ? FusedQkvPreparation : 0) |
               (workgroup_fp8 ? WorkgroupFp8Matrices : 0) |
               (fused_feed_forward && !detailed_profile ? FusedFeedForward : 0) |
               (projected_qkv && !detailed_profile ? ProjectedQkvPreparation : 0) |
               (shape_tiles ? WorkgroupShapeTiles : 0) |
               (workgroup_compatibility ? WorkgroupCompatibility : 0) |
               (fast_math ? ApproximateMath : 0);
    }
    const char* name() const override {
        return "vulkan-cpp";
    }
    Allocation allocate(uint64_t byte_count) override {
        demand(byte_count < (uint64_t(1) << 40), "Vulkan allocation extent");
        int key = call<int>("xmx_buf_create_kind", byte_count, 0);
        check(key);
        live.push_back(key);
        allocation_extents[key] = byte_count;
        check(call<int>("xmx_buf_zero", key));
        return {reinterpret_cast<void*>(uint64_t(key) << 40), byte_count, uint64_t(key)};
    }
    void release(Allocation allocation) override {
        clear_graphs();
        check(call<int>("xmx_buf_destroy", int(allocation.token)));
        allocation_extents.erase(int(allocation.token));
        native_fp8_weights.erase(int(allocation.token));
    }
    void upload(Allocation allocation, const void* source, uint64_t byte_count) override {
        demand(byte_count <= allocation.bytes, "Vulkan upload bounds");
        check(call<int>("xmx_buf_upload", int(allocation.token), source,
                        uint64_t(offset(allocation.address, 1)), byte_count));
        if (workgroup_fp8) {
            // Legacy packed weights use 0x7f/0xff for +/-480; hardware E4M3
            // treats those codes as NaN. Prove byte compatibility once at
            // complete upload, then keep only a buffer-ID marker. Partial
            // uploads invalidate it and any graph that selected this engine.
            const int id = int(allocation.token);
            bool compatible =
                offset(allocation.address, 1) == 0 && allocation_extents.at(id) == byte_count;
            if (compatible) {
                const auto* bytes = static_cast<const uint8_t*>(source);
                for (uint64_t index = 0; index < byte_count; ++index)
                    if ((bytes[index] & 127u) == 127u) {
                        compatible = false;
                        break;
                    }
            }
            const bool previous = native_fp8_weights.contains(id);
            if (previous != compatible)
                clear_graphs();
            if (compatible)
                native_fp8_weights.insert(id);
            else
                native_fp8_weights.erase(id);
        }
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
                for (size_t command_index = 0; command_index < commands.size(); ++command_index) {
                    const auto& command = commands[command_index];
                    auto operation = command.operation;
                    bool combined_qkv = false;
                    // Complete 32-channel projection/preparation group. Aligned
                    // row groups avoid partial tensor stores in the driver;
                    // other shapes retain individually bounded operations.
                    if (projected_qkv && !detailed_profile &&
                        packed_weights_supported(operation.right_input) &&
                        operation.family == phx::MatrixMultiply && operation.row_count &&
                        operation.row_count % 32 == 0 && operation.reduction_count == 32 &&
                        operation.column_count == 96 && operation.batch_count == 1 &&
                        operation.flags == (1u | phx::RightInputFp8 | phx::LeftInputFp8Grid) &&
                        operation.left_row_stride == 32 && operation.right_row_stride == 32 &&
                        operation.output_row_stride == 96 && command_index + 3 < commands.size() &&
                        command_index % submission_size < submission_size - 3) {
                        const auto& query = commands[command_index + 1].operation;
                        const auto& key = commands[command_index + 2].operation;
                        const auto& value = commands[command_index + 3].operation;
                        if (query.family == phx::RowReduction &&
                            query.flags == (phx::StoreHalf | phx::InterleavedQkv) &&
                            key.family == phx::RowReduction &&
                            key.flags == (phx::StoreHalf | phx::InterleavedQkv | (1u << 18)) &&
                            value.family == phx::Elementwise &&
                            value.flags == (phx::SplitQkvHeads | phx::StoreHalf | (5u << 8)) &&
                            query.left_input == operation.output &&
                            key.left_input == operation.output &&
                            value.left_input == operation.output &&
                            query.row_count == operation.row_count &&
                            key.row_count == operation.row_count &&
                            value.row_count == operation.row_count && query.batch_count == 1 &&
                            key.batch_count == 1 && value.batch_count == 1 &&
                            query.column_count == 64 && key.column_count == 64 &&
                            value.column_count == 32 && query.reduction_count == 1 &&
                            key.reduction_count == 0 && query.auxiliary == key.auxiliary &&
                            value.left_batch_stride == 64 && value.right_batch_stride == 2) {
                            uint64_t args[] = {uintptr_t(operation.left_input),
                                               uintptr_t(operation.right_input),
                                               uintptr_t(query.output),
                                               uintptr_t(key.output),
                                               uintptr_t(value.output),
                                               uintptr_t(query.auxiliary),
                                               operation.row_count};
                            check(call<int>("xmx_matrix2_fused_record", 2u, args));
                            command_index += 3;
                            combined_qkv = true;
                        }
                    }

                    // Match a complete local Q/K/V group. Global and split-head
                    // blocks have different rounding rules and keep separate passes.
                    // Never cross a fenced graph boundary, or hide dispatch profiling.
                    if (!combined_qkv && fused_qkv && !detailed_profile &&
                        operation.family == phx::RowReduction &&
                        operation.flags == (phx::StoreHalf | phx::InterleavedQkv) &&
                        operation.reduction_count == 1 && operation.column_count == 64 &&
                        operation.batch_count && operation.batch_count != 16 &&
                        command_index + 2 < commands.size() &&
                        command_index % submission_size < submission_size - 2) {
                        const auto& key = commands[command_index + 1].operation;
                        const auto& value_command = commands[command_index + 2];
                        const auto& value = value_command.operation;
                        if (key.family == phx::RowReduction &&
                            key.flags == (phx::StoreHalf | phx::InterleavedQkv | (1u << 18)) &&
                            key.reduction_count == 0 && key.left_input == operation.left_input &&
                            key.auxiliary == operation.auxiliary &&
                            key.row_count == operation.row_count && key.column_count == 64 &&
                            key.batch_count == operation.batch_count &&
                            value.family == phx::Elementwise &&
                            value.flags == (phx::SplitQkvHeads | phx::StoreHalf | (5u << 8)) &&
                            value.left_input == operation.left_input &&
                            value.row_count == operation.row_count &&
                            value.column_count == operation.batch_count * 32 &&
                            value.batch_count == operation.batch_count &&
                            value.left_batch_stride == 64 && value.right_batch_stride == 2 &&
                            value_command.count == uint64_t(operation.row_count) * 32) {
                            call<void>("phoenix_native_operand_offsets",
                                       uint64_t(offset(operation.left_input, 1)),
                                       uint64_t(offset(operation.output, 1)),
                                       uint64_t(offset(key.output, 1)),
                                       uint64_t(offset(value.output, 1)),
                                       uint64_t(offset(operation.auxiliary, 1)));
                            check(call<int>("xmx_rec_qkv_prepare", buffer_id(operation.left_input),
                                            buffer_id(operation.output), buffer_id(key.output),
                                            buffer_id(value.output), buffer_id(operation.auxiliary),
                                            operation.row_count, 64u, operation.batch_count));
                            command_index += 2;
                            combined_qkv = true;
                        }
                    }
                    // Hide only a dead expansion output, within one submission.
                    // Both packed matrices must have native-compatible bytes.
                    if (fused_feed_forward && !combined_qkv && !detailed_profile &&
                        packed_weights_supported(operation.right_input) &&
                        operation.family == phx::MatrixMultiply &&
                        operation.reduction_count == 32 && operation.column_count == 128 &&
                        operation.batch_count == 1 && operation.row_count &&
                        operation.row_count % 32 == 0 && operation.right_row_stride == 32 &&
                        ((operation.flags >> 8) & 15u) == phx::PublishFusedGateFp8 &&
                        (operation.flags & phx::LeftInputFp8Grid) &&
                        (operation.flags & phx::RightInputFp8) &&
                        operation.flags == (1u | phx::StoreHalf | (phx::PublishFusedGateFp8 << 8) |
                                            phx::RightInputFp8 | phx::LeftInputFp8Grid) &&
                        operation.left_row_stride == 32 && command_index + 1 < commands.size() &&
                        ((command_index + 1) % submission_size) != 0) {
                        const auto& contract = commands[command_index + 1].operation;
                        if (contract.family == phx::MatrixMultiply &&
                            contract.left_input == operation.output &&
                            contract.row_count == operation.row_count &&
                            contract.column_count == 32 && contract.reduction_count == 128 &&
                            contract.batch_count == 1 && contract.left_row_stride == 128 &&
                            contract.right_row_stride == 128 &&
                            packed_weights_supported(contract.right_input) &&
                            (contract.flags & phx::MatrixResidual) &&
                            (contract.flags & phx::RightInputFp8) && (contract.flags & 1u) &&
                            ((contract.flags >> 8) & 15u) == phx::KeepFloat &&
                            !(contract.flags & phx::StoreHalf) &&
                            contract.output_row_stride == 32) {
                            uint64_t args[] = {uintptr_t(operation.left_input),
                                               uintptr_t(operation.right_input),
                                               uintptr_t(contract.output),
                                               uintptr_t(contract.right_input),
                                               uintptr_t(contract.auxiliary),
                                               uintptr_t(contract.secondary_output),
                                               operation.row_count,
                                               operation.left_row_stride,
                                               contract.output_row_stride,
                                               contract.flags};
                            check(call<int>("xmx_matrix2_fused_record", 1u, args));
                            ++command_index;
                            combined_qkv = true;
                        }
                    }
                    if (!combined_qkv) {
                        if (operation.family == phx::MatrixMultiply) {
                            if (operation.flags & phx::MatrixResidual) {
                                demand(
                                    matrix_rows != 0,
                                    "Fused matrix residual requires Vulkan cooperative matrices");
                                call<void>("phoenix_native_matrix_residual",
                                           buffer_id(operation.auxiliary),
                                           buffer_id(operation.secondary_output),
                                           uint64_t(offset(operation.auxiliary, 1)),
                                           uint64_t(offset(operation.secondary_output, 1)));
                            }
                            check(call<int>(
                                "xmx_rec_gemm", buffer_id(operation.left_input),
                                buffer_id(operation.right_input), buffer_id(operation.output),
                                operation.row_count, operation.column_count,
                                operation.reduction_count, operation.batch_count,
                                operation.left_batch_stride, operation.right_batch_stride,
                                operation.output_batch_stride,
                                operation.flags |
                                    (packed_weights_supported(operation.right_input) ? 0x20000000u
                                                                                     : 0u),
                                operation.left_row_stride, operation.right_row_stride,
                                operation.output_row_stride, offset(operation.left_input, 2),
                                offset(operation.right_input,
                                       (operation.flags & 0x2000000) ? 1 : 2),
                                offset(operation.output, (operation.flags & 0x1000) ? 2 : 4)));
                        } else {
                            call<void>("phoenix_native_operand_offsets",
                                       uint64_t(offset(operation.left_input, 1)),
                                       uint64_t(offset(operation.right_input, 1)),
                                       uint64_t(offset(operation.output, 1)),
                                       uint64_t(offset(operation.auxiliary, 1)),
                                       uint64_t(offset(operation.secondary_output, 1)));
                            if (operation.family == phx::WindowAttention)
                                check(call<int>("phoenix_native_window_attention",
                                                buffer_id(operation.left_input),
                                                buffer_id(operation.right_input),
                                                buffer_id(operation.auxiliary),
                                                buffer_id(operation.secondary_output),
                                                buffer_id(operation.output), operation.batch_count,
                                                operation.column_count));
                            else if (operation.family == phx::RowReduction)
                                check(call<int>(
                                    "xmx_rec_row", operation.flags, buffer_id(operation.left_input),
                                    buffer_id(operation.right_input), buffer_id(operation.output),
                                    buffer_id(operation.auxiliary), operation.row_count,
                                    operation.column_count, operation.batch_count,
                                    operation.reduction_count, operation.left_batch_stride, 0.f));
                            else if (operation.secondary_output)
                                check(call<int>(
                                    "xmx_rec_unary2", operation.flags,
                                    buffer_id(operation.left_input),
                                    buffer_id(operation.right_input), buffer_id(operation.output),
                                    buffer_id(operation.auxiliary),
                                    buffer_id(operation.secondary_output), uint32_t(command.count),
                                    operation.column_count, operation.scale, operation.batch_count,
                                    operation.left_batch_stride, operation.right_batch_stride,
                                    operation.output_batch_stride, operation.reduction_count));
                            else
                                check(call<int>(
                                    "xmx_rec_unary", operation.flags,
                                    buffer_id(operation.left_input),
                                    buffer_id(operation.right_input), buffer_id(operation.output),
                                    buffer_id(operation.auxiliary), uint32_t(command.count),
                                    operation.column_count, operation.scale, operation.batch_count,
                                    operation.left_batch_stride, operation.right_batch_stride,
                                    operation.output_batch_stride, operation.reduction_count));
                        }
                    }
                    // Keep submission bounds below the Windows watchdog risk; all dependencies
                    // cross a fenced boundary. Later device validation can increase this batch.
                    if (command_index + 1 == commands.size() ||
                        (command_index + 1) % submission_size == 0) {
                        auto graph = call<int>("xmx_graph_capture");
                        check(graph);
                        graphs.push_back(graph);
                        if (command_index + 1 != commands.size())
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
                library_symbol(module, "xmx_profile_each_kind"));
            auto milliseconds = reinterpret_cast<double (*)(unsigned)>(
                library_symbol(module, "xmx_profile_each_ms"));
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
// Existing Windows factories keep their wide path ABI. Unix callers can use
// the explicit UTF-8 factory; tests find shaders beside this backend library.
static std::filesystem::path shader_folder(const wchar_t* root) {
#ifdef _WIN32
    demand(root != nullptr, "Vulkan shader folder required");
    return root;
#else
    Dl_info info{};
    demand(dladdr(reinterpret_cast<const void*>(&shader_folder), &info) != 0,
           "Vulkan backend location unavailable");
    return std::filesystem::path(info.dli_fname).parent_path() / "vulkan";
#endif
}
extern "C" PHOENIX_EXPORT Backend* phoenix_backend_create_utf8(const char* device_id,
                                                               const char* root, uint32_t options,
                                                               char* error, size_t size) {
    try {
        demand(root != nullptr, "Vulkan shader folder required");
        return new VulkanBackend(device_id ? device_id : "", std::filesystem::u8path(root),
                                 options);
    } catch (const std::exception& exception) {
        copy_error(error, size, exception.what());
        return nullptr;
    }
}
extern "C" PHOENIX_EXPORT Backend*
phoenix_backend_create(const char* device_id, const wchar_t* root, char* error, size_t size) {
    try {
        return new VulkanBackend(device_id ? device_id : "", shader_folder(root));
    } catch (const std::exception& exception) {
        copy_error(error, size, exception.what());
        return nullptr;
    }
}
extern "C" PHOENIX_EXPORT void phoenix_backend_destroy(Backend* backend) {
    delete backend;
}

extern "C" PHOENIX_EXPORT Backend* phoenix_backend_create_configured(const char* device_id,
                                                                     const wchar_t* root,
                                                                     uint32_t options, char* error,
                                                                     size_t size) {
    try {
        return new VulkanBackend(device_id ? device_id : "", shader_folder(root), options);
    } catch (const std::exception& exception) {
        copy_error(error, size, exception.what());
        return nullptr;
    }
}
extern "C" PHOENIX_EXPORT uint32_t phoenix_backend_features(Backend* backend) {
    return static_cast<VulkanBackend*>(backend)->features();
}
