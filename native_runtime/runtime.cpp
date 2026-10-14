// Owns one selected backend, immutable graph, frame preparation and per-pass history.
// A frame commits its new histories only after inference/resolve and finite-value checks complete.

#include "platform.h"
#include "graph.h"
#include "frame.h"
#include "thread_config.h"
#include "backend_options.h"
#include <chrono>
#include <mutex>
using namespace phoenix_cpp;
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}
struct Runtime {
    LibraryHandle module = nullptr;
    Backend* backend = nullptr;
    DestroyBackend destroy = nullptr;
    std::unique_ptr<Model> model;
    std::unique_ptr<Graph> graph;
    std::unique_ptr<FrameState> frame;
    std::array<unsigned, 8> repeats{};
    bool fused_attention = false;
    bool device_features = false;
    bool fused_residual = false;
    unsigned window_batch_limit = 128;
    std::vector<std::vector<float>> histories;
    Json status, signature;
    phoenix_model::WeightSource weight_source;
    std::mutex lock;
    std::string error;
    Runtime(const Json& config, unsigned width, unsigned height) {
        struct Rollback {
            Runtime& runtime;
            bool done = false;
            ~Rollback() {
                if (!done) {
                    runtime.graph.reset();
                    runtime.model.reset();
                    if (runtime.backend && runtime.destroy)
                        runtime.destroy(runtime.backend);
                    if (runtime.module)
                        close_library(runtime.module);
                }
            }
        } rollback{*this};
        demand(width && height && width <= 8192 && height <= 8192, "Unsupported frame extent");
        auto validated = config.value("validated_devices", std::vector<std::string>{});
        auto selected = config.at("device").get<std::string>();
        bool tested = std::find(validated.begin(), validated.end(), selected) != validated.end();
        demand(tested || config.value("allow_unvalidated", false),
               "Selected C++ device has no validation entry; explicit opt-in required");
        auto root = std::filesystem::u8path(config.at("runtime_root").get<std::string>());
        demand(root.is_absolute(), "Native runtime root must be absolute");
        std::string kind = config.at("backend"), device = config.at("device");
        demand(kind == "cpu" || kind == "cuda" || kind == "vulkan" || kind == "hip" ||
                   kind == "sycl" || kind == "metal",
               "Unknown backend");
        module = open_library(root / library_name("phoenix_" + kind + "_cpp"));
        demand(module != nullptr, "Compiled backend library unavailable");
        const bool cpu_device = device == "cpu" || device == "sycl:cpu";
        const unsigned requested_threads = cpu_thread_request(config),
                       cpu_threads = phoenix_threads::resolve(requested_threads);
        auto create =
            reinterpret_cast<CreateBackend>(library_symbol(module, "phoenix_backend_create"));
        destroy =
            reinterpret_cast<DestroyBackend>(library_symbol(module, "phoenix_backend_destroy"));
        demand(create && destroy, "Backend ABI unavailable");
        char message[2048]{};
        // The wide shader-folder argument belongs to the Windows Vulkan ABI.
        // Metal embeds its shader library; Unix CPU/Metal do not consume it.
#ifdef _WIN32
        const auto shader_folder = (root / "vulkan").wstring();
        const wchar_t* shader_folder_argument = shader_folder.c_str();
#else
        const wchar_t* shader_folder_argument = nullptr;
#endif
        if (kind == "cpu") {
            using CreateThreads = Backend* (*)(const char*, unsigned, char*, size_t);
            auto create_threads = reinterpret_cast<CreateThreads>(
                library_symbol(module, "phoenix_backend_create_threads"));
            demand(create_threads != nullptr,
                   "CPU thread control requires the matching CPU backend build");
            backend = create_threads(device.c_str(), cpu_threads, message, sizeof(message));
        } else {
            using ConfiguredCreate =
                Backend* (*)(const char*, const wchar_t*, uint32_t, char*, size_t);
            auto configured_create = reinterpret_cast<ConfiguredCreate>(
                library_symbol(module, "phoenix_backend_create_configured"));
            uint32_t options = 0;
            if (config.value("cuda_direct_fp8", true))
                options |= CudaDirectFp8;
            if (config.value("cuda_fused_feed_forward", true))
                options |= CudaFusedFeedForward;
            if (config.value("cuda_fused_window_block", true))
                options |= CudaFusedWindowBlock;
            if (config.value("cuda_compact_window_block", true))
                options |= CudaCompactWindowBlock;
            if (config.value("cuda_fused_wide_feed_forward", false))
                options |= CudaFusedWideFeedForward;
            if (config.value("cuda_shape_matrices", !config.contains("cuda_fp8_columns")))
                options |= CudaShapeMatrices;
            if (config.value("scalar_matrices", false))
                options |= ScalarMatrices;
            if (config.value("profiling", std::string("frame")) == "dispatches")
                options |= DispatchProfiling;
            if (!config.value("tensor_attention", true))
                options |= OrderedAttention;
            if (config.value("vulkan_fp8_matrices", false))
                options |= LosslessFp8Matrices;
            if (config.value("vulkan_native_half", true))
                options |= VulkanHardwareHalf;
            if (config.value("vulkan_shape_matrices", !config.contains("vulkan_matrix_columns")))
                options |= VulkanShapeMatrices;
            if (!config.value("vulkan_specialize_wide", true))
                options |= VulkanGenericWide;
            const auto vulkan_columns = config.value("vulkan_matrix_columns", 16u);
            demand(vulkan_columns == 16 || vulkan_columns == 32 || vulkan_columns == 64,
                   "Vulkan matrix columns must be 16, 32 or 64");
            if (vulkan_columns == 32)
                options |= VulkanWideColumns32;
            if (vulkan_columns == 64)
                options |= VulkanWideColumns64;
            const auto matrix_warps = config.value("vulkan_matrix_warps", 0u);
            demand(matrix_warps == 0 || matrix_warps == 1 || matrix_warps == 4,
                   "Vulkan matrix warps must be 0 (automatic), 1 or 4");
            if (matrix_warps == 1)
                options |= NarrowMatrixWorkgroups;
            if (matrix_warps == 4)
                options |= WideMatrixWorkgroups;
            const auto fp8_columns = config.value("cuda_fp8_columns", 32u);
            demand(fp8_columns == 32 || fp8_columns == 64 || fp8_columns == 128,
                   "CUDA FP8 columns must be 32, 64 or 128");
            if (fp8_columns == 64)
                options |= Fp8Columns64;
            if (fp8_columns == 128)
                options |= Fp8Columns128;
            const auto matrix_columns = config.value("cuda_matrix_columns", 16u);
            demand(matrix_columns == 16 || matrix_columns == 32 || matrix_columns == 64,
                   "CUDA matrix columns must be 16, 32 or 64");
            if (matrix_columns == 32)
                options |= MatrixColumns32;
            if (matrix_columns == 64)
                options |= MatrixColumns64;
            using Utf8Create = Backend* (*)(const char*, const char*, uint32_t, char*, size_t);
            auto utf8_create = reinterpret_cast<Utf8Create>(
                library_symbol(module, "phoenix_backend_create_utf8"));
            const auto shader_folder_utf8 = phoenix_model::source_path_string(root / "vulkan");
            backend = utf8_create
                ? utf8_create(device.c_str(), shader_folder_utf8.c_str(), options, message, sizeof(message))
                : configured_create
                    ? configured_create(device.c_str(), shader_folder_argument, options, message,
                                        sizeof(message))
                    : create(device.c_str(), shader_folder_argument, message, sizeof(message));
        }
        demand(backend != nullptr, message);
        using Features = uint32_t (*)(Backend*);
        auto features =
            reinterpret_cast<Features>(library_symbol(module, "phoenix_backend_features"));
        fused_residual = config.value("fused_residual", true) && features &&
                         (features(backend) & FusedMatrixResidual);
        device_features = config.value("device_features", true) && features &&
                          (features(backend) & DeviceFeaturePacking);
        fused_attention = config.value("fused_attention", !features ||
                              !(features(backend) & PreferUnfusedWindowAttention)) && features &&
                          (features(backend) & FusedWindowAttention);
        window_batch_limit = config.value("window_batch_limit", 512u);
        demand(window_batch_limit >= 16 && window_batch_limit <= 8192,
               "Window batch limit must be 16..8192");
        weight_source = phoenix_model::WeightSource::config(config.at("weight_source"));
        model = std::make_unique<Model>(weight_source);
        // Six spatial reductions. Preserve the portable plan's minimum/padding policy.
        auto align = [](unsigned value, unsigned alignment_bytes) {
            return (value + alignment_bytes - 1) / alignment_bytes * alignment_bytes;
        };
        auto alignment = [&](unsigned value) {
            unsigned reductions = 0;
            for (unsigned iteration_index = 0; iteration_index < 6; ++iteration_index) {
                auto next = align((value + 1) / 2, 4);
                if (next < value)
                    ++reductions;
                value = next;
            }
            return 1u << reductions;
        };
        auto horizontal_alignment = alignment(width), vertical_alignment = alignment(height);
        unsigned working_width = std::max(320u, align(width, horizontal_alignment)),
                 working_height = std::max(320u, align(height, vertical_alignment));
        if (working_height % (vertical_alignment * 4) == 0 &&
            working_width % (horizontal_alignment * 4) == 0)
            working_width += horizontal_alignment;
        frame = std::make_unique<FrameState>(
            width, height, working_width, working_height,
            cpu_device ? cpu_threads
                       : std::min(16u, std::max(1u, std::thread::hardware_concurrency() / 2)));
        status = {{"backend", backend->name()},
                  {"device", device},
                  {"runtime", "native-cpp"},
                  {"width", width},
                  {"height", height},
                  {"working_width", working_width},
                  {"working_height", working_height},
                  {"inference_validated", tested},
                  {"python_required", false}};
        if (cpu_device)
            status.update({{"cpu_threads", cpu_threads},
                           {"cpu_threads_requested", requested_threads},
                           {"cpu_threads_available", phoenix_threads::available()}});
        repeats.fill(UINT32_MAX);
        rollback.done = true;
    }
    ~Runtime() {
        graph.reset();
        model.reset();
        if (backend && destroy)
            destroy(backend);
        if (module)
            close_library(module);
    }
    // Execute sampling passes against separate histories. Stage timings include host
    // preparation and transfers; operator events alone do not describe game frame time.
    void run(const float* color, const float* motion, const float* depth, const float* validity,
             const Json& metadata, float* output) {
        std::lock_guard guard(lock);
        auto start = Clock::now();
        size_t pixel_count = size_t(frame->width) * frame->height;
        demand(color && output, "Missing frame buffers");
        // Validate disjoint rows in the existing frame pool. Keep finite/range
        // failures separate so rejection semantics do not depend on worker order.
        std::atomic<unsigned> input_errors{0};
        frame->rows.run(frame->height, [&](unsigned row) {
            unsigned errors = 0;
            const size_t first = size_t(row) * frame->width * 3;
            for (size_t index = first; index < first + frame->width * 3; ++index) {
                const float value = color[index];
                if (!std::isfinite(value))
                    errors |= 1;
                if (!(value >= 0 && value <= 1))
                    errors |= 2;
            }
            if (errors)
                input_errors.fetch_or(errors, std::memory_order_relaxed);
        });
        demand(!(input_errors.load() & 1), "Nonfinite frame input");
        demand(!(input_errors.load() & 2), "Frame input requires model proxy range 0..1");
        // Preflight every consumed metadata field and declared input before changing
        // temporal state or rebuilding the graph. Rejected requests must be invisible
        // to the next valid frame, including a reset paired with invalid metadata.
        const bool reset = metadata.value("reset", false);
        auto controls = Controls::from(metadata.value("controls", Json::object()));
        auto requested = metadata.value("repeats", std::array<unsigned, 8>{1, 1, 1, 1, 1, 1, 1, 1});
        unsigned passes = metadata.value("passes", 1u);
        demand(passes >= 1 && passes <= 4, "Network passes outside 1..4");
        for (auto count : requested)
            demand(count <= 4, "Block repetition outside 0..4");
        bool has_motion = metadata.value("has_motion", false),
             has_depth = metadata.value("has_depth", false),
             has_history_validity = metadata.value("has_history_validity", false),
             reversed_depth = metadata.value("reversed_depth", false);
        const auto motion_from_frame = metadata.value("motion_from_frame", uint64_t(0));
        demand(!has_history_validity || has_motion, "History validity requires motion");
        demand(!has_motion || motion, "Missing declared motion");
        demand(!has_depth || depth, "Missing declared depth");
        demand(!has_history_validity || validity, "Missing declared validity");
        std::atomic<unsigned> plane_errors{0};
        frame->rows.run(frame->height, [&](unsigned row) {
            unsigned errors = 0;
            const size_t first = size_t(row) * frame->width;
            for (size_t pixel = first; pixel < first + frame->width; ++pixel) {
                if (has_motion) {
                    const float x = motion[pixel * 2], y = motion[pixel * 2 + 1];
                    if (x != 0 || y != 0)
                        errors |= 1;
                    if (!std::isfinite(x) || !std::isfinite(y))
                        errors |= 2;
                }
                if (has_depth && !std::isfinite(depth[pixel]))
                    errors |= 4;
                if (has_history_validity && validity[pixel] != 0 && validity[pixel] != 1)
                    errors |= 8;
            }
            if (errors)
                plane_errors.fetch_or(errors, std::memory_order_relaxed);
        });
        const unsigned errors = plane_errors.load();
        demand(!(errors & 2), "Nonfinite motion");
        demand(!(errors & 4), "Nonfinite depth");
        demand(!(errors & 8), "Nonbinary validity");
        const bool stationary_motion = has_motion && !(errors & 1);
        double validation_ms = elapsed(start);
        Json next_signature = {
            requested,          passes,        controls.style,     controls.tone,
            controls.structure, controls.skin, controls.auto_mask, reversed_depth};
        if (reset)
            histories.clear();
        if (next_signature != signature) {
            histories.clear();
            signature = next_signature;
        }
        // Backend allocation/build/execution failures are outside preflight: these
        // operations can replace the graph and invalidate history before failing.
        if (requested != repeats) {
            graph.reset();
            repeats.fill(UINT32_MAX);
            if (!model->decoded())
                model = std::make_unique<Model>(weight_source);
            graph = std::make_unique<Graph>(*backend, *model, fused_attention, window_batch_limit,
                                            device_features, fused_residual);
            graph->build(frame->padded_width, frame->padded_height, requested);
            model->discard_decoded();
            repeats = requested;
            histories.clear();
        }
        if (histories.size() != passes)
            histories.assign(passes, {});
        struct HistoryBorrow {
            FrameState& frame;
            ~HistoryBorrow() {
                frame.history_view = nullptr;
            }
        } history_borrow{*frame};
        std::vector<std::vector<float>> next_histories(passes);
        std::vector<float> conditioning_history, resolve_history;
        std::vector<uint8_t> conditioning_valid, resolve_valid;
        // Borrow the caller's first-pass color until resolve publishes a new image.
        // No input is written, including when caller input/output pointers alias.
        std::vector<float> current, next, head;
        const float* current_color = color;
        std::vector<Event> events;
        double prepare_ms = 0, execute_ms = 0, resolve_ms = 0;
        bool history_used = false;
        size_t pre_pixels = 0, post_pixels = 0;
        uint64_t temporary_capacity_high_water = 0;
        auto nested_capacity = [](const std::vector<std::vector<float>>& buffers) {
            uint64_t bytes = buffers.capacity() * sizeof(std::vector<float>);
            for (const auto& buffer : buffers)
                bytes += buffer.capacity() * sizeof(float);
            return bytes;
        };
        auto sample_temporary_capacity = [&](uint64_t feature_bytes = 0) {
            const uint64_t bytes = (conditioning_history.capacity() + resolve_history.capacity() +
                                    current.capacity() + next.capacity() + head.capacity()) *
                                       sizeof(float) +
                                   conditioning_valid.capacity() + resolve_valid.capacity() +
                                   events.capacity() * sizeof(Event) +
                                   nested_capacity(next_histories) + feature_bytes;
            temporary_capacity_high_water = std::max(temporary_capacity_high_water, bytes);
        };
        for (unsigned pass = 0; pass < passes; ++pass) {
            auto stage_started = Clock::now();
            frame->history_view = &histories[pass];
            const std::vector<float>* conditioning_source = &conditioning_history;
            const std::vector<float>* resolve_source = &resolve_history;
            const std::vector<uint8_t>* resolve_mask = &resolve_valid;
            if (stationary_motion && !histories[pass].empty()) {
                // Zero motion samples the original pixel exactly. Depth dilation
                // cannot change those coordinates, so both views borrow the
                // existing history and share one validity mask without filtering
                // or copying two full RGB images.
                conditioning_source = resolve_source = &histories[pass];
                conditioning_valid.resize(pixel_count);
                for (size_t pixel = 0; pixel < pixel_count; ++pixel)
                    conditioning_valid[pixel] = !has_history_validity || validity[pixel] > .9999f;
                resolve_mask = &conditioning_valid;
            } else {
                frame->warp(has_motion ? motion : nullptr, has_depth ? depth : nullptr,
                            has_history_validity ? validity : nullptr, reversed_depth,
                            conditioning_history, resolve_history, conditioning_valid,
                            resolve_valid);
            }
            pre_pixels =
                std::count(conditioning_valid.begin(), conditioning_valid.end(), uint8_t(1));
            post_pixels = std::count(resolve_mask->begin(), resolve_mask->end(), uint8_t(1));
            history_used = history_used || pre_pixels || post_pixels;
            std::vector<uint16_t> features;
            if (device_features)
                frame->prepare_noise(controls.seed);
            else
                features = frame->features(current_color, controls, *conditioning_source,
                                           conditioning_valid);
            sample_temporary_capacity(features.capacity() * sizeof(uint16_t));
            prepare_ms += elapsed(stage_started);
            stage_started = Clock::now();
            if (device_features)
                graph->run_from_planes(
                    current_color, pixel_count * 3, *conditioning_source, conditioning_valid,
                    frame->noise, frame->width, frame->height, controls.style, controls.tone,
                    controls.structure, controls.skin, controls.auto_mask, head, events);
            else
                graph->run(features, head, events);
            sample_temporary_capacity();
            execute_ms += elapsed(stage_started);
            stage_started = Clock::now();
            current =
                frame->resolve(current_color, head, controls, *resolve_source, *resolve_mask, next);
            current_color = current.data();
            resolve_ms += elapsed(stage_started);
            auto validation_started = Clock::now();
            std::atomic<unsigned> result_errors{0};
            frame->rows.run(frame->height, [&](unsigned row) {
                unsigned errors = 0;
                const size_t first = size_t(row) * frame->width * 3;
                for (size_t index = first; index < first + frame->width * 3; ++index) {
                    if (!std::isfinite(current[index]))
                        errors |= 1;
                    if (!std::isfinite(next[index]))
                        errors |= 2;
                }
                if (errors)
                    result_errors.fetch_or(errors, std::memory_order_relaxed);
            });
            demand(!(result_errors.load() & 1), "Nonfinite network result");
            demand(!(result_errors.load() & 2), "Nonfinite history result");
            validation_ms += elapsed(validation_started);
            next_histories[pass] = std::move(next);
            sample_temporary_capacity();
        }
        frame->history_view = nullptr;
        histories = std::move(next_histories);
        std::copy(current.begin(), current.end(), output);
        Json layers = Json::array();
        std::map<std::pair<unsigned, unsigned>, double> sums;
        for (auto event : events)
            sums[{event.block, event.layer}] += event.milliseconds;
        double total = 0;
        for (auto [layer_key, milliseconds] : sums)
            total += milliseconds;
        for (auto [layer_key, milliseconds] : sums) {
            if (layer_key.first == UINT32_MAX && layer_key.second == UINT32_MAX)
                continue;
            layers.push_back({{"block", layer_key.first == UINT32_MAX ? -1 : int(layer_key.first)},
                              {"layer", layer_key.second},
                              {"ms", milliseconds},
                              {"percent", total ? 100 * milliseconds / total : 0}});
        }
        status.update(
            {{"state", "completed"},
             {"fused_attention", fused_attention},
             {"device_features", device_features},
             {"fused_residual", fused_residual},
             {"window_batch_limit", window_batch_limit},
             {"profiling_scope",
              events.empty() || events.front().block != UINT32_MAX ? "layers" : "submission spans"},
             {"validation_ms", validation_ms},
             {"prepare_ms", prepare_ms},
             {"execute_with_transfers_ms", execute_ms},
             {"resolve_ms", resolve_ms},
             {"worker_ms", elapsed(start)},
             {"command_count", graph->commands.size()},
             {"timing_intervals", events.size()},
             {"operator_ms", total},
             {"operator_clock", status.value("device", std::string()) == "cpu" ? "CPU wall"
                                : status.value("device", std::string()) == "sycl:cpu"
                                    ? "CPU queue events"
                                    : "GPU events"},
             {"resident_bytes", graph->memory_bytes()},
             {"resident_weight_bytes", graph->weight_bytes()},
             {"resident_scratch_bytes", graph->memory_bytes() - graph->weight_bytes()},
             {"backend_allocation_bytes", graph->memory_bytes()},
             {"host_frame_buffers",
              {{"noise_capacity_bytes", frame->noise.capacity() * sizeof(uint16_t)},
               {"retained_history_capacity_bytes",
                nested_capacity(histories) + frame->history.capacity() * sizeof(float)},
               {"temporary_capacity_sampled_high_water_bytes", temporary_capacity_high_water},
               {"scope",
                "Vector capacity bytes, sampled at prepare/execute/resolve boundaries; not process RAM or an allocator-wide peak. Backend allocation bytes exclude these host buffers, driver caches, stack and caller transport."}}},
             {"layers", layers},
             {"passes", passes},
             {"repeats", requested},
             {"history_used", history_used},
             {"history_pre_valid_pixels", pre_pixels},
             {"history_post_valid_pixels", post_pixels},
             {"motion_from_frame", motion_from_frame},
             {"host_boundary",
              device_features
                  ? "GPU feature packing; C++ history warp and resolve; host-plane transfers"
                  : "C++ host preparation and resolve; host-plane transfers"}});
    }
};
thread_local std::string last_error;
}
extern "C" PHOENIX_EXPORT void* phoenix_runtime_create(const char* json, unsigned width,
                                                              unsigned height) {
    try {
        demand(json != nullptr, "Missing runtime configuration");
        auto runtime = std::make_unique<Runtime>(Json::parse(json), width, height);
        last_error.clear();
        return runtime.release();
    } catch (const std::exception& exception) {
        last_error = exception.what();
        return nullptr;
    }
}
extern "C" PHOENIX_EXPORT void phoenix_runtime_destroy(void* runtime) {
    delete static_cast<Runtime*>(runtime);
}
extern "C" PHOENIX_EXPORT int phoenix_runtime_frame(void* runtime, const float* color,
                                                           const float* motion, const float* depth,
                                                           const float* validity,
                                                           const char* metadata, float* output) {
    try {
        demand(runtime, "Missing runtime");
        demand(metadata != nullptr, "Missing frame metadata");
        static_cast<Runtime*>(runtime)->run(color, motion, depth, validity, Json::parse(metadata),
                                            output);
        last_error.clear();
        return 0;
    } catch (const std::exception& exception) {
        last_error = exception.what();
        return -1;
    }
}
extern "C" PHOENIX_EXPORT const char* phoenix_runtime_error() {
    return last_error.c_str();
}
extern "C" PHOENIX_EXPORT size_t phoenix_runtime_status(void* handle, char* destination,
                                                               size_t capacity) {
    if (!handle)
        return 0;
    auto& runtime = *static_cast<Runtime*>(handle);
    std::lock_guard guard(runtime.lock);
    auto text = runtime.status.dump();
    if (destination && capacity > text.size())
        std::memcpy(destination, text.c_str(), text.size() + 1);
    return text.size() + 1;
}
