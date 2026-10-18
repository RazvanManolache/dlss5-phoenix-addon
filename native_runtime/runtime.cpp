// Owns one selected backend, immutable graph, frame preparation and per-pass history.
// A frame commits its new histories only after inference/resolve and finite-value checks complete.

#define PHOENIX_RUNTIME_IMPLEMENTATION
#include "platform.h"
#include "graph.h"
#include "frame.h"
#include "thread_config.h"
#include "backend_options.h"
#include "frame_planes.h"
#include "advanced_frame.h"
#include <chrono>
#include <mutex>
using namespace phoenix_cpp;
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}
Json portable_capabilities() {
    return {
        {"schema", "phoenix-portable-capabilities/1"},
        {"host_frame_api_versions", {1, 2, 3}},
        {"legacy_api_control_mask", false},
        {"control_mask", {{"profile", "phoenix-experimental-rgb/1"},
                          {"state", "experimental"}, {"input_api", 2},
                          {"preparation", "device when advertised; otherwise host"}, {"vendor_validated", false}}},
        {"style", {{"conditioning", true}, {"output_grade", "recovered_3108"}, {"vendor_validated", false}}},
        {"ui_composition", {{"state", "experimental"}, {"input_api", 3}, {"contract", "premultiplied SDR proxy"}, {"vendor_validated", false}}},
        {"bidirectional_distortion", {{"state", "experimental"}, {"input_api", 3}, {"contract", "normalized RG/BA canonicalization"}, {"vendor_validated", false}}},
        {"appearance_model_editions", {{"state", "fixed_compatible_architecture"},
                                        {"curated_abc_validated", false}}}};
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
    bool device_noise_cache = false;
    bool device_control_maps = false;
    bool fused_residual = false;
    unsigned window_batch_limit = 128;
    std::vector<std::vector<float>> histories;
    Json status, signature;
    phoenix_model::WeightSource weight_source;
    std::mutex lock;
    std::string error;
    bool validation_stage_hashes = false;
    float model_history_gain = 1;
    bool legacy_history_gain = false;
    uint32_t sampling_index = 0;
    bool legacy_output = false;
    std::vector<float> previous_distortion;
    Runtime(const Json& config, unsigned width, unsigned height) {
        validation_stage_hashes = config.value("validation_stage_hashes", false);
        const auto gain_mode = config.value("history_gain_mode", std::string("model"));
        demand(gain_mode == "model" || gain_mode == "legacy_unit", "Unknown history gain mode");
        legacy_history_gain = gain_mode == "legacy_unit";
        const auto output_profile = config.value("output_profile", std::string("recovered_3108"));
        demand(output_profile == "recovered_3108" || output_profile == "legacy", "Unknown output profile");
        legacy_output = output_profile == "legacy";
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
            if (kind == "vulkan" && config.value("vulkan_fused_qkv", true))
                options |= VulkanFusedQkv;
            if (kind == "vulkan" && config.value("vulkan_workgroup_fp8", true))
                options |= VulkanWorkgroupFp8;
            if (kind == "vulkan" && config.value("vulkan_workgroup_shape_tiles", true))
                options |= VulkanWorkgroupShapeTiles;
            if (kind == "vulkan" && config.value("vulkan_fused_feed_forward", true))
                options |= VulkanFusedFeedForward;
            if (kind == "vulkan" && config.value("vulkan_projected_qkv", true))
                options |= VulkanProjectedQkv;
            if (kind == "vulkan" && config.value("vulkan_cache_noise", true))
                options |= VulkanCacheNoise;
            if (config.value("vulkan_shape_matrices", !config.contains("vulkan_matrix_columns")))
                options |= VulkanShapeMatrices;
            if (!config.value("vulkan_specialize_wide", true))
                options |= VulkanGenericWide;
            if (kind == "vulkan" && config.value("vulkan_tiled_fallback", true))
                options |= VulkanTiledFallback;
            if (kind == "vulkan" && config.value("vulkan_coalesced_fallback", true))
                options |= VulkanCoalescedFallback;
            if (kind == "vulkan" && config.value("vulkan_scalar_attention", false))
                options |= VulkanScalarAttention;
            if (kind == "vulkan" && config.value("vulkan_workgroup_compatibility", false))
                options |= VulkanWorkgroupCompatibility;
            if (kind == "vulkan" && config.value("vulkan_fast_math", false))
                options |= VulkanFastMath | VulkanWorkgroupCompatibility;
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
            auto utf8_create =
                reinterpret_cast<Utf8Create>(library_symbol(module, "phoenix_backend_create_utf8"));
            const auto shader_folder_utf8 = phoenix_model::source_path_string(root / "vulkan");
            backend =
                utf8_create ? utf8_create(device.c_str(), shader_folder_utf8.c_str(), options,
                                          message, sizeof(message))
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
        fused_attention =
            config.value("fused_attention",
                         !features || !(features(backend) & PreferUnfusedWindowAttention)) &&
            features && (features(backend) & FusedWindowAttention);
        device_noise_cache = device_features && features && (features(backend) & DeviceNoiseCache);
        device_control_maps = features && (features(backend) & DeviceControlMaps);
        window_batch_limit = config.value("window_batch_limit", 512u);
        demand(window_batch_limit >= 16 && window_batch_limit <= 8192,
               "Window batch limit must be 16..8192");
        weight_source = phoenix_model::WeightSource::config(config.at("weight_source"));
        model = std::make_unique<Model>(weight_source);
        model_history_gain = checked_model_history_gain(model->history_gain());
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
        status["frame_boundary_contract"] = frame_boundary_contract;
        status["legacy_frame_boundary_emulation"]=legacy_history_gain&&legacy_output;
        status["history_gain_mode"] = legacy_history_gain ? "legacy_unit" : "model";
        status["model_history_gain"] = model_history_gain;
        status["model_resource_sha256"] = model->resource_sha256();
        status["model_architecture"] = "phoenix-network37/1";
        status["effective_history_gain"] = legacy_history_gain ? 1.f : model_history_gain;
        status["vendor_frame_boundary_validated"] = false;
        status["capabilities"] = portable_capabilities();
        status["output_profile"] = legacy_output ? "legacy" : "recovered_3108";
        if (kind == "vulkan") {
            const bool compatibility = features && (features(backend) & WorkgroupCompatibility);
            status["vulkan_workgroup_compatibility"] = compatibility;
            const bool approximate = features && (features(backend) & ApproximateMath);
            status["vulkan_fast_math"] = approximate;
            status["arithmetic"] = approximate ? "approximate-hardware" : "reference";
            status["vulkan_fused_qkv"] = features && (features(backend) & FusedQkvPreparation);
            status["vulkan_workgroup_fp8"] = features && (features(backend) & WorkgroupFp8Matrices);
            status["vulkan_workgroup_shape_tiles"] =
                features && (features(backend) & WorkgroupShapeTiles);
            status["vulkan_fused_feed_forward"] =
                features && (features(backend) & FusedFeedForward);
            status["vulkan_projected_qkv"] =
                features && (features(backend) & ProjectedQkvPreparation);
            status["device_noise_cache"] = device_noise_cache;
            const bool matrix_acceleration = features && (features(backend) & FusedMatrixResidual);
            status["vulkan_matrix_acceleration"] = matrix_acceleration;
            status["vulkan_fallback_tile"] = matrix_acceleration ? "unused"
                                             : config.value("vulkan_tiled_fallback", true) ? "32x32"
                                                                                           : "8x16";
            status["vulkan_coalesced_fallback"] = !matrix_acceleration &&
                                                  config.value("vulkan_tiled_fallback", true) &&
                                                  config.value("vulkan_coalesced_fallback", true);
            status["vulkan_scalar_attention"] =
                !compatibility && !matrix_acceleration && config.value("vulkan_scalar_attention", false);
        }
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
    void replace_model(const Json& source_json) {
        const auto source=phoenix_model::WeightSource::config(source_json);
        auto replacement=std::make_unique<Model>(source);
        const float gain=checked_model_history_gain(replacement->history_gain());
        std::lock_guard guard(lock);
        backend->wait();
        graph.reset();model=std::move(replacement);weight_source=source;
        model_history_gain=gain;repeats.fill(UINT32_MAX);histories.clear();
        previous_distortion.clear();signature=Json();sampling_index=0;
        status["state"]="model_replaced";status["inference_validated"]=false;
        status["model_resource_sha256"]=model->resource_sha256();
        status["model_history_gain"]=gain;status["effective_history_gain"]=legacy_history_gain?1.f:gain;
        for(const auto* key:{"validation_stages","optional_input_hashes","presented_output_sha256","layers",
            "command_count","resident_bytes","resident_weight_bytes","resident_scratch_bytes",
            "backend_allocation_bytes","host_frame_buffers"})status.erase(key);
        status["history_used"]=false;status["sampling_next_index"]=0;
    }
    // Execute sampling passes against separate histories. Stage timings include host
    // preparation and transfers; operator events alone do not describe game frame time.
    void run(const float* color, const float* motion, const float* depth, const float* validity,
             const Json& metadata, float* output, const float* control_mask = nullptr,
             uint64_t bounded_adapter_bytes = 0, const AdvancedFrame* advanced = nullptr) {
        std::lock_guard guard(lock);
        auto start = Clock::now();
        size_t pixel_count = size_t(frame->width) * frame->height;
        const float* original_color = color;
        CanonicalFrame canonical;
        std::vector<float> next_distortion;
        Json input_plane_hashes=Json::object();
        double canonicalize_ms=0,presentation_ms=0;
        demand(color && output, "Missing frame buffers");
        validate_control_mask(control_mask, pixel_count);
        const bool use_device_features = device_features && (!control_mask || device_control_maps);
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
        const auto output_stage=metadata.value("output_stage",std::string("presented"));
        demand(output_stage=="presented"||output_stage=="neural_only","Unknown output stage");
        const bool neural_only=output_stage=="neural_only";
        demand(!neural_only||!legacy_output,"Neural-only output requires recovered_3108 profile");
        demand(!metadata.value("has_control_mask", false) || control_mask,
               "Declared control mask requires the bounded V2 interface");
        auto controls = Controls::from(metadata.value("controls", Json::object()));
        controls.history_strength = legacy_history_gain ? 1.f : model_history_gain;
        controls.legacy_output = legacy_output;
        demand(!advanced || !legacy_output, "Advanced planes require recovered_3108 output profile");
        if (!legacy_output && controls.style)
            demand(status.at("model_resource_sha256") == "836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4",
                   "Selected model has no verified style preset table");
        if (advanced) {
            validate_unit_plane(advanced->ui, pixel_count * 4, "Invalid UI proxy values");
            validate_unit_plane(advanced->ui_alpha, pixel_count, "Invalid UI alpha coverage");
            validate_unit_plane(advanced->backbuffer, pixel_count * 3, "Invalid backbuffer proxy values");
            if (advanced->ui)
                for (size_t p = 0; p < pixel_count; ++p) {
                    const float a = advanced->ui[p*4+3];
                    demand(!advanced->ui_alpha || advanced->ui_alpha[p] == a, "UI and UIAlpha coverage conflict");
                    for (unsigned c = 0; c < 3; ++c)
                        demand(advanced->ui[p*4+c] <= a, "UI must be premultiplied");
                }
            if (advanced->distortion) {
                for (size_t i = 0; i < pixel_count * 4; ++i)
                    demand(std::isfinite(advanced->distortion[i]) && std::fabs(advanced->distortion[i]) <= 1,
                           "Distortion requires finite normalized displacements in -1..1");
                next_distortion.assign(advanced->distortion, advanced->distortion + pixel_count * 4);
            }
            if(validation_stage_hashes) {
                auto pin=[&](const char* name,const float* plane,unsigned channels) {
                    if(plane)input_plane_hashes[name]=phoenix_model::weight_digest(
                        reinterpret_cast<const uint8_t*>(plane),pixel_count*channels*sizeof(float));
                };
                pin("control_mask",control_mask,4);pin("ui",advanced->ui,4);pin("ui_alpha",advanced->ui_alpha,1);
                pin("backbuffer",advanced->backbuffer,3);pin("distortion",advanced->distortion,4);
            }
        }
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
        if (advanced && advanced->distortion) {
            const auto canonical_started=Clock::now();
            canonical.prepare(color, has_motion ? motion : nullptr, has_depth ? depth : nullptr,
                has_history_validity ? validity : nullptr, control_mask, advanced->distortion,
                reset || previous_distortion.empty() ? nullptr : previous_distortion.data(),
                frame->width, frame->height);
            color = canonical.color.data();
            if (has_motion) {
                motion = canonical.motion.data();
                validity = canonical.validity.data();
                has_history_validity = true;
            }
            if (has_depth) depth = canonical.depth.data();
            if (control_mask) control_mask = canonical.mask.data();
            canonicalize_ms=elapsed(canonical_started);
        }
        const bool stationary_motion = has_motion && !(errors & 1) &&
            !(advanced && advanced->distortion);
        double validation_ms = elapsed(start)-canonicalize_ms;
        Json next_signature = {
            requested,          passes,        controls.style,     controls.tone,
            controls.structure, controls.skin, controls.auto_mask, reversed_depth,
            controls.advancing_noise, control_mask != nullptr, advanced && advanced->distortion};
        const SamplingPlan sampling{controls.seed,
                                    reset || next_signature != signature ? 0u : sampling_index,
                                    controls.advancing_noise};
        if (reset)
            histories.clear();
        if (next_signature != signature) {
            histories.clear();
        }
        // Backend allocation/build/execution failures are outside preflight: these
        // operations can replace the graph and invalidate history before failing.
        if (requested != repeats) {
            graph.reset();
            repeats.fill(UINT32_MAX);
            if (!model->decoded())
                model = std::make_unique<Model>(weight_source);
            demand(model->resource_sha256() == status.at("model_resource_sha256").get<std::string>(),
                   "Model resource changed during session; create a new session");
            graph = std::make_unique<Graph>(*backend, *model, fused_attention, window_batch_limit,
                                            device_features, fused_residual, device_noise_cache);
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
        Json validation_stages = Json::array();
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
                                   nested_capacity(next_histories) + feature_bytes + bounded_adapter_bytes +
                                   (canonical.color.capacity()+canonical.motion.capacity()+canonical.depth.capacity()+
                                    canonical.validity.capacity()+canonical.mask.capacity()+next_distortion.capacity())*sizeof(float);
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
            controls.seed = sampling.seed(pass);
            std::vector<uint16_t> features;
            if (use_device_features) {
                if (!graph->cached_noise(controls.seed) || validation_stage_hashes)
                    frame->prepare_noise(controls.seed);
            } else
                features = frame->features(current_color, controls, *conditioning_source,
                                           conditioning_valid, control_mask);
            sample_temporary_capacity(features.capacity() * sizeof(uint16_t));
            prepare_ms += elapsed(stage_started);
            Json validation = Json::object();
            auto digest = [](const auto& buffer) {
                return phoenix_model::weight_digest(reinterpret_cast<const uint8_t*>(buffer.data()),
                    buffer.size() * sizeof(typename std::remove_reference_t<decltype(buffer)>::value_type));
            };
            if (validation_stage_hashes) {
                validation["reference_producer"] = "portable_phoenix";
                validation["frame_boundary_contract"] = frame_boundary_contract;
                validation["effective_history_gain"] = controls.history_strength;
                validation["sampling_seed"] = controls.seed;
                validation["feature_packing"] = use_device_features ? "device" : "host";
                if (!use_device_features)
                    validation["prepared_features_fp16"] = digest(features);
                validation["color"] = phoenix_model::weight_digest(reinterpret_cast<const uint8_t*>(current_color), pixel_count * 3 * sizeof(float));
                validation["conditioning_history"] = digest(*conditioning_source);
                validation["conditioning_validity"] = digest(conditioning_valid);
                validation["resolve_history"] = digest(*resolve_source);
                validation["resolve_validity"] = digest(*resolve_mask);
                validation["noise"] = digest(frame->noise);
                if (control_mask)
                    validation["control_mask"] = phoenix_model::weight_digest(
                        reinterpret_cast<const uint8_t*>(control_mask), pixel_count * 4 * sizeof(float));
            }
            stage_started = Clock::now();
            if (use_device_features)
                graph->run_from_planes(current_color, pixel_count * 3, *conditioning_source,
                                       conditioning_valid, frame->noise, frame->width,
                                       frame->height, controls.style, controls.tone,
                                       controls.structure, controls.skin, controls.auto_mask, head,
                                       events, controls.seed, control_mask);
            else
                graph->run(features, head, events);
            if (graph->cached_noise(controls.seed))
                std::vector<uint16_t>().swap(frame->noise);
            sample_temporary_capacity();
            execute_ms += elapsed(stage_started);
            if (validation_stage_hashes) validation["network_head"] = digest(head);
            stage_started = Clock::now();
            current =
                frame->resolve(current_color, head, controls, *resolve_source, *resolve_mask, next,
                               control_mask, advanced != nullptr || neural_only);
            current_color = current.data();
            resolve_ms += elapsed(stage_started);
            if (validation_stage_hashes) {
                validation[advanced ? "canonical_neural_output" : "display_output"] = digest(current);
                validation["next_history"] = digest(next);
                std::vector<float> gates(pixel_count);
                for (unsigned y = 0; y < frame->height; ++y)
                    for (unsigned x = 0; x < frame->width; ++x) {
                        const size_t pixel = size_t(y) * frame->width + x;
                        gates[pixel] = resolve_gate(
                            head[(size_t(y) * frame->padded_width + x) * 4 + 3],
                            (*resolve_mask)[pixel] != 0, controls.history_strength);
                    }
                validation["resolve_gate"] = digest(gates);
                validation_stages.push_back(std::move(validation));
            }
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
        if (advanced) {
            const auto presentation_started=Clock::now();
            if (advanced->distortion)
                current = warp_plane(current.data(), advanced->distortion, frame->width, frame->height, 3, 0);
            for (size_t p = 0; !neural_only && p < pixel_count; ++p) {
                RGB original{}, neural{};
                for (unsigned c = 0; c < 3; ++c) {
                    original[c] = original_color[p*3+c]; neural[c] = current[p*3+c];
                }
                const auto result = compose_display(original, neural, p, controls.intensity,
                    controls.style, controls.tone, *advanced);
                for (unsigned c = 0; c < 3; ++c) current[p*3+c] = result[c];
            }
            presentation_ms=elapsed(presentation_started);
        }
        histories = std::move(next_histories);
        previous_distortion = std::move(next_distortion);
        std::copy(current.begin(), current.end(), output);
        signature = std::move(next_signature);
        sampling_index = sampling.next(passes);
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
        Json operator_kinds = Json::array();
        if (status.value("backend", std::string()) == "vulkan-cpp") {
            std::map<unsigned, std::pair<double, unsigned>> kinds;
            for (const auto& event : events) {
                if (event.gpu && event.block != UINT32_MAX) {
                    kinds[event.operation].first += event.milliseconds;
                    ++kinds[event.operation].second;
                }
            }
            for (auto [code, timing] : kinds)
                operator_kinds.push_back({{"kind", code}, {"family", code / 32},
                                         {"subkind", code % 32}, {"calls", timing.second},
                                         {"gpu_ms", timing.first}});
        }
        status.update(
            {{"state", "completed"},
             {"fused_attention", fused_attention},
             {"device_features", use_device_features},
             {"device_features_available", device_features},
             {"control_mask_profile", control_mask ? "phoenix-experimental-rgb/1" : "none"},
             {"control_mask_vendor_validated", false},
             {"frame_api_version", advanced ? 3 : bounded_adapter_bytes ? 2 : 1},
             {"ui_composition_used", advanced && (advanced->ui || advanced->ui_alpha || advanced->backbuffer)},
             {"distortion_used", advanced && advanced->distortion},
             {"presentation_deferred",neural_only},
             {"fused_residual", fused_residual},
             {"window_batch_limit", window_batch_limit},
             {"profiling_scope",
              events.empty() || events.front().block != UINT32_MAX ? "layers" : "submission spans"},
             {"validation_ms", validation_ms},
             {"canonicalize_ms",canonicalize_ms},
             {"presentation_ms",presentation_ms},
             {"prepare_ms", prepare_ms},
             {"execute_with_transfers_ms", execute_ms},
             {"resolve_ms", resolve_ms},
             {"worker_ms", elapsed(start)},
             {"command_count", graph->commands.size()},
             {"timing_intervals", events.size()},
             {"operator_kinds",operator_kinds},
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
               {"bounded_adapter_capacity_bytes", bounded_adapter_bytes},
               {"retained_distortion_capacity_bytes",previous_distortion.capacity()*sizeof(float)},
               {"device_map_upload_staging_bytes",use_device_features&&control_mask?
                    ((pixel_count+3)&~size_t(3))+pixel_count*4*sizeof(float):0},
               {"retained_history_capacity_bytes",
                nested_capacity(histories) + frame->history.capacity() * sizeof(float)},
               {"temporary_capacity_sampled_high_water_bytes", temporary_capacity_high_water},
               {"scope",
                "Vector capacity bytes, sampled at prepare/execute/resolve boundaries, including bounded V2 input/output snapshots; not process RAM or an allocator-wide peak. Backend allocation bytes exclude these host buffers, driver caches, stack and caller transport."}}},
             {"layers", layers},
             {"passes", passes},
             {"sampling_mode", sampling.advancing ? "advancing" : "fixed"},
             {"sampling_first_seed", sampling.seed(0)},
             {"sampling_next_index", sampling_index},
             {"repeats", requested},
             {"history_used", history_used},
             {"history_pre_valid_pixels", pre_pixels},
             {"history_post_valid_pixels", post_pixels},
             {"motion_from_frame", motion_from_frame},
             {"host_boundary",
              use_device_features
                  ? "GPU feature packing; C++ history warp and resolve; host-plane transfers"
                  : "C++ host preparation and resolve; host-plane transfers"}});
        if (validation_stage_hashes) {
            status["validation_stages"] = std::move(validation_stages);
            status["optional_input_hashes"]=std::move(input_plane_hashes);
            status["presented_output_sha256"]=phoenix_model::weight_digest(
                reinterpret_cast<const uint8_t*>(current.data()),current.size()*sizeof(float));
        }
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
                                                    const float* validity, const char* metadata,
                                                    float* output) {
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
static int bounded_frame(void* handle, const phoenix_frame_v2* request,
                         const phoenix_frame_v3* extended = nullptr) {
    try {
        demand(handle, "Missing runtime");
        demand(request && request->struct_bytes == (extended ? sizeof(phoenix_frame_v3) : sizeof(phoenix_frame_v2)) &&
                   request->version == (extended ? PHOENIX_FRAME_VERSION_3 : PHOENIX_FRAME_VERSION_2) && !request->reserved,
               "Unsupported bounded frame structure/version");
        auto& runtime = *static_cast<Runtime*>(handle);
        const unsigned width = runtime.frame->width, height = runtime.frame->height;
        const size_t output_stride = plane_stride(request->output, width, height, 3);
        demand(request->metadata_json, "Missing frame metadata");
        auto metadata = Json::parse(request->metadata_json);
        demand(metadata.is_object(), "Frame metadata must be an object");
        auto color = copy_plane(request->color, width, height, 3, true);
        auto motion = copy_plane(request->motion, width, height, 2, false);
        auto depth = copy_plane(request->depth, width, height, 1, false);
        auto validity = copy_plane(request->validity, width, height, 1, false);
        auto mask = copy_plane(request->control_mask, width, height, 4, false);
        std::vector<float> ui, ui_alpha, backbuffer, distortion;
        AdvancedFrame advanced;
        if (extended) {
            ui = copy_plane(extended->ui, width, height, 4, false);
            ui_alpha = copy_plane(extended->ui_alpha, width, height, 1, false);
            backbuffer = copy_plane(extended->backbuffer, width, height, 3, false);
            distortion = copy_plane(extended->distortion, width, height, 4, false);
            advanced.ui = ui.empty() ? nullptr : ui.data();
            advanced.ui_alpha = ui_alpha.empty() ? nullptr : ui_alpha.data();
            advanced.backbuffer = backbuffer.empty() ? nullptr : backbuffer.data();
            advanced.distortion = distortion.empty() ? nullptr : distortion.data();
            advanced.control_mask = mask.empty() ? nullptr : mask.data();
        }
        demand((mask.empty() && request->control_profile == PHOENIX_CONTROL_NONE) ||
                   (!mask.empty() && request->control_profile == PHOENIX_CONTROL_EXPERIMENTAL_RGB_1),
               "Control plane requires the explicit experimental RGB profile");
        auto declare = [&](const char* key, bool present) {
            demand(!metadata.contains(key) || metadata.at(key).get<bool>() == present,
                   "Frame metadata conflicts with bounded plane declarations");
            metadata[key] = present;
        };
        declare("has_motion", !motion.empty());
        declare("has_depth", !depth.empty());
        declare("has_history_validity", !validity.empty());
        declare("has_control_mask", !mask.empty());
        std::vector<float> output(size_t(width) * height * 3);
        const uint64_t adapter_bytes = (color.capacity() + motion.capacity() + depth.capacity() +
            validity.capacity() + mask.capacity() + output.capacity() + ui.capacity() +
            ui_alpha.capacity() + backbuffer.capacity() + distortion.capacity()) * sizeof(float);
        runtime.run(color.data(), motion.empty() ? nullptr : motion.data(),
                    depth.empty() ? nullptr : depth.data(),
                    validity.empty() ? nullptr : validity.data(), metadata, output.data(),
                    mask.empty() ? nullptr : mask.data(), adapter_bytes, extended ? &advanced : nullptr);
        auto* destination = reinterpret_cast<unsigned char*>(request->output.data);
        for (unsigned y = 0; y < height; ++y)
            std::memcpy(destination + size_t(y) * output_stride,
                        output.data() + size_t(y) * width * 3, size_t(width) * 3 * sizeof(float));
        last_error.clear();
        return 0;
    } catch (const std::exception& exception) {
        last_error = exception.what();
        return -1;
    }
}
extern "C" PHOENIX_EXPORT int phoenix_runtime_frame_v2(void* handle, const phoenix_frame_v2* request) {
    return bounded_frame(handle, request);
}
extern "C" PHOENIX_EXPORT int phoenix_runtime_frame_v3(void* handle, const phoenix_frame_v3* request) {
    return bounded_frame(handle, request ? &request->base : nullptr, request);
}
extern "C" PHOENIX_EXPORT const char* phoenix_runtime_error() {
    return last_error.c_str();
}
extern "C" PHOENIX_EXPORT int phoenix_runtime_replace_model(void* handle,const char* source) {
    try {
        demand(handle&&source,"Missing runtime/model source");
        static_cast<Runtime*>(handle)->replace_model(Json::parse(source));last_error.clear();return 0;
    } catch(const std::exception& exception) {last_error=exception.what();return -1;}
}
extern "C" PHOENIX_EXPORT size_t phoenix_runtime_capabilities(char* destination, size_t capacity) {
    try {
        const auto text = portable_capabilities().dump();
        if (destination && capacity > text.size())
            std::memcpy(destination, text.c_str(), text.size() + 1);
        last_error.clear();
        return text.size() + 1;
    } catch (const std::exception& exception) {
        last_error = exception.what();
        return 0;
    }
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
