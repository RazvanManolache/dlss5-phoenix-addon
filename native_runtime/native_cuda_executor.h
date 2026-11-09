// Native model schedule on Linux CUDA. No D3D12, NVAPI or Windows DLL execution.
#pragma once
#include "../src/native/model/bundle_loader.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "half_rgba.h"
#include "native_cuda_pixels.h"
#include <chrono>
#include <map>
#include <cmath>

namespace phoenix_native_cuda {
using Json = nlohmann::json;
inline void check(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}
inline void driver(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        throw std::runtime_error(message ? message : "CUDA driver error");
    }
}
using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct Resources {
    void* workspace_raw{};
    void* weight_raw{};
    void* workspace{};
    void* weights{};
    cudaArray_t input{}, output{};
    cudaTextureObject_t color{};
    cudaSurfaceObject_t result{}, input_surface{};
    uint8_t* byte_buffer{};
    uint8_t* byte_staging{};
    uint32_t* invalid{};
    cudaStream_t stream{};
    cudaEvent_t start{}, end{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    std::vector<CUmodule> modules;
    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    ~Resources() {
        if (stream)
            cudaStreamSynchronize(stream);
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
        for (auto module : modules)
            cuModuleUnload(module);
        if (start)
            cudaEventDestroy(start);
        if (end)
            cudaEventDestroy(end);
        if (color)
            cudaDestroyTextureObject(color);
        if (result)
            cudaDestroySurfaceObject(result);
        if (input_surface)
            cudaDestroySurfaceObject(input_surface);
        if (byte_buffer)
            cudaFree(byte_buffer);
        if (byte_staging)
            cudaFreeHost(byte_staging);
        if (invalid)
            cudaFree(invalid);
        if (input)
            cudaFreeArray(input);
        if (output)
            cudaFreeArray(output);
        if (workspace_raw)
            cudaFree(workspace_raw);
        if (weight_raw)
            cudaFree(weight_raw);
        if (stream)
            cudaStreamDestroy(stream);
    }
    void allocate(void*& raw, void*& aligned, size_t bytes) {
        if (!bytes || bytes > SIZE_MAX - 511)
            throw std::runtime_error("Native arena extent");
        check(cudaMalloc(&raw, bytes + 511));
        aligned =
            reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(raw) + 511) & ~uintptr_t(511));
        check(cudaMemset(aligned, 0, bytes));
    }
};
class Executor {
    int selected_ = -1;
    Resources resources_;
    struct Step {
        CUfunction function{};
        std::array<uint32_t, 3> grid, block;
        std::vector<uint8_t> bytes;
    };
    std::vector<Step> steps_;
    std::vector<float4> rgba_;
    std::vector<__half> raw_;
    std::vector<float> rgb_;
    bool failed_ = false;

  public:
    const unsigned width, height;
    Json status;
    Executor(const Json& config, unsigned frame_width, unsigned frame_height)
        : width(frame_width), height(frame_height) {
        if (!width || !height || width > 8192 || height > 8192 ||
            uint64_t(width) * height > 16777216)
            throw std::runtime_error("Native CUDA frame extent exceeds budget");
        if (config.at("backend") != "native-cuda")
            throw std::runtime_error("Expected native-cuda backend");
        const std::string requested = config.at("device");
        if (requested.size() != 37 || requested.rfind("cuda:", 0) != 0)
            throw std::runtime_error("Native CUDA requires cuda:<32 hexadecimal UUID digits>");
        const auto validated = config.value("validated_devices", std::vector<std::string>{});
        if (!config.value("allow_unvalidated", false) &&
            std::find(validated.begin(), validated.end(), requested) == validated.end())
            throw std::runtime_error("Native CUDA device requires explicit validation opt-in");
        // This frontend implements the reset-only final-color contract. Refuse
        // unsupported controls rather than silently changing their meaning.
        const std::vector<std::string> allowed{"backend",           "device",
                                               "runtime_root",      "native_bundle",
                                               "allow_unvalidated", "validated_devices",
                                               "weight_source",     "local_tone",
                                               "local_structure",   "skin_structure",
                                               "auto_mask"};
        for (const auto& item : config.items())
            if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
                throw std::runtime_error("Unsupported native CUDA configuration: " + item.key());
        int count = 0;
        check(cudaGetDeviceCount(&count));
        for (int index = 0; index < count; ++index) {
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, index));
            std::string uuid = "cuda:";
            for (unsigned char byte : properties.uuid.bytes) {
                char hex[3];
                std::snprintf(hex, sizeof(hex), "%02x", byte);
                uuid += hex;
            }
            if (uuid == requested) {
                if (selected_ >= 0)
                    throw std::runtime_error("Ambiguous CUDA UUID");
                selected_ = index;
            }
        }
        if (selected_ < 0)
            throw std::runtime_error("Requested CUDA UUID missing");
        check(cudaSetDevice(selected_));
        cudaDeviceProp gpu{};
        check(cudaGetDeviceProperties(&gpu, selected_));
        if (gpu.major != 12 || gpu.minor != 0)
            throw std::runtime_error("Native CUDA preview currently requires SM120");
        auto& s = resources_;
        check(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        auto root = phoenix_fs::from_utf8(config.at("runtime_root").get<std::string>());
        auto folder = phoenix_fs::from_utf8(
            config.value("native_bundle", phoenix_model::source_path_string(root / "native-cuda")));
        if (!root.is_absolute() || !folder.is_absolute())
            throw std::runtime_error("Native runtime/bundle paths must be absolute");
        phoenix_bundle::Bundle bundle(
            folder, phoenix_model::mapped_file_digest,
            phoenix_model::WeightSource::config(config.at("weight_source")), false, "auto", 120,
            "cuda-texture-objects/1");
        if (bundle.manifest.at("architectures") != Json::array({"sm_120"}))
            throw std::runtime_error("Native CUDA bundle architecture mismatch");
        auto frame = bundle.select_frame(width, height, [](const Json&) {});
        auto packed = bundle.load_weights();
        const size_t workspace_bytes = frame.at("arenas").at("workspace").at("bytes");
        if (workspace_bytes > uint64_t(8) * 1024 * 1024 * 1024 ||
            packed.size() > uint64_t(2) * 1024 * 1024 * 1024)
            throw std::runtime_error("Native CUDA allocation budget exceeded");
        s.allocate(s.workspace_raw, s.workspace, workspace_bytes);
        s.allocate(s.weight_raw, s.weights, packed.size());
        check(cudaMemcpyAsync(s.weights, packed.data(), packed.size(), cudaMemcpyHostToDevice,
                              s.stream));
        const auto input_desc = cudaCreateChannelDesc<float4>();
        check(cudaMallocArray(&s.input, &input_desc, width, height, cudaArraySurfaceLoadStore));
        const cudaChannelFormatDesc output_desc{16, 16, 16, 16, cudaChannelFormatKindFloat};
        check(cudaMallocArray(&s.output, &output_desc, width, height, cudaArraySurfaceLoadStore));
        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = s.input;
        cudaTextureDesc texture{};
        texture.addressMode[0] = texture.addressMode[1] = cudaAddressModeClamp;
        texture.filterMode = cudaFilterModePoint;
        texture.readMode = cudaReadModeElementType;
        texture.normalizedCoords = 1;
        check(cudaCreateTextureObject(&s.color, &resource, &texture, nullptr));
        check(cudaCreateSurfaceObject(&s.input_surface, &resource));
        resource.res.array.array = s.output;
        check(cudaCreateSurfaceObject(&s.result, &resource));
        std::map<std::string, CUfunction> functions;
        for (const auto& item : bundle.kernels.items()) {
            const auto& kernel = item.value();
            CUmodule module{};
            driver(cuModuleLoad(&module, kernel.at("binary").get<std::string>().c_str()));
            s.modules.push_back(module);
            std::string entry = kernel.at("entry");
            if (kernel.contains("large_input_entry") &&
                uint64_t(width) * height >= kernel.at("large_input_min_pixels").get<uint64_t>())
                entry = kernel.at("large_input_entry");
            driver(cuModuleGetFunction(&functions[item.key()], module, entry.c_str()));
        }
        const std::array<phoenix_launch_plan::Arena, 2> arenas{
            {{reinterpret_cast<uint64_t>(s.workspace), workspace_bytes},
             {reinterpret_cast<uint64_t>(s.weights), packed.size()}}};
        for (size_t index = 1; index + 1 < frame.at("launches").size(); ++index) {
            const auto& source = frame.at("launches")[index];
            auto plan = phoenix_bundle::launch(source, [&](const Json& binding) -> uint64_t {
                const std::string role = binding.at("role");
                if (role == "color")
                    return s.color;
                if (role == "history_previous" || role == "motion")
                    return 0;
                if (role == "neural_output_surface")
                    return s.result;
                throw std::runtime_error("Unsupported native CUDA descriptor: " + role);
            });
            steps_.push_back({functions.at(source.at("name").get<std::string>()), plan.grid,
                              plan.block, phoenix_launch_plan::materialize(plan, arenas)});
        }
        phoenix_addon::Controls controls;
        controls.reset = 1;
        controls.inverted = 0;
        controls.tone = config.value("local_tone", 1.f);
        controls.structure = config.value("local_structure", 1.f);
        controls.skin = config.value("skin_structure", -1.f);
        controls.auto_mask = config.value("auto_mask", true);
        const auto resolved =
            phoenix_addon::resolve_controls(controls, width, height, false, false, 0);
        phoenix_addon::patch_stem(steps_.front().bytes.data(), steps_.front().bytes.size(),
                                  resolved);
        phoenix_addon::patch_head(steps_.back().bytes.data(), steps_.back().bytes.size(), resolved);
        auto clear = phoenix_bundle::hex(frame.at("launches")[0].at("template_hex"));
        uint32_t words{};
        std::memcpy(&words, clear.data() + 8, sizeof(words));
        if (!words || size_t(words) * 4 > workspace_bytes)
            throw std::runtime_error("Native counter extent");
        check(cudaStreamSynchronize(s.stream));
        bool capturing = false;
        try {
            check(cudaStreamBeginCapture(s.stream, cudaStreamCaptureModeThreadLocal));
            capturing = true;
            check(cudaMemsetAsync(s.workspace, 0xff, size_t(words) * 4, s.stream));
            for (auto& step : steps_) {
                size_t bytes = step.bytes.size();
                void* extra[]{CU_LAUNCH_PARAM_BUFFER_POINTER, step.bytes.data(),
                              CU_LAUNCH_PARAM_BUFFER_SIZE, &bytes, CU_LAUNCH_PARAM_END};
                driver(cuLaunchKernel(step.function, step.grid[0], step.grid[1], step.grid[2],
                                      step.block[0], step.block[1], step.block[2], 0,
                                      reinterpret_cast<CUstream>(s.stream), nullptr, extra));
            }
            auto result = cudaStreamEndCapture(s.stream, &s.graph);
            capturing = false;
            check(result);
            check(cudaGraphInstantiate(&s.executable, s.graph, nullptr, nullptr, 0));
        } catch (...) {
            if (capturing) {
                cudaGraph_t abandoned{};
                cudaStreamEndCapture(s.stream, &abandoned);
                if (abandoned)
                    cudaGraphDestroy(abandoned);
            }
            throw;
        }
        check(cudaEventCreate(&s.start));
        check(cudaEventCreate(&s.end));
        status = {{"backend", "native-cuda"},
                  {"state", "ready"},
                  {"device", requested},
                  {"gpu", gpu.name},
                  {"model", bundle.weight_source.model_id},
                  {"model_resource_sha256", bundle.model_resource.sha256},
                  {"boundary_abi", "cuda-texture-objects/1"},
                  {"frame_contract", "reset-rgb/1"},
                  {"width", width},
                  {"height", height},
                  {"neural_launches", steps_.size()},
                  {"workspace_bytes", workspace_bytes},
                  {"weight_bytes", packed.size()},
                  {"frames", 0},
                  {"timing_scope", "GPU counter clear and full native stem/model/head; host "
                                   "transfers reported separately"}};
    }
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    ~Executor() {
        if (selected_ >= 0)
            cudaSetDevice(selected_);
    }
    const uint8_t* run_rgba8(const uint8_t* bytes, size_t stride, bool bgra) {
        if (failed_)
            throw std::runtime_error("Native CUDA session failed; recreate it");
        check(cudaSetDevice(selected_));
        auto& s = resources_;
        try {
            const size_t count = size_t(width) * height * 4;
            if (!s.byte_buffer) {
                check(cudaMalloc(&s.byte_buffer, count));
                check(cudaMalloc(&s.invalid, sizeof(uint32_t)));
                check(cudaHostAlloc(&s.byte_staging, count, cudaHostAllocDefault));
            }
            auto started = Clock::now();
            for (unsigned y = 0; y < height; ++y)
                std::memcpy(s.byte_staging + size_t(y) * width * 4, bytes + size_t(y) * stride,
                            size_t(width) * 4);
            const double prepare_ms = milliseconds(started);
            started = Clock::now();
            check(cudaMemcpyAsync(s.byte_buffer, s.byte_staging, count, cudaMemcpyHostToDevice,
                                  s.stream));
            check(unpack_rgba8(s.byte_buffer, s.input_surface, width, height, bgra, s.stream));
            check(cudaEventRecord(s.start, s.stream));
            check(cudaGraphLaunch(s.executable, s.stream));
            check(cudaEventRecord(s.end, s.stream));
            check(cudaMemsetAsync(s.invalid, 0, sizeof(uint32_t), s.stream));
            check(pack_rgba8(s.result, s.byte_buffer, s.invalid, width, height, bgra, s.stream));
            check(cudaMemcpyAsync(s.byte_staging, s.byte_buffer, count, cudaMemcpyDeviceToHost,
                                  s.stream));
            uint32_t invalid{};
            check(cudaMemcpyAsync(&invalid, s.invalid, sizeof(invalid), cudaMemcpyDeviceToHost,
                                  s.stream));
            check(cudaStreamSynchronize(s.stream));
            if (invalid)
                throw std::runtime_error("Nonfinite native output");
            const double transfers_ms = milliseconds(started);
            float gpu_ms{};
            check(cudaEventElapsedTime(&gpu_ms, s.start, s.end));
            status.update({{"state", "completed"},
                           {"operator_ms", gpu_ms},
                           {"prepare_ms", prepare_ms},
                           {"execute_with_transfers_ms", transfers_ms},
                           {"resolve_ms", 0.0},
                           {"frame_api", "bounded-rgba8/1"},
                           {"frames", status.at("frames").get<uint64_t>() + 1}});
            return s.byte_staging;
        } catch (...) {
            failed_ = true;
            status["state"] = "failed";
            throw;
        }
    }
    const std::vector<float>& run(const float* color) {
        if (failed_)
            throw std::runtime_error("Native CUDA session failed; recreate it");
        check(cudaSetDevice(selected_));
        // Packed presentation never needs these host float planes. Allocate
        // them only when an explicit bounded float V2 caller uses that route.
        if (rgba_.empty()) {
            rgba_.resize(size_t(width) * height);
            raw_.resize(size_t(width) * height * 4);
            rgb_.resize(size_t(width) * height * 3);
        }
        auto started = Clock::now();
        for (size_t pixel = 0; pixel < rgba_.size(); ++pixel) {
            const float r = color[pixel * 3], g = color[pixel * 3 + 1], b = color[pixel * 3 + 2];
            if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b))
                throw std::runtime_error("Native color must be finite");
            rgba_[pixel] = make_float4(r, g, b, 1.f);
        }
        const double prepare_ms = milliseconds(started);
        auto& s = resources_;
        try {
            started = Clock::now();
            check(cudaMemcpy2DToArrayAsync(s.input, 0, 0, rgba_.data(), width * 16, width * 16,
                                           height, cudaMemcpyHostToDevice, s.stream));
            check(cudaEventRecord(s.start, s.stream));
            check(cudaGraphLaunch(s.executable, s.stream));
            check(cudaEventRecord(s.end, s.stream));
            check(cudaMemcpy2DFromArrayAsync(raw_.data(), width * 8, s.output, 0, 0, width * 8,
                                             height, cudaMemcpyDeviceToHost, s.stream));
            check(cudaStreamSynchronize(s.stream));
            const double transfers_ms = milliseconds(started);
            float gpu_ms{};
            check(cudaEventElapsedTime(&gpu_ms, s.start, s.end));
            started = Clock::now();
            static_assert(sizeof(__half) == sizeof(uint16_t));
            half_rgba_to_rgb(reinterpret_cast<const uint16_t*>(raw_.data()), rgb_.data(),
                             rgba_.size());
            for (const float value : rgb_)
                if (!std::isfinite(value))
                    throw std::runtime_error("Nonfinite native output");
            status.update({{"state", "completed"},
                           {"operator_ms", gpu_ms},
                           {"prepare_ms", prepare_ms},
                           {"execute_with_transfers_ms", transfers_ms},
                           {"resolve_ms", milliseconds(started)},
                           {"timing_intervals", 1},
                           {"frames", status.at("frames").get<uint64_t>() + 1}});
            return rgb_;
        } catch (...) {
            failed_ = true;
            status["state"] = "failed";
            throw;
        }
    }
};
} // namespace phoenix_native_cuda
