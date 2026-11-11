#define NOMINMAX
#include "legacy_depth_api.h"
#include <windows.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <cstring>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
struct AxisSample {
    unsigned first, second;
    float fraction;
};

AxisSample sample_at(unsigned pixel, unsigned source, unsigned destination) {
    const float position = std::clamp((float(pixel) + .5f) * float(source) / float(destination) - .5f,
                                      0.f, float(source - 1));
    const auto first = unsigned(position);
    return {first, std::min(first + 1, source - 1), position - float(first)};
}

template<class Sample>
float interpolate(AxisSample x, AxisSample y, Sample sample) {
    return std::lerp(std::lerp(sample(x.first, y.first), sample(x.second, y.first), x.fraction),
                     std::lerp(sample(x.first, y.second), sample(x.second, y.second), x.fraction), y.fraction);
}

int directml_adapter_index(uint64_t luid) {
    if (!luid) return 0;
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
        throw std::runtime_error("Cannot enumerate the DirectML adapter");
    for (UINT index = 0;; ++index) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if (FAILED(factory->EnumAdapters1(index, &adapter))) break;
        DXGI_ADAPTER_DESC1 info{};
        uint64_t id = 0;
        if (SUCCEEDED(adapter->GetDesc1(&info))) {
            std::memcpy(&id, &info.AdapterLuid, sizeof(id));
            if (id == luid && !(info.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) return int(index);
        }
    }
    throw std::runtime_error("The game's hardware adapter is unavailable to DirectML");
}

Ort::SessionOptions options(PhoenixDepthBackend backend, uint64_t luid) {
    Ort::SessionOptions config;
    config.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    config.SetIntraOpNumThreads(4);
    if (backend == PhoenixDepthBackend::Cuda) {
        const auto providers = Ort::GetAvailableProviders();
        if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end())
            throw std::runtime_error("ONNX Runtime CUDA provider unavailable");
        OrtCUDAProviderOptions provider;
        config.AppendExecutionProvider_CUDA(provider);
    } else if (backend == PhoenixDepthBackend::DirectML) {
        using AppendDml = OrtStatus* (ORT_API_CALL *)(OrtSessionOptions*, int);
        const auto runtime = GetModuleHandleW(L"onnxruntime.dll");
        const auto append = runtime ? reinterpret_cast<AppendDml>(
            GetProcAddress(runtime, "OrtSessionOptionsAppendExecutionProvider_DML")) : nullptr;
        if (!append) throw std::runtime_error("ONNX Runtime DirectML provider unavailable");
        config.DisableMemPattern();
        config.SetExecutionMode(ORT_SEQUENTIAL);
        Ort::ThrowOnError(append(config, directml_adapter_index(luid)));
    }
    return config;
}

struct Estimator {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "Phoenix legacy depth"};
    Ort::Session session;
    std::string input_name, output_name;
    std::vector<int64_t> input_shape;
    ONNXTensorElementDataType input_type;
    std::vector<float> input, disparity;

    Estimator(const wchar_t* path, PhoenixDepthBackend backend, uint64_t luid)
        : session(env, path, options(backend, luid)) {
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
            throw std::runtime_error("Expected one RGB input and one depth output");
        Ort::AllocatorWithDefaultOptions allocator;
        input_name = session.GetInputNameAllocated(0, allocator).get();
        output_name = session.GetOutputNameAllocated(0, allocator).get();
        const auto input_info = session.GetInputTypeInfo(0);
        const auto tensor = input_info.GetTensorTypeAndShapeInfo();
        input_shape = tensor.GetShape();
        input_type = tensor.GetElementType();
        if (input_shape.size() != 4 || (input_shape[0] != 1 && input_shape[0] >= 0) ||
            input_shape[1] != 3 || (input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
                                   input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16))
            throw std::runtime_error("Expected float RGB tensor [1,3,H,W]");
    }

    void estimate(const uint8_t* rgba, unsigned width, unsigned height, float* depth) {
        if (!rgba || !depth || !width || !height || width > 8192 || height > 8192)
            throw std::runtime_error("Unsupported captured image");
        auto shape = input_shape;
        shape[0] = 1;
        auto patch_size = [](unsigned dimension, unsigned largest) {
            return std::max(14, int(std::lround(double(dimension) * 37 / largest)) * 14);
        };
        if (shape[2] < 0) shape[2] = patch_size(height, std::max(width, height));
        if (shape[3] < 0) shape[3] = patch_size(width, std::max(width, height));
        if (shape[2] <= 0 || shape[3] <= 0 || shape[2] > 2048 || shape[3] > 2048)
            throw std::runtime_error("Unsupported depth model input size");
        const auto iw = unsigned(shape[3]), ih = unsigned(shape[2]);
        const auto pixels = size_t(iw) * ih;
        input.resize(3 * pixels);
        constexpr std::array mean{.485f, .456f, .406f}, deviation{.229f, .224f, .225f};
        for (unsigned y = 0; y < ih; ++y) {
            const auto sy = sample_at(y, height, ih);
            for (unsigned x = 0; x < iw; ++x) {
                const auto sx = sample_at(x, width, iw);
                for (unsigned channel = 0; channel < 3; ++channel) {
                    const float value = interpolate(sx, sy, [&](unsigned px, unsigned py) {
                        return float(rgba[(size_t(py) * width + px) * 4 + channel]) / 255.f;
                    });
                    input[channel * pixels + size_t(y) * iw + x] = (value - mean[channel]) / deviation[channel];
                }
            }
        }
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Float16_t> half;
        Ort::Value tensor{nullptr};
        if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            half.reserve(input.size());
            for (float value : input) half.emplace_back(value);
            tensor = Ort::Value::CreateTensor(memory, half.data(), half.size(), shape.data(), shape.size());
        } else {
            tensor = Ort::Value::CreateTensor(memory, input.data(), input.size(), shape.data(), shape.size());
        }
        const char* inputs[]{input_name.c_str()};
        const char* outputs[]{output_name.c_str()};
        const auto output = session.Run(Ort::RunOptions{}, inputs, &tensor, 1, outputs, 1);
        const auto info = output.front().GetTensorTypeAndShapeInfo();
        const auto dimensions = info.GetShape();
        if (dimensions.size() < 2 || dimensions.size() > 4)
            throw std::runtime_error("Expected one scalar depth plane");
        const auto dw = dimensions.back(), dh = dimensions[dimensions.size() - 2];
        if (dw <= 0 || dh <= 0 || dw > 2048 || dh > 2048 || info.GetElementCount() != size_t(dw) * dh)
            throw std::runtime_error("Expected one scalar depth plane");
        disparity.resize(info.GetElementCount());
        if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            const auto* data = output.front().GetTensorData<float>();
            std::copy_n(data, disparity.size(), disparity.begin());
        } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const auto* data = output.front().GetTensorData<Ort::Float16_t>();
            std::transform(data, data + disparity.size(), disparity.begin(), [](auto v) { return v.ToFloat(); });
        } else {
            throw std::runtime_error("Expected float depth output");
        }
        if (!std::all_of(disparity.begin(), disparity.end(), [](float value) { return std::isfinite(value); }))
            throw std::runtime_error("Depth model returned nonfinite values");
        // Clip extreme predictions before converting relative inverse depth to
        // the feeder's 0..1 convention. No camera calibration is implied.
        std::vector<float> distribution;
        const auto step = std::max<size_t>(1, disparity.size() / 4096);
        for (size_t i = 0; i < disparity.size(); i += step) distribution.push_back(disparity[i]);
        std::sort(distribution.begin(), distribution.end());
        const auto trim = distribution.size() / 50;
        const float low = distribution[trim], high = distribution[distribution.size() - 1 - trim];
        if (high - low <= 1e-6f || !std::isfinite(high - low))
            throw std::runtime_error("Depth model returned a flat or invalid map");
        for (unsigned y = 0; y < height; ++y) {
            const auto sy = sample_at(y, unsigned(dh), height);
            for (unsigned x = 0; x < width; ++x) {
                const auto sx = sample_at(x, unsigned(dw), width);
                const float value = interpolate(sx, sy, [&](unsigned px, unsigned py) {
                    return disparity[size_t(py) * size_t(dw) + px];
                });
                depth[size_t(y) * width + x] = 1.f - std::clamp((value - low) / (high - low), 0.f, 1.f);
            }
        }
    }
};
}

extern "C" __declspec(dllexport) void* PhoenixDepthOpenV1(const wchar_t* path, PhoenixDepthBackend backend,
                                                         uint64_t luid, char* error, uint32_t size) {
    try { return new Estimator(path, backend, luid); }
    catch (const std::exception& e) { if (error && size) std::snprintf(error, size, "%s", e.what()); return nullptr; }
}
extern "C" __declspec(dllexport) int PhoenixDepthEstimateV1(void* handle, const uint8_t* rgba, uint32_t width,
                                                         uint32_t height, float* depth, char* error, uint32_t size) {
    try {
        if (!handle) throw std::runtime_error("Depth session is unavailable");
        static_cast<Estimator*>(handle)->estimate(rgba, width, height, depth);
        return 1;
    } catch (const std::exception& e) { if (error && size) std::snprintf(error, size, "%s", e.what()); return 0; }
}
extern "C" __declspec(dllexport) void PhoenixDepthCloseV1(void* handle) { delete static_cast<Estimator*>(handle); }
