#pragma once
#include "routing.h"
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace phoenix_addon::phoenix {
inline constexpr const char* comparison_names[] = {
    "Three-way", "Pair wipe", "Absolute RGB", "Signed RGB", "Heatmap", "Luminance difference"};
inline constexpr const char* source_names[] = {"Input", "NVIDIA DLSS5", "Phoenix DLSS5"};
inline constexpr char comparison_shader[] = R"hlsl(
Texture2D<float4> inputImage : register(t0);
Texture2D<float4> nativeImage : register(t1);
Texture2D<float4> phoenixImage : register(t2);
RWTexture2D<float4> destination : register(u0);
cbuffer Params : register(b0) { uint width, height, mode, sourceA; uint sourceB, divider; float gain; uint unused; };
float4 source(uint index, uint2 p) {
    float4 value = phoenixImage.Load(int3(p,0));
    if (index == 0) value = inputImage.Load(int3(p,0));
    if (index == 1) value = nativeImage.Load(int3(p,0));
    return value;
}
[numthreads(16,16,1)] void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) return;
    float4 a = source(sourceA, tid.xy), b = source(sourceB, tid.xy);
    if (mode == 1) { destination[tid.xy] = tid.x < divider ? a : b; return; }
    float3 d = (a.rgb - b.rgb) * gain;
    float3 result = saturate(abs(d));
    if (mode == 3) result = saturate(0.5 + 0.5*d);
    if (mode == 4) {
        float v = saturate(max(abs(d.r), max(abs(d.g), abs(d.b))));
        result = float3(saturate(4*v-2), saturate(4*v)-saturate(4*v-3), saturate(4*v)-saturate(4*v-1));
    }
    if (mode == 5) result = saturate(abs(dot(d, float3(0.2126,0.7152,0.0722)))).xxx;
    destination[tid.xy] = float4(result,1);
}
)hlsl";
inline void comparison_hr(HRESULT value) {
    if (FAILED(value))
        throw std::runtime_error("Phoenix DLSS5 comparison D3D12 error " +
                                 std::to_string(unsigned(value)));
}
inline Microsoft::WRL::ComPtr<ID3DBlob> compile_comparison() {
    Microsoft::WRL::ComPtr<ID3DBlob> shader, errors;
    const auto status = D3DCompile(comparison_shader, sizeof(comparison_shader) - 1,
                                   "phoenix_comparison", nullptr, nullptr, "main", "cs_5_1",
                                   D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS,
                                   0, &shader, &errors);
    if (FAILED(status))
        throw std::runtime_error(errors ? static_cast<const char*>(errors->GetBufferPointer())
                                        : "Comparison shader compile failed");
    return shader;
}
// These resources share the owning Session's submission fence and retirement.
// Snapshot sources first: never read the engine output while writing it.
// In wipe mode the displayed bank holds the previous frame (see record):
// sources_ doubles as the lag bank, so no extra texture or copy is needed.
class Comparison {
    using Resource = Microsoft::WRL::ComPtr<ID3D12Resource>;
    std::array<Resource, 3> sources_;
    bool first_ = true;
    Resource result_;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline_;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> heap_;
    unsigned width_, height_;
    DXGI_FORMAT format_;
    static void transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* image,
                           D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        if (before == after)
            return;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {image, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        cmd->ResourceBarrier(1, &barrier);
    }
    Resource texture(ID3D12Device* device, bool uav) {
        D3D12_RESOURCE_DESC description{};
        description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        description.Width = width_;
        description.Height = height_;
        description.DepthOrArraySize = 1;
        description.MipLevels = 1;
        description.Format = format_;
        description.SampleDesc.Count = 1;
        description.Flags =
            uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        Resource result;
        comparison_hr(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
                                                      uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                                          : D3D12_RESOURCE_STATE_COPY_DEST,
                                                      nullptr, IID_PPV_ARGS(&result)));
        return result;
    }

  public:
    Comparison(ID3D12Device* device, unsigned width, unsigned height, DXGI_FORMAT format)
        : width_(width), height_(height), format_(format) {
        if (!device || !width || !height)
            throw std::invalid_argument("Comparison geometry");
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
        comparison_hr(
            device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)));
        if (!(support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) ||
            !(support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_LOAD))
            throw std::invalid_argument("Comparison requires shader-load and typed-store support");
        for (auto& source : sources_)
            source = texture(device, false);
        result_ = texture(device, true);
        D3D12_DESCRIPTOR_RANGE ranges[2]{{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0},
                                         {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 3}};
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable = {2, ranges};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[1].Constants = {0, 0, 8};
        D3D12_ROOT_SIGNATURE_DESC desc{2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        Microsoft::WRL::ComPtr<ID3DBlob> blob, error;
        comparison_hr(
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error));
        comparison_hr(device->CreateRootSignature(0, blob->GetBufferPointer(),
                                                  blob->GetBufferSize(), IID_PPV_ARGS(&root_)));
        auto shader = compile_comparison();
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
        pipeline.pRootSignature = root_.Get();
        pipeline.CS = {shader->GetBufferPointer(), shader->GetBufferSize()};
        comparison_hr(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&pipeline_)));
        D3D12_DESCRIPTOR_HEAP_DESC heap_description{};
        heap_description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_description.NumDescriptors = 4;
        heap_description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        comparison_hr(device->CreateDescriptorHeap(&heap_description, IID_PPV_ARGS(&heap_)));
        auto handle = heap_->GetCPUDescriptorHandleForHeapStart();
        auto stride = device->GetDescriptorHandleIncrementSize(heap_description.Type);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        for (auto& source : sources_) {
            device->CreateShaderResourceView(source.Get(), &srv, handle);
            handle.ptr += stride;
        }
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = format;
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(result_.Get(), nullptr, &uav, handle);
    }
    void record(ID3D12GraphicsCommandList* cmd, const std::array<ID3D12Resource*, 3>& sources,
                const std::array<D3D12_RESOURCE_STATES, 3>& states, ID3D12Resource* output,
                D3D12_RESOURCE_STATES output_state, unsigned mode, unsigned source_a,
                unsigned source_b, float divider, float gain) {
        if (!cmd || !output || mode < 1 || mode > 5 || source_a > 2 || source_b > 2 ||
            !std::isfinite(gain) || gain < 1 || gain > 64 || !std::isfinite(divider) ||
            divider < .1f || divider > .9f)
            throw std::invalid_argument("Comparison settings");
        auto validate = [&](ID3D12Resource* image) {
            if (!image)
                throw std::invalid_argument("Comparison missing source");
            auto description = image->GetDesc();
            if (description.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
                description.Width != width_ || description.Height != height_ ||
                description.Format != format_ || description.MipLevels != 1 ||
                description.DepthOrArraySize != 1 || description.SampleDesc.Count != 1)
                throw std::invalid_argument("Comparison source mismatch");
        };
        validate(output);
        for (auto* source : sources)
            validate(source);
        const bool lag = wipe_uses_previous_frame(mode);
        if (!lag || first_) {
            for (unsigned iteration_index = 0; iteration_index < 3; ++iteration_index) {
                transition(cmd, sources[iteration_index], states[iteration_index],
                           D3D12_RESOURCE_STATE_COPY_SOURCE);
                cmd->CopyResource(sources_[iteration_index].Get(), sources[iteration_index]);
                transition(cmd, sources[iteration_index], D3D12_RESOURCE_STATE_COPY_SOURCE,
                           states[iteration_index]);
                transition(cmd, sources_[iteration_index].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
        } else {
            // Display bank already holds the previous frame: make it readable.
            for (unsigned iteration_index = 0; iteration_index < 3; ++iteration_index)
                transition(cmd, sources_[iteration_index].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        struct Constants {
            unsigned width, height, mode, source_a, source_b, divider;
            float gain;
            unsigned unused;
        };
        Constants constants{
            width_, height_, mode, source_a, source_b, unsigned(float(width_) * divider), gain, 0};
        auto* heap = heap_.Get();
        cmd->SetDescriptorHeaps(1, &heap);
        cmd->SetComputeRootSignature(root_.Get());
        cmd->SetPipelineState(pipeline_.Get());
        cmd->SetComputeRootDescriptorTable(0, heap_->GetGPUDescriptorHandleForHeapStart());
        cmd->SetComputeRoot32BitConstants(1, 8, &constants, 0);
        cmd->Dispatch((width_ + 15) / 16, (height_ + 15) / 16, 1);
        transition(cmd, result_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        transition(cmd, output, output_state, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(output, result_.Get());
        transition(cmd, output, D3D12_RESOURCE_STATE_COPY_DEST, output_state);
        transition(cmd, result_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (lag && !first_) {
            // Store the current frame for the next wipe. The game textures
            // are untouched by our dispatch above, so they are as valid now
            // as at entry; the bank ends COPY_DEST like every other path.
            for (unsigned iteration_index = 0; iteration_index < 3; ++iteration_index) {
                transition(cmd, sources_[iteration_index].Get(),
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_COPY_DEST);
                transition(cmd, sources[iteration_index], states[iteration_index],
                           D3D12_RESOURCE_STATE_COPY_SOURCE);
                cmd->CopyResource(sources_[iteration_index].Get(), sources[iteration_index]);
                transition(cmd, sources[iteration_index], D3D12_RESOURCE_STATE_COPY_SOURCE,
                           states[iteration_index]);
            }
        } else {
            for (auto& source : sources_)
                transition(cmd, source.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_COPY_DEST);
        }
        first_ = false;
    }
};
}
