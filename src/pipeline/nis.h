#pragma once
#include "../native/network/production_network.h"
#include "../../third_party/nis/NIS_Config.h"
#include "../../third_party/nis/shader_source.h"
#include <d3dcompiler.h>
#include <cfloat>

namespace phoenix_pipeline {
// Own one scaler per stage instance. The caller retires its consumer fence
// before updating constants/descriptors. Source/output states are restored.
class NisScaler {
    template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pipeline_;
    ComPtr<ID3D12DescriptorHeap> descriptors_;
    ComPtr<ID3D12Resource> constants_;
    std::array<ComPtr<ID3D12Resource>, 2> coefficients_, uploads_;
    std::array<D3D12_PLACED_SUBRESOURCE_FOOTPRINT, 2> footprints_{};
    static void check(HRESULT result) { phoenix_native::hr(result); }
public:
    explicit NisScaler(ID3D12Device* device) : device_(device) {
        D3D12_DESCRIPTOR_RANGE ranges[]{{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0},
            {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 3}};
        D3D12_ROOT_PARAMETER parameters[2]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[0].DescriptorTable = {2, ranges};
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        parameters[1].Descriptor.ShaderRegister = 0;
        D3D12_STATIC_SAMPLER_DESC sampler{};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MaxLOD = FLT_MAX; sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC signature{2, parameters, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> serialized, errors;
        check(D3D12SerializeRootSignature(&signature, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors));
        check(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&root_)));
        D3D_SHADER_MACRO defines[]{{"NIS_SCALER", "1"}, {"NIS_HDR_MODE", "0"},
            {"NIS_USE_HALF_PRECISION", "0"}, {"NIS_BLOCK_WIDTH", "32"}, {"NIS_BLOCK_HEIGHT", "24"},
            {"NIS_THREAD_GROUP_SIZE", "128"}, {"NIS_VIEWPORT_SUPPORT", "1"}, {nullptr, nullptr}};
        ComPtr<ID3DBlob> shader;
        const auto compiled = D3DCompile(phoenix_nis_shader, sizeof(phoenix_nis_shader) - 1,
            "Phoenix NVIDIA Image Scaling", defines, nullptr, "main", "cs_5_0",
            D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &shader, &errors);
        if (FAILED(compiled)) throw std::runtime_error(errors
            ? std::string(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize())
            : "NIS shader compilation failed");
        D3D12_COMPUTE_PIPELINE_STATE_DESC description{};
        description.pRootSignature = root_.Get(); description.CS = {shader->GetBufferPointer(), shader->GetBufferSize()};
        check(device->CreateComputePipelineState(&description, IID_PPV_ARGS(&pipeline_)));
        D3D12_DESCRIPTOR_HEAP_DESC heap{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        check(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&descriptors_)));
        constants_ = phoenix_native::buffer(device, 256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        const float* sources[]{&coef_scale[0][0], &coef_usm[0][0]};
        for (unsigned i = 0; i != 2; ++i) {
            coefficients_[i] = phoenix_native::image(device, 2, 64, DXGI_FORMAT_R32G32B32A32_FLOAT,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, false);
            auto shape = coefficients_[i]->GetDesc(); UINT64 bytes = 0;
            device->GetCopyableFootprints(&shape, 0, 1, 0, &footprints_[i], nullptr, nullptr, &bytes);
            uploads_[i] = phoenix_native::buffer(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            uint8_t* mapped = nullptr; D3D12_RANGE empty{};
            check(uploads_[i]->Map(0, &empty, reinterpret_cast<void**>(&mapped)));
            for (unsigned row = 0; row != 64; ++row)
                std::memcpy(mapped + footprints_[i].Offset + row * footprints_[i].Footprint.RowPitch, sources[i] + row * 8, 32);
            uploads_[i]->Unmap(0, nullptr);
        }
    }
    NisScaler(const NisScaler& other)
        :device_(other.device_),root_(other.root_),pipeline_(other.pipeline_),
         coefficients_(other.coefficients_),uploads_(other.uploads_),footprints_(other.footprints_) {
        D3D12_DESCRIPTOR_HEAP_DESC heap{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,4,D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,0};
        check(device_->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&descriptors_)));
        constants_=phoenix_native::buffer(device_.Get(),256,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    }
    bool record(ID3D12GraphicsCommandList* command, ID3D12Resource* input, D3D12_RESOURCE_STATES input_state,
                ID3D12Resource* output, D3D12_RESOURCE_STATES output_state, float sharpness,
                unsigned left = 0, unsigned top = 0, unsigned active_width = 0, unsigned active_height = 0) {
        const auto source = input->GetDesc(), target = output->GetDesc();
        if (!active_width) active_width = unsigned(source.Width);
        if (!active_height) active_height = source.Height;
        NISConfig configuration{};
        if (!NVScalerUpdateConfig(configuration, sharpness, left, top, active_width, active_height,
            unsigned(source.Width), source.Height, 0, 0, unsigned(target.Width), target.Height,
            unsigned(target.Width), target.Height, NISHDRMode::None)) return false;
        void* mapped = nullptr; D3D12_RANGE empty{};
        check(constants_->Map(0, &empty, &mapped)); std::memcpy(mapped, &configuration, sizeof(configuration));
        constants_->Unmap(0, nullptr);
        auto cpu = descriptors_->GetCPUDescriptorHandleForHeapStart();
        const auto stride = device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        for (auto* texture : {input, coefficients_[0].Get(), coefficients_[1].Get()}) {
            D3D12_SHADER_RESOURCE_VIEW_DESC view{}; view.Format = texture->GetDesc().Format;
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D; view.Texture2D.MipLevels = 1;
            view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            device_->CreateShaderResourceView(texture, &view, cpu); cpu.ptr += stride;
        }
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{}; view.Format = target.Format; view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        device_->CreateUnorderedAccessView(output, nullptr, &view, cpu);
        for (unsigned i = 0; i != 2; ++i) {
            phoenix_native::transition(command, coefficients_[i].Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION from{uploads_[i].Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, footprints_[i]};
            D3D12_TEXTURE_COPY_LOCATION to{coefficients_[i].Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
            command->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            phoenix_native::transition(command, coefficients_[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        phoenix_native::transition(command, input, input_state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(command, output, output_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto* heap = descriptors_.Get(); command->SetDescriptorHeaps(1, &heap);
        command->SetComputeRootSignature(root_.Get()); command->SetPipelineState(pipeline_.Get());
        command->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        command->SetComputeRootConstantBufferView(1, constants_->GetGPUVirtualAddress());
        command->Dispatch((unsigned(target.Width) + 31) / 32, (target.Height + 23) / 24, 1);
        phoenix_native::transition(command, input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, input_state);
        phoenix_native::transition(command, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, output_state);
        return true;
    }
};
} // namespace phoenix_pipeline
