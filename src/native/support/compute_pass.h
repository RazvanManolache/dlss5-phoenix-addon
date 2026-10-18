#pragma once
#include "../network/production_network.h"
#include <d3dcompiler.h>
namespace phoenix_addon::phoenix::direct {
using Microsoft::WRL::ComPtr;
// Diagnostic crumb hook: the probes leave it null (silent); the addon points
// it at its crash-safe logger to bracket GPU-stage recording.
inline void (*phoenix_crumb)(const char*) = nullptr;
inline void crumb(const char* text) {
    if (phoenix_crumb)
        phoenix_crumb(text);
}
struct Constants {
    unsigned width, height, reset, mode;
    // Two stable root-constant words have pass-specific meanings. Comparison
    // shaders read source indices; meter/normalize shaders reinterpret pixel
    // jitter as float bits. Aliases describe the use without changing offsets.
    union {
        unsigned source_a;
        unsigned jitter_x_bits;
    };
    union {
        unsigned source_b;
        unsigned jitter_y_bits;
    };
    float divider, gain;
    unsigned resolve;
};
static_assert(sizeof(Constants) == 9 * sizeof(unsigned), "Direct root-constant ABI changed");
class Pass {
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pso_;
    std::vector<ComPtr<ID3D12DescriptorHeap>> heaps_;
    std::vector<std::array<ID3D12Resource*, 4>> bindings_;

  public:
    void account(const std::shared_ptr<phoenix_native::memory::Generation>& generation) {
        phoenix_native::memory::track(root_.Get(), generation,
                                      phoenix_native::memory::Kind::pipeline);
        phoenix_native::memory::track(pso_.Get(), generation,
                                      phoenix_native::memory::Kind::pipeline);
        for (const auto& heap : heaps_)
            phoenix_native::memory::track(heap.Get(), generation,
                                          phoenix_native::memory::Kind::views);
    }
    static ComPtr<ID3DBlob> compile(const char* source) {
        ComPtr<ID3DBlob> bytes, errors;
        HRESULT result = D3DCompile(
            source, strlen(source), "phoenix_direct", nullptr, nullptr, "main", "cs_5_1",
            D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_WARNINGS_ARE_ERRORS, 0, &bytes, &errors);
        if (FAILED(result))
            throw std::runtime_error(errors ? static_cast<const char*>(errors->GetBufferPointer())
                                            : "Direct shader compile failed");
        return bytes;
    }
    Pass(ID3D12Device* device, const char* source,
         std::shared_ptr<phoenix_native::memory::Generation> accounting = {}, unsigned slots = 2)
        : device_(device), heaps_(slots), bindings_(slots) {
        phoenix_bundle::require(slots>=1 && slots<=32,"Compute descriptor ring extent");
        D3D12_DESCRIPTOR_RANGE ranges[] = {{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0},
                                           {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 3}};
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable = {2, ranges};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[1].Constants = {0, 0, 9};
        D3D12_ROOT_SIGNATURE_DESC root_description{2, params, 0, nullptr,
                                                   D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob, errors;
        phoenix_native::hr(D3D12SerializeRootSignature(
            &root_description, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors));
        phoenix_native::hr(device->CreateRootSignature(
            0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_)));
        phoenix_native::memory::track(root_.Get(), accounting,
                                      phoenix_native::memory::Kind::pipeline);
        auto code = compile(source);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_description{};
        pipeline_description.pRootSignature = root_.Get();
        pipeline_description.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        phoenix_native::hr(
            device->CreateComputePipelineState(&pipeline_description, IID_PPV_ARGS(&pso_)));
        phoenix_native::memory::track(pso_.Get(), accounting,
                                      phoenix_native::memory::Kind::pipeline);
        create_heaps(accounting);
    }
    // Rotating frame bindings need their own descriptors, but the compiled
    // shader and root signature are identical. Never recompile them per texture.
    Pass(const Pass& pipeline,
         const std::shared_ptr<phoenix_native::memory::Generation>& accounting)
        : device_(pipeline.device_), root_(pipeline.root_), pso_(pipeline.pso_),
          heaps_(pipeline.heaps_.size()),bindings_(pipeline.heaps_.size()) {
        create_heaps(accounting);
    }

  private:
    void create_heaps(const std::shared_ptr<phoenix_native::memory::Generation>& accounting) {
        D3D12_DESCRIPTOR_HEAP_DESC heap_description{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                                    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        for (auto& heap : heaps_) {
            phoenix_native::hr(
                device_->CreateDescriptorHeap(&heap_description, IID_PPV_ARGS(&heap)));
            phoenix_native::memory::track(heap.Get(), accounting,
                                          phoenix_native::memory::Kind::views);
        }
    }

  public:
    template <typename Parameters = Constants>
    void record(ID3D12GraphicsCommandList* cmd, std::array<ID3D12Resource*, 3> inputs,
                ID3D12Resource* output, const Parameters& constants, bool meter = false,
                unsigned slot = 0) {
        static_assert(sizeof(Parameters) == 9 * sizeof(unsigned),
                      "Direct root-constant ABI changed");
        check(slot < heaps_.size(), "Direct descriptor slot");
        auto& heap = heaps_[slot];
        const std::array<ID3D12Resource*, 4> binding{inputs[0], inputs[1], inputs[2], output};
        auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
        auto stride =
            device_->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        if (!(bindings_[slot] == binding)) {
            // Engine textures rotate (double-buffered outputs, feature
            // re-creation): rebuild the views instead of refusing mid-frame,
            // which would strand already-recorded barriers and kill the driver.
            if (bindings_[slot][3])
                crumb("record: rebound rotated textures");
            for (auto* input : inputs) {
                D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
                srv.Format = input ? input->GetDesc().Format : DXGI_FORMAT_R32_FLOAT;
                if (srv.Format == DXGI_FORMAT_R32G8X24_TYPELESS)
                    srv.Format = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
                srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                srv.Texture2D.MipLevels = 1;
                device_->CreateShaderResourceView(input, &srv, cpu);
                cpu.ptr += stride;
            }
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
            uav.Format = output->GetDesc().Format;
            uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
            device_->CreateUnorderedAccessView(output, nullptr, &uav, cpu);
            bindings_[slot] = binding;
        }
        auto* raw = heap.Get();
        cmd->SetDescriptorHeaps(1, &raw);
        cmd->SetComputeRootSignature(root_.Get());
        cmd->SetPipelineState(pso_.Get());
        cmd->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        cmd->SetComputeRoot32BitConstants(1, 9, &constants, 0);
        cmd->Dispatch(meter ? 1 : (constants.width + 15) / 16,
                      meter ? 1 : (constants.height + 15) / 16, 1);
    }
};
}
