#pragma once
// Multiplies the network result back into linear HDR on the way out.
//
// Why this exists: the HDR bridge runs the network in the SDR-normalized
// domain (paper white at 1.0) while the engine's output stays linear HDR
// R16G16B16A16_FLOAT, so the result must be rescaled by the frame's paper
// white before the engine sees it. The white derivation is
// character-identical to the bridge's (1/max(e, 1e-6) from the same meter
// value), so normalize/denormalize round-trips up to the two fp16 stores,
// whose bound the CPU tests pin; see denormalize shader (embedded bytecode).
//
// What this class does NOT do: it does not decide the output's resting
// state. The caller transitions the engine output to UNORDERED_ACCESS
// before `record` and names the state it must be left in; the pass leaves
// it there with the unconditional post-write transition, the same form the
// network output uses (network.h) and Bonsai proves. The
// observer hooks ResourceBarrier on the list, so both transitions are
// epoch-observed exactly like the copy path's were -- the output-state
// contract does not change, only the bytes written under it. The caller
// restores its own bindings afterwards, the same contract as the bridge.
#include <d3d12.h>
#include <wrl/client.h>
#include <stdexcept>
#include <string>
#include "../shaders/denormalize_bytecode.h"

namespace phoenix_addon::denormalize {
using Microsoft::WRL::ComPtr;

inline void hr(HRESULT value) {
    if (FAILED(value))
        throw std::runtime_error("phoenix denormalize D3D12 failure " +
                                 std::to_string(unsigned(value)));
}

// Formats this pass reads and writes. The network result and the engine
// output are both plan-native RGBA16F; anything else is refused rather
// than converted on a guess. Exposure takes ExposureTexture's slot
// unchanged (1x1 single-channel half or float).
inline bool result_readable(DXGI_FORMAT format) noexcept {
    return format == DXGI_FORMAT_R16G16B16A16_FLOAT;
}
inline bool output_writable(DXGI_FORMAT format) noexcept {
    return format == DXGI_FORMAT_R16G16B16A16_FLOAT;
}
inline bool exposure_readable(DXGI_FORMAT format) noexcept {
    return format == DXGI_FORMAT_R16_FLOAT || format == DXGI_FORMAT_R32_FLOAT;
}

class Denormalize {
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pipeline_;
    ComPtr<ID3D12DescriptorHeap> heap_;
    unsigned width_ = 0, height_ = 0, stride_ = 0;

  public:
    Denormalize(ID3D12Device* device, unsigned width, unsigned height)
        : device_(device), width_(width), height_(height) {
        if (!device || !width || !height || width > 16384 || height > 16384)
            throw std::invalid_argument("phoenix denormalize extent");

        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0};
        ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 2};
        D3D12_ROOT_PARAMETER parameters[2]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[0].DescriptorTable = {2, ranges};
        parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[1].Constants = {0, 0, 4};
        D3D12_ROOT_SIGNATURE_DESC description{2, parameters, 0, nullptr,
                                              D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> serialized, errors;
        hr(D3D12SerializeRootSignature(&description, D3D_ROOT_SIGNATURE_VERSION_1, &serialized,
                                       &errors));
        hr(device->CreateRootSignature(0, serialized->GetBufferPointer(),
                                       serialized->GetBufferSize(), IID_PPV_ARGS(&root_)));

        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
        pipeline.pRootSignature = root_.Get();
        pipeline.CS = {denormalize_bytes, sizeof(denormalize_bytes)};
        hr(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&pipeline_)));

        D3D12_DESCRIPTOR_HEAP_DESC heap{};
        heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap.NumDescriptors = 3;
        heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&heap_)));
        stride_ = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        // No UAV is created here: the engine output is a different resource
        // every session and a UAV bakes the resource, so record() creates
        // the output UAV per call in the reserved third slot.
    }

    Denormalize(const Denormalize&) = delete;
    unsigned width() const noexcept {
        return width_;
    }
    unsigned height() const noexcept {
        return height_;
    }

    // `result` is the network output in NON_PIXEL_SHADER_RESOURCE;
    // `exposure` is the frame's 1x1 exposure value in the same state;
    // `output` is the engine output already transitioned to
    // UNORDERED_ACCESS by the caller. The pass leaves `output` in `after`
    // via the unconditional post-write transition. The caller restores its
    // own bindings afterwards; this pass sets a descriptor heap, root
    // signature and PSO.
    void record(ID3D12GraphicsCommandList* cmd, ID3D12Resource* result, ID3D12Resource* exposure,
                ID3D12Resource* output, D3D12_RESOURCE_STATES after) {
        if (!cmd || !result || !exposure || !output)
            throw std::invalid_argument("phoenix denormalize null binding");
        const auto rdesc = result->GetDesc();
        if (rdesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || rdesc.Width != width_ ||
            rdesc.Height != height_ || rdesc.MipLevels != 1 || rdesc.DepthOrArraySize != 1 ||
            rdesc.SampleDesc.Count != 1)
            throw std::invalid_argument("phoenix denormalize result geometry");
        if (!result_readable(rdesc.Format))
            throw std::invalid_argument("phoenix denormalize refuses this result format");
        const auto edesc = exposure->GetDesc();
        if (edesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || edesc.Width != 1 ||
            edesc.Height != 1 || edesc.MipLevels != 1 || edesc.DepthOrArraySize != 1 ||
            edesc.SampleDesc.Count != 1)
            throw std::invalid_argument("phoenix denormalize exposure geometry");
        if (!exposure_readable(edesc.Format))
            throw std::invalid_argument("phoenix denormalize refuses this exposure format");
        const auto odesc = output->GetDesc();
        if (odesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || odesc.Width != width_ ||
            odesc.Height != height_ || odesc.MipLevels != 1 || odesc.DepthOrArraySize != 1 ||
            odesc.SampleDesc.Count != 1)
            throw std::invalid_argument("phoenix denormalize output geometry");
        if (!output_writable(odesc.Format))
            throw std::invalid_argument("phoenix denormalize refuses this output format");

        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        const auto heap_start = heap_->GetCPUDescriptorHandleForHeapStart();
        srv.Format = rdesc.Format;
        device_->CreateShaderResourceView(result, &srv, heap_start);
        srv.Format = edesc.Format;
        auto exposure_slot = heap_start;
        exposure_slot.ptr += stride_;
        device_->CreateShaderResourceView(exposure, &srv, exposure_slot);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = odesc.Format;
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        auto output_slot = heap_start;
        output_slot.ptr += size_t(stride_) * 2;
        device_->CreateUnorderedAccessView(output, nullptr, &uav, output_slot);

        auto* raw = heap_.Get();
        const unsigned extent[4]{width_, height_, 0, 0};
        cmd->SetDescriptorHeaps(1, &raw);
        cmd->SetComputeRootSignature(root_.Get());
        cmd->SetPipelineState(pipeline_.Get());
        cmd->SetComputeRootDescriptorTable(0, heap_->GetGPUDescriptorHandleForHeapStart());
        cmd->SetComputeRoot32BitConstants(1, 4, extent, 0);
        cmd->Dispatch((width_ + 15) / 16, (height_ + 15) / 16, 1);
        // Flush the write so the engine's next read sees this frame: the
        // unconditional post-write transition, the same form the network
        // output uses (network.h) and Bonsai proves. UAV
        // writes land regardless, so there is no pre-transition.
        D3D12_RESOURCE_BARRIER written{};
        written.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        written.Transition.pResource = output;
        written.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        written.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        written.Transition.StateAfter = after;
        cmd->ResourceBarrier(1, &written);
    }
};
} // namespace production_game41::denormalize
