#pragma once
// Widens a packed HDR colour buffer into the format the captured plans expect.
//
// Why this exists: RE Requiem was first seen handing DLSSNR an R11G11B10_FLOAT
// colour buffer (later menu runs observe R16G16B16A16_FLOAT instead -- see
// rescale_only below), while
// `network.h`'s `validate_input` accepts only
// R16G16B16A16_FLOAT or R32G32B32A32_FLOAT. That gate is right to refuse --
// every plan, arena offset and kernel argument we hold was captured against
// RGBA16F inputs, so feeding the kernels a different layout would be running
// them on something they were never captured with.
//
// The fix is to convert rather than to widen the gate. The FORMAT widening is
// EXACT: R11G11B10_FLOAT and fp16 share a 5-bit exponent with the same bias
// and range; the 6-bit (R,G) and 5-bit (B) mantissas are strict subsets of
// fp16's 10-bit mantissa; denormals and Inf/NaN encode identically. The
// hardware unpacks 11/11/10 to fp32 exactly on read and every such value is
// representable in fp16 on write, so the widening itself never rounds or
// clamps. The VALUE rescale (divide by paper white, from the NGX
// ExposureTexture sampled in-shader) is not exact: it is two fp16 roundings
// whose relative error the CPU tests bound over the measured white range
// rather than assert away. See colour_bridge shader (embedded bytecode) for
// why the rescale exists and where paper white comes from.
//
// What this class does NOT do: it does not decide resource states. The caller
// owns the source and destination and must have them in
// NON_PIXEL_SHADER_RESOURCE and UNORDERED_ACCESS respectively before `record`,
// and must restore whatever the engine expects afterwards. That is deliberate
// -- this project refuses to infer states it has not observed.
#include <d3d12.h>
#include <wrl/client.h>
#include <stdexcept>
#include <string>
#include "../shaders/colour_bridge_bytecode.h"

namespace phoenix_addon::colour_bridge {
using Microsoft::WRL::ComPtr;

inline void hr(HRESULT value) {
    if (FAILED(value))
        throw std::runtime_error("phoenix colour bridge D3D12 failure " +
                                 std::to_string(unsigned(value)));
}

// Formats this bridge will widen, and the one it widens them to. Anything not
// listed is refused rather than converted on a guess.
inline bool widenable(DXGI_FORMAT format) noexcept {
    return format == DXGI_FORMAT_R11G11B10_FLOAT;
}
constexpr DXGI_FORMAT target_format = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Formats that need no widening, only the value rescale. RE9 menus observed
// R16G16B16A16_FLOAT native input (five live runs, formats 10/34/10/19):
// fp16 loads exactly, so the bridge shader's float4 divide applies
// unchanged -- only this CPU gate learns the new path. R32G32B32A32_FLOAT
// stays refused: fp32->fp16 would round, breaking the exactness story, and
// no observed game path needs it.
inline bool rescale_only(DXGI_FORMAT format) noexcept {
    return format == DXGI_FORMAT_R16G16B16A16_FLOAT;
}

// Exposure textures this bridge will read. Streamline documents a 1x1
// resource; both half and float single-channel carriers load as float4,
// anything else is refused rather than sampled on a guess.
inline bool exposure_readable(DXGI_FORMAT format) noexcept {
    return format == DXGI_FORMAT_R16_FLOAT || format == DXGI_FORMAT_R32_FLOAT;
}

// Sources this bridge's record() converts. The shader loads float4 and
// divides by exposure either way, so packed R11G11B10 (widened exactly)
// and already-wide RGBA16F (rescale-only, observed live in RE9 menus)
// share the one path; the CPU tests pin this gate.
inline bool bridge_accepts_source(DXGI_FORMAT format) noexcept {
    return widenable(format) || rescale_only(format);
}

class Bridge {
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pipeline_;
    ComPtr<ID3D12DescriptorHeap> heap_;
    ComPtr<ID3D12Resource> target_;
    unsigned width_ = 0, height_ = 0, stride_ = 0;

  public:
    Bridge(ID3D12Device* device, unsigned width, unsigned height)
        : device_(device), width_(width), height_(height) {
        if (!device || !width || !height || width > 16384 || height > 16384)
            throw std::invalid_argument("phoenix colour bridge extent");

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
        pipeline.CS = {colour_bridge_bytes, sizeof(colour_bridge_bytes)};
        hr(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&pipeline_)));

        D3D12_DESCRIPTOR_HEAP_DESC heap{};
        heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap.NumDescriptors = 3;
        heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&heap_)));
        stride_ = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        // Our own destination, in the format the plans were captured with.
        D3D12_HEAP_PROPERTIES properties{};
        properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = width;
        texture.Height = height;
        texture.DepthOrArraySize = 1;
        texture.MipLevels = 1;
        texture.Format = target_format;
        texture.SampleDesc.Count = 1;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        hr(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &texture,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                           IID_PPV_ARGS(&target_)));

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = target_format;
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        auto slot = heap_->GetCPUDescriptorHandleForHeapStart();
        slot.ptr += size_t(stride_) * 2;
        device->CreateUnorderedAccessView(target_.Get(), nullptr, &uav, slot);
    }

    Bridge(const Bridge&) = delete;
    ID3D12Resource* target() const noexcept {
        return target_.Get();
    }
    unsigned width() const noexcept {
        return width_;
    }
    unsigned height() const noexcept {
        return height_;
    }

    // `source` must already be in NON_PIXEL_SHADER_RESOURCE and must match the
    // extent this bridge was built for; `exposure` is the NGX ExposureTexture
    // (1x1, same state) whose red channel carries the frame's game exposure.
    // The caller restores its own bindings afterwards; this pass sets a
    // descriptor heap, root signature and PSO.
    void record(ID3D12GraphicsCommandList* cmd, ID3D12Resource* source, ID3D12Resource* exposure) {
        if (!cmd || !source || !exposure)
            throw std::invalid_argument("phoenix colour bridge null binding");
        const auto desc = source->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width != width_ ||
            desc.Height != height_ || desc.MipLevels != 1 || desc.DepthOrArraySize != 1 ||
            desc.SampleDesc.Count != 1)
            throw std::invalid_argument("phoenix colour bridge source geometry");
        if (!bridge_accepts_source(desc.Format))
            throw std::invalid_argument("phoenix colour bridge refuses this source format");
        const auto edesc = exposure->GetDesc();
        if (edesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || edesc.Width != 1 ||
            edesc.Height != 1 || edesc.MipLevels != 1 || edesc.DepthOrArraySize != 1 ||
            edesc.SampleDesc.Count != 1)
            throw std::invalid_argument("phoenix colour bridge exposure geometry");
        if (!exposure_readable(edesc.Format))
            throw std::invalid_argument("phoenix colour bridge refuses this exposure format");

        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = desc.Format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        const auto heap_start = heap_->GetCPUDescriptorHandleForHeapStart();
        device_->CreateShaderResourceView(source, &srv, heap_start);
        srv.Format = edesc.Format;
        auto exposure_slot = heap_start;
        exposure_slot.ptr += stride_;
        device_->CreateShaderResourceView(exposure, &srv, exposure_slot);

        auto* raw = heap_.Get();
        const unsigned extent[4]{width_, height_, 0, 0};
        cmd->SetDescriptorHeaps(1, &raw);
        cmd->SetComputeRootSignature(root_.Get());
        cmd->SetPipelineState(pipeline_.Get());
        cmd->SetComputeRootDescriptorTable(0, heap_->GetGPUDescriptorHandleForHeapStart());
        cmd->SetComputeRoot32BitConstants(1, 4, extent, 0);
        cmd->Dispatch((width_ + 15) / 16, (height_ + 15) / 16, 1);
        // Flush the write so the network's SRV read sees this frame, not
        // the last one: the unconditional post-write transition, the same
        // form the network output uses (network.h) and
        // Bonsai proves. UAV writes land regardless, so there is no
        // pre-transition; first-frame targets start in UAV from creation.
        D3D12_RESOURCE_BARRIER written{};
        written.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        written.Transition.pResource = target_.Get();
        written.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        written.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        written.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        cmd->ResourceBarrier(1, &written);
    }
};
} // namespace production_game41::colour_bridge
