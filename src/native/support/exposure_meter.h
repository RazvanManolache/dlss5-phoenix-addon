#pragma once
// Log-average luminance meter feeding the exposure bridge. See
// exposure_meter shader (embedded bytecode) for why metering exists at all
// (no exposure at NGX level on RE9) and what the two passes compute.
//
// The 1x1 value target takes ExposureTexture's slot in the bridge and
// split shaders unchanged: it is 1x1 R32_FLOAT, satisfies their exposure
// validation, and carries an exposure-equivalent E = key / Lavg that the
// bridge inverts to paper white exactly as it would an NGX exposure.
//
// Barriers follow the network-output precedent (network.h:
// the unconditional post-write transition that Bonsai proves): after each
// pass the written texture is transitioned UAV -> NON_PIXEL_SHADER_RESOURCE
// so the next reader sees flushed data. No pre-transitions: UAV writes
// land regardless (same proof), and first-frame targets start in UAV from
// creation, which is exactly what the first dispatch wants.
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include "../shaders/meter_bytecode.h"

namespace phoenix_addon::exposure_meter {
using Microsoft::WRL::ComPtr;

// Middle-grey key: scene log-average is assumed to sit at 23% of paper
// white. Calibrated by the c16g replay sweep (b14 minimum on the 0.23-0.24
// plateau, validated on b13/b15 at -25% mean_rel vs 0.18); CNR_PHOENIX_METER_KEY
// still overrides it at runtime, and the replay/ASI name the effective key.
constexpr float meter_key = 0.23f;
constexpr float meter_key_min = 0.01f;
constexpr float meter_key_max = 1.0f;
// Parses the override. Anything absent, unparseable, or outside
// [0.01, 1.0] falls back to the default: a typo must never silently
// rescale every frame. The range check subsumes finiteness (NaN and
// infinities fail both comparisons). CPU tests pin each case.
inline float parse_meter_key(const char* text) noexcept {
    if (!text || !*text)
        return meter_key;
    char* end = nullptr;
    const float parsed_value = std::strtof(text, &end);
    if (end == text || !(parsed_value >= meter_key_min) || !(parsed_value <= meter_key_max))
        return meter_key;
    return parsed_value;
}
inline float resolve_meter_key() noexcept {
    // Win32 read (windows.h arrives via d3d12.h): the CRT getenv is a
    // deprecation warning under /WX. Overlong values read as absent.
    char value[64]{};
    if (!GetEnvironmentVariableA("CNR_PHOENIX_METER_KEY", value, DWORD(sizeof(value))))
        return meter_key;
    return parse_meter_key(value);
}
// Temporal smoothing rate: E may move at most this ratio per frame toward
// its target (b13 frame 88: a one-frame 64%-black transient spiked raw E
// 15x, throwing the visible region out of the network's domain). The
// shader implements this formula; the CPU tests pin it here.
constexpr float meter_smoothing_rate = 1.15f;
inline float smooth_exposure(float previous, float target, bool first) noexcept {
    if (first || !(previous > 0.0f))
        return target;
    const float minimum_exposure = previous / meter_smoothing_rate;
    const float maximum_exposure = previous * meter_smoothing_rate;
    if (target < minimum_exposure)
        return minimum_exposure;
    if (target > maximum_exposure)
        return maximum_exposure;
    return target;
}

inline void hr(HRESULT value) {
    if (FAILED(value))
        throw std::runtime_error("phoenix exposure meter D3D12 failure " +
                                 std::to_string(unsigned(value)));
}

inline void transition(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource,
                       D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    if (before == after)
        return;
    D3D12_RESOURCE_BARRIER transition{};
    transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    transition.Transition.pResource = resource;
    transition.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    transition.Transition.StateBefore = before;
    transition.Transition.StateAfter = after;
    cmd->ResourceBarrier(1, &transition);
}

class Meter {
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pipeline_;
    ComPtr<ID3D12DescriptorHeap> heap_;
    ComPtr<ID3D12Resource> tiles_;
    ComPtr<ID3D12Resource> value_;
    ComPtr<ID3D12Resource> prev_;
    unsigned width_ = 0, height_ = 0, stride_ = 0;
    bool first_ = true;
    float key_ = meter_key;

    static void make_target(ID3D12Device* device, unsigned width, unsigned height,
                            DXGI_FORMAT format, ComPtr<ID3D12Resource>& out) {
        D3D12_HEAP_PROPERTIES properties{};
        properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = width;
        texture.Height = height;
        texture.DepthOrArraySize = 1;
        texture.MipLevels = 1;
        texture.Format = format;
        texture.SampleDesc.Count = 1;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        hr(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &texture,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                           IID_PPV_ARGS(&out)));
    }

  public:
    Meter(ID3D12Device* device, unsigned width, unsigned height)
        : device_(device), width_(width), height_(height), key_(resolve_meter_key()) {
        if (!device || !width || !height || width > 16384 || height > 16384)
            throw std::invalid_argument("phoenix exposure meter extent");

        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0};
        ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 0, 0, 3};
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
        pipeline.CS = {meter_bytes, sizeof(meter_bytes)};
        hr(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&pipeline_)));

        D3D12_DESCRIPTOR_HEAP_DESC heap{};
        heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap.NumDescriptors = 6;
        heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&heap_)));
        stride_ = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        make_target(device_.Get(), 64, 64, DXGI_FORMAT_R32G32B32A32_FLOAT, tiles_);
        make_target(device_.Get(), 1, 1, DXGI_FORMAT_R32_FLOAT, value_);
        make_target(device_.Get(), 1, 1, DXGI_FORMAT_R32_FLOAT, prev_);

        const auto cpu = heap_->GetCPUDescriptorHandleForHeapStart();
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        srv.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        auto tiles_srv = cpu;
        tiles_srv.ptr += stride_;
        device_->CreateShaderResourceView(tiles_.Get(), &srv, tiles_srv);
        srv.Format = DXGI_FORMAT_R32_FLOAT;
        auto prev_srv = cpu;
        prev_srv.ptr += size_t(stride_) * 2;
        device_->CreateShaderResourceView(prev_.Get(), &srv, prev_srv);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uav.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        auto tiles_uav = cpu;
        tiles_uav.ptr += size_t(stride_) * 3;
        device_->CreateUnorderedAccessView(tiles_.Get(), nullptr, &uav, tiles_uav);
        uav.Format = DXGI_FORMAT_R32_FLOAT;
        auto value_uav = cpu;
        value_uav.ptr += size_t(stride_) * 4;
        device_->CreateUnorderedAccessView(value_.Get(), nullptr, &uav, value_uav);
        auto prev_uav = cpu;
        prev_uav.ptr += size_t(stride_) * 5;
        device_->CreateUnorderedAccessView(prev_.Get(), nullptr, &uav, prev_uav);
    }

    Meter(const Meter&) = delete;
    ID3D12Resource* target() const noexcept {
        return value_.Get();
    }
    float key() const noexcept {
        return key_;
    }

    // `source` is R11G11B10_FLOAT or already-wide R16G16B16A16_FLOAT (the meter
    // reads float4 either way), in whatever state the epoch left it in;
    // the caller transitions it to NON_PIXEL_SHADER_RESOURCE and back, the
    // same contract as the bridge. Only R11G11B10_FLOAT and R16G16B16A16_FLOAT
    // are metered: the meter measures what the bridge converts, nothing else.
    void record(ID3D12GraphicsCommandList* cmd, ID3D12Resource* source) {
        if (!cmd || !source)
            throw std::invalid_argument("phoenix exposure meter null binding");
        const auto desc = source->GetDesc();
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Width != width_ ||
            desc.Height != height_ || desc.MipLevels != 1 || desc.DepthOrArraySize != 1 ||
            desc.SampleDesc.Count != 1)
            throw std::invalid_argument("phoenix exposure meter source geometry");
        if (desc.Format != DXGI_FORMAT_R11G11B10_FLOAT &&
            desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT)
            throw std::invalid_argument("phoenix exposure meter refuses this source format");

        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = desc.Format;
        srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Texture2D.MipLevels = 1;
        device_->CreateShaderResourceView(source, &srv,
                                          heap_->GetCPUDescriptorHandleForHeapStart());

        uint32_t key_bits = 0;
        const float key = key_;
        std::memcpy(&key_bits, &key, sizeof(key_bits));
        auto* raw = heap_.Get();
        cmd->SetDescriptorHeaps(1, &raw);
        cmd->SetComputeRootSignature(root_.Get());
        cmd->SetPipelineState(pipeline_.Get());
        cmd->SetComputeRootDescriptorTable(0, heap_->GetGPUDescriptorHandleForHeapStart());
        const unsigned pass1[4]{width_, height_, key_bits, 1};
        cmd->SetComputeRoot32BitConstants(1, 4, pass1, 0);
        cmd->Dispatch(8, 8, 1);
        transition(cmd, tiles_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        // First frame takes the raw target (no previous E exists yet); every
        // later frame rate-limits toward it. The smoothed E lands in both
        // value_ (this frame's answer) and prev_ (next frame's reference).
        const unsigned reduce_phase = first_ ? 3u : 2u;
        first_ = false;
        const unsigned pass2[4]{width_, height_, key_bits, reduce_phase};
        cmd->SetComputeRoot32BitConstants(1, 4, pass2, 0);
        cmd->Dispatch(1, 1, 1);
        transition(cmd, value_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(cmd, prev_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
};
} // namespace production_game41::exposure_meter
