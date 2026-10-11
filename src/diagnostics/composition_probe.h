#pragma once
#include "../native/support/compute_pass.h"
namespace phoenix_addon::phoenix::portable {
// Sample actual input/output differences once per second without a CPU/GPU wait.
// This is coverage evidence, not proof of perceptual image quality.
class CompositionProbe {
    static constexpr unsigned probe_width = 64, probe_height = 36;
    static constexpr const char* shader = R"(
Texture2D<float4> input:register(t0);Texture2D<float4> output:register(t1);
Texture2D<float4> map:register(t2);RWTexture2D<float4> probe:register(u0);
cbuffer Params:register(b0){uint width,height,reset,mode;uint a,b;float divider,gain;uint resolve;};
[numthreads(16,16,1)]void main(uint3 t:SV_DispatchThreadID){
 if(t.x>=width||t.y>=height)return;
 uint w,h;input.GetDimensions(w,h);uint2 p=min(uint2((float2(t.xy)+.5)*float2(w,h)/float2(width,height)),uint2(w-1,h-1));
 float3 delta=abs(output.Load(int3(p,0)).rgb-input.Load(int3(p,0)).rgb);
 float4 m=map.Load(int3(p,0));
 probe[t.xy]=float4(m.z>=.9999?1:0,(delta.x+delta.y+delta.z)/3,max(delta.x,max(delta.y,delta.z)),all(isfinite(delta))?1:0);
})";
    direct::Pass pass_;
    ComPtr<ID3D12Resource> image_, read_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp_{};
    UINT64 bytes_ = 0;
    uint64_t fence_ = 0, tick_ = 0;
    uint32_t generation_ = 0;
    Json result_ = Json::object();

  public:
    void account(ID3D12Device* device,
                 const std::shared_ptr<phoenix_native::memory::Generation>& generation) {
        pass_.account(generation);
        phoenix_native::memory::resource(device, image_.Get(), generation,
                                         phoenix_native::memory::Kind::transport);
        phoenix_native::memory::resource(device, read_.Get(), generation,
                                         phoenix_native::memory::Kind::transport);
    }
    explicit CompositionProbe(ID3D12Device* device) : pass_(device, shader) {
        image_ =
            phoenix_native::image(device, probe_width, probe_height, DXGI_FORMAT_R32G32B32A32_FLOAT,
                                  D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
        auto desc = image_->GetDesc();
        device->GetCopyableFootprints(&desc, 0, 1, 0, &fp_, nullptr, nullptr, &bytes_);
        read_ = phoenix_native::buffer(device, bytes_, D3D12_HEAP_TYPE_READBACK,
                                       D3D12_RESOURCE_STATE_COPY_DEST);
    }
    const Json& result() const {
        return result_;
    }
    void invalidate() {
        result_ = Json::object();
    }
    void retire(uint64_t done, uint32_t generation) {
        if (!fence_ || done < fence_)
            return;
        if (generation_ == generation) {
            void* mapped_bytes = nullptr;
            D3D12_RANGE range{0, SIZE_T(bytes_)};
            phoenix_native::hr(read_->Map(0, &range, &mapped_bytes));
            unsigned valid = 0, changed = 0, finite = 0;
            double mean = 0, maximum = 0;
            for (unsigned row_index = 0; row_index < probe_height; ++row_index)
                for (unsigned column_index = 0; column_index < probe_width; ++column_index) {
                    auto* channels =
                        reinterpret_cast<const float*>(static_cast<const char*>(mapped_bytes) +
                                                       row_index * fp_.Footprint.RowPitch) +
                        4 * column_index;
                    valid += channels[0] > .5f;
                    finite += channels[3] > .5f;
                    changed += channels[2] > 1e-5f;
                    mean += channels[1];
                    maximum = std::max(maximum, double(channels[2]));
                }
            D3D12_RANGE none{0, 0};
            read_->Unmap(0, &none);
            result_ = {{"frame", fence_},
                       {"samples", probe_width * probe_height},
                       {"valid_samples", valid},
                       {"changed_samples", changed},
                       {"finite_samples", finite},
                       {"valid_percent", 100.0 * valid / (probe_width * probe_height)},
                       {"changed_percent", 100.0 * changed / (probe_width * probe_height)},
                       {"mean_abs_delta", mean / (probe_width * probe_height)},
                       {"max_abs_delta", maximum}};
        }
        fence_ = 0;
    }
    void record(ID3D12GraphicsCommandList* cmd, ID3D12Resource* input, ID3D12Resource* output,
                ID3D12Resource* map, uint64_t serial, uint32_t generation) {
        const auto tick = GetTickCount64();
        if (fence_ || tick - tick_ < 1000)
            return;
        constexpr auto shader_read_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        direct::Constants constants{probe_width, probe_height};
        phoenix_native::transition(cmd, image_.Get(), shader_read_state,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        pass_.record(cmd, {input, output, map}, image_.Get(), constants);
        phoenix_native::transition(cmd, image_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION from{}, to{};
        from.pResource = image_.Get();
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        to.pResource = read_.Get();
        to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint = fp_;
        cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        phoenix_native::transition(cmd, image_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                   shader_read_state);
        fence_ = serial;
        generation_ = generation;
        tick_ = tick;
    }
};
}
