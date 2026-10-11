#pragma once
#include "compute_pass.h"
#include "resolution.h"
namespace phoenix_addon::phoenix {
// Copy independently observed depth (plane zero for combined depth/stencil).
// The owned snapshot keeps descriptors stable when the engine rotates buffers.
class GuideAdapter {
    std::shared_ptr<phoenix_native::memory::Generation> accounting_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12Resource> depth_, guides_, nr_depth_;
    Resolution geometry_;
    std::unique_ptr<direct::Pass> extract_depth_;
    direct::Pass pack_;
    unsigned width_, height_, depth_width_ = 0, depth_height_ = 0;
    DXGI_FORMAT depth_format_ = DXGI_FORMAT_UNKNOWN;
    static constexpr const char* shader = R"(
Texture2D<float2> motion:register(t0);Texture2D<float> depth:register(t1);
RWTexture2D<float4> output:register(u0);
cbuffer Params:register(b0){uint width,height,reset,mode;uint a,b;float divider,gain;uint resolve;};
[numthreads(16,16,1)]void main(uint3 tid:SV_DispatchThreadID){
 if(tid.x>=width||tid.y>=height)return;
 uint dw,dh;depth.GetDimensions(dw,dh);
 if(tid.x<reset||tid.y<mode||tid.x>=reset+a||tid.y>=mode+b){output[tid.xy]=0;return;}
 uint2 q=min(uint2((float2(tid.xy)-float2(reset,mode)+.5)*float2(dw,dh)/float2(a,b)),uint2(dw-1,dh-1));
 float2 mv=motion.Load(int3(tid.xy,0))*float2(divider,gain);
 float z=depth.Load(int3(q,0));bool valid=all(isfinite(mv))&&isfinite(z)&&z>=0&&z<=1;
 output[tid.xy]=float4(mv,z,valid?1:0);
})";

  public:
    static DXGI_FORMAT copy_format(DXGI_FORMAT format) {
        return format == DXGI_FORMAT_R32G8X24_TYPELESS || format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT
                   ? DXGI_FORMAT_R32G8X24_TYPELESS
                   : DXGI_FORMAT_R32_FLOAT;
    }
    GuideAdapter(ID3D12Device* device, unsigned width, unsigned height,
                 std::shared_ptr<phoenix_native::memory::Generation> accounting = {},
                 Resolution geometry = {})
        : accounting_(accounting), device_(device), pack_(device, shader, accounting),
          width_(width), height_(height) {
        geometry_ = geometry.width ? geometry : fit_resolution(width, height, width, height);
        guides_ = phoenix_native::image(device, width, height, DXGI_FORMAT_R32G32B32A32_FLOAT,
                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, true);
        phoenix_native::memory::resource(device, guides_.Get(), accounting_,
                                         phoenix_native::memory::Kind::guides);
    }
    static bool supported(const D3D12_RESOURCE_DESC& description) {
        return description.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
               description.MipLevels == 1 && description.DepthOrArraySize == 1 &&
               description.SampleDesc.Count == 1 &&
               (description.Format == DXGI_FORMAT_R32_FLOAT ||
                description.Format == DXGI_FORMAT_R32_TYPELESS ||
                description.Format == DXGI_FORMAT_D32_FLOAT ||
                description.Format == DXGI_FORMAT_R32G8X24_TYPELESS ||
                description.Format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT);
    }
    bool matches(ID3D12Resource* depth) const {
        if (!depth)
            return true;
        const auto description = depth->GetDesc();
        return !depth_ ||
               (description.Width == depth_width_ && description.Height == depth_height_ &&
                copy_format(description.Format) == depth_format_);
    }
    ID3D12Resource* native_depth(ID3D12GraphicsCommandList* cmd, ID3D12Resource* motion,
                                 ID3D12Resource* source, D3D12_RESOURCE_STATES state) {
        auto* packed = record(cmd, motion, source, state, 0, 0);
        if (!packed)
            return nullptr;
        if (!extract_depth_) {
            const char* extract_shader = R"(
Texture2D<float4> guides:register(t0); RWTexture2D<float> output:register(u0);
cbuffer Params:register(b0){uint width,height,reset,mode,a,b;float divider,gain;uint resolve;};
[numthreads(16,16,1)]void main(uint3 tid:SV_DispatchThreadID){
 if(tid.x<width&&tid.y<height)output[tid.xy]=guides.Load(int3(tid.xy,0)).z;
})";
            extract_depth_ =
                std::make_unique<direct::Pass>(device_.Get(), extract_shader, accounting_);
            nr_depth_ = phoenix_native::image(device_.Get(), width_, height_, DXGI_FORMAT_R32_FLOAT,
                                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            phoenix_native::memory::resource(device_.Get(), nr_depth_.Get(), accounting_,
                                             phoenix_native::memory::Kind::guides);
        }
        phoenix_native::transition(cmd, nr_depth_.Get(),
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        extract_depth_->record(cmd, {packed, nullptr, nullptr}, nr_depth_.Get(),
                               {width_, height_, 0, 0, 0, 0, 0, 0, 0});
        phoenix_native::transition(cmd, nr_depth_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        return nr_depth_.Get();
    }
    ID3D12Resource* record(ID3D12GraphicsCommandList* cmd, ID3D12Resource* motion,
                           ID3D12Resource* source, D3D12_RESOURCE_STATES state, float motion_x,
                           float motion_y) {
        if (!source || !supported(source->GetDesc()))
            return nullptr;
        const auto desc = source->GetDesc();
        check(matches(source), "Depth geometry changed; retire and recreate guide adapter");
        if (!depth_) {
            depth_width_ = unsigned(desc.Width);
            depth_height_ = desc.Height;
            depth_format_ = copy_format(desc.Format);
            // Keep the depth/stencil family and copy only plane 0. Stencil is never sampled.
            auto owned = desc;
            owned.Format = depth_format_;
            owned.Flags = depth_format_ == DXGI_FORMAT_R32G8X24_TYPELESS
                              ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
                              : D3D12_RESOURCE_FLAG_NONE;
            D3D12_HEAP_PROPERTIES heap_properties{};
            heap_properties.Type = D3D12_HEAP_TYPE_DEFAULT;
            phoenix_native::hr(device_->CreateCommittedResource(
                &heap_properties, D3D12_HEAP_FLAG_NONE, &owned,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&depth_)));
            phoenix_native::memory::resource(device_.Get(), depth_.Get(), accounting_,
                                             phoenix_native::memory::Kind::guides);
        }
        constexpr auto read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        auto transition_source = [&](D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
            if (before == after)
                return;
            D3D12_RESOURCE_BARRIER transition{};
            transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            transition.Transition = {source,
                                     copy_format(desc.Format) == DXGI_FORMAT_R32G8X24_TYPELESS
                                         ? 0u
                                         : D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                     before, after};
            cmd->ResourceBarrier(1, &transition);
        };
        transition_source(state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        phoenix_native::transition(cmd, depth_.Get(), read, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = source;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.pResource = depth_.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        cmd->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        phoenix_native::transition(cmd, depth_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, read);
        transition_source(D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        phoenix_native::transition(cmd, guides_.Get(), read, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        direct::Constants constants{width_,
                                    height_,
                                    geometry_.left,
                                    geometry_.top,
                                    geometry_.image_width,
                                    geometry_.image_height,
                                    motion_x,
                                    motion_y,
                                    0};
        pack_.record(cmd, {motion, depth_.Get(), nullptr}, guides_.Get(), constants);
        phoenix_native::transition(cmd, guides_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, read);
        return guides_.Get();
    }
};
}
