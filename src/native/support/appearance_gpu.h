#pragma once
#include "compute_pass.h"
namespace phoenix_addon::phoenix::direct {
// Independent SDR proxy composition. Style constants match the inspected
// 310.8 table; this does not claim equivalence for other model editions.
inline constexpr const char* appearance_delta_shader = R"hlsl(
Texture2D<float4> Original:register(t0);
Texture2D<float4> Neural:register(t1);
Texture2D<float4> Mask:register(t2);
RWTexture2D<float4> Output:register(u0);
cbuffer C:register(b0) { uint W,H,Style,Flags,ToneBits,Unused; float Intensity,Gain; uint Spare; };
float3 saturation(float3 c,float scale) {
 float lo=min(c.x,min(c.y,c.z)),hi=max(c.x,max(c.y,c.z)),l=(lo+hi)*.5;
 float d=hi-lo,den=1-abs(2*l-1),s=den>0?d/den:0;
 return d>0?saturate(l+(c-l)*(s>0?saturate(s*scale)/s:0)):c;
}
[numthreads(16,16,1)] void main(uint3 p:SV_DispatchThreadID) {
 if(p.x>=W||p.y>=H)return;
 float3 original=Original.Load(int3(p.xy,0)).rgb,c=Neural.Load(int3(p.xy,0)).rgb;
 float t=saturate(asfloat(ToneBits));
 if(Style==1) { c=saturate(c*exp2(-.1*t)); c=saturate(c+(-.25*t)*(c*c*(3-2*c)-c)); c=saturation(c,1-.1*t); }
 if(Style==2)c=saturation(c,1-.15*t);
 float amount=Intensity*((Flags&1)!=0?Mask.Load(int3(p.xy,0)).r:1);
 Output[p.xy]=float4(amount*(c-original),0);
})hlsl";
inline constexpr const char* appearance_compose_shader = R"hlsl(
Texture2D<float4> Delta:register(t0);
Texture2D<float4> Base:register(t1);
Texture2D<float4> UI:register(t2);
RWTexture2D<float4> Output:register(u0);
cbuffer C:register(b0) { uint W,H,Unused,Flags,A,B; float X,Y; uint Z; };
[numthreads(16,16,1)] void main(uint3 p:SV_DispatchThreadID) {
 if(p.x>=W||p.y>=H)return;
 float4 base=Base.Load(int3(p.xy,0)),ui=0;
 float a=0;
 if((Flags&1)!=0) {ui=UI.Load(int3(p.xy,0)); a=saturate((Flags&2)!=0?ui.r:ui.a);}
 if((Flags&4)!=0)base.rgb=ui.rgb+base.rgb*(1-a);
 base.rgb=saturate(base.rgb+(1-a)*Delta.Load(int3(p.xy,0)).rgb);
 Output[p.xy]=base;
})hlsl";
inline constexpr const char* appearance_warp_shader = R"hlsl(
Texture2D<float4> Source:register(t0);
Texture2D<float4> Field:register(t1);
Texture2D<float4> Previous:register(t2);
RWTexture2D<float4> Output:register(u0);
cbuffer C:register(b0) {uint W,H,Unused,Mode,A,B;float MotionX,MotionY;uint PriorValid;};
float4 sampleSource(float2 p) {
 p=clamp(p,0,float2(W-1,H-1)); int2 b=int2(floor(p));float2 f=frac(p);int2 hi=int2(W-1,H-1);
 return lerp(lerp(Source.Load(int3(b,0)),Source.Load(int3(min(b+int2(1,0),hi),0)),f.x),
 lerp(Source.Load(int3(min(b+int2(0,1),hi),0)),Source.Load(int3(min(b+1,hi),0)),f.x),f.y);
}
float2 previousOffset(float2 p) {
 p=clamp(p,0,float2(W-1,H-1));int2 b=int2(floor(p));float2 f=frac(p);int2 hi=int2(W-1,H-1);
 return lerp(lerp(Previous.Load(int3(b,0)).rg,Previous.Load(int3(min(b+int2(1,0),hi),0)).rg,f.x),
 lerp(Previous.Load(int3(min(b+int2(0,1),hi),0)).rg,Previous.Load(int3(min(b+1,hi),0)).rg,f.x),f.y);
}
[numthreads(16,16,1)]void main(uint3 p:SV_DispatchThreadID) {
 if(p.x>=W||p.y>=H)return;
 float4 f=Field.Load(int3(p.xy,0));float2 extent=float2(W,H);
 float2 s=float2(p.xy)+(Mode==1?f.rg:f.ba)*extent;
 if(Mode!=2) {Output[p.xy]=sampleSource(s);return;}
 float2 q=s+sampleSource(s).rg*float2(MotionX,MotionY);
 float2 old=q+previousOffset(q)*extent;
 if(any(s<0)||any(s>extent-1)||any(q<0)||any(q>extent-1)) old=float2(p.xy)+extent*2;
 Output[p.xy]=float4(old-float2(p.xy),0,0);
})hlsl";
struct AppearanceTextures {
    ID3D12Resource *mask=nullptr,*ui=nullptr,*alpha=nullptr,*backbuffer=nullptr,*field=nullptr;
    bool any() const {return mask||ui||alpha||backbuffer||field;}
};
// Per-executor resources; all inputs enter/leave shader-read state. Owners
// retain caller textures to their submission fence and select a retired ring
// slot. Network history is never an output of this presentation stage.
class AppearanceStage {
    ComPtr<ID3D12Device> device_;
    std::shared_ptr<phoenix_native::memory::Generation> accounting_;
    unsigned w_,h_;
    ComPtr<ID3D12Resource> mask_,original_,motion_,prior_,delta_,output_,unwarped_;
    std::unique_ptr<Pass> grade_,compose_,color_warp_,motion_warp_,mask_warp_,output_warp_;
    bool had_mask_=false,had_field_=false;
    uint32_t generation_=0;
    ComPtr<ID3D12Resource> image(DXGI_FORMAT format) {
        auto value=phoenix_native::image(device_.Get(),w_,h_,format,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::memory::resource(device_.Get(),value.Get(),accounting_,phoenix_native::memory::Kind::bridge);
        return value;
    }
    void ensure(DXGI_FORMAT motion_format) {
        if(grade_)return;
        grade_=std::make_unique<Pass>(device_.Get(),appearance_delta_shader,accounting_,32);
        compose_=std::make_unique<Pass>(device_.Get(),appearance_compose_shader,accounting_,32);
        color_warp_=std::make_unique<Pass>(device_.Get(),appearance_warp_shader,accounting_,32);
        motion_warp_=std::make_unique<Pass>(*color_warp_,accounting_);
        mask_warp_=std::make_unique<Pass>(*color_warp_,accounting_);
        output_warp_=std::make_unique<Pass>(*color_warp_,accounting_);
        original_=image(DXGI_FORMAT_R16G16B16A16_FLOAT);motion_=image(motion_format);
        prior_=image(DXGI_FORMAT_R32G32B32A32_FLOAT);delta_=image(DXGI_FORMAT_R32G32B32A32_FLOAT);
        output_=image(DXGI_FORMAT_R16G16B16A16_FLOAT);unwarped_=image(DXGI_FORMAT_R16G16B16A16_FLOAT);
    }
    static void copy(ID3D12GraphicsCommandList* cmd,ID3D12Resource* from,ID3D12Resource* to) {
        const auto read=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        phoenix_native::transition(cmd,from,read,D3D12_RESOURCE_STATE_COPY_SOURCE);
        phoenix_native::transition(cmd,to,read,D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(to,from);
        phoenix_native::transition(cmd,to,D3D12_RESOURCE_STATE_COPY_DEST,read);
        phoenix_native::transition(cmd,from,D3D12_RESOURCE_STATE_COPY_SOURCE,read);
    }
    void warp(ID3D12GraphicsCommandList* cmd,Pass& pass,ID3D12Resource* source,ID3D12Resource* field,
              ID3D12Resource* previous,ID3D12Resource* target,unsigned mode,unsigned slot,const Controls& controls) {
        Constants c{};c.width=w_;c.height=h_;c.mode=mode;c.divider=controls.motion_x;c.gain=controls.motion_y;
        phoenix_native::transition(cmd,target,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        pass.record(cmd,{source,field,previous},target,c,false,slot);
        phoenix_native::transition(cmd,target,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
  public:
    AppearanceStage(ID3D12Device* device,unsigned width,unsigned height,
                    std::shared_ptr<phoenix_native::memory::Generation> accounting={})
        :device_(device),accounting_(accounting),w_(width),h_(height) {mask_=image(DXGI_FORMAT_R32G32B32A32_FLOAT);}
    ID3D12Resource* mask_target() const {return mask_.Get();}
    bool changed(const AppearanceTextures& textures,uint32_t generation=0) const {return had_mask_!=bool(textures.mask)||had_field_!=bool(textures.field)||generation_!=generation;}
    void absent() {had_mask_=had_field_=false;generation_=0;}
    void remember_mode(const AppearanceTextures& textures,uint32_t generation=0) {had_mask_=textures.mask;had_field_=textures.field;generation_=generation;}
    static bool requested(const Controls& controls,const AppearanceTextures& textures) {
        return textures.any()||controls.intensity!=1||controls.style!=0||controls.ui_correction!=0;
    }
    void prepare(ID3D12GraphicsCommandList* cmd,ID3D12Resource* color,ID3D12Resource* motion,
                 const AppearanceTextures& textures,const Controls& controls,unsigned slot) {
        ensure(motion?motion->GetDesc().Format:DXGI_FORMAT_R16G16_FLOAT);
        copy(cmd,color,original_.Get());
        if(textures.field) {
            warp(cmd,*color_warp_,original_.Get(),textures.field,nullptr,color,0,slot,controls);
            if(motion) {copy(cmd,motion,motion_.Get());warp(cmd,*motion_warp_,motion_.Get(),textures.field,
                had_field_&&!controls.reset?prior_.Get():textures.field,motion,2,slot,controls);}
        }
        if(textures.mask) {
            if(textures.field)warp(cmd,*mask_warp_,textures.mask,textures.field,nullptr,mask_.Get(),0,slot,controls);
            else copy(cmd,textures.mask,mask_.Get());
        }
    }
    ID3D12Resource* present(ID3D12GraphicsCommandList* cmd,ID3D12Resource* neural,
                            const AppearanceTextures& textures,const Controls& controls,unsigned slot) {
        const auto read=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if(textures.field) {warp(cmd,*output_warp_,neural,textures.field,nullptr,unwarped_.Get(),1,slot,controls);neural=unwarped_.Get();}
        const bool post=controls.intensity<1||textures.mask||textures.ui||textures.alpha||textures.backbuffer||(controls.style&&controls.tone>0);
        if(post) {
            Constants c{};c.width=w_;c.height=h_;c.reset=unsigned(controls.style);c.mode=textures.mask?1u:0u;
            std::memcpy(&c.source_a,&controls.tone,sizeof(float));c.divider=controls.intensity;
            phoenix_native::transition(cmd,delta_.Get(),read,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            grade_->record(cmd,{original_.Get(),neural,textures.mask},delta_.Get(),c,false,slot);
            phoenix_native::transition(cmd,delta_.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,read);
            auto* ui=textures.ui?textures.ui:textures.alpha;
            c.mode=(ui?1u:0u)|(textures.alpha?2u:0u)|(textures.ui&&!textures.backbuffer?4u:0u);
            phoenix_native::transition(cmd,output_.Get(),read,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            compose_->record(cmd,{delta_.Get(),textures.backbuffer?textures.backbuffer:original_.Get(),ui},output_.Get(),c,false,slot);
            phoenix_native::transition(cmd,output_.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,read);neural=output_.Get();
        }
        if(textures.field)copy(cmd,textures.field,prior_.Get());
        had_mask_=textures.mask;had_field_=textures.field;return neural;
    }
};
}
