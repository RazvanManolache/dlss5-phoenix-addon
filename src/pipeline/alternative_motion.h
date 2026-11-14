#pragma once
#include "motion_plugin.h"
#include "video_motion.h"
#include "luma_block_match.h"
#include <filesystem>

namespace phoenix_pipeline {
using MotionPass = phoenix_addon::phoenix::direct::Pass;
using MotionConstants = phoenix_addon::phoenix::direct::Constants;

template<class Interface>
class MotionPlugin {
    HMODULE library_ = nullptr;
    typename Interface::Destroy destroy_ = nullptr;
public:
    void* context = nullptr;
    typename Interface::Run run = nullptr;

    template<class... Arguments>
    explicit MotionPlugin(const std::filesystem::path& directory, Arguments... arguments) {
        HMODULE module = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&MotionPlugin::anchor), &module))
            throw std::runtime_error("Cannot locate Phoenix motion provider");
        wchar_t path[32768]{};
        const auto length = GetModuleFileNameW(module, path, ARRAYSIZE(path));
        if (!length || length >= ARRAYSIZE(path))
            throw std::runtime_error("Cannot locate Phoenix motion provider");
        namespace fs = std::filesystem;
        auto file = fs::path(path).parent_path() / Interface::filename;
        if (!directory.empty()) file = directory / Interface::filename;
        library_ = LoadLibraryExW(file.c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!library_) throw std::runtime_error("Optional Phoenix motion provider DLL is unavailable");
        auto create = reinterpret_cast<typename Interface::Create>(GetProcAddress(library_, "PhoenixMotionCreateV1"));
        destroy_ = reinterpret_cast<typename Interface::Destroy>(GetProcAddress(library_, "PhoenixMotionDestroyV1"));
        run = reinterpret_cast<typename Interface::Run>(GetProcAddress(library_, "PhoenixMotionRunV1"));
        if (!create || !destroy_ || !run || !(context = create(arguments...))) {
            FreeLibrary(library_);
            throw std::runtime_error("Motion provider initialization failed");
        }
    }

    MotionPlugin(const MotionPlugin&) = delete;
    MotionPlugin& operator=(const MotionPlugin&) = delete;
    ~MotionPlugin() {
        destroy_(context);
        FreeLibrary(library_);
    }
private:
    static void anchor() {}
};

class AlternativeMotion {
    template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
    MotionSource source_;
    unsigned width_, height_, analysis_width_, analysis_height_;
    std::unique_ptr<MotionPass> resize_, convert_, block_match_;
    ComPtr<ID3D12Resource> input_, vectors_, scene_, output_, readback_, upload_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT input_footprint_{}, vector_footprint_{};
    std::unique_ptr<MotionPlugin<phoenix_motion_plugin::FidelityFX>> ffx_;
    std::unique_ptr<MotionPlugin<phoenix_motion_plugin::DIS>> dis_;
    std::unique_ptr<VideoMotion> video_;
    std::vector<unsigned char> previous_;
    ComPtr<ID3D12Resource> previous_luma_;
    unsigned luma_slot_ = 0;
    bool primed_ = false;
    unsigned fidelity_frames_ = 0;

    static constexpr char resize_shader[] = R"hlsl(
Texture2D<float4> Source:register(t0);
RWTexture2D<float4> Output:register(u0);
cbuffer C:register(b0){uint W,H,Gray;uint unused[6];};
[numthreads(16,16,1)] void main(uint3 p:SV_DispatchThreadID) {
 if(p.x>=W||p.y>=H)return;
 uint w,h;Source.GetDimensions(w,h);
 float2 q=(float2(p.xy)+.5)*float2(w,h)/float2(W,H)-.5;
 int2 base=int2(floor(q)), edge=int2(w-1,h-1);
 float2 f=frac(q);
 float3 a=Source.Load(int3(clamp(base,0,edge),0)).rgb;
 float3 b=Source.Load(int3(clamp(base+int2(1,0),0,edge),0)).rgb;
 float3 c=Source.Load(int3(clamp(base+int2(0,1),0,edge),0)).rgb;
 float3 d=Source.Load(int3(clamp(base+1,0,edge),0)).rgb;
 float3 rgb=lerp(lerp(a,b,f.x),lerp(c,d,f.x),f.y);
 if(Gray)rgb=dot(saturate(rgb),float3(.2126,.7152,.0722));
 Output[p.xy]=float4(rgb,1);
})hlsl";
    static constexpr char convert_integer[] = R"hlsl(
Texture2D<int2> Flow:register(t0);
Texture2D<uint> Scene:register(t1);
RWTexture2D<float2> Output:register(u0);
cbuffer C:register(b0){uint W,H,Block,AnalysisW,AnalysisH,SceneCheck;float Factor;uint unused,Nearest;};
[numthreads(16,16,1)] void main(uint3 p:SV_DispatchThreadID) {
 if(p.x>=W||p.y>=H)return;
 uint w,h;Flow.GetDimensions(w,h);
 float2 scale=float2(W,H)/float2(AnalysisW,AnalysisH);
 float2 q=(float2(p.xy)+.5)/scale/Block-.5;
 int2 base=int2(floor(q)), edge=int2(w-1,h-1);
 float2 f=frac(q);
 float2 a=Flow.Load(int3(clamp(base,0,edge),0));
 float2 b=Flow.Load(int3(clamp(base+int2(1,0),0,edge),0));
 float2 c=Flow.Load(int3(clamp(base+int2(0,1),0,edge),0));
 float2 d=Flow.Load(int3(clamp(base+1,0,edge),0));
 float2 v=Nearest ? float2(Flow.Load(int3(clamp(int2(floor(q+.5)),0,edge),0))) :
                    lerp(lerp(a,b,f.x),lerp(c,d,f.x),f.y);
 v*=scale*Factor;
 float2 samplePos=float2(p.xy)+v;
 bool reject=any(samplePos<0)||any(samplePos>=float2(W,H));
 if(SceneCheck)reject=reject||((Scene.Load(int3(1,0,0))&15)!=0);
 Output[p.xy]=reject?float2(W*2,H*2):v;
})hlsl";
    static constexpr char convert_float[] = R"hlsl(
Texture2D<float2> Flow:register(t0);
RWTexture2D<float2> Output:register(u0);
cbuffer C:register(b0){uint W,H,Block,AnalysisW,AnalysisH;uint unused[4];};
[numthreads(16,16,1)] void main(uint3 p:SV_DispatchThreadID) {
 if(p.x>=W||p.y>=H)return;
 float2 scale=float2(W,H)/float2(AnalysisW,AnalysisH);
 float2 q=(float2(p.xy)+.5)/scale-.5;
 int2 base=int2(floor(q)), edge=int2(AnalysisW-1,AnalysisH-1);
 float2 f=frac(q);
 float2 a=Flow.Load(int3(clamp(base,0,edge),0));
 float2 b=Flow.Load(int3(clamp(base+int2(1,0),0,edge),0));
 float2 c=Flow.Load(int3(clamp(base+int2(0,1),0,edge),0));
 float2 d=Flow.Load(int3(clamp(base+1,0,edge),0));
 float2 v=lerp(lerp(a,b,f.x),lerp(c,d,f.x),f.y)*scale;
 float2 samplePos=float2(p.xy)+v;
 Output[p.xy]=any(!isfinite(v))||any(samplePos<0)||any(samplePos>=float2(W,H))?float2(W*2,H*2):v;
})hlsl";

public:
    AlternativeMotion(ID3D12Device* device, MotionSource source, unsigned width, unsigned height,
                      std::array<unsigned,2> extent, const std::filesystem::path& directory)
        : source_(source), width_(width), height_(height), analysis_width_(extent[0]), analysis_height_(extent[1]) {
        const bool fidelity = source_ == MotionSource::FidelityFX;
        if (source_ == MotionSource::Video) {
            // NV12 requires an even extent. Preserve the realized X/Y scale in conversion.
            analysis_width_ = (analysis_width_ + 1) & ~1u;
            analysis_height_ = (analysis_height_ + 1) & ~1u;
            video_ = std::make_unique<VideoMotion>(device, analysis_width_, analysis_height_);
        } else if (source_ == MotionSource::BlockMatch) {
            block_match_ = std::make_unique<MotionPass>(device, luma_block_match, nullptr, 2);
            vectors_ = phoenix_native::image(device, (analysis_width_ + 7) / 8, (analysis_height_ + 7) / 8,
                DXGI_FORMAT_R16G16_SINT, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            previous_luma_ = phoenix_native::image(device, analysis_width_, analysis_height_,
                DXGI_FORMAT_R8_UNORM, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        } else if (fidelity) {
            ffx_ = std::make_unique<MotionPlugin<phoenix_motion_plugin::FidelityFX>>(directory, device, analysis_width_, analysis_height_);
            vectors_ = phoenix_native::image(device, (analysis_width_ + 7) / 8, (analysis_height_ + 7) / 8,
                DXGI_FORMAT_R16G16_SINT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            scene_ = phoenix_native::image(device, 3, 1, DXGI_FORMAT_R32_UINT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        } else {
            if (analysis_width_ < 16 || analysis_height_ < 16)
                throw std::runtime_error("OpenCV DIS requires at least 16 pixels per axis");
            dis_ = std::make_unique<MotionPlugin<phoenix_motion_plugin::DIS>>(directory);
            vectors_ = phoenix_native::image(device, analysis_width_, analysis_height_, DXGI_FORMAT_R32G32_FLOAT,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, false);
            previous_.resize(size_t(analysis_width_) * analysis_height_);
        }
        input_ = phoenix_native::image(device, analysis_width_, analysis_height_,
            fidelity ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8_UNORM, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        output_ = phoenix_native::image(device, width_, height_, DXGI_FORMAT_R16G16_FLOAT,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        resize_ = std::make_unique<MotionPass>(device, resize_shader);
        convert_ = std::make_unique<MotionPass>(device, dis_ ? convert_float : convert_integer);
        if (dis_) {
            const auto input_desc = input_->GetDesc(), vector_desc = vectors_->GetDesc();
            UINT64 bytes = 0;
            device->GetCopyableFootprints(&input_desc, 0, 1, 0, &input_footprint_, nullptr, nullptr, &bytes);
            readback_ = phoenix_native::buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
            device->GetCopyableFootprints(&vector_desc, 0, 1, 0, &vector_footprint_, nullptr, nullptr, &bytes);
            upload_ = phoenix_native::buffer(device, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        }
    }

    std::array<unsigned,2> extent() const { return {analysis_width_, analysis_height_}; }
    bool warming() const { return ffx_ && fidelity_frames_ > 0 && fidelity_frames_ <= 6; }
    ID3D12Resource* output() const { return output_.Get(); }

    // begin and finish use the caller's direct list, fence and allocator.
    // DIS waits for luma readback. GPU routes stay GPU-ordered until
    // the normal caller synchronization point.
    template<class Begin, class Finish>
    bool prepare(ID3D12GraphicsCommandList* list, ID3D12CommandQueue* queue, ID3D12Fence* fence,
                 const uint64_t& serial, ID3D12Resource* color, D3D12_RESOURCE_STATES state,
                 bool reset, bool scene_detection, bool ordered, Begin begin, Finish finish) {
        using phoenix_native::transition;
        reset = reset || !primed_;
        if (reset) fidelity_frames_ = 0;
        begin();
        MotionConstants resize{};
        resize.width = analysis_width_;
        resize.height = analysis_height_;
        resize.reset = !ffx_;
        transition(list, color, state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        auto* current_luma = luma_slot_ ? previous_luma_.Get() : input_.Get();
        resize_->record(list, {color, nullptr, nullptr}, current_luma, resize, false, luma_slot_);
        transition(list, color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, state);
        const auto read = (ffx_ || block_match_) ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE :
                                                  D3D12_RESOURCE_STATE_COPY_SOURCE;
        transition(list, current_luma, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, read);

        bool ready = false;
        if (block_match_) {
            auto* previous_luma = luma_slot_ ? input_.Get() : previous_luma_.Get();
            if (!reset) {
                const auto desc = vectors_->GetDesc();
                MotionConstants c{};
                c.width = unsigned(desc.Width);
                c.height = desc.Height;
                transition(list, vectors_.Get(), read, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                block_match_->record(list, {current_luma, previous_luma, nullptr}, vectors_.Get(), c, false, luma_slot_);
                transition(list, vectors_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, read);
                ready = true;
            }
            if (primed_) transition(list, previous_luma, read, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            luma_slot_ ^= 1;
        } else if (ffx_) {
            const bool ok = ffx_->run(ffx_->context, list, input_.Get(), vectors_.Get(), scene_.Get(), reset);
            transition(list, input_.Get(), read, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!ok) {
                finish(true);
                primed_ = false;
                return false;
            }
            // SDK 1.x explicitly suppresses motion at frame indices 0..5.
            if (fidelity_frames_ <= 6) ++fidelity_frames_;
            ready = fidelity_frames_ > 6;
        } else {
            if (video_) video_->record_luma(list, input_.Get());
            else {
                const D3D12_TEXTURE_COPY_LOCATION from{input_.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
                const D3D12_TEXTURE_COPY_LOCATION to{readback_.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, input_footprint_};
                list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            }
            transition(list, input_.Get(), read, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            finish(!video_);
            ready = video_ ? video_->submit(queue, fence, serial, reset) : compute_dis(reset);
            if (!ready) {
                primed_ = true;
                if (video_ && !ordered) {
                    begin();
                    finish(true);
                }
                return false;
            }
            begin();
            if (dis_) {
                transition(list, vectors_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
                const D3D12_TEXTURE_COPY_LOCATION from{upload_.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, vector_footprint_};
                const D3D12_TEXTURE_COPY_LOCATION to{vectors_.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
                list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
                transition(list, vectors_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
        }
        primed_ = true;
        if (ready) {
            auto* vectors = video_ ? video_->vectors() : vectors_.Get();
            const auto vector_state = video_ ? D3D12_RESOURCE_STATE_COMMON :
                ffx_ ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            transition(list, vectors, vector_state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (scene_) transition(list, scene_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            transition(list, output_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            MotionConstants c{};
            c.width = width_;
            c.height = height_;
            c.reset = video_ ? video_->block() : 8;
            c.mode = analysis_width_;
            c.source_a = analysis_height_;
            c.source_b = ffx_ && scene_detection;
            c.divider = video_ ? .25f : 1.f;
            c.resolve = bool(block_match_);
            convert_->record(list, {vectors, scene_.Get(), nullptr}, output_.Get(), c);
            transition(list, output_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (scene_) transition(list, scene_.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            transition(list, vectors, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, vector_state);
        }
        finish(!ordered);
        return ready;
    }
private:
    bool compute_dis(bool reset) {
        const D3D12_RANGE pixels{0, size_t(readback_->GetDesc().Width)}, empty{};
        unsigned char* input = nullptr;
        phoenix_native::hr(readback_->Map(0, &pixels, reinterpret_cast<void**>(&input)));
        auto* first = input + input_footprint_.Offset;
        const auto pitch = input_footprint_.Footprint.RowPitch;
        bool ready = false;
        if (!reset) {
            float* vectors = nullptr;
            const auto hr = upload_->Map(0, &empty, reinterpret_cast<void**>(&vectors));
            if (FAILED(hr)) {
                readback_->Unmap(0, &empty);
                phoenix_native::hr(hr);
            }
            ready = dis_->run(dis_->context, analysis_width_, analysis_height_, first, pitch,
                previous_.data(), vectors, vector_footprint_.Footprint.RowPitch);
            upload_->Unmap(0, nullptr);
        }
        for (unsigned y = 0; y < analysis_height_; ++y)
            std::memcpy(previous_.data() + size_t(y) * analysis_width_, first + size_t(y) * pitch, analysis_width_);
        readback_->Unmap(0, &empty);
        return ready;
    }
};
}
