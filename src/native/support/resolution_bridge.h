#pragma once
#include "compute_pass.h"
#include "resolution.h"

namespace phoenix_addon::phoenix::direct {
// Independently authored spatial resampling. This wraps NR inference; it is
// not NVIDIA's separate temporal DLSS Super Resolution feature.
inline constexpr const char* resolution_shader = R"hlsl(
Texture2D<float4> Source : register(t0);
Texture2D<float4> Original : register(t1);
RWTexture2D<float4> Destination : register(u0);
cbuffer Geometry : register(b0) {
    uint Width, Height, Left, Top, ImageWidth, ImageHeight;
    float MotionX, MotionY;
    uint Operation;
};
float4 interpolate(float2 position, int2 lower, int2 upper) {
    int2 base = int2(floor(position));
    float2 fraction = frac(position);
    return lerp(lerp(Source.Load(int3(clamp(base, lower, upper), 0)),
                     Source.Load(int3(clamp(base + int2(1,0), lower, upper), 0)), fraction.x),
                lerp(Source.Load(int3(clamp(base + int2(0,1), lower, upper), 0)),
                     Source.Load(int3(clamp(base + 1, lower, upper), 0)), fraction.x), fraction.y);
}
float original_alpha(uint2 pixel) {
    uint original_width = 0, original_height = 0;
    Original.GetDimensions(original_width, original_height);
    float alpha = Original.Load(int3(pixel,0)).a;
    // Library clients may supply a smaller source than the output. Resample
    // alpha independently; the trained network does not reconstruct opacity.
    if (original_width != Width || original_height != Height) {
    float2 position = (float2(pixel)+0.5) * float2(original_width,original_height) /
                      float2(Width,Height) - 0.5;
    int2 base = int2(floor(position)), upper = int2(original_width-1,original_height-1);
    float2 fraction = frac(position);
    alpha = lerp(lerp(Original.Load(int3(clamp(base,int2(0,0),upper),0)).a,
                     Original.Load(int3(clamp(base+int2(1,0),int2(0,0),upper),0)).a,fraction.x),
                lerp(Original.Load(int3(clamp(base+int2(0,1),int2(0,0),upper),0)).a,
                     Original.Load(int3(clamp(base+1,int2(0,0),upper),0)).a,fraction.x),fraction.y);
    }
    return alpha;
}
[numthreads(16,16,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint2 pixel = thread.xy;
    if (pixel.x >= Width || pixel.y >= Height) return;
    uint source_width, source_height;
    Source.GetDimensions(source_width, source_height);
    if (Operation == 3) {
        // Vendor reconstruction already has output dimensions. Convert its
        // format while retaining the game's alpha, which temporal APIs omit.
        float4 result = Source.Load(int3(pixel,0));
        result.a = original_alpha(pixel);
        Destination[pixel] = result;
        return;
    }
    if (Operation == 2) {
        // Sample only the active image; padding never bleeds into the output.
        float2 position = float2(Left,Top) + (float2(pixel)+0.5) *
                          float2(ImageWidth,ImageHeight) / float2(Width,Height) - 0.5;
        float4 result = interpolate(position, int2(Left,Top),
                                   int2(Left+ImageWidth-1,Top+ImageHeight-1));
        result.a = original_alpha(pixel);
        Destination[pixel] = result;
        return;
    }
    if (pixel.x < Left || pixel.y < Top ||
        pixel.x >= Left+ImageWidth || pixel.y >= Top+ImageHeight) {
        Destination[pixel] = Operation == 1 ? float4(0,0,0,0) : float4(0,0,0,1);
        return;
    }
    float2 footprint = float2(source_width,source_height) / float2(ImageWidth,ImageHeight);
    float2 position = (float2(pixel)-float2(Left,Top)+0.5) * footprint - 0.5;
    if (Operation == 1) {
        // Do not blend vectors across object boundaries. Raw pixel-space
        // displacement scales with the fitted image, not the padded canvas.
        int2 nearest = clamp(int2(floor(position+0.5)), int2(0,0),
                             int2(source_width-1,source_height-1));
        float4 motion = Source.Load(int3(nearest,0));
        motion.xy *= float2(MotionX,MotionY);
        Destination[pixel] = motion;
    } else {
        // Four bilinear taps integrate the source footprint before reduction.
        float2 offset = max(footprint-1,0) * 0.25;
        int2 upper = int2(source_width-1,source_height-1);
        Destination[pixel] = 0.25 *
            (interpolate(position+offset, int2(0,0), upper) +
             interpolate(position-offset, int2(0,0), upper) +
             interpolate(position+float2(offset.x,-offset.y), int2(0,0), upper) +
             interpolate(position+float2(-offset.x,offset.y), int2(0,0), upper));
    }
}
)hlsl";

class ResolutionBridge {
    Resolution geometry_;
    std::shared_ptr<phoenix_native::memory::Generation> accounting_;
    // One shader pipeline serves every frame. Descriptor tables remain immutable
    // until the queue fence confirms that all commands using them have finished.
    Pass pipeline_;
    struct Binding {
        std::array<ComPtr<ID3D12Resource>, 4> resources;
        std::unique_ptr<Pass> pass;
        uint64_t submission = 0;
    };
    std::vector<Binding> bindings_;
    ComPtr<ID3D12Resource> output_;
    uint64_t completed_ = 0, recording_ = 1;

    Pass& pass(std::array<ID3D12Resource*, 4> resources) {
        for (auto& binding : bindings_) {
            bool matches = true;
            for (size_t index = 0; index < resources.size(); ++index)
                matches &= binding.resources[index].Get() == resources[index];
            if (matches) {
                binding.submission = recording_;
                return *binding.pass;
            }
        }
        auto reusable = std::find_if(bindings_.begin(), bindings_.end(), [&](const auto& binding) {
            return binding.submission <= completed_;
        });
        if (reusable == bindings_.end()) {
            bindings_.push_back({{}, std::make_unique<Pass>(pipeline_, accounting_), 0});
            reusable = std::prev(bindings_.end());
        }
        for (size_t index = 0; index < resources.size(); ++index)
            reusable->resources[index] = resources[index];
        reusable->submission = recording_;
        return *reusable->pass;
    }

  public:
    ResolutionBridge(ID3D12Device* device, Resolution geometry, DXGI_FORMAT format,
                     const std::shared_ptr<phoenix_native::memory::Generation>& accounting)
        : geometry_(geometry), accounting_(accounting),
          pipeline_(device, resolution_shader, accounting) {
        output_ = phoenix_native::image(device, geometry.output_width, geometry.output_height,
                                        format, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        phoenix_native::memory::resource(device, output_.Get(), accounting,
                                         phoenix_native::memory::Kind::bridge);
    }
    void begin_frame(uint64_t completed, uint64_t submission) {
        check(completed >= completed_ && submission >= recording_ && submission > completed,
              "Resolution descriptor fence order");
        completed_ = completed;
        recording_ = submission;
    }
    size_t binding_count() const {
        return bindings_.size();
    }
    Json status() const {
        return {{"compiled_pipelines", 1},
                {"descriptor_tables", bindings_.size()},
                {"completed_submission", completed_},
                {"recording_submission", recording_}};
    }
    Constants constants(unsigned operation) const {
        return {operation >= 2 ? geometry_.output_width : geometry_.width,
                operation >= 2 ? geometry_.output_height : geometry_.height,
                geometry_.left,
                geometry_.top,
                geometry_.image_width,
                geometry_.image_height,
                geometry_.motion_x(),
                geometry_.motion_y(),
                operation};
    }
    void reduce(ID3D12GraphicsCommandList* cmd, ID3D12Resource* source, D3D12_RESOURCE_STATES state,
                ID3D12Resource* destination, bool motion, float motion_scale_x = 0,
                float motion_scale_y = 0) {
        auto& dispatch = pass({source, nullptr, nullptr, destination});
        phoenix_native::transition(cmd, source, state,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(cmd, destination, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto geometry = constants(motion ? 1 : 0);
        if (motion && motion_scale_x > 0 && motion_scale_y > 0) {
            geometry.divider = motion_scale_x;
            geometry.gain = motion_scale_y;
        }
        dispatch.record(cmd, {source, nullptr, nullptr}, destination, geometry);
        phoenix_native::transition(cmd, destination, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(cmd, source, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   state);
    }
    void present(ID3D12GraphicsCommandList* cmd, ID3D12Resource* low, ID3D12Resource* game,
                 D3D12_RESOURCE_STATES state) {
        present_source(cmd, low, game, state, game, state);
    }
    // The addon resolves over its original game image; library clients can
    // provide a distinct source image for alpha and a differently sized target.
    void present_source(ID3D12GraphicsCommandList* cmd, ID3D12Resource* low,
                        ID3D12Resource* original, D3D12_RESOURCE_STATES original_state,
                        ID3D12Resource* game, D3D12_RESOURCE_STATES state) {
        auto& dispatch = pass({low, original, nullptr, output_.Get()});
        phoenix_native::transition(cmd, low, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(cmd, game, state,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (original != game)
            phoenix_native::transition(cmd, original, original_state,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        // The low-resolution surface is stable. The engine output rotates.
        dispatch.record(cmd, {low, original, nullptr}, output_.Get(), constants(2));
        if (original != game)
            phoenix_native::transition(cmd, original, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                       original_state);
        phoenix_native::transition(cmd, output_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        phoenix_native::transition(cmd, game, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(game, output_.Get());
        phoenix_native::transition(cmd, game, D3D12_RESOURCE_STATE_COPY_DEST, state);
        phoenix_native::transition(cmd, output_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        phoenix_native::transition(cmd, low, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    void present_full(ID3D12GraphicsCommandList* cmd, ID3D12Resource* reconstructed,
                      ID3D12Resource* game, D3D12_RESOURCE_STATES state) {
        auto& dispatch = pass({reconstructed, game, nullptr, output_.Get()});
        phoenix_native::transition(cmd, reconstructed, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(cmd, game, state,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dispatch.record(cmd, {reconstructed, game, nullptr}, output_.Get(), constants(3));
        phoenix_native::transition(cmd, output_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        phoenix_native::transition(cmd, game, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(game, output_.Get());
        phoenix_native::transition(cmd, game, D3D12_RESOURCE_STATE_COPY_DEST, state);
        phoenix_native::transition(cmd, output_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        phoenix_native::transition(cmd, reconstructed,
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
};
} // namespace phoenix_addon::phoenix::direct
