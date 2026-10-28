#pragma once
#include "host_client.h"
#include "pixel_transport.h"
#include "../../../third_party/reshade/reshade.hpp"
#include <d3d9.h>
#include <wincodec.h>

namespace phoenix_compat {
struct ScreenshotLayout {
    uint32_t width, height, row_pitch;
    bool buffers, bgra, flip_rows, rgb10;
};
inline ScreenshotLayout screenshot_layout(reshade::api::device_api api,
                                          const reshade::api::resource_desc& description) {
    using namespace reshade::api;
    require(description.type == resource_type::texture_2d ||
                description.type == resource_type::surface,
            "Screenshot requires a 2D presentation surface");
    const auto width = description.texture.width, height = description.texture.height;
    require(width && width <= 8192 && height && height <= 8192 &&
                (description.texture.samples == 1 || api == device_api::d3d9 && description.texture.samples > 1) &&
                description.texture.depth_or_layers == 1 &&
                description.texture.levels == 1,
            "Screenshot requires SDR up to 8192 pixels per side; multisampling requires D3D9");
    const auto pixel_format = description.texture.format;
    const bool rgb10 = supports_rgb10_readback(api) &&
        (pixel_format == format::r10g10b10a2_unorm || pixel_format == format::r10g10b10a2_typeless);
    const bool bgra =
        pixel_format == format::b8g8r8a8_typeless || pixel_format == format::b8g8r8x8_typeless ||
        pixel_format == format::b8g8r8a8_unorm || pixel_format == format::b8g8r8x8_unorm ||
        pixel_format == format::b8g8r8a8_unorm_srgb || pixel_format == format::b8g8r8x8_unorm_srgb;
    require(
        rgb10 || bgra || pixel_format == format::r8g8b8a8_typeless ||
            pixel_format == format::r8g8b8a8_unorm || pixel_format == format::r8g8b8a8_unorm_srgb ||
            pixel_format == format::r8g8b8x8_unorm || pixel_format == format::r8g8b8x8_unorm_srgb,
        "Screenshot supports SDR RGBA8/BGRA8 or DXGI RGB10A2; HDR capture is rejected");
    return {width,
            height,
            transfer_row_pitch(api, width),
            uses_buffer_readback(api),
            bgra,
            api == device_api::opengl, rgb10};
}
inline reshade::api::resource_desc screenshot_readback_description(
    reshade::api::device_api api, const reshade::api::resource_desc& description) {
    using namespace reshade::api;
    const auto layout = screenshot_layout(api, description);
    if (layout.buffers)
        return resource_desc(uint64_t(layout.row_pitch) * layout.height,
                             memory_heap::readback, resource_usage::copy_dest);
    auto read = description;
    read.texture.samples = 1;
    read.heap = memory_heap::readback;
    read.usage = resource_usage::copy_dest;
    read.flags = resource_flags::none;
    // GetRenderTargetData requires IDirect3DSurface9, including when ReShade
    // describes the game's backbuffer as a texture rather than a surface.
    if (api == device_api::d3d9)
        read.type = resource_type::surface;
    return read;
}
// Accept only normalized top-down RGBA8. Native HDR bytes never enter this
// function, and size checks precede all encoder allocations.
inline void save_rgba8_png(std::vector<uint8_t> pixels, uint32_t width, uint32_t height,
                           const fs::path& path) {
    require(width && height && uint64_t(width) * height <= UINT_MAX / 4 &&
                pixels.size() == uint64_t(width) * height * 4,
            "Screenshot dimensions exceed the encoder limit");
    const auto apartment = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    require(SUCCEEDED(apartment) || apartment == RPC_E_CHANGED_MODE, "Screenshot COM unavailable");
    struct Uninitialize {
        bool owned;
        ~Uninitialize() {
            if (owned)
                CoUninitialize();
        }
    } com{SUCCEEDED(apartment)};
    Ptr<IWICImagingFactory> factory;
    hr(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                        IID_PPV_ARGS(&factory)));
    Ptr<IWICStream> stream;
    hr(factory->CreateStream(&stream));
    hr(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE));
    Ptr<IWICBitmapEncoder> encoder;
    hr(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder));
    hr(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache));
    Ptr<IWICBitmapFrameEncode> frame;
    Ptr<IPropertyBag2> properties;
    hr(encoder->CreateNewFrame(&frame, &properties));
    hr(frame->Initialize(properties.Get()));
    hr(frame->SetSize(width, height));
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppBGRA;
    hr(frame->SetPixelFormat(&format));
    require(IsEqualGUID(format, GUID_WICPixelFormat32bppBGRA), "Screenshot encoder format changed");
    for (size_t pixel = 0; pixel < pixels.size(); pixel += 4)
        std::swap(pixels[pixel], pixels[pixel + 2]);
    hr(frame->WritePixels(height, width * 4, UINT(pixels.size()), pixels.data()));
    hr(frame->Commit());
    hr(encoder->Commit());
}

// Capture the resource supplied by PRESENT. DXGI may rotate to a different
// backbuffer before finish_present; never look up its current image again.
class Snapshot {
    reshade::api::device* device_;
    reshade::api::command_queue* queue_;
    reshade::api::resource readback_{}, d3d9_snapshot_{};
    reshade::api::fence retired_{};
    uint32_t width_ = 0, height_ = 0, row_pitch_ = 0;
    bool buffers_ = false, bgra_ = false, flip_rows_ = false, rgb10_ = false;
    bool recorded_ = false, submitted_ = false, complete_ = false;
    uint64_t recorded_tick_ = 0;
    fs::path path_;
    void release() noexcept {
        if (recorded_ && !submitted_) {
            // The owner must retain this object until finish_present. Flushing
            // a Vulkan list here would omit the game's wait semaphores. If an
            // exceptional owner teardown breaks that contract, retain resources
            // for device destruction instead of freeing queued GPU references.
            reshade::log::message(
                reshade::log::level::error,
                "Screenshot destroyed before presentation submission; resources retained for device teardown.");
            return;
        }
        if (submitted_ && retired_.handle && device_->get_completed_fence_value(retired_) < 1)
            queue_->wait_idle();
        if (retired_.handle)
            device_->destroy_fence(retired_);
        for (auto resource : {readback_, d3d9_snapshot_})
            if (resource.handle)
                device_->destroy_resource(resource);
        retired_ = {};
        readback_ = d3d9_snapshot_ = {};
    }

  public:
    explicit Snapshot(reshade::api::effect_runtime* runtime)
        : device_(runtime->get_device()), queue_(runtime->get_command_queue()) {}
    Snapshot(const Snapshot&) = delete;
    Snapshot& operator=(const Snapshot&) = delete;
    ~Snapshot() {
        release();
    }
    const fs::path& path() const {
        return path_;
    }
    bool pending() const {
        return recorded_ && !complete_;
    }
    bool submitted() const {
        return submitted_;
    }
    void record(reshade::api::command_list* command, reshade::api::resource back,
                const fs::path& path) {
        using namespace reshade::api;
        require(!recorded_ && !readback_.handle, "Screenshot is already recorded");
        require(command && command == queue_->get_immediate_command_list(),
                "Screenshot requires the presentation immediate queue");
        const auto description = device_->get_resource_desc(back);
        const auto layout = screenshot_layout(device_->get_api(), description);
        width_ = layout.width;
        height_ = layout.height;
        row_pitch_ = layout.row_pitch;
        buffers_ = layout.buffers;
        bgra_ = layout.bgra;
        flip_rows_ = layout.flip_rows;
        rgb10_ = layout.rgb10;
        const auto read = screenshot_readback_description(device_->get_api(), description);
        // Allocate before recording: unsupported formats or allocation failures
        // must not leave a partial capture in the shared immediate command list.
        require(device_->create_resource(read, nullptr, resource_usage::copy_dest, &readback_),
                "Cannot create screenshot readback storage");
        if (device_->get_api() == device_api::d3d9) {
            auto snapshot = description;
            snapshot.texture.samples = 1;
            snapshot.type = resource_type::surface;
            snapshot.heap = memory_heap::gpu_only;
            snapshot.usage = resource_usage::render_target | resource_usage::copy_source |
                             resource_usage::copy_dest;
            snapshot.flags = resource_flags::none;
            require(device_->create_resource(snapshot, nullptr, resource_usage::copy_dest,
                                             &d3d9_snapshot_),
                    "Cannot create D3D9 screenshot snapshot");
        }
        require(device_->create_fence(0, fence_flags::none, &retired_),
                "Cannot create screenshot retirement fence");
        path_ = path;
        recorded_tick_ = GetTickCount64();
        recorded_ = true;
        command->barrier(back, resource_usage::present, resource_usage::copy_source);
        if (buffers_)
            command->copy_texture_to_buffer(back, 0, nullptr, readback_, 0, row_pitch_ / 4,
                                            height_);
        else
            command->copy_resource(back, d3d9_snapshot_.handle ? d3d9_snapshot_ : readback_);
        command->barrier(back, resource_usage::copy_source, resource_usage::present);
        if (device_->get_api() != device_api::vulkan)
            signal_after_present();
    }
    void signal_after_present() {
        if (!recorded_ || submitted_)
            return;
        const bool signalled = queue_->signal(retired_, 1);
        // signal() flushes the list even when the native fence signal fails.
        // At finish_present it is already submitted by ReShade in any case.
        // Preserve that fact so error teardown never treats it as unsubmitted.
        submitted_ = true;
        require(signalled, "Cannot signal screenshot retirement");
    }
    bool poll() {
        using namespace reshade::api;
        if (complete_)
            return true;
        if (!submitted_)
            return false;
        const auto completed = device_->get_completed_fence_value(retired_);
        require(completed != UINT64_MAX, "Screenshot graphics device was lost");
        require(completed >= 1 || GetTickCount64() - recorded_tick_ <= 15000,
                "Screenshot GPU readback did not retire within 15 seconds");
        if (completed < 1)
            return false;
        if (d3d9_snapshot_.handle) {
            auto* native = reinterpret_cast<IDirect3DDevice9*>(uintptr_t(device_->get_native()));
            hr(native->GetRenderTargetData(
                reinterpret_cast<IDirect3DSurface9*>(uintptr_t(d3d9_snapshot_.handle)),
                reinterpret_cast<IDirect3DSurface9*>(uintptr_t(readback_.handle))));
        }
        std::vector<uint8_t> rgba(size_t(width_) * height_ * 4);
        subresource_data mapped{};
        if (buffers_) {
            require(device_->map_buffer_region(readback_, 0, uint64_t(row_pitch_) * height_,
                                               map_access::read_only, &mapped.data),
                    "Cannot map retired screenshot buffer");
            mapped.row_pitch = row_pitch_;
        } else
            require(
                device_->map_texture_region(readback_, 0, nullptr, map_access::read_only, &mapped),
                "Cannot map retired screenshot texture");
        struct Unmap {
            device* owner;
            resource image;
            bool buffer;
            ~Unmap() {
                if (buffer)
                    owner->unmap_buffer_region(image);
                else
                    owner->unmap_texture_region(image, 0);
            }
        };
        {
            Unmap scope{device_, readback_, buffers_};
            if (rgb10_)
                copy_rgb10a2_rows(rgba.data(), width_ * 4, static_cast<const uint8_t*>(mapped.data),
                                 mapped.row_pitch, width_, height_, true);
            else
                copy_pixel_rows(rgba.data(), width_ * 4, static_cast<const uint8_t*>(mapped.data),
                                mapped.row_pitch, width_, height_, flip_rows_, bgra_);
        }
        // Presented game images are opaque; an unused X8 alpha byte may be zero.
        for (size_t pixel = 0; pixel < rgba.size(); pixel += 4)
            rgba[pixel + 3] = 255;
        fs::create_directories(path_.parent_path());
        save_rgba8_png(std::move(rgba), width_, height_, path_);
        complete_ = true;
        return true;
    }
};
} // namespace phoenix_compat
