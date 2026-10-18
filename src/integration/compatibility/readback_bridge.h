// Staging textures on D3D9/10/11; pixel buffers on D3D12/Vulkan/OpenGL.
// CPU mapping follows a GPU fence. The helper owns neural execution.
#pragma once
#include "host_client.h"
#include "pixel_transport.h"
#include "d3d9_compositor.h"
namespace phoenix_compat {
class ReadbackBridge {
    reshade::api::device* device_;
    reshade::api::command_queue* queue_;
    reshade::api::resource readback_{}, upload_{}, upload_buffer_{};
    reshade::api::fence retired_{};
    std::unique_ptr<HostClient> host_;
    std::unique_ptr<D3D9Compositor> compositor9_;
    unsigned width_, height_, row_pitch_;
    bool bgra_, rgb10_, upload_bgra_, buffer_readback_, flip_rows_, d3d9_upload_, pending_ = false,
                                                                          delivered_image_ = false;
    uint64_t serial_ = 0, capture_serial_ = 0, upload_serial_ = 0, queued_signal_ = 0;
    uint64_t capture_frame_ = 0, submitted_frame_ = 0, delivered_frame_ = 0;
    uint64_t delivered_ = 0, presented_ = 0;
    uint64_t presented_input_frame_ = 0, presented_output_frame_ = 0, composed_at_frame_ = 0;
    uint64_t capture_tick_ = 0, submitted_tick_ = 0, delivered_tick_ = 0;
    std::vector<uint8_t> submitted_pixels_;
    double capture_ms_ = 0, upload_ms_ = 0, compose_ms_ = 0;
    double deferred_readback_ms_ = 0;
    uint64_t capture_copies_ = 0, deferred_readbacks_ = 0;
    static double elapsed_ms(std::chrono::steady_clock::time_point begin) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
            .count();
    }
    struct MappedResource {
        reshade::api::device* device;
        reshade::api::resource resource;
        bool buffer;
        ~MappedResource() {
            if (buffer)
                device->unmap_buffer_region(resource);
            else
                device->unmap_texture_region(resource, 0);
        }
    };

  public:
    ReadbackBridge(reshade::api::effect_runtime* runtime,
                   const reshade::api::resource_desc& description, HMODULE addon,
                   const phoenix_model::WeightSource& source,
                   const phoenix_addon::phoenix::Settings& settings,
                   std::optional<PortableOptions> portable = std::nullopt,
                   std::optional<NativeOptions> native = std::nullopt)
        : device_(runtime->get_device()), queue_(runtime->get_command_queue()),
          width_(description.texture.width), height_(description.texture.height),
          row_pitch_(transfer_row_pitch(device_->get_api(), width_)),
          buffer_readback_(uses_buffer_readback(device_->get_api())),
          flip_rows_(device_->get_api() == reshade::api::device_api::opengl),
          d3d9_upload_(device_->get_api() == reshade::api::device_api::d3d9) {
        using namespace reshade::api;
        require(height_ && height_ <= 8192 &&
                    (description.texture.samples == 1 || d3d9_upload_ && description.texture.samples > 1) &&
                    description.texture.depth_or_layers == 1 && description.texture.levels == 1,
                "Unsupported readback canvas (multisampling currently requires D3D9)");
        bgra_ = description.texture.format == format::b8g8r8a8_typeless ||
                description.texture.format == format::b8g8r8x8_typeless ||
                description.texture.format == format::b8g8r8a8_unorm ||
                description.texture.format == format::b8g8r8x8_unorm ||
                description.texture.format == format::b8g8r8a8_unorm_srgb ||
                description.texture.format == format::b8g8r8x8_unorm_srgb;
        rgb10_ = device_->get_api() == device_api::d3d12 &&
                 (description.texture.format == format::r10g10b10a2_unorm ||
                  description.texture.format == format::r10g10b10a2_typeless);
        require(rgb10_ || bgra_ || description.texture.format == format::r8g8b8a8_typeless ||
                    description.texture.format == format::r8g8b8a8_unorm ||
                    description.texture.format == format::r8g8b8a8_unorm_srgb ||
                    description.texture.format == format::r8g8b8x8_unorm ||
                    description.texture.format == format::r8g8b8x8_unorm_srgb,
                "Readback compatibility requires SDR RGBA8/BGRA8 or D3D12 RGB10A2");
        auto read = description;
        // D3D9Compositor captures via StretchRect, which resolves the game's
        // multisampled render target. CPU staging and sampled upload textures
        // always contain the resulting single-sample image.
        read.texture.samples = 1;
        read.heap = memory_heap::readback;
        read.usage = resource_usage::copy_dest;
        read.flags = resource_flags::none;
        if (buffer_readback_)
            read = resource_desc(uint64_t(row_pitch_) * height_, memory_heap::readback,
                                 resource_usage::copy_dest);
        auto write = description;
        write.texture.samples = 1;
        // Mutable Vulkan swapchain images are described as typeless. Their
        // memory still contains four UNORM bytes; private upload textures use
        // a concrete view format so no implicit gamma conversion is introduced.
        write.texture.format = rgb10_ ? format::r10g10b10a2_unorm
                                     : bgra_ ? format::b8g8r8a8_unorm : format::r8g8b8a8_unorm;
        if (d3d9_upload_ && (description.texture.format == format::b8g8r8x8_unorm ||
                             description.texture.format == format::b8g8r8x8_unorm_srgb))
            write.texture.format = format::b8g8r8x8_unorm;
        write.type = resource_type::texture_2d;
        write.heap = memory_heap::gpu_only;
        write.usage = resource_usage::copy_source | resource_usage::copy_dest |
                      resource_usage::shader_resource;
        if (device_->get_api() == device_api::d3d9 || flip_rows_)
            write.usage |= resource_usage::render_target;
        write.flags = resource_flags::none;
        try {
            require(device_->create_resource(read, nullptr, resource_usage::copy_dest, &readback_),
                    "API cannot create image readback storage");
            require(device_->create_resource(write, nullptr, resource_usage::copy_source, &upload_),
                    "API cannot create presentation upload");
            // OpenGL textures canonicalize a BGRA backbuffer format to RGBA8.
            // Buffer uploads use the allocated texture's actual format, rather
            // than assuming its byte order matches the presentation surface.
            const auto upload_format = device_->get_resource_desc(upload_).texture.format;
            upload_bgra_ = upload_format == format::b8g8r8a8_typeless ||
                           upload_format == format::b8g8r8x8_typeless ||
                           upload_format == format::b8g8r8a8_unorm ||
                           upload_format == format::b8g8r8x8_unorm ||
                           upload_format == format::b8g8r8a8_unorm_srgb ||
                           upload_format == format::b8g8r8x8_unorm_srgb;
            if (buffer_readback_) {
                resource_desc buffer(uint64_t(row_pitch_) * height_, memory_heap::upload,
                                     resource_usage::copy_source);
                require(device_->create_resource(buffer, nullptr, resource_usage::cpu_access,
                                                 &upload_buffer_),
                        "API cannot create upload pixel buffer");
            } else if (d3d9_upload_) {
                auto staging = description;
                staging.texture.samples = 1;
                staging.type = resource_type::surface;
                staging.heap = memory_heap::upload;
                staging.usage = resource_usage::copy_source;
                staging.flags = resource_flags::none;
                require(device_->create_resource(staging, nullptr, resource_usage::cpu_access,
                                                 &upload_buffer_),
                        "D3D9 cannot create upload surface");
                auto* texture = reinterpret_cast<IDirect3DTexture9*>(uintptr_t(upload_.handle));
                D3DSURFACE_DESC native_description{};
                hr(texture->GetLevelDesc(0, &native_description));
                compositor9_ = std::make_unique<D3D9Compositor>(
                    reinterpret_cast<IDirect3DDevice9*>(uintptr_t(device_->get_native())), width_,
                    height_, native_description.Format);
            }
            require(device_->create_fence(0, fence_flags::none, &retired_),
                    "API cannot provide transfer fencing");
            FeedBuild build{};
            build.width = build.output_width = width_;
            build.height = build.output_height = height_;
            build.color_fmt = build.output_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
            build.cpu_bridge = 1;
            build.mv_scale_x = build.mv_scale_y = 1;
            host_ = std::make_unique<HostClient>(addon, source, settings, build, portable,native);
        } catch (...) {
            release();
            throw;
        }
    }
    void release() {
        host_.reset();
        if (serial_ || presented_)
            queue_->wait_idle();
        compositor9_.reset();
        if (retired_.handle)
            device_->destroy_fence(retired_);
        for (auto resource : {readback_, upload_, upload_buffer_})
            if (resource.handle)
                device_->destroy_resource(resource);
        retired_ = {};
        readback_ = upload_ = upload_buffer_ = {};
    }
    ~ReadbackBridge() {
        release();
    }
    HostClient* host() const {
        return host_.get();
    }
    uint64_t delivered() const {
        return delivered_;
    }
    uint64_t presented() const {
        return presented_;
    }
    bool presents_current_frame() const {
        return compositor9_ != nullptr;
    }
    Json transfer_timings() const {
        return {{"last_capture_ms", capture_ms_},
                {"last_upload_ms", upload_ms_},
                {"last_compose_ms", compose_ms_},
                {"last_deferred_readback_ms", deferred_readback_ms_},
                {"capture_snapshot_copies", capture_copies_},
                {"deferred_readbacks", deferred_readbacks_},
                {"input_frame", presented_input_frame_}, {"output_frame", presented_output_frame_},
                {"composed_at_frame", composed_at_frame_}, {"width", width_}, {"height", height_}};
    }
    void signal_after_present() {
        if (!queued_signal_)
            return;
        const auto begin = std::chrono::steady_clock::now();
        require(queue_->signal(retired_, queued_signal_),
                "Cannot signal graphics transfer completion");
        if (elapsed_ms(begin)>100)
            reshade::log::message(reshade::log::level::warning,
                ("Phoenix transfer fence signal stalled " + std::to_string(elapsed_ms(begin)) + " ms").c_str());
        queued_signal_ = 0;
    }
    void frame(reshade::api::command_list* command, reshade::api::resource back,
               uint64_t frame_number, const phoenix_addon::phoenix::Settings& settings,
               bool coherent_frames = false) {
        using namespace reshade::api;
        const auto frame_begin = std::chrono::steady_clock::now();
        require(command == queue_->get_immediate_command_list(),
                "Image transport requires the runtime immediate queue");
        const auto completed = device_->get_completed_fence_value(retired_);
        require(completed != UINT64_MAX, "Readback graphics device was lost");
        // A persistent upload buffer cannot be rewritten while the GPU reads it.
        if (host_->phase() == Phase::Finished && completed >= upload_serial_) {
            const auto upload_begin = std::chrono::steady_clock::now();
            auto pixels = host_->take();
            require(pixels.size() == size_t(width_) * height_ * 4,
                    "Helper returned an incomplete image");
            if (buffer_readback_) {
                void* mapped = nullptr;
                require(device_->map_buffer_region(upload_buffer_, 0,
                                                   uint64_t(row_pitch_) * height_,
                                                   map_access::write_only, &mapped),
                        "Cannot map upload pixel buffer");
                {
                    MappedResource mapping{device_, upload_buffer_, true};
                    if (rgb10_)
                        copy_rgb10a2_rows(static_cast<uint8_t*>(mapped), row_pitch_, pixels.data(),
                                         width_ * 4, width_, height_, false);
                    else
                        copy_pixel_rows(static_cast<uint8_t*>(mapped), row_pitch_, pixels.data(),
                                        width_ * 4, width_, height_, false, upload_bgra_);
                }
                // ReShade's texture-to-default-framebuffer blit already flips
                // OpenGL Y. Upload textures therefore keep helper row order;
                // only raw GL framebuffer readback needs the CPU row reversal.
                command->barrier(upload_, resource_usage::copy_source, resource_usage::copy_dest);
                command->copy_buffer_to_texture(upload_buffer_, 0, row_pitch_ / 4, height_, upload_,
                                                0);
                command->barrier(upload_, resource_usage::copy_dest, resource_usage::copy_source);
                upload_serial_ = queued_signal_ = ++serial_;
            } else if (d3d9_upload_) {
                // ReShade's D3D9 update_texture_region swizzles 4-channel
                // uploads to BGRA. Mapping the system surface avoids a second
                // swizzle when our backbuffer already uses a BGRA format.
                subresource_data mapped{};
                require(device_->map_texture_region(upload_buffer_, 0, nullptr,
                                                    map_access::write_only, &mapped),
                        "Cannot map D3D9 upload surface");
                {
                    MappedResource mapping{device_, upload_buffer_, false};
                    copy_pixel_rows(static_cast<uint8_t*>(mapped.data), mapped.row_pitch,
                                    pixels.data(), width_ * 4, width_, height_, false, bgra_);
                }
                command->copy_resource(upload_buffer_, upload_);
                upload_serial_ = queued_signal_ = ++serial_;
            } else {
                std::vector<uint8_t> native_pixels(pixels.size());
                copy_pixel_rows(native_pixels.data(), width_ * 4, pixels.data(), width_ * 4, width_,
                                height_, false, bgra_);
                device_->update_texture_region(
                    {native_pixels.data(), width_ * 4, width_ * height_ * 4}, upload_, 0);
            }
            delivered_image_ = true;
            if (compositor9_)
                compositor9_->match_completed_input();
            delivered_frame_ = submitted_frame_;
            delivered_tick_ = submitted_tick_;
            ++delivered_;
            upload_ms_ = elapsed_ms(upload_begin);
        }
        if (pending_ && completed >= capture_serial_) {
            if (compositor9_) {
                const auto readback_begin = std::chrono::steady_clock::now();
                compositor9_->readback_completed_capture(
                    reinterpret_cast<IDirect3DSurface9*>(uintptr_t(readback_.handle)));
                deferred_readback_ms_ = elapsed_ms(readback_begin);
                ++deferred_readbacks_;
            }
            // Allocate before mapping so an allocation failure cannot leave a
            // resource mapped while runtime teardown tries to release it.
            std::vector<uint8_t> pixels(size_t(width_) * height_ * 4);
            subresource_data mapped{};
            if (buffer_readback_) {
                require(device_->map_buffer_region(readback_, 0, uint64_t(row_pitch_) * height_,
                                                   map_access::read_only, &mapped.data),
                        "Cannot map retired pixel buffer");
                mapped.row_pitch = row_pitch_;
            } else
                require(device_->map_texture_region(readback_, 0, nullptr, map_access::read_only,
                                                    &mapped),
                        "Cannot map retired texture readback");
            {
                MappedResource mapping{device_, readback_, buffer_readback_};
                if (rgb10_)
                    copy_rgb10a2_rows(pixels.data(), width_ * 4, static_cast<const uint8_t*>(mapped.data),
                                     mapped.row_pitch, width_, height_, true);
                else
                    copy_pixel_rows(pixels.data(), width_ * 4, static_cast<const uint8_t*>(mapped.data),
                                    mapped.row_pitch, width_, height_, flip_rows_, bgra_);
            }
            pending_ = false;
            if (settings.mode == phoenix_addon::phoenix::split && !compositor9_)
                submitted_pixels_ = pixels;
            if (host_->submit(capture_serial_, true, std::move(pixels))) {
                submitted_frame_ = capture_frame_;
                submitted_tick_ = capture_tick_;
            }
        }
        if (!pending_ && host_->phase() == Phase::Ready &&
            settings.mode != phoenix_addon::phoenix::input) {
            const auto capture_begin = std::chrono::steady_clock::now();
            if (compositor9_)
                compositor9_->capture(reinterpret_cast<IDirect3DSurface9*>(uintptr_t(back.handle)));
            else {
                command->barrier(back, resource_usage::present, resource_usage::copy_source);
                if (buffer_readback_)
                    command->copy_texture_to_buffer(back, 0, nullptr, readback_, 0, row_pitch_ / 4,
                                                    height_);
                else
                    command->copy_resource(back, readback_);
                command->barrier(back, resource_usage::copy_source, resource_usage::present);
            }
            ++capture_copies_;
            capture_serial_ = queued_signal_ = ++serial_;
            pending_ = true;
            capture_frame_ = frame_number;
            capture_tick_ = GetTickCount64();
            capture_ms_ = elapsed_ms(capture_begin);
        }
        if (delivered_image_ && settings.mode != phoenix_addon::phoenix::input &&
            (coherent_frames ||
             (frame_number - delivered_frame_ <= uint64_t(settings.offload_age_frames) &&
              GetTickCount64() - delivered_tick_ <= uint64_t(settings.offload_age_ms)))) {
            const auto compose_begin = std::chrono::steady_clock::now();
            command->barrier(back, resource_usage::present, resource_usage::copy_dest);
            if (compositor9_)
                compositor9_->draw(reinterpret_cast<IDirect3DSurface9*>(uintptr_t(back.handle)),
                                   reinterpret_cast<IDirect3DTexture9*>(uintptr_t(upload_.handle)),
                                   helper_presentation_mode(settings), coherent_frames);
            else
                command->copy_resource(upload_, back);
            command->barrier(back, resource_usage::copy_dest, resource_usage::present);
            ++presented_;
            // A plain upload replaces the complete image. Only the D3D9
            // compositor can mix a live input with an older processed output.
            presented_input_frame_ = compositor9_ && !coherent_frames ? frame_number : delivered_frame_;
            presented_output_frame_ = delivered_frame_;
            composed_at_frame_ = frame_number;
            compose_ms_ = elapsed_ms(compose_begin);
        }
        // Vulkan submits with application wait semaphores AFTER this callback.
        // Let ReShade flush first; finish_present then signals the retirement fence.
        if (device_->get_api() != device_api::vulkan)
            signal_after_present();
        if (elapsed_ms(frame_begin)>100)
            reshade::log::message(reshade::log::level::warning,
                ("Phoenix transfer frame stalled " + std::to_string(elapsed_ms(frame_begin)) +
                 " ms; capture=" + std::to_string(capture_ms_) + " upload=" + std::to_string(upload_ms_) +
                 " compose=" + std::to_string(compose_ms_)).c_str());
    }
};
} // namespace phoenix_compat
