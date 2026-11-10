// Staging textures on D3D9/10/11; pixel buffers on D3D12/Vulkan/OpenGL.
// CPU mapping follows a GPU fence. Execution can be local or in a helper.
#pragma once
#include "host_client.h"
#include "pixel_transport.h"
#include "d3d9_compositor.h"
#include "retirement_serial.h"
#include "capture_retirement.h"
#include "capture_queue.h"
#include "../../pipeline/paired_screenshot.h"
namespace phoenix_compat {
class ReadbackBridge {
    reshade::api::device* device_;
    reshade::api::command_queue* queue_;
    reshade::api::resource readback_{}, upload_{}, upload_buffer_{},gpu_output_{};
    bool gpu_transport_=false,gpu_settings_=false;
    std::string gpu_fallback_;
    mutable Json gpu_pair_status_=Json::object();
    reshade::api::fence retired_{};
    std::unique_ptr<HostClient> host_;
    std::unique_ptr<D3D9Compositor> compositor9_;
    unsigned width_, height_, row_pitch_;
    bool bgra_, rgb10_, upload_bgra_, buffer_readback_, flip_rows_, d3d9_upload_,
                                                                          delivered_image_ = false;
    CaptureQueue captures_;
    bool history_started_ = false;
    uint64_t serial_ = 0, capture_serial_ = 0, upload_serial_ = 0, queued_signal_ = 0;
    uint64_t capture_frame_ = 0, submitted_frame_ = 0, delivered_frame_ = 0;
    uint64_t delivered_ = 0, presented_ = 0;
    uint64_t presented_input_frame_ = 0, presented_output_frame_ = 0, composed_at_frame_ = 0;
    uint64_t capture_tick_ = 0, submitted_tick_ = 0, delivered_tick_ = 0;
    std::vector<uint8_t> submitted_pixels_;
    std::vector<uint8_t> queued_pixels_, upload_pixels_;
    const char* transfer_stage_="initialization";
    phoenix_pipeline::screenshots::CompletedPair screenshot_pair_;
    double capture_ms_ = 0, upload_ms_ = 0, compose_ms_ = 0;
    double cpu_readback_ms_ = 0;
    double deferred_readback_ms_ = 0;
    uint64_t capture_copies_ = 0, deferred_readbacks_ = 0;
    uint64_t post_present_retirements_ = 0, post_present_submissions_ = 0, post_present_calls_=0;
    uint64_t capture_not_ready_frames_=0, upload_not_ready_frames_=0, worker_not_ready_frames_=0;
    uint64_t last_pre_completed_=0,last_post_completed_=0;
    double frame_period_ms_=0,last_post_wait_ms_=0;
    std::chrono::steady_clock::time_point preceding_frame_{};
    bool prefetch_available() const {
        // Non-DX9 routes retire directly into independent CPU input storage.
        // Only DX9 needs a second GPU snapshot: its compositor must retain the
        // in-flight input until it can pair that image with its neural output.
        return !gpu_transport_&&(!compositor9_ || compositor9_->prefetch_available());
    }
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

    // Retire only already-completed GPU work, on the graphics thread. Present
    // can make a capture ready after frame() has polled it. Submitting here
    // removes that avoidable frame of worker idle time; uploading/composition
    // still belongs exclusively to the next pre-Present frame().
    void retire_and_submit(const phoenix_addon::phoenix::Settings& settings,
                           bool after_present) {
        using namespace reshade::api;
        while(auto* capture=captures_.front()) {
          if (!capture->retired) {
            auto completed = device_->get_completed_fence_value(retired_);
            if(after_present) {
                last_post_completed_=completed;
                const auto begin=std::chrono::steady_clock::now();
                // Keep this below an eighth of the observed frame period and
                // 0.5ms. A busy neural worker cannot benefit, so never wait then.
                const double budget_ms=std::min(.5,frame_period_ms_*.125);
                completed=poll_capture_retirement(capture->serial,completed,
                    host_->phase()==Phase::Ready && budget_ms>0,
                    [&]{return device_->get_completed_fence_value(retired_);},
                    [&]{return elapsed_ms(begin)>=budget_ms;},[]{SwitchToThread();});
                last_post_wait_ms_=elapsed_ms(begin);last_post_completed_=completed;
            }
            require(completed != UINT64_MAX, "Readback graphics device was lost");
            if(!after_present&&completed<capture->serial)++capture_not_ready_frames_;
            if (completed >= capture->serial) {
                if(!gpu_transport_){
                if (compositor9_) {
                    const auto readback_begin = std::chrono::steady_clock::now();
                    compositor9_->readback_completed_capture(
                        reinterpret_cast<IDirect3DSurface9*>(uintptr_t(readback_.handle)),capture->slot);
                    deferred_readback_ms_ = elapsed_ms(readback_begin);
                    ++deferred_readbacks_;
                }
                // Allocate before mapping so an allocation failure cannot leave a
                // resource mapped while runtime teardown tries to release it.
                transfer_stage_="capture CPU storage allocation";
                auto pixels=host_->take_recycled_input();
                pixels.resize(size_t(width_) * height_ * 4);
                subresource_data mapped{};
                const auto cpu_readback_started=std::chrono::steady_clock::now();
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
                cpu_readback_ms_=elapsed_ms(cpu_readback_started);
                capture->retired = true;
                // Queue one retired capture without replacing the current worker's
                // screenshot input or its matched GPU snapshot.
                queued_pixels_=std::move(pixels);
                }else capture->retired=true;
                if (after_present) ++post_present_retirements_;
            } else break;
          }
          if(host_->phase()!=Phase::Ready){++worker_not_ready_frames_;break;}
          require(gpu_transport_||!queued_pixels_.empty(),"Retired capture lost its CPU pixels");
            if (!gpu_transport_&&settings.mode == phoenix_addon::phoenix::split && !compositor9_)
                submitted_pixels_ = queued_pixels_;
            transfer_stage_="screenshot input copy";
            screenshot_pair_.remember(capture->frame,queued_pixels_);
            transfer_stage_="capture worker submission";
            if (host_->submit(capture->serial, !history_started_, std::move(queued_pixels_))) {
                if(compositor9_)compositor9_->promote_submitted_capture(capture->slot);
                if (after_present) ++post_present_submissions_;
                history_started_ = true;
                submitted_frame_ = capture->frame;
                submitted_tick_ = capture->tick;
                captures_.pop();
            } else {
                screenshot_pair_.rejected(capture->frame);
                throw std::runtime_error("Worker rejected a retired capture");
            }
        }
    }

  public:
    bool request_screenshot_pair(const fs::path& folder) { return gpu_transport_?host_->request_gpu_pair(folder):screenshot_pair_.output.request(folder); }
    const Json& screenshot_pair_status() const { if(gpu_transport_){gpu_pair_status_=host_->gpu_pair_status();return gpu_pair_status_;}return screenshot_pair_.output.result; }
    bool gpu_transport()const{return gpu_transport_;}
    bool matches_gpu_settings(const Settings& settings)const{return gpu_settings_==portable_gpu_settings(settings,width_,height_);}
    ReadbackBridge(reshade::api::effect_runtime* runtime,
                   const reshade::api::resource_desc& description, HMODULE addon,
                   const phoenix_model::WeightSource& source,
                   const phoenix_addon::phoenix::Settings& settings,
                   std::optional<PortableOptions> portable = std::nullopt,
                   std::optional<NativeOptions> native = std::nullopt,
                   std::optional<LocalGpuOptions> local = std::nullopt)
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
        rgb10_ = supports_rgb10_readback(device_->get_api()) &&
                 (description.texture.format == format::r10g10b10a2_unorm ||
                  description.texture.format == format::r10g10b10a2_typeless);
        require(rgb10_ || bgra_ || description.texture.format == format::r8g8b8a8_typeless ||
                    description.texture.format == format::r8g8b8a8_unorm ||
                    description.texture.format == format::r8g8b8a8_unorm_srgb ||
                    description.texture.format == format::r8g8b8x8_unorm ||
                    description.texture.format == format::r8g8b8x8_unorm_srgb,
                "Readback compatibility requires SDR RGBA8/BGRA8 or DXGI RGB10A2");
        auto read = description;
        // D3D9Compositor captures via StretchRect, which resolves the game's
        // multisampled render target. CPU staging and sampled upload textures
        // always contain the resulting single-sample image.
        read.texture.samples = 1;
        read.heap = memory_heap::readback;
        read.usage = resource_usage::copy_dest;
        read.flags = resource_flags::none;
        // A D3D9 canvas may be described as a texture. GetRenderTargetData
        // requires a system-memory SURFACE, never the texture COM interface.
        // The upload staging path already makes the same distinction below.
        if (d3d9_upload_)
            read.type = resource_type::surface;
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
        gpu_settings_=portable_gpu_settings(settings,width_,height_);
        std::shared_ptr<PortableGpuHandles> gpu_handles;
        // The first implementation admits byte-exact RGBA8 temporal SDK views.
        // Other formats, comparison views and host color transforms keep staging.
        if(portable&&portable->gpu_transport&&gpu_settings_&&in_process_api_available()&&!bgra_&&!rgb10_&&
           (device_->get_api()==device_api::d3d11||device_->get_api()==device_api::d3d12)&&
           (device_->get_api()==device_api::d3d11||device_->check_capability(device_caps::shared_resource_nt_handle))&&
           fs::is_regular_file(portable->root/"phoenix_native.dll")){
            HANDLE input=nullptr,output=nullptr;
            try{
                auto shared=write;shared.flags=resource_flags::shared|resource_flags::shared_nt_handle;
                shared.usage|=resource_usage::general;
                if(device_->get_api()==device_api::d3d11){
                    auto* game=reinterpret_cast<ID3D11Device*>(uintptr_t(device_->get_native()));
                    auto color=portable_gpu_texture11(game,width_,height_,&input);
                    readback_={uint64_t(uintptr_t(color.Detach()))};
                    auto result=portable_gpu_texture11(game,width_,height_,&output);
                    gpu_output_={uint64_t(uintptr_t(result.Detach()))};
                }else{
                    require(device_->create_resource(shared,nullptr,resource_usage::general,&readback_,&input),"Cannot export GPU color mailbox");
                    require(device_->create_resource(shared,nullptr,resource_usage::general,&gpu_output_,&output),"Cannot export GPU output mailbox");
                }
                LUID luid{};
                if(device_->get_api()==device_api::d3d12)luid=reinterpret_cast<ID3D12Device*>(uintptr_t(device_->get_native()))->GetAdapterLuid();
                else {Ptr<IDXGIDevice> dxgi;hr(reinterpret_cast<ID3D11Device*>(uintptr_t(device_->get_native()))->QueryInterface(IID_PPV_ARGS(&dxgi)));
                    Ptr<IDXGIAdapter> adapter;hr(dxgi->GetAdapter(&adapter));DXGI_ADAPTER_DESC desc{};hr(adapter->GetDesc(&desc));luid=desc.AdapterLuid;}
                gpu_handles=std::make_shared<PortableGpuHandles>(input,output,luid);portable->gpu=gpu_handles;
                gpu_transport_=true;buffer_readback_=false;
            }catch(const std::exception& error){
                gpu_fallback_=error.what();
                if(readback_.handle)device_->destroy_resource(readback_);
                if(gpu_output_.handle)device_->destroy_resource(gpu_output_);
                readback_=gpu_output_={};
            }
            if(input)CloseHandle(input);if(output)CloseHandle(output);
        }
        try {
            if(!gpu_transport_)require(device_->create_resource(read, nullptr, resource_usage::copy_dest, &readback_),
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
                captures_.capacity(compositor9_->capture_capacity());
            }
            require(device_->create_fence(0, fence_flags::none, &retired_),
                    "API cannot provide transfer fencing");
            FeedBuild build{};
            build.width = build.output_width = width_;
            build.height = build.output_height = height_;
            build.color_fmt = build.output_fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
            build.cpu_bridge = gpu_transport_?0:1;
            build.mv_scale_x = build.mv_scale_y = 1;
            host_ = std::make_unique<HostClient>(addon, source, settings, build, portable,native,local);
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
        for (auto resource : {readback_, upload_, upload_buffer_,gpu_output_})
            if (resource.handle)
                device_->destroy_resource(resource);
        retired_ = {};
        readback_ = upload_ = upload_buffer_ = gpu_output_ = {};
    }
    ~ReadbackBridge() {
        release();
    }
    const char* transfer_stage()const{return transfer_stage_;}
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
                {"last_cpu_readback_ms",cpu_readback_ms_},
                {"readback_bytes",gpu_transport_?0:uint64_t(width_)*height_*4},
                {"capture_pixel_transport",gpu_transport_?"shared-gpu-mailbox":"cpu-staging"},
                {"capture_host_transfer_bytes",gpu_transport_?0:uint64_t(width_)*height_*8},
                {"gpu_transport_fallback",gpu_fallback_},
                {"readback_memory",gpu_transport_?"Shared GPU texture; no pixel mapping":"API-managed readback heap; CPU mapping follows GPU retirement"},
                {"last_upload_ms", upload_ms_},
                {"last_compose_ms", compose_ms_},
                {"last_deferred_readback_ms", deferred_readback_ms_},
                {"capture_snapshot_copies", capture_copies_},
                {"post_present_calls",post_present_calls_},
                {"post_present_retirements", post_present_retirements_},
                {"last_post_wait_ms",last_post_wait_ms_},
                {"frame_period_ms",frame_period_ms_},
                {"capture_not_ready_frames",capture_not_ready_frames_},
                {"worker_not_ready_frames",worker_not_ready_frames_},
                {"upload_not_ready_frames",upload_not_ready_frames_},
                {"last_pre_completed_serial",last_pre_completed_},
                {"last_post_completed_serial",last_post_completed_},
                {"capture_serial",capture_serial_},{"upload_serial",upload_serial_},
                {"post_present_submissions", post_present_submissions_},
                {"deferred_readbacks", deferred_readbacks_},
                {"capture_slots",captures_.capacity()},
                {"pending_captures",captures_.size()},
                {"capture_prefetch",prefetch_available()},
                {"queued_input",!queued_pixels_.empty()},
                {"input_frame", presented_input_frame_}, {"output_frame", presented_output_frame_},
                {"composed_at_frame", composed_at_frame_}, {"width", width_}, {"height", height_}};
    }
    void signal_after_present() {
        if (!queued_signal_)
            return;
        const auto begin = std::chrono::steady_clock::now();
        require(queue_->signal(retired_, queued_signal_),
                "Cannot signal graphics transfer completion");
        // DX11 can retain a signal in its immediate-context batch while an
        // occluded/background swapchain presents. Submit that batch without
        // waiting for GPU completion; polling alone must not starve captures.
        if(device_->get_api()==reshade::api::device_api::d3d11)
            queue_->flush_immediate_command_list();
        if (elapsed_ms(begin)>100)
            reshade::log::message(reshade::log::level::warning,
                ("Phoenix transfer fence signal stalled " + std::to_string(elapsed_ms(begin)) + " ms").c_str());
        queued_signal_ = 0;
    }
    void retire_after_present(const phoenix_addon::phoenix::Settings& settings) {
        ++post_present_calls_;
        retire_and_submit(settings, true);
    }
    void frame(reshade::api::command_list* command, reshade::api::resource back,
               uint64_t frame_number, const phoenix_addon::phoenix::Settings& settings,
               bool coherent_frames = false) {
        using namespace reshade::api;
        const auto frame_begin = std::chrono::steady_clock::now();
        if(preceding_frame_!=std::chrono::steady_clock::time_point{})
            frame_period_ms_=std::chrono::duration<double,std::milli>(frame_begin-preceding_frame_).count();
        preceding_frame_=frame_begin;
        require(command == queue_->get_immediate_command_list(),
                "Image transport requires the runtime immediate queue");
        const auto completed = device_->get_completed_fence_value(retired_);
        require(completed != UINT64_MAX, "Readback graphics device was lost");
        last_pre_completed_=completed;
        if(host_->phase()==Phase::Finished&&completed<upload_serial_)++upload_not_ready_frames_;
        // A persistent upload buffer cannot be rewritten while the GPU reads it.
        if (host_->phase() == Phase::Finished && completed >= upload_serial_) {
            const auto upload_begin = std::chrono::steady_clock::now();
            transfer_stage_="completed output acquisition";
            auto pixels = host_->take();
            if(gpu_transport_){
                command->barrier(gpu_output_,resource_usage::general,resource_usage::copy_source);
                command->barrier(upload_,resource_usage::copy_source,resource_usage::copy_dest);
                command->copy_resource(gpu_output_,upload_);
                command->barrier(upload_,resource_usage::copy_dest,resource_usage::copy_source);
                command->barrier(gpu_output_,resource_usage::copy_source,resource_usage::general);
                upload_serial_=reserve_retirement_serial(serial_,queued_signal_);
            }else{
            screenshot_pair_.complete(submitted_frame_,pixels,width_,height_);
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
                upload_serial_ = reserve_retirement_serial(serial_, queued_signal_);
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
                upload_serial_ = reserve_retirement_serial(serial_, queued_signal_);
            } else {
                transfer_stage_="presentation CPU byte conversion";
                const uint8_t* native_pixels=pixels.data();
                if(rgb10_||bgra_) {
                    upload_pixels_.resize(pixels.size());
                    if(rgb10_)
                        copy_rgb10a2_rows(upload_pixels_.data(),width_*4,pixels.data(),width_*4,width_,height_,false);
                    else
                        copy_pixel_rows(upload_pixels_.data(),width_*4,pixels.data(),width_*4,width_,height_,false,true);
                    native_pixels=upload_pixels_.data();
                }
                device_->update_texture_region(
                    {const_cast<uint8_t*>(native_pixels), width_ * 4, width_ * height_ * 4}, upload_, 0);
            }
            }
            delivered_image_ = true;
            if (compositor9_)
                compositor9_->match_completed_input();
            delivered_frame_ = submitted_frame_;
            delivered_tick_ = submitted_tick_;
            ++delivered_;
            upload_ms_ = elapsed_ms(upload_begin);
            // Neither GPU staging nor the completed screenshot references the
            // CPU vector now. Reuse it instead of clearing/allocating another
            // full image on the native worker's next frame.
            host_->recycle_output(std::move(pixels));
        }
        retire_and_submit(settings, false);
        const int capture_slot=captures_.free_slot();
        if (capture_slot>=0 &&
            (host_->phase() == Phase::Ready ||
             (prefetch_available() && (host_->phase()==Phase::Busy||host_->phase()==Phase::Finished))) &&
            settings.mode != phoenix_addon::phoenix::input) {
            transfer_stage_="GPU capture copy";
            const auto capture_begin = std::chrono::steady_clock::now();
            if (compositor9_)
                compositor9_->capture(reinterpret_cast<IDirect3DSurface9*>(uintptr_t(back.handle)),unsigned(capture_slot));
            else {
                command->barrier(back, resource_usage::present, resource_usage::copy_source);
                if(gpu_transport_)command->barrier(readback_,resource_usage::general,resource_usage::copy_dest);
                if (buffer_readback_)
                    command->copy_texture_to_buffer(back, 0, nullptr, readback_, 0, row_pitch_ / 4,
                                                    height_);
                else
                    command->copy_resource(back, readback_);
                if(gpu_transport_)command->barrier(readback_,resource_usage::copy_dest,resource_usage::general);
                command->barrier(back, resource_usage::copy_source, resource_usage::present);
            }
            ++capture_copies_;
            capture_serial_ = reserve_retirement_serial(serial_, queued_signal_);
            capture_frame_ = frame_number;
            capture_tick_ = GetTickCount64();
            captures_.push({capture_serial_,capture_frame_,capture_tick_,unsigned(capture_slot),false});
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
