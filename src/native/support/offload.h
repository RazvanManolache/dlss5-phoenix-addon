#pragma once
#include "executor.h"
#include "compute_pass.h"
#include "async_shader.h"
#include "worker_client.h"
#include "../../diagnostics/composition_probe.h"
#include "../../pipeline/recipe.h"

namespace phoenix_addon::phoenix::portable {
// Bounded GPU -> worker -> GPU execution. The caller must submit command lists
// in order on one queue and retain this object until its last serial completes.
// No GPU fence wait, inference wait or frame-file I/O occurs in record().
class OffloadExecutor final : public NeuralExecutor {
    static constexpr auto read_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    static constexpr const char* copy_shader = R"(
Texture2D<float4> source:register(t0); RWTexture2D<float4> target:register(u0);
cbuffer Params:register(b0){uint width,height,reset,mode;uint a,b;float divider,gain;uint resolve;};
[numthreads(16,16,1)] void main(uint3 p:SV_DispatchThreadID){
 if(p.x<width&&p.y<height)target[p.xy]=source.Load(int3(p.xy,0));
})";
    struct Slot {
        ComPtr<ID3D12Resource> maps[2], residual;
        std::unique_ptr<direct::Pass> initialize, advance, compose;
        uint64_t serial = 0, tick = 0;
        uint64_t locked_input_token = 0;
        uint32_t generation = 0;
        unsigned map = 0;
        bool tracking = false;
    } slots_[2];
    unsigned width_, height_, max_frames_;
    uint64_t max_ms_, last_serial_ = 0, capture_fence_ = 0, upload_fence_ = 0;
    uint64_t current_locked_input_ = 0;
    uint32_t generation_ = 1;
    int pending_ = -1, ready_ = -1;
    bool submitted_ = false, readback_decoded_ = false, failed_ = false;
    bool capture_device_depth_ = false;
    bool capture_mask_ = false, capture_distortion_ = false;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12Resource> current_, guides_, output_, readback_, upload_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint_{};
    UINT64 texture_bytes_ = 0;
    ID3D12Resource* color_identity_ = nullptr;
    std::filesystem::path python_, script_, weights_, logs_, vulkan_build_, hip_build_, sycl_build_;
    std::wstring worker_device_, vulkan_engine_;
    unsigned cpu_threads_ = 0;
    std::unique_ptr<WorkerClient> worker_;
    phoenix_model::WeightSource weight_source_;
    static constexpr unsigned max_restarts_ = 3;
    unsigned restart_attempts_ = 0, worker_exits_ = 0;
    bool accounted_ = false;
    DWORD last_exit_code_ = STILL_ACTIVE;
    uint64_t retry_at_ = 0;
    std::string recovery_error_;
    direct::Pass copy_;
    CompositionProbe composition_;
    double residual_mean_ = 0, residual_max_ = 0;
    std::vector<float> color_, motion_, depth_, history_validity_, result_;
    std::vector<float> captured_mask_,captured_distortion_;
    Json capture_request_, signature_, last_result_ = Json::object();
    std::string state_ = "worker_starting", error_;
    uint64_t captures_ = 0, delivered_ = 0, discarded_ = 0, presented_ = 0;
    uint64_t age_discarded_ = 0, last_result_age_ms_ = 0, last_result_age_frames_ = 0;
    std::string last_discard_reason_;

    void start_worker() {
        worker_ = std::make_unique<WorkerClient>(
            width_, height_, python_, script_, weights_, worker_device_, logs_, vulkan_build_,
            vulkan_engine_, hip_build_, sycl_build_, weight_source_, cpu_threads_);
    }
    void schedule_restart(uint64_t tick) {
        // Only process death is retried. Do not hide resource/IPC contract faults.
        last_exit_code_ = worker_->exit_code();
        ++worker_exits_;
        recovery_error_ = "C++ runner exited with code " + std::to_string(last_exit_code_);
        auto detail = worker_->status().value("error", std::string());
        if (!detail.empty())
            recovery_error_ += ": " + detail;
        worker_.reset();
        invalidate();
        if (pending_ >= 0)
            ++discarded_;
        pending_ = -1;
        submitted_ = false;
        readback_decoded_ = false;
        result_.clear();
        capture_request_ = Json::object();
        last_result_ = Json::object();
        // Keep capture/upload fences: an abandoned request does not retire GPU work.
        if (restart_attempts_ >= max_restarts_) {
            failed_ = true;
            error_ = recovery_error_ + "; automatic restart budget exhausted";
            state_ = "worker_restart_exhausted_input_preserved";
        } else {
            retry_at_ = tick + (uint64_t(500) << restart_attempts_);
            state_ = "worker_restart_backoff_input_preserved";
        }
    }
    bool ensure_worker(uint64_t tick) {
        if (worker_) {
            if (worker_->alive())
                return true;
            schedule_restart(tick);
            return false;
        }
        if (tick < retry_at_) {
            state_ = "worker_restart_backoff_input_preserved";
            return false;
        }
        ++restart_attempts_;
        try {
            start_worker();
            state_ = "worker_starting";
        } catch (const std::exception& exception) {
            recovery_error_ = exception.what();
            if (restart_attempts_ >= max_restarts_) {
                failed_ = true;
                error_ = recovery_error_ + "; automatic restart budget exhausted";
                state_ = "worker_restart_exhausted_input_preserved";
            } else {
                retry_at_ = tick + (uint64_t(500) << restart_attempts_);
                state_ = "worker_restart_backoff_input_preserved";
            }
        }
        return false; // Never present or submit an old capture on the launch frame.
    }

    ComPtr<ID3D12Resource> resource(D3D12_RESOURCE_DESC description, D3D12_HEAP_TYPE heap,
                                    D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES heap_properties{};
        heap_properties.Type = heap;
        ComPtr<ID3D12Resource> allocation;
        phoenix_native::hr(device_->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE,
                                                            &description, state, nullptr,
                                                            IID_PPV_ARGS(&allocation)));
        return allocation;
    }
    static void barrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource,
                        D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER transition{};
        transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        transition.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
        cmd->ResourceBarrier(1, &transition);
    }
    void buffer_copy(ID3D12GraphicsCommandList* cmd, ID3D12Resource* texture,
                     ID3D12Resource* buffer, UINT64 offset, bool download) {
        D3D12_TEXTURE_COPY_LOCATION texture_location{}, buffer_location{};
        texture_location.pResource = texture;
        texture_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        buffer_location.pResource = buffer;
        buffer_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        buffer_location.PlacedFootprint = footprint_;
        buffer_location.PlacedFootprint.Offset = offset;
        if (download)
            cmd->CopyTextureRegion(&buffer_location, 0, 0, 0, &texture_location, nullptr);
        else
            cmd->CopyTextureRegion(&texture_location, 0, 0, 0, &buffer_location, nullptr);
    }
    bool eligible(const Slot& slot, uint64_t serial, uint64_t tick) const {
        return slot.tracking && slot.generation == generation_ && serial >= slot.serial &&
               capture_age_valid(slot.locked_input_token, current_locked_input_,
                                 serial-slot.serial, tick-slot.tick, max_frames_, max_ms_);
    }
    Json request(const ExecutorFrame& frame) const {
        const auto& controls = frame.controls;
        Json request_body{{"has_motion", true},
                          {"budget_epoch", frame.budget_epoch},
                          {"transformer_blocks", frame.transformer_blocks},
                          {"output_stage","neural_only"},
                          {"has_control_mask",frame.appearance.mask!=nullptr},
                          {"has_distortion",frame.appearance.field!=nullptr},
                          {"appearance_generation",frame.appearance_generation},
                          {"has_depth", true},
                          {"has_history_validity", true},
                          {"reversed_depth", frame.device_depth && controls.inverted != 0},
                          {"controls",
                           {{"intensity", 1.f},
                            {"style", controls.style},
                            {"local_tone", controls.tone},
                            {"local_structure", controls.structure},
                            {"skin_structure", controls.skin},
                            {"auto_mask", controls.auto_mask != 0},
                            {"seed", 0}}}};
        if (frame.settings) {
            auto repeats = frame.settings->block_repeats;
            for (unsigned iteration_index = frame.transformer_blocks; iteration_index < 8;
                 ++iteration_index)
                repeats[iteration_index] = 0;
            request_body["repeats"] = repeats;
            request_body["passes"] = frame.settings->refinement_passes;
            if(!phoenix_pipeline::default_chain(frame.settings->pipeline,width_,height_,width_,height_))
                request_body["pipeline"]=phoenix_pipeline::options_json(phoenix_pipeline::without_comparison(frame.settings->pipeline));
        }
        return request_body;
    }
    // Decode the completed readback using the D3D12 padded row pitch, rather
    // than assuming tightly packed texture rows. No GPU storage is read before
    // the capture fence signals completion.
    bool decode_capture() {
        void* mapped_bytes = nullptr;
        D3D12_RANGE range{0, SIZE_T(texture_bytes_ * 5)};
        phoenix_native::hr(readback_->Map(0, &range, &mapped_bytes));
        auto decode_optional=[&](bool present,unsigned offset,std::vector<float>& destination) {
            destination.resize(present?size_t(width_)*height_*4:0);
            if(present)for(unsigned y=0;y<height_;++y)
                std::memcpy(destination.data()+size_t(y)*width_*4,
                    static_cast<const char*>(mapped_bytes)+texture_bytes_*offset+size_t(y)*footprint_.Footprint.RowPitch,
                    size_t(width_)*16);
        };
        decode_optional(capture_mask_,3,captured_mask_);decode_optional(capture_distortion_,4,captured_distortion_);
        // Both snapshots were copied in the same command list. They cannot be
        // touched until capture_fence_ has completed on the host's queue.
        for (unsigned pixel_y = 0; pixel_y < height_; ++pixel_y)
            for (unsigned pixel_x = 0; pixel_x < width_; ++pixel_x) {
                const size_t iteration_index = size_t(pixel_y) * width_ + pixel_x;
                auto* rgb =
                    reinterpret_cast<const float*>(static_cast<const char*>(mapped_bytes) +
                                                   pixel_y * footprint_.Footprint.RowPitch) +
                    pixel_x * 4;
                auto* guide = reinterpret_cast<const float*>(
                                  static_cast<const char*>(mapped_bytes) + texture_bytes_ +
                                  pixel_y * footprint_.Footprint.RowPitch) +
                              pixel_x * 4;
                for (unsigned channel = 0; channel < 3; ++channel)
                    color_[iteration_index * 3 + channel] = rgb[channel];
                // Invalid guides cannot become invented motion or depth for the
                // network. Reject the capture; per-pixel invalidation still applies
                // when a guide becomes invalid after the capture.
                if (!std::isfinite(guide[0]) || !std::isfinite(guide[1]) ||
                    !std::isfinite(guide[2]) ||
                    (capture_device_depth_ ? (guide[2] < 0 || guide[2] > 1) : guide[2] <= 0) ||
                    !std::isfinite(guide[3]) || guide[3] <= 0) {
                    D3D12_RANGE empty{0, 0};
                    readback_->Unmap(0, &empty);
                    return false;
                }
                motion_[iteration_index * 2] = guide[0];
                motion_[iteration_index * 2 + 1] = guide[1];
                depth_[iteration_index] = guide[2];
            }
        const bool has_history_map = capture_request_.contains("motion_from_frame");
        bool history_valid = false;
        // A history warp needs motion back to the last evaluated source frame,
        // not just last display frame. Reuse the same accumulated GPU map.
        // Preserve valid correspondences elsewhere when a region disoccludes.
        // Invalid pixels carry an explicit mask, not invented zero-motion history.
        std::fill(history_validity_.begin(), history_validity_.end(), 0.0f);
        if (has_history_map) {
            for (unsigned pixel_y = 0; pixel_y < height_; ++pixel_y)
                for (unsigned pixel_x = 0; pixel_x < width_; ++pixel_x) {
                    auto* motion_row =
                        reinterpret_cast<const float*>(static_cast<const char*>(mapped_bytes) +
                                                       texture_bytes_ * 2 +
                                                       pixel_y * footprint_.Footprint.RowPitch) +
                        4 * pixel_x;
                    const bool valid =
                        std::isfinite(motion_row[0]) && std::isfinite(motion_row[1]) &&
                        std::isfinite(motion_row[2]) && std::isfinite(motion_row[3]) &&
                        motion_row[2] >= .9999f && motion_row[0] >= .5f && motion_row[1] >= .5f &&
                        motion_row[0] <= width_ - .5f && motion_row[1] <= height_ - .5f;
                    const size_t iteration_index = size_t(pixel_y) * width_ + pixel_x;
                    history_validity_[iteration_index] = valid ? 1.0f : 0.0f;
                    motion_[2 * iteration_index] = valid ? motion_row[0] - (pixel_x + .5f) : 0.0f;
                    motion_[2 * iteration_index + 1] =
                        valid ? motion_row[1] - (pixel_y + .5f) : 0.0f;
                    history_valid = history_valid || valid;
                }
        }
        if (!history_valid)
            capture_request_.erase("motion_from_frame");
        capture_request_["reset"] = !history_valid;
        D3D12_RANGE empty{0, 0};
        readback_->Unmap(0, &empty);
        readback_decoded_ = true;
        return true;
    }

  public:
    void account(const std::shared_ptr<phoenix_native::memory::Generation>& generation) override {
        if (!generation || accounted_)
            return;
        accounted_ = true;
        // GPU snapshots, motion maps, residuals and CPU-visible transfer heaps
        // belong to the game process; worker model/scratch is reported separately.
        for (auto* resource :
             {current_.Get(), guides_.Get(), output_.Get(), readback_.Get(), upload_.Get()})
            phoenix_native::memory::resource(device_.Get(), resource, generation,
                                             phoenix_native::memory::Kind::transport);
        copy_.account(generation);
        for (auto& slot : slots_) {
            for (const auto& map : slot.maps)
                phoenix_native::memory::resource(device_.Get(), map.Get(), generation,
                                                 phoenix_native::memory::Kind::transport);
            phoenix_native::memory::resource(device_.Get(), slot.residual.Get(), generation,
                                             phoenix_native::memory::Kind::transport);
            slot.initialize->account(generation);
            slot.advance->account(generation);
            slot.compose->account(generation);
        }
        composition_.account(device_.Get(), generation);
    }
    OffloadExecutor(ID3D12Device* device, unsigned width, unsigned height,
                    const std::filesystem::path& python, const std::filesystem::path& script,
                    const std::filesystem::path& weights, const std::wstring& worker_device,
                    const std::filesystem::path& logs, unsigned max_frames = 4,
                    uint64_t max_ms = 100, const std::filesystem::path& vulkan_build = {},
                    const std::wstring& vulkan_engine = L"auto",
                    const std::filesystem::path& hip_build = {},
                    const std::filesystem::path& sycl_build = {},
                    const phoenix_model::WeightSource& weight_source = {}, unsigned cpu_threads = 0)
        : width_(width), height_(height), max_frames_(max_frames), max_ms_(max_ms), device_(device),
          python_(python), script_(script), weights_(weights), logs_(logs),
          vulkan_build_(vulkan_build), hip_build_(hip_build), sycl_build_(sycl_build),
          weight_source_(weight_source), worker_device_(worker_device),
          vulkan_engine_(vulkan_engine), cpu_threads_(cpu_threads), copy_(device, copy_shader),
          composition_(device) {
        check(max_frames && max_frames <= kMaxOffloadAgeFrames && max_ms &&
                  max_ms <= kMaxOffloadAgeMs,
              "Offload age budget out of range");
        start_worker();
        D3D12_RESOURCE_DESC texture_description{};
        texture_description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture_description.Width = width;
        texture_description.Height = height;
        texture_description.DepthOrArraySize = texture_description.MipLevels = 1;
        texture_description.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        texture_description.SampleDesc.Count = 1;
        device_->GetCopyableFootprints(&texture_description, 0, 1, 0, &footprint_, nullptr, nullptr,
                                       &texture_bytes_);
        texture_bytes_ = (texture_bytes_ + 511) & ~UINT64(511);
        texture_description.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        current_ = resource(texture_description, D3D12_HEAP_TYPE_DEFAULT, read_state);
        guides_ = resource(texture_description, D3D12_HEAP_TYPE_DEFAULT, read_state);
        output_ = resource(texture_description, D3D12_HEAP_TYPE_DEFAULT, read_state);
        for (auto& slot : slots_) {
            for (auto& map : slot.maps)
                map = resource(texture_description, D3D12_HEAP_TYPE_DEFAULT, read_state);
            slot.residual = resource(texture_description, D3D12_HEAP_TYPE_DEFAULT, read_state);
            slot.initialize = std::make_unique<direct::Pass>(device, async_shader);
            slot.advance = std::make_unique<direct::Pass>(device, async_shader);
            slot.compose = std::make_unique<direct::Pass>(device, async_shader);
        }
        D3D12_RESOURCE_DESC buffer_description{};
        buffer_description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer_description.Width = texture_bytes_ * 5;
        buffer_description.Height = 1;
        buffer_description.DepthOrArraySize = buffer_description.MipLevels = 1;
        buffer_description.SampleDesc.Count = 1;
        buffer_description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        readback_ =
            resource(buffer_description, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        buffer_description.Width = texture_bytes_;
        upload_ =
            resource(buffer_description, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        color_.resize(size_t(width) * height * 3);
        motion_.resize(size_t(width) * height * 2);
        depth_.resize(size_t(width) * height);
        history_validity_.resize(size_t(width) * height);
    }
    void invalidate() noexcept override {
        composition_.invalidate();
        ++generation_;
        ready_ = -1;
        for (auto& slot : slots_)
            slot.tracking = false;
        // Do not recycle mapped/copy buffers here: their fences still matter.
    }
    Json status() const override {
        const auto tick = GetTickCount64();
        return {{"host_transport_vector_capacity_bytes",
                 (color_.capacity() + motion_.capacity() + depth_.capacity() +
                  history_validity_.capacity() + result_.capacity()) *
                     sizeof(float)},
                {"kind", "portable_worker"},
                {"state", state_},
                {"error", error_},
                {"generation", generation_},
                {"captures", captures_},
                {"delivered", delivered_},
                {"discarded", discarded_},
                {"presented", presented_},
                {"composition", composition_.result()},
                {"residual_mean_abs", residual_mean_},
                {"residual_max_abs", residual_max_},
                {"age_discarded", age_discarded_},
                {"last_discard_reason", last_discard_reason_},
                {"last_result_age_ms", last_result_age_ms_},
                {"last_result_age_frames", last_result_age_frames_},
                {"ready_frame", ready_ < 0 || !slots_[ready_].tracking ? 0 : slots_[ready_].serial},
                {"last_result", last_result_},
                {"capture_fence", capture_fence_},
                {"worker_submitted", submitted_},
                {"worker_pid", worker_ ? worker_->process_id() : 0},
                {"worker_exits", worker_exits_},
                {"restart_attempts", restart_attempts_},
                {"restart_limit", max_restarts_},
                {"last_exit_code", last_exit_code_},
                {"recovery_error", recovery_error_},
                {"requested_device", std::string(worker_device_.begin(), worker_device_.end())},
                {"restart_delay_remaining_ms",
                 worker_ || failed_ ? uint64_t(0) : (retry_at_ > tick ? retry_at_ - tick : 0)},
                {"max_age_frames", max_frames_},
                {"max_age_ms", max_ms_},
                {"timing_scope",
                 "Worker elapsed time is separate from GPU transfer/presentation time"}};
    }
    ID3D12Resource* record(const ExecutorFrame& frame) override {
        check(frame.command && frame.color && frame.serial > last_serial_ &&
                  frame.completed_serial < frame.serial,
              "Offload frame sequence/command invalid");
        const bool gap = last_serial_ && frame.serial != last_serial_ + 1;
        last_serial_ = frame.serial;
        current_locked_input_ = frame.locked_input_token;
        if (failed_)
            return frame.color;
        if (!frame.guides) {
            invalidate();
            state_ = "missing_linear_depth_guides";
            return frame.color;
        }
        const auto guide_description = frame.guides->GetDesc(),
                   color_description = frame.color->GetDesc();
        check(guide_description.Width == width_ && guide_description.Height == height_ &&
                  guide_description.Format == DXGI_FORMAT_R32G32B32A32_FLOAT &&
                  guide_description.MipLevels == 1 && guide_description.DepthOrArraySize == 1 &&
                  guide_description.SampleDesc.Count == 1,
              "Offload guide contract");
        check(color_description.Width == width_ && color_description.Height == height_ &&
                  (color_description.Format == DXGI_FORMAT_R16G16B16A16_FLOAT ||
                   color_description.Format == DXGI_FORMAT_R32G32B32A32_FLOAT),
              "Offload model proxy contract");
        // Stable owned inputs keep shader descriptors immutable while recorded
        // lists are in flight. Recreate the executor after host texture changes.
        check(!color_identity_ || color_identity_ == frame.color,
              "Offload color resource changed; recreate executor after retiring GPU work");
        color_identity_ = frame.color;
        auto desired = request(frame);
        if (gap || frame.controls.reset || desired != signature_ ||
            frame.device_depth != capture_device_depth_) {
            invalidate();
            signature_ = desired;
            capture_device_depth_ = frame.device_depth;
        }
        if (!frame.controls.enabled) {
            invalidate();
            state_ = "disabled";
            return frame.color;
        }
        const uint64_t tick = GetTickCount64();
        auto* cmd = frame.command;
        composition_.retire(frame.completed_serial, generation_);
        if (!ensure_worker(tick))
            return frame.color;
        state_ = "waiting_for_fresh_result";
        direct::Constants constants{width_,
                                    height_,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    frame.device_depth ? .002f : .02f,
                                    frame.device_depth ? 1u : 0u};
        barrier(cmd, current_.Get(), read_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        copy_.record(cmd, {frame.color, nullptr, nullptr}, current_.Get(), constants);
        barrier(cmd, current_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, read_state);
        barrier(cmd, frame.guides, read_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        barrier(cmd, guides_.Get(), read_state, D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(guides_.Get(), frame.guides);
        barrier(cmd, guides_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, read_state);
        barrier(cmd, frame.guides, D3D12_RESOURCE_STATE_COPY_SOURCE, read_state);
        for (auto& slot : slots_)
            if (slot.tracking) {
                if (!eligible(slot, frame.serial, tick)) {
                    slot.tracking = false;
                    continue;
                }
                const unsigned next = 1 - slot.map;
                constants.mode = 1;
                barrier(cmd, slot.maps[next].Get(), read_state,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                slot.advance->record(cmd, {slot.maps[slot.map].Get(), guides_.Get(), nullptr},
                                     slot.maps[next].Get(), constants, false, next);
                barrier(cmd, slot.maps[next].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        read_state);
                slot.map = next;
            }
        try {
            if (pending_ >= 0 && submitted_ && result_.empty()) {
                uint64_t serial = 0;
                uint32_t generation = 0;
                Json meta;
                if (worker_->poll(serial, generation, result_, meta)) {
                    last_result_ = meta;
                    const auto& captured = slots_[pending_];
                    last_result_age_ms_ = tick - captured.tick;
                    last_result_age_frames_ = frame.serial - captured.serial;
                    last_discard_reason_.clear();
                    if (!meta.value("ok", false) || serial != slots_[pending_].serial ||
                        generation != generation_ ||
                        !eligible(slots_[pending_], frame.serial, tick)) {
                        last_discard_reason_ =
                            !meta.value("ok", false)    ? "worker_error"
                            : serial != captured.serial ? "frame_identity"
                            : generation != generation_ ? "invalidated_generation"
                            : last_result_age_ms_ > max_ms_ || last_result_age_frames_ > max_frames_
                                ? "result_age_limit"
                                : "invalidated_history";
                        if (last_discard_reason_ == "result_age_limit")
                            ++age_discarded_;
                        result_.clear();
                        slots_[pending_].tracking = false;
                        pending_ = -1;
                        submitted_ = false;
                        ++discarded_;
                        state_ =
                            meta.value("ok", false) ? "discarded_stale_result" : "worker_error";
                    }
                }
            }
            if (pending_ >= 0 && !result_.empty() && frame.completed_serial >= upload_fence_) {
                auto& slot = slots_[pending_];
                if (eligible(slot, frame.serial, tick)) {
                    void* data = nullptr;
                    D3D12_RANGE empty{0, 0};
                    phoenix_native::hr(upload_->Map(0, &empty, &data));
                    residual_mean_ = residual_max_ = 0;
                    for (unsigned pixel_y = 0; pixel_y < height_; ++pixel_y)
                        for (unsigned pixel_x = 0; pixel_x < width_; ++pixel_x) {
                            const size_t iteration_index = size_t(pixel_y) * width_ + pixel_x;
                            auto* pixel =
                                reinterpret_cast<float*>(static_cast<char*>(data) +
                                                         pixel_y * footprint_.Footprint.RowPitch) +
                                4 * pixel_x;
                            for (unsigned channel = 0; channel < 3; ++channel) {
                                const float residual = result_[iteration_index * 3 + channel] -
                                                       color_[iteration_index * 3 + channel];
                                pixel[channel] = residual;
                                residual_mean_ += std::abs(residual);
                                residual_max_ =
                                    std::max(residual_max_, double(std::abs(residual)));
                            }
                            pixel[3] = 0;
                        }
                    residual_mean_ /= double(width_) * height_ * 3;
                    upload_->Unmap(0, nullptr);
                    barrier(cmd, slot.residual.Get(), read_state, D3D12_RESOURCE_STATE_COPY_DEST);
                    buffer_copy(cmd, slot.residual.Get(), upload_.Get(), 0, false);
                    barrier(cmd, slot.residual.Get(), D3D12_RESOURCE_STATE_COPY_DEST, read_state);
                    upload_fence_ = frame.serial;
                    if (ready_ >= 0)
                        slots_[ready_].tracking = false;
                    ready_ = pending_;
                    ++delivered_;
                } else {
                    slot.tracking = false;
                    ++discarded_;
                }
                result_.clear();
                pending_ = -1;
                submitted_ = false;
            }
            if (pending_ >= 0 && !submitted_ && frame.completed_serial >= capture_fence_) {
                auto& slot = slots_[pending_];
                if (!eligible(slot, frame.serial, tick)) {
                    slot.tracking = false;
                    pending_ = -1;
                    ++discarded_;
                } else {
                    if (!readback_decoded_ && !decode_capture()) {
                        invalidate();
                        pending_ = -1;
                        submitted_ = false;
                        ++discarded_;
                        state_ = "invalid_capture_input_preserved";
                        return frame.color;
                    }
                    submitted_ =
                        worker_->submit(slot.serial, slot.generation, color_.data(), motion_.data(),
                                        depth_.data(), capture_request_, history_validity_.data(),
                                        captured_mask_.empty()?nullptr:captured_mask_.data(),nullptr,nullptr,nullptr,
                                        captured_distortion_.empty()?nullptr:captured_distortion_.data());
                    state_ = submitted_ ? "inference_running" : "worker_starting";
                }
            }
            const auto worker_status = worker_->status();
            if (pending_ < 0 && frame.completed_serial >= capture_fence_ &&
                worker_status.value("state", 99) == uint32_t(WorkerState::idle) &&
                worker_status.value("initialized", false)) {
                pending_ = ready_ == 0 ? 1 : 0;
                auto& slot = slots_[pending_];
                slot.serial = frame.serial;
                slot.tick = tick;
                slot.locked_input_token = frame.locked_input_token;
                slot.generation = generation_;
                slot.map = 0;
                slot.tracking = true;
                constants.mode = 0;
                barrier(cmd, slot.maps[0].Get(), read_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                slot.initialize->record(cmd, {nullptr, guides_.Get(), nullptr}, slot.maps[0].Get(),
                                        constants);
                barrier(cmd, slot.maps[0].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, read_state);
                for (unsigned iteration_index = 0; iteration_index < 2; ++iteration_index) {
                    auto* source = iteration_index ? guides_.Get() : current_.Get();
                    barrier(cmd, source, read_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    buffer_copy(cmd, source, readback_.Get(), iteration_index * texture_bytes_,
                                true);
                    barrier(cmd, source, D3D12_RESOURCE_STATE_COPY_SOURCE, read_state);
                }
                capture_fence_ = frame.serial;
                capture_request_ = desired;
                capture_mask_=frame.appearance.mask;capture_distortion_=frame.appearance.field;
                ID3D12Resource* optional[]={frame.appearance.mask,frame.appearance.field};
                for(unsigned i=0;i<2;++i)if(optional[i]) {
                    const auto d=optional[i]->GetDesc();
                    check(d.Width==width_&&d.Height==height_&&d.Format==DXGI_FORMAT_R32G32B32A32_FLOAT,"Offload appearance plane geometry/format");
                    barrier(cmd,optional[i],read_state,D3D12_RESOURCE_STATE_COPY_SOURCE);
                    buffer_copy(cmd,optional[i],readback_.Get(),texture_bytes_*(3+i),true);
                    barrier(cmd,optional[i],D3D12_RESOURCE_STATE_COPY_SOURCE,read_state);
                }
                if (ready_ >= 0 && eligible(slots_[ready_], frame.serial, tick) &&
                    last_result_.value("ok", false) &&
                    last_result_.value("frame_id", uint64_t(0)) == slots_[ready_].serial) {
                    auto& previous = slots_[ready_];
                    auto* map = previous.maps[previous.map].Get();
                    barrier(cmd, map, read_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    buffer_copy(cmd, map, readback_.Get(), texture_bytes_ * 2, true);
                    barrier(cmd, map, D3D12_RESOURCE_STATE_COPY_SOURCE, read_state);
                    capture_request_["motion_from_frame"] = previous.serial;
                }
                readback_decoded_ = false;
                submitted_ = false;
                ++captures_;
            }
        } catch (const std::exception& exception) {
            if (worker_ && !worker_->alive())
                schedule_restart(tick);
            else {
                failed_ = true;
                error_ = exception.what();
                state_ = "failed_input_preserved";
                invalidate();
            }
            return frame.color;
        }
        if (ready_ >= 0 && eligible(slots_[ready_], frame.serial, tick)) {
            auto& slot = slots_[ready_];
            constants.mode = 2;
            constants.reset = 0;
            barrier(cmd, output_.Get(), read_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            slot.compose->record(cmd,
                                 {current_.Get(), slot.residual.Get(), slot.maps[slot.map].Get()},
                                 output_.Get(), constants, false, slot.map);
            barrier(cmd, output_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, read_state);
            composition_.record(cmd, current_.Get(), output_.Get(), slot.maps[slot.map].Get(),
                                frame.serial, generation_);
            ++presented_;
            state_ = "composing_reprojected_result";
            return output_.Get();
        }
        return frame.color;
    }
};
}
