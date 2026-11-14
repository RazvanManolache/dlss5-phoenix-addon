#pragma once
#include <array>
#include <atomic>
#include <thread>

// One copy queue serves a bounded ring of color and game-depth textures. A slot stays
// immutable until its consumer retires and the client receives its frame ACK.
class PhoenixCpuPrefetch {
    template<class T> using Ptr = Microsoft::WRL::ComPtr<T>;
    struct Slot {
        Ptr<ID3D12Resource> texture, upload, depth_texture, depth_upload;
        void* mapped = nullptr;
        void* depth_mapped = nullptr;
        unsigned depth_width = 0, depth_height = 0;
        bool depth_uploaded = false;
        std::atomic<uint64_t> ready{0};
        phoenix_compat::DepthRange depth{};
        double upload_ms = 0, depth_ms = 0;
    };
    std::array<Slot, phoenix_compat::CpuFrameMapping::max_slots()> slots_{};
    Ptr<ID3D12CommandQueue> queue_;
    Ptr<ID3D12CommandAllocator> allocator_;
    Ptr<ID3D12GraphicsCommandList> list_;
    Ptr<ID3D12Fence> fence_;
    HANDLE event_ = nullptr, wake_ = nullptr, ready_event_ = nullptr;
    std::atomic<bool> stop_{false}, failed_{false};
    std::thread worker_;

public:
    ~PhoenixCpuPrefetch() {
        stop_ = true;
        if (wake_) SetEvent(wake_);
        if (worker_.joinable()) worker_.join();
        if (failed_) {
            // A device failure or hung copy may leave a submission in flight.
            // Keep its objects alive until the helper exits.
            queue_.Detach(); allocator_.Detach(); list_.Detach(); fence_.Detach();
            for (auto& slot : slots_) {
                slot.texture.Detach(); slot.upload.Detach();
                slot.depth_texture.Detach(); slot.depth_upload.Detach();
            }
        }
        if (event_) CloseHandle(event_);
        if (wake_) CloseHandle(wake_);
        if (ready_event_) CloseHandle(ready_event_);
    }
    bool start(ID3D12Device* device, const phoenix_compat::CpuFrameMapping& mapping,
               unsigned width, unsigned height, bool depth_inverted, HANDLE wake) {
        wake_ = wake;
        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue_))) ||
            FAILED(device->CreateCommandAllocator(desc.Type, IID_PPV_ARGS(&allocator_))) ||
            FAILED(device->CreateCommandList(0, desc.Type, allocator_.Get(), nullptr, IID_PPV_ARGS(&list_))) ||
            FAILED(list_->Close()) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)))) return false;
        event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        ready_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event_ || !ready_event_) return false;
        const unsigned row = width * 4, pitch = (row + 255) & ~255u;
        try {
            for (unsigned i = 0; i < mapping.slots(); ++i) {
                auto& slot = slots_[i];
                slot.texture = phoenix_native::image(device, width, height, DXGI_FORMAT_R8G8B8A8_UNORM,
                                                      D3D12_RESOURCE_STATE_COMMON, false);
                slot.upload = phoenix_native::buffer(device, uint64_t(pitch) * height, D3D12_HEAP_TYPE_UPLOAD,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ);
                const D3D12_RANGE empty{};
                if (FAILED(slot.upload->Map(0, &empty, &slot.mapped))) return false;
            }
            worker_ = std::thread([this, &mapping, width, height, row, pitch, depth_inverted,
                                   device = Ptr<ID3D12Device>(device)] {
                uint64_t serial = 0;
                while (!stop_) {
                    bool progressed = false;
                    for (unsigned i = 0; i < mapping.slots() && !stop_; ++i) {
                        auto& slot = slots_[i];
                        const auto frame = mapping.ready_frame(i);
                        if (!frame || frame == slot.ready.load(std::memory_order_relaxed)) continue;
                        const auto started = std::chrono::steady_clock::now();
                        auto* dst = static_cast<uint8_t*>(slot.mapped);
                        const auto* src = mapping.input(i).data();
                        if (row == pitch) std::memcpy(dst, src, size_t(row) * height);
                        else for (unsigned y = 0; y < height; ++y)
                            std::memcpy(dst + size_t(y) * pitch, src + size_t(y) * row, row);

                        if (FAILED(allocator_->Reset()) || FAILED(list_->Reset(allocator_.Get(), nullptr))) {
                            failed_ = true;
                            return;
                        }
                        D3D12_TEXTURE_COPY_LOCATION from{};
                        from.pResource = slot.upload.Get();
                        from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        from.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, pitch};
                        D3D12_TEXTURE_COPY_LOCATION to{slot.texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
                        // COMMON promotes to COPY_DEST and decays on completion
                        // of this copy-queue submission before DIRECT reads it.
                        list_->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
                        std::span<const float> depth;
                        slot.depth_uploaded = false;
                        if (mapping.depth_ready(i)) {
                            const auto dims = mapping.depth_dims(i);
                            const unsigned w = dims ? feed_depth_width(dims) : width;
                            const unsigned h = dims ? feed_depth_height(dims) : height;
                            if (w && h && w <= width && h <= height) {
                                depth = mapping.depth(i).first(size_t(w) * h);
                                slot.depth_uploaded = record_depth(device.Get(), slot, depth, w, h, depth_inverted);
                            }
                        }
                        if (FAILED(list_->Close())) { failed_ = true; return; }
                        ID3D12CommandList* lists[]{list_.Get()};
                        queue_->ExecuteCommandLists(1, lists);
                        if (FAILED(queue_->Signal(fence_.Get(), ++serial)) ||
                            FAILED(fence_->SetEventOnCompletion(serial, event_))) { failed_ = true; return; }

                        const auto depth_started = std::chrono::steady_clock::now();
                        slot.depth = depth.empty() ? phoenix_compat::DepthRange{} : phoenix_compat::depth_range_stats(depth);
                        slot.depth_ms = milliseconds(depth_started);
                        if (WaitForSingleObject(event_, 4000) != WAIT_OBJECT_0 ||
                            fence_->GetCompletedValue() == UINT64_MAX) { failed_ = true; return; }
                        slot.upload_ms = milliseconds(started);
                        slot.ready.store(frame, std::memory_order_release);
                        SetEvent(ready_event_);
                        progressed = true;
                    }
                    if (!progressed) WaitForSingleObject(wake_, INFINITE);
                }
            });
        } catch (...) { return false; }
        return true;
    }

    struct Frame {
        ID3D12Resource* texture = nullptr;
        ID3D12Resource* depth_texture = nullptr;
        phoenix_compat::DepthRange depth{};
        double upload_ms = 0, depth_ms = 0, wait_ms = 0;
        bool ready_at_arrival = false;
    };
    Frame take(unsigned slot, uint64_t frame) const {
        const auto started = std::chrono::steady_clock::now();
        const auto& source = slots_[slot];
        const bool ready = source.ready.load(std::memory_order_acquire) == frame;
        const auto deadline = GetTickCount64() + 4000;
        while (!failed_ && source.ready.load(std::memory_order_acquire) != frame && GetTickCount64() < deadline)
            WaitForSingleObject(ready_event_, 20);
        if (failed_ || source.ready.load(std::memory_order_acquire) != frame) return {};
        return {source.texture.Get(), source.depth_uploaded ? source.depth_texture.Get() : nullptr,
                source.depth, source.upload_ms, source.depth_ms, milliseconds(started), ready};
    }
private:
    bool record_depth(ID3D12Device* device, Slot& slot, std::span<const float> depth,
                      unsigned width, unsigned height, bool invert) {
        const unsigned row = width * sizeof(float), pitch = (row + 255) & ~255u;
        if (width != slot.depth_width || height != slot.depth_height) {
            try {
                auto texture = phoenix_native::image(device, width, height, DXGI_FORMAT_R32_FLOAT,
                                                     D3D12_RESOURCE_STATE_COMMON, false);
                auto upload = phoenix_native::buffer(device, uint64_t(pitch) * height, D3D12_HEAP_TYPE_UPLOAD,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ);
                void* mapped = nullptr;
                const D3D12_RANGE empty{};
                if (FAILED(upload->Map(0, &empty, &mapped))) return false;
                slot.depth_texture = std::move(texture);
                slot.depth_upload = std::move(upload);
                slot.depth_mapped = mapped;
                slot.depth_width = width;
                slot.depth_height = height;
            } catch (...) { return false; }
        }
        auto* mapped = static_cast<uint8_t*>(slot.depth_mapped);
        for (unsigned y = 0; y < height; ++y) {
            auto* dst = reinterpret_cast<float*>(mapped + size_t(y) * pitch);
            const auto* src = depth.data() + size_t(y) * width;
            if (invert) for (unsigned x = 0; x < width; ++x) dst[x] = 1.f - src[x];
            else std::memcpy(dst, src, row);
        }
        D3D12_TEXTURE_COPY_LOCATION from{};
        from.pResource = slot.depth_upload.Get();
        from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        from.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_FLOAT, width, height, 1, pitch};
        D3D12_TEXTURE_COPY_LOCATION to{slot.depth_texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
        list_->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        return true;
    }
    static double milliseconds(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};
