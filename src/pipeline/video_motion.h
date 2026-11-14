#pragma once
#include <d3d12video.h>
#include "../native/support/compute_pass.h"

namespace phoenix_pipeline {
class VideoMotion {
    template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
    ComPtr<ID3D12VideoMotionEstimator> estimator_;
    ComPtr<ID3D12VideoMotionVectorHeap> heap_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12VideoEncodeCommandList> list_;
    ComPtr<ID3D12Fence> completion_;
    ComPtr<ID3D12Resource> frames_[2], vectors_, transfer_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT luma_{};
    unsigned width_, height_, block_, current_ = 0;
    uint64_t serial_ = 0;
    bool primed_ = false;
public:
    VideoMotion(ID3D12Device* device, unsigned width, unsigned height)
        : width_(width), height_(height) {
        using phoenix_native::hr;
        ComPtr<ID3D12VideoDevice1> video;
        hr(device->QueryInterface(IID_PPV_ARGS(&video)));
        D3D12_FEATURE_DATA_VIDEO_MOTION_ESTIMATOR caps{};
        caps.InputFormat = DXGI_FORMAT_NV12;
        hr(video->CheckFeatureSupport(D3D12_FEATURE_VIDEO_MOTION_ESTIMATOR, &caps, sizeof(caps)));
        if (!(caps.PrecisionFlags & D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_FLAG_QUARTER_PEL) ||
            !(caps.BlockSizeFlags & (D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_FLAG_8X8 |
                                     D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_FLAG_16X16)) ||
            width < caps.SizeRange.MinWidth || height < caps.SizeRange.MinHeight ||
            width > caps.SizeRange.MaxWidth || height > caps.SizeRange.MaxHeight)
            throw std::runtime_error("D3D12 video motion unavailable at the selected analysis size");
        const bool use_8x8 = (caps.BlockSizeFlags & D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_FLAG_8X8) != 0;
        block_ = use_8x8 ? 8 : 16;
        const auto size = use_8x8 ? D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_8X8 :
                                  D3D12_VIDEO_MOTION_ESTIMATOR_SEARCH_BLOCK_SIZE_16X16;
        const auto precision = D3D12_VIDEO_MOTION_ESTIMATOR_VECTOR_PRECISION_QUARTER_PEL;
        // The driver may allocate for the entire declared range.
        const D3D12_VIDEO_SIZE_RANGE range{width, height, width, height};
        const D3D12_VIDEO_MOTION_ESTIMATOR_DESC desc{0, DXGI_FORMAT_NV12, size, precision, range};
        const D3D12_VIDEO_MOTION_VECTOR_HEAP_DESC heap{0, DXGI_FORMAT_NV12, size, precision, range};
        hr(video->CreateVideoMotionEstimator(&desc, nullptr, IID_PPV_ARGS(&estimator_)));
        hr(video->CreateVideoMotionVectorHeap(&heap, nullptr, IID_PPV_ARGS(&heap_)));
        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE;
        hr(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue_)));
        hr(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&allocator_)));
        hr(device->CreateCommandList(0, q.Type, allocator_.Get(), nullptr, IID_PPV_ARGS(&list_)));
        hr(list_->Close());
        hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completion_)));
        for (auto& frame : frames_)
            frame = phoenix_native::image(device, width, height, DXGI_FORMAT_NV12, D3D12_RESOURCE_STATE_COMMON, false);
        vectors_ = phoenix_native::image(device, (width + block_ - 1) / block_, (height + block_ - 1) / block_,
            DXGI_FORMAT_R16G16_SINT, D3D12_RESOURCE_STATE_COMMON, false);
        const auto shape = frames_[0]->GetDesc();
        UINT64 bytes = 0;
        device->GetCopyableFootprints(&shape, 0, 1, 0, &luma_, nullptr, nullptr, &bytes);
        transfer_ = phoenix_native::buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
    }

    void record_luma(ID3D12GraphicsCommandList* list, ID3D12Resource* gray) {
        // The motion API searches the Y plane only. Use a buffer to copy
        // R8 into plane zero without an RGB-to-NV12 video-process pass.
        auto luma = luma_;
        luma.Footprint.Format = DXGI_FORMAT_R8_UNORM;
        const D3D12_TEXTURE_COPY_LOCATION from{gray, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
        const D3D12_TEXTURE_COPY_LOCATION transfer{transfer_.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, luma};
        list->CopyTextureRegion(&transfer, 0, 0, 0, &from, nullptr);
        phoenix_native::transition(list, transfer_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto* frame = frames_[current_].Get();
        phoenix_native::transition(list, frame, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        const D3D12_TEXTURE_COPY_LOCATION to{frame, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
        list->CopyTextureRegion(&to, 0, 0, 0, &transfer, nullptr);
        phoenix_native::transition(list, frame, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        phoenix_native::transition(list, transfer_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    }

    bool submit(ID3D12CommandQueue* direct, ID3D12Fence* prepared, uint64_t value, bool reset) {
        auto* current = frames_[current_].Get();
        auto* previous = frames_[current_ ^ 1u].Get();
        current_ ^= 1u;
        if (!primed_ || reset) {
            primed_ = true;
            return false;
        }
        using phoenix_native::hr;
        hr(allocator_->Reset());
        hr(list_->Reset(allocator_.Get()));
        auto barrier = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES state, bool enter) {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                enter ? D3D12_RESOURCE_STATE_COMMON : state, enter ? state : D3D12_RESOURCE_STATE_COMMON};
            list_->ResourceBarrier(1, &b);
        };
        for (auto* input : {current, previous}) barrier(input, D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ, true);
        barrier(vectors_.Get(), D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE, true);
        const D3D12_VIDEO_MOTION_ESTIMATOR_OUTPUT out{heap_.Get()};
        const D3D12_VIDEO_MOTION_ESTIMATOR_INPUT in{current, 0, previous, 0, nullptr};
        list_->EstimateMotion(estimator_.Get(), &out, &in);
        const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_OUTPUT destination{vectors_.Get(), {}};
        const D3D12_RESOLVE_VIDEO_MOTION_VECTOR_HEAP_INPUT source{heap_.Get(), width_, height_};
        list_->ResolveMotionVectorHeap(&destination, &source);
        for (auto* input : {current, previous}) barrier(input, D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ, false);
        barrier(vectors_.Get(), D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE, false);
        hr(list_->Close());
        hr(queue_->Wait(prepared, value));
        ID3D12CommandList* lists[]{list_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        hr(queue_->Signal(completion_.Get(), ++serial_));
        hr(direct->Wait(completion_.Get(), serial_));
        return true;
    }

    ID3D12Resource* vectors() const { return vectors_.Get(); }
    unsigned block() const { return block_; }
};
}
