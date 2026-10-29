#include "d3d12_test_context.h"
#include "../../src/native/support/present_guides.h"
#include "../../src/native/support/ledger.h"
#include "../../src/integration/compatibility/completed_presentation.h"
#include <iostream>

int main() {
    try {
        phoenix_test::Gpu gpu;
        auto completed = gpu.texture(17, 9), live = gpu.texture(17, 9);
        // The next raw game image is black, while the completed image contains
        // the fixture gradient. Exercise the actual GPU copy, not a mock sink.
        void* mapped = nullptr; D3D12_RANGE empty{};
        auto& upload = gpu.uploads.back();
        phoenix_native::hr(upload->Map(0, &empty, &mapped));
        memset(mapped, 0, size_t(upload->GetDesc().Width));
        upload->Unmap(0, nullptr);
        auto expected = gpu.readback(completed.Get());
        auto raw = gpu.readback(live.Get());
        phoenix_native::transition(gpu.cmd.Get(), completed.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        phoenix_native::transition(gpu.cmd.Get(), live.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
        phoenix_compat::CompletedPresentation presentation;
        phoenix_bundle::require(!presentation.record_hold(gpu.cmd.Get(), completed.Get(), live.Get(), true), "Cold presentation copied uninitialized output");
        presentation.completed();
        phoenix_bundle::require(!presentation.record_hold(gpu.cmd.Get(), completed.Get(), live.Get(), false), "Noncoherent mode unexpectedly froze the game");
        phoenix_bundle::require(presentation.record_hold(gpu.cmd.Get(), completed.Get(), live.Get(), true), "Completed frame not held during deferral");
        phoenix_native::transition(gpu.cmd.Get(), live.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        auto held = gpu.readback(live.Get());
        presentation.invalidate();
        phoenix_bundle::require(!presentation.record_hold(gpu.cmd.Get(), completed.Get(), live.Get(), true), "Previous generation leaked after model switch");
        gpu.submit();
        auto pixels = [](auto& readback) {
            std::vector<uint8_t> result(17 * 9 * 8); void* data = nullptr;
            D3D12_RANGE read{0, SIZE_T(readback.first->GetDesc().Width)};
            phoenix_native::hr(readback.first->Map(0, &read, &data));
            for (unsigned row = 0; row < 9; ++row)
                memcpy(result.data() + row * 17 * 8, static_cast<uint8_t*>(data) + row * readback.second.Footprint.RowPitch, 17 * 8);
            D3D12_RANGE no_write{}; readback.first->Unmap(0, &no_write); return result;
        };
        const auto wanted = pixels(expected), original = pixels(raw), actual = pixels(held);
        phoenix_bundle::require(wanted != original && actual == wanted, "A raw frame escaped the completed presentation hold");
        gpu.validate();
        gpu.next();
        // No successful output exists during startup or after invalidation.
        // Populate a real render target with game pixels, then prove that the
        // loading gate hides them rather than presenting them accidentally.
        Microsoft::WRL::ComPtr<ID3D12Resource> startup;
        auto description = completed->GetDesc();
        description.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        phoenix_native::hr(gpu.device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&description,
            D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&startup)));
        phoenix_compat::CompletedPresentation::record_loading(gpu.cmd.Get(),startup.Get());
        phoenix_native::transition(gpu.cmd.Get(),startup.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        auto loading = gpu.readback(startup.Get());
        gpu.submit(); gpu.validate();
        const auto blank = pixels(loading);
        for (size_t pixel=0;pixel<17*9;++pixel) {
            const uint8_t black_half[8]={0,0,0,0,0,0,0,0x3c};
            phoenix_bundle::require(memcmp(blank.data()+pixel*8,black_half,8)==0,
                "Cold-start loading gate exposed unprocessed pixels");
        }
        gpu.next();
        auto motion = gpu.texture(17, 9, true);
        auto depth = phoenix_native::image(gpu.device.Get(), 17, 9, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_COMMON);
        phoenix_addon::phoenix::direct::PresentGuides guides(gpu.device.Get(), gpu.queue.Get());
        phoenix_bundle::require(guides.record(gpu.cmd.Get(), motion.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            depth.Get(), D3D12_RESOURCE_STATE_COMMON, 17, 9, 42, 17.f, 9.f, true), "Guide snapshot refused");
        ID3D12CommandList* recorded[]{gpu.cmd.Get()};
        gpu.submit(); guides.submitted(gpu.queue.Get(), 1, recorded);
        gpu.next();
        struct Frame { ID3D12Resource *output, *motion=nullptr, *depth=nullptr; float motion_scale_x=0, motion_scale_y=0; unsigned depth_inverted=0; } frame{completed.Get()};
        phoenix_bundle::require(!guides.acquire(gpu.cmd.Get(), gpu.queue.Get(), 41, frame), "Guide snapshot used before its frame");
        phoenix_bundle::require(!guides.acquire(gpu.cmd.Get(), gpu.queue.Get(), 43, frame), "Stale guide snapshot reused on the next frame");
        phoenix_bundle::require(guides.acquire(gpu.cmd.Get(), gpu.queue.Get(), 42, frame), "Same-frame guide snapshot unavailable");
        phoenix_bundle::require(frame.motion != motion.Get() && frame.depth != depth.Get() && frame.motion_scale_x == 17.f && frame.motion_scale_y == 9.f && frame.depth_inverted == 1,
            "Guide ownership, units or depth convention changed");
        gpu.submit(); gpu.validate();
        gpu.next();
        phoenix_bundle::require(guides.record(gpu.cmd.Get(), motion.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            depth.Get(), D3D12_RESOURCE_STATE_COMMON, 17, 9, 43, 17.f, 9.f, true, true), "Pending guide snapshot refused");
        ID3D12CommandList* unconfirmed[]{gpu.cmd.Get()};
        gpu.submit(); guides.submitted(gpu.queue.Get(), 1, unconfirmed);
        gpu.next();
        phoenix_bundle::require(!guides.acquire(gpu.cmd.Get(), gpu.queue.Get(), 43, frame),
            "Failed/unconfirmed game evaluation admitted guide history");
        phoenix_bundle::require(guides.record(gpu.cmd.Get(), motion.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            depth.Get(), D3D12_RESOURCE_STATE_COMMON, 17, 9, 44, 17.f, 9.f, true, true), "Confirmation snapshot refused");
        phoenix_bundle::require(!guides.confirm(gpu.cmd.Get(),43) && guides.confirm(gpu.cmd.Get(),44),
            "Guide confirmation accepted a different frame");
        ID3D12CommandList* confirmed[]{gpu.cmd.Get()};
        gpu.submit(); guides.submitted(gpu.queue.Get(),1,confirmed);
        gpu.next();
        phoenix_bundle::require(guides.acquire(gpu.cmd.Get(),gpu.queue.Get(),44,frame),
            "Successful provider evaluation did not admit its guides");
        gpu.submit(); gpu.validate();
        gpu.next();
        auto engine_motion=gpu.texture(17,9,true);
        auto& motion_upload=gpu.uploads.back();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        const auto motion_description=engine_motion->GetDesc();
        gpu.device->GetCopyableFootprints(&motion_description,0,1,0,&footprint,nullptr,nullptr,nullptr);
        phoenix_native::hr(motion_upload->Map(0,&empty,&mapped));
        for(unsigned y=0;y<9;++y)for(unsigned x=0;x<17;++x){
            auto* values=reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(mapped)+y*footprint.Footprint.RowPitch)+x*2;
            values[0]=DirectX::PackedVector::XMConvertFloatToHalf(1.25f);
            values[1]=DirectX::PackedVector::XMConvertFloatToHalf(-.5f);
        }
        motion_upload->Unmap(0,nullptr);
        phoenix_native::transition(gpu.cmd.Get(),depth.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        const uint64_t prefix=uint64_t(gpu.cmd.Get()), queue_identity=uint64_t(gpu.queue.Get());
        phoenix_addon::ledger::Ledger ordering;
        ordering.transition(prefix,uint64_t(engine_motion.Get()),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        ordering.transition(prefix,uint64_t(depth.Get()),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.submit(); ordering.submitted(prefix,queue_identity);
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> provider_allocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> provider;
        phoenix_native::hr(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&provider_allocator)));
        phoenix_native::hr(gpu.device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,provider_allocator.Get(),nullptr,IID_PPV_ARGS(&provider)));
        phoenix_native::hr(provider->Close());
        ordering.transition(0xfeed,uint64_t(engine_motion.Get()),D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        phoenix_bundle::require(!ordering.state(uint64_t(engine_motion.Get()),uint64_t(provider.Get()),queue_identity).known,
            "Pending future work should block a recording-time state answer");
        phoenix_bundle::require(guides.stage(provider.Get(),engine_motion.Get(),depth.Get(),17,9,45,1,1,false)&&
            !guides.pending_before(provider.Get())&&guides.confirm(provider.Get(),45),"Deferred guide confirmation failed");
        const auto boundary=[&](ID3D12Resource* resource){return ordering.at_submission_boundary(uint64_t(resource),queue_identity);};
        const auto execute=[](ID3D12CommandQueue* queue,unsigned count,ID3D12CommandList* const* commands){queue->ExecuteCommandLists(count,commands);};
        phoenix_bundle::require(guides.capture_before(gpu.queue.Get(),provider.Get(),boundary,execute),
            "Ordered prefix did not make pending game guides available");
        ID3D12CommandList* provider_lists[]{provider.Get()};
        execute(gpu.queue.Get(),1,provider_lists);
        gpu.next();
        // Future game work overwrites its source. The owned snapshot must
        // still contain the values captured before the provider was queued.
        phoenix_native::hr(motion_upload->Map(0,&empty,&mapped));
        memset(mapped,0,size_t(motion_upload->GetDesc().Width));motion_upload->Unmap(0,nullptr);
        phoenix_native::transition(gpu.cmd.Get(),engine_motion.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION source{motion_upload.Get(),D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,footprint},target{engine_motion.Get(),D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,0};
        gpu.cmd->CopyTextureRegion(&target,0,0,0,&source,nullptr);
        phoenix_native::transition(gpu.cmd.Get(),engine_motion.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_bundle::require(guides.acquire(gpu.cmd.Get(),gpu.queue.Get(),45,frame),"Submission snapshot was not acquired");
        phoenix_native::transition(gpu.cmd.Get(),frame.motion,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        auto snapshot_pixels=gpu.readback(frame.motion);
        auto overwritten=gpu.readback(engine_motion.Get());
        phoenix_native::transition(gpu.cmd.Get(),frame.motion,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
        gpu.submit();gpu.validate();
        for(auto* image:{&snapshot_pixels,&overwritten}){
            void* data=nullptr;D3D12_RANGE read{0,SIZE_T(image->first->GetDesc().Width)};
            phoenix_native::hr(image->first->Map(0,&read,&data));
            for(unsigned y=0;y<9;++y)for(unsigned x=0;x<17;++x){
                const auto* values=reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(data)+y*image->second.Footprint.RowPitch)+x*2;
                const bool saved=image==&snapshot_pixels;
                phoenix_bundle::require(values[0]==(saved?DirectX::PackedVector::XMConvertFloatToHalf(1.25f):0)&&
                    values[1]==(saved?DirectX::PackedVector::XMConvertFloatToHalf(-.5f):0),"Guide snapshot read future game pixels");
            }
            image->first->Unmap(0,&empty);
        }
        phoenix_bundle::require(guides.stage(provider.Get(),engine_motion.Get(),depth.Get(),17,9,46,1,1,false)&&guides.confirm(provider.Get(),46),"Rejection setup failed");
        const auto writable=[](ID3D12Resource*){return phoenix_addon::ledger::Ledger::Answer{true,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,phoenix_addon::ledger::Basis::submitted_transition};};
        phoenix_bundle::require(!guides.capture_before(gpu.queue.Get(),provider.Get(),writable,execute),"Writable list-entry guides were accepted");
        phoenix_bundle::require(guides.stage(provider.Get(),engine_motion.Get(),depth.Get(),17,9,47,1,1,false),"Cancel setup failed");
        guides.abandoned(provider.Get());
        phoenix_bundle::require(!guides.pending_before(provider.Get())&&guides.status().at("pending_submission")==0,"Abandoned request kept borrowed resources");
        phoenix_bundle::require(guides.stage(provider.Get(),engine_motion.Get(),depth.Get(),17,9,48,1,1,false),"Expiry setup failed");
        guides.expire_before(49);
        phoenix_bundle::require(guides.status().at("pending_submission")==0,"Old frame request kept engine resources alive");
        gpu.next();
        const auto motion_ready=[&](ID3D12Resource* resource){return resource==engine_motion.Get()?boundary(resource):phoenix_addon::ledger::Ledger::Answer{};};
        phoenix_bundle::require(guides.stage(provider.Get(),engine_motion.Get(),depth.Get(),17,9,50,1,1,false,true)&&guides.confirm(provider.Get(),50),"Motion-only setup failed");
        phoenix_bundle::require(guides.capture_before(gpu.queue.Get(),provider.Get(),motion_ready,execute),"Native motion-only consumer was blocked by missing depth");
        phoenix_bundle::require(guides.acquire(gpu.cmd.Get(),gpu.queue.Get(),50,frame)&&frame.motion&&frame.depth==nullptr,
            "Motion-only capture exposed missing or stale depth");
        gpu.submit();gpu.validate();
        phoenix_bundle::require(guides.stage(provider.Get(),engine_motion.Get(),depth.Get(),17,9,51,1,1,false,false)&&guides.confirm(provider.Get(),51),"Full-guide consumer setup failed");
        phoenix_bundle::require(!guides.capture_before(gpu.queue.Get(),provider.Get(),motion_ready,execute),"Depth-requiring consumer accepted motion-only guides");
        auto stencil=phoenix_native::image(gpu.device.Get(),17,9,DXGI_FORMAT_R32G8X24_TYPELESS,D3D12_RESOURCE_STATE_COMMON,false);
        const auto all_read=[](ID3D12Resource*){return phoenix_addon::ledger::Ledger::Answer{true,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,phoenix_addon::ledger::Basis::submitted_transition};};
        phoenix_bundle::require(guides.stage(provider.Get(),engine_motion.Get(),stencil.Get(),17,9,52,1,1,false,true)&&guides.confirm(provider.Get(),52),"Stencil fallback setup failed");
        phoenix_bundle::require(guides.capture_before(gpu.queue.Get(),provider.Get(),all_read,execute),"Unsupported stencil depth blocked native motion capture");
        gpu.next();
        phoenix_bundle::require(!guides.acquire(gpu.cmd.Get(),gpu.queue.Get(),52,frame,true),
            "A newly selected depth-requiring consumer borrowed an earlier motion-only result");
        phoenix_bundle::require(guides.acquire(gpu.cmd.Get(),gpu.queue.Get(),52,frame,false,true)&&frame.depth==nullptr,
            "Stencil data was incorrectly supplied as scalar depth");
        auto* borrowed=frame.motion;
        phoenix_bundle::require(guides.stage(provider.Get(),engine_motion.Get(),depth.Get(),17,9,53,1,1,false,true)&&guides.confirm(provider.Get(),53),"Concurrent capture setup failed");
        phoenix_bundle::require(guides.capture_before(gpu.queue.Get(),provider.Get(),motion_ready,execute)&&
            guides.acquire(gpu.cmd.Get(),gpu.queue.Get(),53,frame)&&frame.motion!=borrowed,
            "An unsubmitted consumer's borrowed guide texture was recycled");
        guides.expire_before(54);
        phoenix_bundle::require(guides.status().at("active_readers")==1,"Frame expiry released an active GPU reader");
        gpu.submit();gpu.validate();
        guides.release(borrowed);
        phoenix_bundle::require(guides.status().at("active_readers")==0&&guides.status().at("motion_only_captures")==3,
            "Borrowed guide release or motion-only diagnostics are incorrect");
        std::cout << "{\"passed\":true,\"software_d3d12\":true,\"raw_frame_replaced\":true,\"loading_hides_raw_pixels\":true,\"cold_and_generation_invalidations\":true,\"same_frame_guide_lifetime\":true,\"submission_prefix_snapshot\":true,\"future_source_writes_isolated\":true,\"writable_entry_refused\":true,\"motion_without_depth\":true,\"required_depth_refused\":true}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
