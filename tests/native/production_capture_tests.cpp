// Exercise real capture allocation, copying and fence retirement on WARP.
// No NVIDIA DLL, model, kernel bundle or physical GPU is required.
#include "../../src/native/support/direct.h"
#include <dxgi1_6.h>
#include <iostream>

using namespace phoenix_addon;
using Microsoft::WRL::ComPtr;

class PassthroughExecutor final : public phoenix::NeuralExecutor {
  public:
    ID3D12Resource* record(const phoenix::ExecutorFrame& frame) override {
        return frame.color;
    }
    Json status() const override {
        return {{"kind", "test_passthrough"}};
    }
    void invalidate() noexcept override {}
};

int main() {
    try {
        ComPtr<IDXGIFactory4> factory;
        phoenix_native::hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter> adapter;
        phoenix_native::hr(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
        ComPtr<ID3D12Device> device;
        phoenix_native::hr(
            D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
        D3D12_COMMAND_QUEUE_DESC queue_description{};
        queue_description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> queue;
        phoenix_native::hr(device->CreateCommandQueue(&queue_description, IID_PPV_ARGS(&queue)));
        phoenix::Settings settings;
        settings.execution_device = "cpu";
        settings.host_unjitter = false;
        uint64_t hd_capture_bytes = 0;
        {
            phoenix::direct::Runner hd(
                device.Get(), queue.Get(), {{"width", 1920}, {"height", 1080}}, {}, {},
                DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, false, 1,
                {{1, 1, 1, 1, 1, 1, 1, 1}}, std::make_unique<PassthroughExecutor>(), &settings);
            check(hd.capture_readback_bytes() == 0, "Idle HD runner allocated capture memory");
            hd.arm_capture("test", 1);
            hd_capture_bytes = hd.capture_readback_bytes();
            check(hd_capture_bytes == 1920ull * 1080 * 20, "Unexpected HD capture footprint");
        }
        constexpr unsigned width = 96, height = 64;
        phoenix::direct::Runner runner(
            device.Get(), queue.Get(), {{"width", width}, {"height", height}}, {}, {},
            DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, false, 1,
            {{1, 1, 1, 1, 1, 1, 1, 1}}, std::make_unique<PassthroughExecutor>(), &settings);
        check(runner.capture_readback_bytes() == 0, "Idle runner allocated capture memory");
        auto color =
            phoenix_native::image(device.Get(), width, height, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                  D3D12_RESOURCE_STATE_COPY_DEST);
        auto motion = phoenix_native::image(device.Get(), width, height, DXGI_FORMAT_R16G16_FLOAT,
                                            D3D12_RESOURCE_STATE_COPY_DEST);
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> command;
        phoenix_native::hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                          IID_PPV_ARGS(&allocator)));
        phoenix_native::hr(device->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&command)));
        std::vector<ComPtr<ID3D12Resource>> uploads;
        auto initialize = [&](ID3D12Resource* texture, unsigned bytes_per_pixel, uint16_t value) {
            const auto description = texture->GetDesc();
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
            UINT64 bytes = 0;
            device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, nullptr, nullptr,
                                          &bytes);
            auto upload = phoenix_native::buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ);
            void* mapped = nullptr;
            phoenix_native::hr(upload->Map(0, nullptr, &mapped));
            std::memset(mapped, 0, size_t(bytes));
            for (unsigned row = 0; row < height; ++row) {
                auto* pixels =
                    reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(mapped) + footprint.Offset +
                                                size_t(row) * footprint.Footprint.RowPitch);
                std::fill(pixels, pixels + width * bytes_per_pixel / 2, value);
            }
            upload->Unmap(0, nullptr);
            D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
            source.pResource = upload.Get();
            source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            source.PlacedFootprint = footprint;
            destination.pResource = texture;
            destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            command->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
            phoenix_native::transition(command.Get(), texture, D3D12_RESOURCE_STATE_COPY_DEST,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            uploads.push_back(std::move(upload));
        };
        initialize(color.Get(), 8, 0x3800); // FP16 0.5, including alpha for this fixture.
        initialize(motion.Get(), 4, 0);
        runner.arm_capture("test", 2);
        const auto active_bytes = runner.capture_readback_bytes();
        check(active_bytes > 0, "Capture request did not allocate readbacks");
        FrameEvidence evidence;
        evidence.queue = queue.Get();
        evidence.color_state = evidence.motion_state =
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        evidence.states_verified = true;
        evidence.restore_bindings = [](void*, ID3D12GraphicsCommandList*) {};
        runner.record(command.Get(), color.Get(), motion.Get(), evidence, {}, settings, 8);
        runner.poll_captures();
        check(runner.capture_readback_bytes() == active_bytes,
              "Capture buffers released before submission retired");
        phoenix_native::hr(command->Close());
        ID3D12CommandList* lists[]{command.Get()};
        queue->ExecuteCommandLists(1, lists);
        check(runner.after_submit(queue.Get(), 1, lists), "Capture submission was not tracked");
        const auto deadline = GetTickCount64() + 10000;
        while (runner.completed() < runner.submitted() && GetTickCount64() < deadline)
            Sleep(1);
        check(runner.completed() == runner.submitted(), "WARP capture fence did not retire");
        runner.poll_captures();
        check(runner.capture_left() == 1 && runner.capture_readback_bytes() == active_bytes,
              "Capture burst lost reusable readbacks before its final frame");
        phoenix::direct::CaptureFrame captured;
        check(runner.take_capture(captured), "Completed capture missing");
        check(captured.color.size() == width * height * 8 &&
                  captured.motion.size() == width * height * 4 &&
                  captured.output.size() == width * height * 8,
              "Capture dimensions changed");
        for (size_t offset = 0; offset < captured.color.size(); offset += 2) {
            uint16_t value{};
            std::memcpy(&value, captured.color.data() + offset, 2);
            check(value == 0x3800, "Captured input bytes changed");
        }
        check(std::all_of(captured.motion.begin(), captured.motion.end(),
                          [](uint8_t byte) { return byte == 0; }),
              "Captured motion changed");
        // A burst reuses the same readbacks, then releases them after its final fence.
        phoenix_native::hr(allocator->Reset());
        phoenix_native::hr(command->Reset(allocator.Get(), nullptr));
        runner.record(command.Get(), color.Get(), motion.Get(), evidence, {}, settings, 8);
        runner.poll_captures();
        check(runner.capture_readback_bytes() == active_bytes,
              "Final burst frame released readbacks before submission");
        phoenix_native::hr(command->Close());
        queue->ExecuteCommandLists(1, lists);
        check(runner.after_submit(queue.Get(), 1, lists), "Final burst submission was not tracked");
        const auto final_deadline = GetTickCount64() + 10000;
        while (runner.completed() < runner.submitted() && GetTickCount64() < final_deadline)
            Sleep(1);
        check(runner.completed() == runner.submitted(), "Final burst fence did not retire");
        runner.poll_captures();
        check(runner.capture_readback_bytes() == 0, "Completed burst retained readbacks");
        check(runner.take_capture(captured) && captured.meta.frame == 1,
              "Final burst snapshot missing or out of order");
        runner.arm_capture("next", 1);
        check(runner.capture_readback_bytes() == active_bytes,
              "Capture buffers could not be recreated for another request");
        // Exercise the real asynchronous display wrapper and optional-plane
        // capture on software D3D12. The executor is deliberately model-free.
        phoenix_native::hr(allocator->Reset());
        phoenix_native::hr(command->Reset(allocator.Get(),nullptr));
        auto mask=phoenix_native::image(device.Get(),width,height,DXGI_FORMAT_R32G32B32A32_FLOAT,D3D12_RESOURCE_STATE_COPY_DEST);
        auto ui=phoenix_native::image(device.Get(),width,height,DXGI_FORMAT_R32G32B32A32_FLOAT,D3D12_RESOURCE_STATE_COPY_DEST);
        auto field=phoenix_native::image(device.Get(),width,height,DXGI_FORMAT_R32G32B32A32_FLOAT,D3D12_RESOURCE_STATE_COPY_DEST);
        auto floats=[&](ID3D12Resource* target,std::array<float,4> value) {
            const auto desc=target->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT64 bytes=0;
            device->GetCopyableFootprints(&desc,0,1,0,&fp,nullptr,nullptr,&bytes);
            auto upload=phoenix_native::buffer(device.Get(),bytes,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
            void* mapped=nullptr;phoenix_native::hr(upload->Map(0,nullptr,&mapped));
            for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)
                std::memcpy(static_cast<uint8_t*>(mapped)+fp.Offset+size_t(y)*fp.Footprint.RowPitch+x*16,value.data(),16);
            upload->Unmap(0,nullptr);
            D3D12_TEXTURE_COPY_LOCATION from{},to{};from.pResource=upload.Get();from.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;from.PlacedFootprint=fp;
            to.pResource=target;to.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            command->CopyTextureRegion(&to,0,0,0,&from,nullptr);
            phoenix_native::transition(command.Get(),target,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            uploads.push_back(std::move(upload));
        };
        floats(mask.Get(),{0,1,1,1});floats(ui.Get(),{.25f,.25f,.25f,1});floats(field.Get(),{0,0,0,0});
        phoenix::direct::AppearanceTextures developer;developer.mask=mask.Get();developer.ui=ui.Get();developer.field=field.Get();
        {phoenix::direct::AppearanceStage epoch(device.Get(),width,height);
            check(epoch.changed(developer,17),"New map/field epoch not detected");
            epoch.remember_mode(developer,17);
            check(!epoch.changed(developer,17),"Stable producer epoch unexpectedly resets history");
            check(epoch.changed(developer,18),"Changed producer epoch did not request reset");}
        evidence.experimental_appearance_contract=true;
        for(const char* key:{"DLSSNR.ControlMask","DLSSNR.UI","DLSSNR.BidirectionalDistortionField"})
            evidence.optional_states[key]=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        runner.record(command.Get(),color.Get(),motion.Get(),evidence,{},settings,8,nullptr,
                      D3D12_RESOURCE_STATE_COMMON,&developer,17);
        phoenix_native::hr(command->Close());queue->ExecuteCommandLists(1,lists);
        check(runner.after_submit(queue.Get(),1,lists),"Advanced capture submission not tracked");
        const auto advanced_deadline=GetTickCount64()+10000;
        while(runner.completed()<runner.submitted()&&GetTickCount64()<advanced_deadline)Sleep(1);
        check(runner.completed()==runner.submitted(),"Advanced capture fence did not retire");
        runner.poll_captures();check(runner.take_capture(captured),"Advanced capture missing");
        check(captured.meta.appearance_flags==19&&captured.meta.appearance_generation==17,"Advanced capture identity lost");
        const std::array<float,4> expected_map{0,1,1,1},expected_ui{.25f,.25f,.25f,1};
        for(unsigned p=0;p<width*height;++p) {
            check(std::memcmp(captured.appearance[0].data()+p*16,expected_map.data(),16)==0,"Map snapshot changed");
            check(std::memcmp(captured.appearance[1].data()+p*16,expected_ui.data(),16)==0,"UI snapshot changed");
        }
        check(captured.appearance[4].size()==width*height*16&&
            std::all_of(captured.appearance[4].begin(),captured.appearance[4].end(),[](uint8_t b){return b==0;}),"Field snapshot changed");
        check(runner.capture_readback_bytes()==0,"Advanced capture retained buffers after retirement");
        std::cout << Json({{"passed", true},
                           {"software_d3d12", "WARP"},
                           {"model_inference_executed", false},
                           {"idle_capture_readback_bytes", 0},
                           {"hd_capture_readback_bytes", hd_capture_bytes},
                           {"retired_capture_readback_bytes", 0},
                           {"capture_input_bytes_exact", true},
                           {"burst_readbacks_reused", true},
                           {"optional_snapshots_exact",true},{"producer_epochs_checked",true}})
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
