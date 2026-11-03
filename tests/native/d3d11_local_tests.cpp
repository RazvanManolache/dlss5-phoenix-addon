// Software-only interop and native evaluator plumbing. The evaluator is a
// synthetic clear, not model inference: no NVIDIA driver/model is exercised.
#include "d3d12_test_context.h"
#include "../../src/integration/compatibility/d3d11_local_surface.h"
#include "../../src/integration/compatibility/d3d12_bridge.h"
#include "../../src/integration/compatibility/d3d12_local_pixels.h"
#include "../../src/integration/compatibility/host_client.h"
#include "../../src/integration/compatibility/d3d11_bridge.h"
#include <iostream>
#include <source_location>

using namespace phoenix_compat;
using Microsoft::WRL::ComPtr;
template<class T> concept HasRuntime = requires(T& value) { value.runtime; };
static_assert(!HasRuntime<LocalGpuOptions>, "Detached options must not borrow an effect runtime");
static uint64_t completed = 0;
static int behavior = 0, mode = 2;
static std::atomic<bool> block_evaluation = false, evaluation_entered = false, resume_evaluation = false;
static ComPtr<ID3D12DescriptorHeap> cpu_heap, gpu_heap;
static BOOL WINAPI action(uint32_t, uint32_t) { return TRUE; }
static UINT WINAPI metadata(char* out, UINT capacity) {
    const auto text = Json{{"schema","production-game-phoenix/2"},{"input_provider","feeder_d3d12"},
        {"terminal",false},{"completed_frames",completed},{"phoenix_routing",{{"frames",completed},{"active_mode",mode}}},
        {"direct_game",{{"enabled",true},{"status",mode == 1 ?
            "Native NR output (resolved) written back to the game DLSS output" :
            "Own network output written back to the game DLSS output"}}},
        {"worker_execution",{{"kind","native"}}}}.dump();
    if (out && capacity > text.size()) memcpy(out, text.c_str(), text.size()+1);
    return UINT(text.size()+1);
}
static HRESULT WINAPI evaluate(const PhoenixFeederFrameV1* frame, PhoenixFeederResultV1* result) {
    if (block_evaluation) {
        evaluation_entered = true;
        const auto deadline = GetTickCount64() + 5000;
        while (!resume_evaluation && GetTickCount64() < deadline) Sleep(1);
        if (!resume_evaluation) return E_ABORT;
    }
    if (behavior == 1) { result->status = PhoenixFeederStatusV1::Deferred; return S_FALSE; }
    if (behavior == 2) { result->status = PhoenixFeederStatusV1::Bypass; return S_OK; }
    ComPtr<ID3D12Device> device; phoenix_native::hr(frame->queue->GetDevice(IID_PPV_ARGS(&device)));
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{}; view.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(frame->output, nullptr, &view, cpu_heap->GetCPUDescriptorHandleForHeapStart());
    device->CopyDescriptorsSimple(1, gpu_heap->GetCPUDescriptorHandleForHeapStart(),
        cpu_heap->GetCPUDescriptorHandleForHeapStart(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    ID3D12DescriptorHeap* heaps[]{gpu_heap.Get()}; frame->command_list->SetDescriptorHeaps(1, heaps);
    phoenix_native::transition(frame->command_list, frame->output, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const float red[]{1,0,0,1};
    frame->command_list->ClearUnorderedAccessViewFloat(gpu_heap->GetGPUDescriptorHandleForHeapStart(),
        cpu_heap->GetCPUDescriptorHandleForHeapStart(), frame->output, red, 0, nullptr);
    phoenix_native::transition(frame->command_list, frame->output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    ++completed; result->status = PhoenixFeederStatusV1::Recorded; return S_OK;
}
int main() try {
    // An OpenGL frontend need not observe our private D3D12 creation at all.
    // Create this unhooked WARP queue BEFORE Gpu enables the debug-layer proxy.
    // Debug proxies are intentionally not classified as raw native objects.
    {
        ComPtr<IDXGIFactory4> factory; phoenix_native::hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter> adapter; phoenix_native::hr(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
        ComPtr<ID3D12Device> device;
        phoenix_native::hr(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)));
        ComPtr<ID3D12CommandQueue> queue; D3D12_COMMAND_QUEUE_DESC description{};
        phoenix_native::hr(device->CreateCommandQueue(&description,IID_PPV_ARGS(&queue)));
        phoenix_bundle::require(is_unwrapped_system_d3d12_queue(queue.Get()),
            "Unobserved native system queue was rejected");
    }
    phoenix_bundle::require(!is_unwrapped_system_d3d12_queue(nullptr) &&
        !system_d3d12_implementation(reinterpret_cast<const void*>(&metadata)),
        "Non-system implementation was accepted as a native queue");
    phoenix_test::Gpu gpu; // WARP only; hardware selection is never enabled.
    {
        ComPtr<ID3D12Device> device=gpu.device,wrapper_device;
        ComPtr<ID3D12CommandQueue> queue=gpu.queue,wrapper_queue;
        unsigned resolved=0;
        in_process_api.native_queue=[&](ID3D12CommandQueue* supplied){
            phoenix_bundle::require(supplied==gpu.queue.Get(),"Unexpected queue identity");
            ++resolved;return gpu.queue.Get();
        };
        resolve_local_d3d12_identity(device,queue,wrapper_device,wrapper_queue);
        phoenix_bundle::require(resolved==1&&device.Get()==gpu.device.Get()&&queue.Get()==gpu.queue.Get()&&
            wrapper_device.Get()==gpu.device.Get()&&wrapper_queue.Get()==gpu.queue.Get(),"Native identity/lifetime not retained");
        in_process_api.native_queue=[](ID3D12CommandQueue*)->ID3D12CommandQueue*{return nullptr;};
        bool refused=false;try{resolve_local_d3d12_identity(device,queue,wrapper_device,wrapper_queue);}
        catch(const std::exception&){refused=true;}
        phoenix_bundle::require(refused&&device.Get()==gpu.device.Get()&&queue.Get()==gpu.queue.Get(),
            "Unknown identity was accepted or changed the active context");
        in_process_api.native_queue={};
    }
    ComPtr<ID3D11Device> game; ComPtr<ID3D11DeviceContext> context;
    phoenix_native::hr(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,
        D3D11_SDK_VERSION,&game,nullptr,&context));
    D3D11_TEXTURE2D_DESC desc{}; desc.Width=17; desc.Height=9;
    desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1; desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.BindFlags=D3D11_BIND_RENDER_TARGET;
    ComPtr<ID3D11Texture2D> back; phoenix_native::hr(game->CreateTexture2D(&desc,nullptr,&back));
    D3D11LocalSurface surface(game.Get(),17,9,desc.Format,gpu.device.Get());
    D3D12_DESCRIPTOR_HEAP_DESC heap{}; heap.NumDescriptors=1;
    phoenix_native::hr(surface.device()->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&cpu_heap)));
    heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    phoenix_native::hr(surface.device()->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&gpu_heap)));
    in_process_api.prepare=[](auto*,const Settings& settings){mode=settings.mode;};
    in_process_api.release=[](auto*){}; in_process_api.observe=[](auto*,auto*,auto*){};
    in_process_api.forget=[](ID3D12GraphicsCommandList*) noexcept {};
    in_process_api.evaluate=evaluate; in_process_api.progress=in_process_api.ui=metadata;
    in_process_api.action=action;
    in_process_api.update_tuning=[](const Settings&){};
    reshade::api::resource_desc description(17,9,1,1,reshade::api::format::r8g8b8a8_unorm,1,
        reshade::api::memory_heap::gpu_only,reshade::api::resource_usage::render_target);
    Settings settings; settings.mode=2; settings.uplift=true; settings.pipeline={};
    settings.pipeline.motion_enabled=false; // Synthetic interop test, not hardware flow.
    settings.pipeline.motion_enabled=false; settings.pipeline.scene_detection=false;
    const reshade::api::resource image{uint64_t(uintptr_t(surface.image()))};
    std::vector<uint32_t> blue(17*9,0xffff0000);
    unsigned checks=0;
    auto verify=[&](bool ok,std::source_location location=std::source_location::current()){
        ++checks;
        if(!ok)throw std::runtime_error("Local DX11 interop assertion failed at line "+std::to_string(location.line()));
    };
    {
        int leases=0, acquired=0;
        in_process_api.local_queue_lifetime=[&](ID3D12CommandQueue* queue,bool created) {
            verify(queue!=nullptr); leases+=created?1:-1; acquired+=created?1:0;
        };
        { D3D11LocalSurface owned(game.Get(),17,9,desc.Format,gpu.device.Get()); verify(leases==1); }
        verify(leases==0 && acquired==1);
        bool failed=false;
        try { D3D11LocalSurface invalid(game.Get(),0,9,desc.Format,gpu.device.Get()); }
        catch(const std::exception&) { failed=true; }
        verify(failed && leases==0 && acquired==2);
        in_process_api.local_queue_lifetime={};
    }
    {
        // A warm helper handoff must release every backbuffer RTV; otherwise
        // ResizeBuffers fails even though the model itself survived the reset.
        // Use an invisible, non-activating software swapchain, not COM public
        // refcounts (D3D11 views retain resources through internal references).
        struct Window {
            HWND handle=CreateWindowExW(WS_EX_NOACTIVATE|WS_EX_TOOLWINDOW,L"STATIC",L"Phoenix WARP resize test",
                WS_POPUP,0,0,17,9,nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);
            ~Window(){if(handle)DestroyWindow(handle);}
        } window;
        verify(window.handle!=nullptr);
        ComPtr<IDXGIFactory2> factory;phoenix_native::hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        DXGI_SWAP_CHAIN_DESC1 swap_description{};
        swap_description.Width=17;swap_description.Height=9;swap_description.Format=desc.Format;
        swap_description.SampleDesc.Count=1;swap_description.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swap_description.BufferCount=2;swap_description.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
        ComPtr<IDXGISwapChain1> swapchain;
        phoenix_native::hr(factory->CreateSwapChainForHwnd(game.Get(),window.handle,&swap_description,nullptr,nullptr,&swapchain));
        ComPtr<ID3D11Texture2D> swap_back;phoenix_native::hr(swapchain->GetBuffer(0,IID_PPV_ARGS(&swap_back)));
        D3D11Bridge helper_bridge(game.Get(),17,9,desc.Format,gpu.device.Get());
        helper_bridge.frame(swap_back.Get(),1,settings,true); // No host: caches only the target view.
        swap_back.Reset();
        verify(FAILED(swapchain->ResizeBuffers(0,31,17,DXGI_FORMAT_UNKNOWN,0)));
        helper_bridge.release_presentation_targets();
        context->Flush();
        verify(SUCCEEDED(swapchain->ResizeBuffers(0,31,17,DXGI_FORMAT_UNKNOWN,0)));
    }
    {
    D3D12Bridge bridge(nullptr,description,surface.device(),surface.queue());
    auto run=[&](int disposition,bool apply=false){
        behavior=disposition; context->UpdateSubresource(back.Get(),0,nullptr,blue.data(),17*4,0);
        surface.capture(back.Get());
        const bool shown=bridge.frame(image,settings,apply);
        if(shown)surface.publish(back.Get());else surface.retire();
        verify(surface.retired()&&bridge.safe_to_fallback()); return shown;
    };
    verify(!run(1)); verify(!run(2)); verify(bridge.delivered()==0);
    for(int renderer:{2,1,3}) {
        settings.mode=renderer;
        verify(run(0,true)); verify(!bridge.held_last_frame());
        const auto count=bridge.delivered();
        verify(run(1)); verify(bridge.held_last_frame()&&bridge.delivered()==count);
        verify(run(2)); verify(bridge.held_last_frame()&&bridge.delivered()==count);
    }
    desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> readback;phoenix_native::hr(game->CreateTexture2D(&desc,nullptr,&readback));
    context->CopyResource(readback.Get(),back.Get()); D3D11_MAPPED_SUBRESOURCE map{};
    phoenix_native::hr(context->Map(readback.Get(),0,D3D11_MAP_READ,0,&map));
    bool all_red=true;
    for(unsigned y=0;y<9;++y)for(unsigned x=0;x<17;++x)
        all_red &= reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(map.pData)+map.RowPitch*y)[x]==0xff0000ff;
    context->Unmap(readback.Get(),0); verify(all_red); verify(completed==3);
    // A second frontend cannot reconfigure the singleton native session while
    // its previous bridge (possibly an asynchronous worker) is still alive.
    bool refused=false;
    try { D3D12Bridge second(nullptr,description,surface.device(),surface.queue()); }
    catch(const std::exception&) { refused=true; }
    verify(refused);
    }
    {
        unsigned local_queue_leases=0;
        in_process_api.local_queue_lifetime=[&](ID3D12CommandQueue* queue,bool created){
            phoenix_bundle::require(queue!=nullptr,"Private queue lease lost its owner");
            if(created)++local_queue_leases;else --local_queue_leases;
        };
        {
            D3D12LocalPixels lifetime(nullptr,17,9,gpu.device.Get());
            verify(local_queue_leases==1);
        }
        verify(local_queue_leases==0);
        in_process_api.local_queue_lifetime={};
    }
    {
        D3D12LocalPixels pixels(nullptr,17,9,gpu.device.Get());
        std::vector<uint8_t> input(17*9*4);
        for(size_t i=0;i<input.size();i+=4){input[i+2]=255;input[i+3]=255;}
        settings.mode=2; pixels.upload(input); behavior=1;
        verify(!pixels.frame(settings,true)); behavior=2;
        verify(!pixels.frame(settings,false)); behavior=0;
        verify(pixels.frame(settings,false));
        const auto output=pixels.download();
        bool correct=output.size()==input.size()&&output!=input;
        for(size_t i=0;i<output.size();i+=4)
            correct &= output[i]==255&&output[i+1]==0&&output[i+2]==0&&output[i+3]==255;
        verify(correct); verify(pixels.bridge().delivered()==1);
    }
    {
        FeedBuild build{}; build.width=build.output_width=17;build.height=build.output_height=9;
        build.cpu_bridge=1;
        HostClient client(GetModuleHandleW(nullptr),{},settings,build,std::nullopt,std::nullopt,
            LocalGpuOptions{gpu.device});
        const auto deadline=GetTickCount64()+10000;
        while(client.phase()==Phase::Starting&&GetTickCount64()<deadline)Sleep(1);
        verify(client.phase()==Phase::Ready);
        verify(client.in_process()&&client.process_id()==GetCurrentProcessId());
        verify(std::string(client.native_execution())=="in_process_staged");
        std::vector<uint8_t> input(17*9*4,0);for(size_t i=3;i<input.size();i+=4)input[i]=255;
        verify(client.submit(1,true,input));
        while(client.phase()==Phase::Busy&&GetTickCount64()<deadline)Sleep(1);
        if(client.phase()==Phase::Failed)throw std::runtime_error(client.error());
        verify(client.phase()==Phase::Finished);
        const auto output=client.take();
        verify(output.size()==input.size()&&output!=input&&output[0]==255&&output[2]==0);
        verify(client.ui_snapshot().at("processing_location")=="in_process_staged");
        verify(client.ui_snapshot().at("inference_pid")==GetCurrentProcessId());
        auto updated=settings;updated.intensity=0.5f;
        verify(client.update_tuning(updated));
        verify(client.submit(2,true,input));
        while(client.phase()==Phase::Busy&&GetTickCount64()<deadline)Sleep(1);
        verify(client.phase()==Phase::Finished);
        verify(client.take()==output);
        verify(client.tuning_current_output());
    }
    // The asynchronous lease must cancel and retire before another frontend can
    // own the singleton. This loop does not launch any helper or use hardware.
    const auto retirement_deadline=GetTickCount64()+5000;
    while(!in_process_executor_mutex.try_lock()) {
        if(GetTickCount64()>retirement_deadline)throw std::runtime_error("Local worker did not retire");
        Sleep(1);
    }
    in_process_executor_mutex.unlock();
    while(!can_publish_in_process_api()&&GetTickCount64()<retirement_deadline)Sleep(1);
    verify(can_publish_in_process_api());
    {
        // Real software command submission paused inside the provider: destroy
        // the client and retire the addon while that detached worker is busy.
        struct Callbacks {
            std::atomic<unsigned> wrong_runtime=0, released=0, queues=0;
        };
        auto observed=std::make_shared<Callbacks>();
        auto provider=std::make_shared<int>(1);
        std::weak_ptr<int> provider_weak=provider;
        in_process_api.prepare=[observed](auto* runtime,const Settings& value){
            if(runtime)++observed->wrong_runtime;mode=value.mode;
        };
        in_process_api.observe=[observed](auto* runtime,auto*,auto*){if(runtime)++observed->wrong_runtime;};
        in_process_api.release=[observed,provider](auto* runtime){
            if(runtime)++observed->wrong_runtime;++observed->released;
        };
        // A function-pointer hook cannot capture the fixture; use the private
        // queue callback to verify cleanup AFTER the bridge's release hook.
        in_process_api.local_queue_lifetime=[observed](auto*,bool created){
            if(created)++observed->queues;else --observed->queues;
        };
        provider.reset();
        FeedBuild build{};build.width=build.output_width=17;build.height=build.output_height=9;build.cpu_bridge=1;
        auto client=std::make_unique<HostClient>(GetModuleHandleW(nullptr),phoenix_model::WeightSource{},settings,
            build,std::nullopt,std::nullopt,LocalGpuOptions{gpu.device});
        struct Resume { ~Resume(){resume_evaluation=true;} } unblock;
        const auto ready_deadline=GetTickCount64()+10000;
        while(client->phase()==Phase::Starting&&GetTickCount64()<ready_deadline)Sleep(1);
        verify(client->phase()==Phase::Ready);
        block_evaluation=true;
        std::vector<uint8_t> input(17*9*4,255);
        verify(client->submit(1,true,input));
        const auto entered_deadline=GetTickCount64()+5000;
        while(!evaluation_entered&&GetTickCount64()<entered_deadline)Sleep(1);
        verify(evaluation_entered&&observed->queues==1);
        client.reset();
        retire_in_process_api();
        verify(!in_process_api_available()&&!can_publish_in_process_api()&&!provider_weak.expired());
        verify(!publish_in_process_api({}));
        resume_evaluation=true;
        const auto stopped_deadline=GetTickCount64()+10000;
        while(!can_publish_in_process_api()&&GetTickCount64()<stopped_deadline)Sleep(1);
        verify(can_publish_in_process_api());
        verify(observed->wrong_runtime==0&&observed->released==1&&observed->queues==0);
        verify(provider_weak.expired()&&!in_process_api.release&&!in_process_api.local_queue_lifetime);
    }
    std::cout<<Json{{"passed",true},{"software_only",true},{"hardware_gpu_inference",false},
        {"checks",checks},{"synthetic_completed_frames",completed}}.dump()<<'\n';
} catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
