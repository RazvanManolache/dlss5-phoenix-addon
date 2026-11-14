// Model-free DX9 -> D3D11 -> D3D12 -> DX9 transfer. Compile as x86 for the
// fixture and x64 for the worker. The D3D11 device lives in the x86 fixture.
#define NOMINMAX
#include <windows.h>
#include <d3d9.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "../../third_party/nlohmann/json.hpp"

template<class T> using Ptr = Microsoft::WRL::ComPtr<T>;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr UINT width = 2560, height = 1080;
constexpr DWORD timeout_ms = 5000;

void check(bool condition, const char* stage) {
    if (!condition) throw std::runtime_error(stage);
}
void hr(HRESULT result, const char* stage) {
    if (SUCCEEDED(result)) return;
    std::ostringstream error;
    error << stage << " failed (0x" << std::hex << static_cast<unsigned>(result) << ")";
    throw std::runtime_error(error.str());
}
struct Handle {
    HANDLE value;
    explicit Handle(HANDLE handle = nullptr) : value(handle) {}
    ~Handle() { if (value) CloseHandle(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};
void signal(HANDLE event) { check(SetEvent(event) != 0, "SetEvent"); }
void wait(HANDLE event, HANDLE worker = nullptr) {
    const HANDLE handles[]{event, worker};
    const auto result = WaitForMultipleObjects(worker ? 2 : 1, handles, FALSE, timeout_ms);
    if (result == WAIT_OBJECT_0 + 1) {
        DWORD code = 0;
        GetExitCodeProcess(worker, &code);
        throw std::runtime_error("Worker exited before acknowledgement (code " + std::to_string(code) + ")");
    }
    check(result == WAIT_OBJECT_0, "Timed out waiting for transfer event");
}
template<class Poll> void retire(Poll poll, const char* stage) {
    const auto end = Clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        const auto result = poll();
        if (result == S_OK) return;
        hr(result, stage);
        check(Clock::now() < end, stage);
        SwitchToThread();
    }
}
HANDLE argument_handle(const wchar_t* text) {
    return reinterpret_cast<HANDLE>(static_cast<uintptr_t>(std::stoull(text)));
}

Ptr<IDXGIAdapter> match_adapter(LUID luid) {
    Ptr<IDXGIFactory4> factory;
    hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
    Ptr<IDXGIAdapter> adapter;
    hr(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)), "Match DX9 adapter LUID");
    return adapter;
}
void run_worker(wchar_t** args) {
    check(sizeof(void*) == 8, "Worker must be x64");
    const auto texture_handle = argument_handle(args[2]);
    const auto fence_handle = argument_handle(args[3]);
    const auto request = argument_handle(args[4]);
    const auto complete = argument_handle(args[5]);
    const LUID luid{static_cast<DWORD>(std::stoul(args[6])), static_cast<LONG>(std::stol(args[7]))};
    const auto frames = std::stoul(args[8]);
    const auto adapter = match_adapter(luid);
    Ptr<ID3D12Device> device;
    hr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)), "Create x64 D3D12 device");
    Ptr<ID3D12Resource> texture;
    Ptr<ID3D12Fence> fence;
    hr(device->OpenSharedHandle(texture_handle, IID_PPV_ARGS(&texture)), "Import shared texture in x64");
    hr(device->OpenSharedHandle(fence_handle, IID_PPV_ARGS(&fence)), "Import shared fence in x64");
    const D3D12_COMMAND_QUEUE_DESC queue_description{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ptr<ID3D12CommandQueue> queue;
    hr(device->CreateCommandQueue(&queue_description, IID_PPV_ARGS(&queue)), "Create x64 D3D12 queue");
    Ptr<ID3D12DescriptorHeap> targets;
    const D3D12_DESCRIPTOR_HEAP_DESC heap{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    hr(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&targets)), "Create D3D12 RTV heap");
    const auto target = targets->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(texture.Get(), nullptr, target);
    Ptr<ID3D12CommandAllocator> allocator;
    Ptr<ID3D12GraphicsCommandList> commands;
    hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
       "Create D3D12 allocator");
    hr(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                         IID_PPV_ARGS(&commands)), "Create D3D12 command list");
    hr(commands->Close(), "Close initial D3D12 command list");
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {texture.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                          D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET};
    const D3D12_RECT marker{0, 0, 64, 64};
    const float white[]{1, 1, 1, 1};
    signal(complete);
    for (unsigned frame = 0; frame < frames; ++frame) {
        wait(request);
        const uint64_t input_ready = 2ull * frame + 1;
        hr(queue->Wait(fence.Get(), input_ready), "Order D3D12 after the DX11 copy");
        hr(allocator->Reset(), "Reset D3D12 allocator");
        hr(commands->Reset(allocator.Get(), nullptr), "Reset D3D12 command list");
        commands->ResourceBarrier(1, &barrier);
        commands->ClearRenderTargetView(target, white, 1, &marker);
        auto restore = barrier;
        std::swap(restore.Transition.StateBefore, restore.Transition.StateAfter);
        commands->ResourceBarrier(1, &restore);
        hr(commands->Close(), "Close D3D12 command list");
        ID3D12CommandList* lists[]{commands.Get()};
        queue->ExecuteCommandLists(1, lists);
        hr(queue->Signal(fence.Get(), input_ready + 1), "Publish the D3D12 result");
        signal(complete);
    }
    Handle retired(CreateEventW(nullptr, FALSE, FALSE, nullptr));
    check(retired.value != nullptr, "Create worker retirement event");
    hr(fence->SetEventOnCompletion(2ull * frames, retired.value), "Watch final D3D12 completion");
    wait(retired.value);
    hr(device->GetDeviceRemovedReason(), "D3D12 worker device");
}

struct SharedTextures {
    D3DFORMAT format;
    HANDLE input_handle = nullptr, output_handle = nullptr;
    Ptr<IDirect3DTexture9> input, output;
};
SharedTextures make_shared(IDirect3DDevice9* device) {
    for (const auto format : {D3DFMT_A8B8G8R8, D3DFMT_A8R8G8B8}) {
        SharedTextures pair{format};
        if (SUCCEEDED(device->CreateTexture(width, height, 1, D3DUSAGE_RENDERTARGET, format,
                                            D3DPOOL_DEFAULT, &pair.input, &pair.input_handle)) &&
            SUCCEEDED(device->CreateTexture(width, height, 1, D3DUSAGE_RENDERTARGET, format,
                                            D3DPOOL_DEFAULT, &pair.output, &pair.output_handle)) &&
            pair.input_handle && pair.output_handle) return pair;
    }
    throw std::runtime_error("DX9 could not create a pair of shareable RGBA8 render targets");
}

// The game process owns the D3D11 device. The provisioning D3D12 device
// creates NT handles, following the same sharing pattern as D3D11Bridge.
class GameRelay {
    Ptr<ID3D11Texture2D> input_, output_, shared_;
    Ptr<ID3D11DeviceContext4> context_;
    Ptr<ID3D11Fence> fence11_;
    Ptr<ID3D11Query> retired_;
    Ptr<ID3D12Resource> texture12_;
    Ptr<ID3D12Fence> fence12_;

public:
    Handle texture_handle, fence_handle;
    GameRelay(LUID luid, const SharedTextures& game) {
        const auto adapter = match_adapter(luid);
        Ptr<ID3D11Device> device11;
        Ptr<ID3D11DeviceContext> immediate;
        hr(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0,
                            D3D11_SDK_VERSION, &device11, nullptr, &immediate), "Create x86 relay device");
        hr(immediate.As(&context_), "Open fence-capable x86 relay context");
        Ptr<ID3D11Device5> interop;
        hr(device11.As(&interop), "Open fence-capable x86 relay device");
        hr(device11->OpenSharedResource(game.input_handle, IID_PPV_ARGS(&input_)), "Import DX9Ex input in x86");
        hr(device11->OpenSharedResource(game.output_handle, IID_PPV_ARGS(&output_)), "Import DX9Ex output in x86");

        D3D11_TEXTURE2D_DESC source{};
        input_->GetDesc(&source);
        Ptr<ID3D12Device> provision;
        hr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&provision)),
           "Create x86 shared-resource provisioning device");
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC description{};
        description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        description.Width = width;
        description.Height = height;
        description.DepthOrArraySize = description.MipLevels = 1;
        description.Format = source.Format;
        description.SampleDesc.Count = 1;
        description.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
        hr(provision->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &description,
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&texture12_)), "Allocate shared relay texture");
        hr(provision->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence12_)), "Create shared relay fence");
        SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
        hr(provision->CreateSharedHandle(texture12_.Get(), &inherit, GENERIC_ALL, nullptr, &texture_handle.value),
           "Export relay texture NT handle");
        hr(provision->CreateSharedHandle(fence12_.Get(), &inherit, GENERIC_ALL, nullptr, &fence_handle.value),
           "Export relay fence NT handle");
        hr(interop->OpenSharedResource1(texture_handle.value, IID_PPV_ARGS(&shared_)), "Open relay texture in D3D11");
        hr(interop->OpenSharedFence(fence_handle.value, IID_PPV_ARGS(&fence11_)), "Open relay fence in D3D11");
        const D3D11_QUERY_DESC query{D3D11_QUERY_EVENT};
        hr(device11->CreateQuery(&query, &retired_), "Create relay retirement query");
    }
    void capture(uint64_t input_ready) {
        context_->CopyResource(shared_.Get(), input_.Get());
        hr(context_->Signal(fence11_.Get(), input_ready), "Signal the D3D11 input copy");
        context_->Flush();
    }
    void compose(uint64_t output_ready) {
        hr(context_->Wait(fence11_.Get(), output_ready), "Order D3D11 after the D3D12 edit");
        context_->CopyResource(output_.Get(), shared_.Get());
        context_->End(retired_.Get());
        context_->Flush();
        retire([&] { return context_->GetData(retired_.Get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH); },
               "Retire the D3D11 output copy before DX9 reads it");
    }
};

unsigned gray(unsigned frame) { return 32 + frame % 192; }
DWORD gray_pixel(unsigned frame) { return 0xff000000u | (gray(frame) * 0x010101u); }

void verify_output(IDirect3DDevice9* device, IDirect3DSurface9* surface, D3DFORMAT format,
                   unsigned frame) {
    Ptr<IDirect3DSurface9> staging;
    hr(device->CreateOffscreenPlainSurface(width, height, format, D3DPOOL_SYSTEMMEM, &staging, nullptr),
       "Create verification staging surface");
    hr(device->GetRenderTargetData(surface, staging.Get()), "Read back verification output");
    D3DLOCKED_RECT mapped{};
    hr(staging->LockRect(&mapped, nullptr, D3DLOCK_READONLY), "Map verification output");
    const auto* bytes = static_cast<const unsigned char*>(mapped.pBits);
    unsigned mismatches = 0;
    for (UINT y = 0; y < height; ++y) {
        const auto* row = reinterpret_cast<const DWORD*>(bytes + size_t(y) * mapped.Pitch);
        for (UINT x = 0; x < width; ++x) {
            const auto expected = x < 64 && y < 64 ? 0xffffffffu : gray_pixel(frame);
            mismatches += (row[x] & 0x00ffffffu) != (expected & 0x00ffffffu);
        }
    }
    hr(staging->UnlockRect(), "Unmap verification output");
    if (mismatches)
        throw std::runtime_error("Shared texture verification found " + std::to_string(mismatches) +
                                 " mismatched pixels");
}

Json run_generation(IDirect3DDevice9* device, LUID luid, const std::wstring& worker_path,
                    unsigned warmup, unsigned measured);

Json run_fixture(const std::wstring& worker_path, bool use_ex) {
    check(sizeof(void*) == 4, "Fixture must be x86");
    const auto window = CreateWindowExW(0, L"STATIC", L"Phoenix DX9 shared textures", WS_POPUP,
                                        0, 0, 320, 180, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
    check(window != nullptr, "Create fixture window");
    Ptr<IDirect3D9Ex> d3d_ex;
    hr(Direct3DCreate9Ex(D3D_SDK_VERSION, &d3d_ex), "Direct3DCreate9Ex");
    D3DADAPTER_IDENTIFIER9 adapter{};
    hr(d3d_ex->GetAdapterIdentifier(D3DADAPTER_DEFAULT, 0, &adapter), "Identify DX9 adapter");
    LUID luid{};
    hr(d3d_ex->GetAdapterLUID(D3DADAPTER_DEFAULT, &luid), "GetAdapterLUID");
    D3DPRESENT_PARAMETERS parameters{};
    parameters.Windowed = TRUE;
    parameters.SwapEffect = D3DSWAPEFFECT_DISCARD;
    parameters.hDeviceWindow = window;
    parameters.BackBufferWidth = width;
    parameters.BackBufferHeight = height;
    parameters.BackBufferFormat = D3DFMT_X8R8G8B8;
    parameters.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
    Ptr<IDirect3DDevice9> device;
    if (use_ex) {
        Ptr<IDirect3DDevice9Ex> device_ex;
        hr(d3d_ex->CreateDeviceEx(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, window,
                                  D3DCREATE_HARDWARE_VERTEXPROCESSING, &parameters, nullptr, &device_ex),
           "Create DX9Ex device");
        device = device_ex;
    } else {
        Ptr<IDirect3D9> d3d;
        d3d.Attach(Direct3DCreate9(D3D_SDK_VERSION));
        check(bool(d3d), "Direct3DCreate9");
        hr(d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, window,
                             D3DCREATE_HARDWARE_VERTEXPROCESSING, &parameters, &device),
           "Create legacy DX9 device");
    }
    auto result = run_generation(device.Get(), luid, worker_path, 20, 120);
    if (use_ex) {
        Ptr<IDirect3DDevice9Ex> reset_device;
        hr(device.As(&reset_device), "Query DX9Ex device for reset");
        hr(reset_device->ResetEx(&parameters, nullptr), "Reset DX9Ex swapchain");
        const auto recovered = run_generation(device.Get(), luid, worker_path, 0, 0);
        result["verified_after_swapchain_reset"] = recovered.at("pixel_verified_frames");
    }
    device.Reset();
    DestroyWindow(window);
    result["adapter"] = adapter.Description;
    return result;
}

Json run_generation(IDirect3DDevice9* device, LUID luid, const std::wstring& worker_path,
                    unsigned warmup, unsigned measured) {
    auto textures = make_shared(device);
    GameRelay relay(luid, textures);
    Ptr<IDirect3DSurface9> input, output, backbuffer;
    hr(textures.input->GetSurfaceLevel(0, &input), "Get shared input surface");
    hr(textures.output->GetSurfaceLevel(0, &output), "Get shared output surface");
    hr(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &backbuffer), "Get XRGB game backbuffer");
    Ptr<IDirect3DQuery9> fence;
    hr(device->CreateQuery(D3DQUERYTYPE_EVENT, &fence), "Create DX9 event query");
    const auto retire_dx9 = [&] {
        hr(fence->Issue(D3DISSUE_END), "Issue DX9 event query");
        retire([&] { return fence->GetData(nullptr, 0, D3DGETDATA_FLUSH); }, "Retire DX9 texture access");
    };

    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    Handle request(CreateEventW(&inherit, FALSE, FALSE, nullptr));
    Handle complete(CreateEventW(&inherit, FALSE, FALSE, nullptr));
    check(request.value && complete.value, "Create helper command events");
    constexpr unsigned verification = 3;
    const unsigned total = warmup + measured + verification;
    std::wstring command = L"\"" + worker_path + L"\" --worker";
    for (const auto value : {relay.texture_handle.value, relay.fence_handle.value, request.value, complete.value})
        command += L" " + std::to_wstring(reinterpret_cast<uintptr_t>(value));
    command += L" " + std::to_wstring(luid.LowPart) + L" " + std::to_wstring(luid.HighPart) +
               L" " + std::to_wstring(total);
    STARTUPINFOW startup{sizeof(startup)};
    PROCESS_INFORMATION process{};
    check(CreateProcessW(worker_path.c_str(), command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                         nullptr, nullptr, &startup, &process) != 0, "Start x64 worker");
    Handle worker(process.hProcess);
    CloseHandle(process.hThread);
    wait(complete.value, worker.value);
    std::vector<double> milliseconds;
    for (unsigned frame = 0; frame < total; ++frame) {
        const auto start = Clock::now();
        hr(device->ColorFill(backbuffer.Get(), nullptr, gray_pixel(frame)), "Draw fixture frame");
        hr(device->StretchRect(backbuffer.Get(), nullptr, input.Get(), nullptr, D3DTEXF_NONE),
           "Capture DX9 frame into shared input");
        retire_dx9();
        const uint64_t input_ready = 2ull * frame + 1;
        relay.capture(input_ready);
        signal(request.value);
        wait(complete.value, worker.value);
        relay.compose(input_ready + 1);
        hr(device->StretchRect(output.Get(), nullptr, backbuffer.Get(), nullptr, D3DTEXF_NONE),
           "Compose shared output in DX9");
        retire_dx9(); // Keep the helper from overwriting an output still being sampled.
        if (frame >= warmup && frame < warmup + measured)
            milliseconds.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        if (frame >= warmup + measured) verify_output(device, backbuffer.Get(), D3DFMT_X8R8G8B8, frame);
    }
    wait(worker.value);
    DWORD exit_code = 0;
    check(GetExitCodeProcess(worker.value, &exit_code) && exit_code == 0, "Worker exited unsuccessfully");
    std::sort(milliseconds.begin(), milliseconds.end());
    Json result{{"passed", true}, {"fixture_bits", 32}, {"relay_api", "D3D11"}, {"relay_bits", 32},
            {"worker_bits", 64}, {"worker_api", "D3D12"},
            {"width", width}, {"height", height},
            {"shared_d3d9_format", textures.format}, {"measured_frames", measured},
            {"pixel_verified_frames", verification}, {"cpu_pixel_bytes_per_measured_frame", 0}};
    if (measured) {
        result["p50_ms"] = milliseconds[measured / 2];
        result["p95_ms"] = milliseconds[(measured * 95 / 100) - 1];
    }
    return result;
}

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc == 9 && std::wstring_view(argv[1]) == L"--worker") {
            run_worker(argv);
        } else if (argc == 3) {
            const bool use_ex = std::wstring_view(argv[1]) == L"--ex";
            check(use_ex || std::wstring_view(argv[1]) == L"--legacy", "Expected --legacy or --ex");
            std::cout << run_fixture(argv[2], use_ex).dump() << '\n';
        } else {
            throw std::runtime_error("Usage: probe --ex|--legacy <x64 worker path>");
        }
        return 0;
    } catch (const std::exception& error) {
        if (argc == 9) std::cerr << error.what() << '\n';
        else std::cout << Json{{"passed", false}, {"error", error.what()}}.dump() << '\n';
        return 1;
    }
}
