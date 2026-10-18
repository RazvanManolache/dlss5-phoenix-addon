// Isolates module creation without ReShade hooks, model state, or GPU dispatch.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include "../../third_party/nvapi/nvapi.h"
#include "../../third_party/nlohmann/json.hpp"
#include "../../src/native/support/module_capability.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using Microsoft::WRL::ComPtr;
using Json = nlohmann::json;

static void event(const char* stage, Json details = Json::object()) {
    details["stage"] = stage;
    std::cout << details.dump() << std::endl;
}
static void checked(HRESULT result, const char* stage) {
    if (FAILED(result)) {
        event(stage, {{"hresult", uint32_t(result)}});
        throw std::runtime_error(stage);
    }
}

int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    try {
        event("abi", {{"pointer_bytes", sizeof(void*)}, {"object_handle_bytes", sizeof(NVDX_ObjectHandle)},
              {"launch_params_bytes", sizeof(NVAPI_CU_KERNEL_LAUNCH_PARAMS)},
              {"params_offset", offsetof(NVAPI_CU_KERNEL_LAUNCH_PARAMS, pParams)}});
        if (argc == 2 && std::wstring(argv[1]) == L"--describe") return 0;
        const bool shader = argc == 7 && std::wstring(argv[1]) == L"--shader";
        const bool arithmetic = argc == 3 && std::wstring(argv[1]) == L"--arithmetic";
        if (!shader && !arithmetic && (argc != 3 || (std::wstring(argv[1]) != L"--module" && std::wstring(argv[1]) != L"--safe-module" &&
                                     std::wstring(argv[1]) != L"--capability" && std::wstring(argv[1]) != L"--cuda")))
            throw std::runtime_error("Use --describe, --module/--capability BLOB, or --shader BLOB ENTRY X Y Z");
        std::ifstream stream(std::filesystem::path(argv[2]), std::ios::binary | std::ios::ate);
        if (!stream || stream.tellg() <= 0 || stream.tellg() > 16 * 1024 * 1024)
            throw std::runtime_error("Missing or oversized kernel blob");
        std::vector<char> blob(size_t(stream.tellg()), 0);
        stream.seekg(0);
        if (!stream.read(blob.data(), std::streamsize(blob.size())))
            throw std::runtime_error("Short kernel blob read");
        if(std::wstring(argv[1])==L"--cuda") {
            auto cuda=LoadLibraryExW(L"nvcuda.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
            if(!cuda) throw std::runtime_error("CUDA driver unavailable");
            auto init=reinterpret_cast<int(WINAPI*)(unsigned)>(GetProcAddress(cuda,"cuInit"));
            if(!init) throw std::runtime_error("CUDA initialization API unavailable");
            event("cuda_initialize_call");
            const auto status=init(0);
            event("cuda_initialize_result",{{"status",status}});
            using DumpLogs=int(WINAPI*)(unsigned*,char*,size_t*,unsigned);
            if(auto logs=reinterpret_cast<DumpLogs>(GetProcAddress(cuda,"cuLogsDumpToMemory"))) {
                char buffer[25600]{}; size_t bytes=sizeof(buffer);
                auto dumped=logs(nullptr,buffer,&bytes,0);
                event("cuda_logs",{{"status",dumped},{"text",bytes<sizeof(buffer)?std::string(buffer,bytes):""}});
            }
            return status==0?0:2;
        }
        ComPtr<IDXGIFactory4> factory;
        checked(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "factory");
        ComPtr<ID3D12Device> device;
        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> adapter;
            auto result = factory->EnumAdapters1(index, &adapter);
            if (result == DXGI_ERROR_NOT_FOUND) break;
            checked(result, "adapter");
            DXGI_ADAPTER_DESC1 desc{};
            checked(adapter->GetDesc1(&desc), "adapter_desc");
            if (desc.VendorId != 0x10de || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
            checked(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&device)), "device");
            event("adapter", {{"vendor_id", desc.VendorId}, {"device_id", desc.DeviceId}});
            break;
        }
        if (!device) throw std::runtime_error("No NVIDIA DLSS5 D3D12 adapter");
        ComPtr<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        checked(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)), "queue");
        const auto library = LoadLibraryExW(sizeof(void*) == 8 ? L"nvapi64.dll" : L"nvapi.dll",
                                            nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!library) throw std::runtime_error("No matching system NVAPI library");
        auto query = reinterpret_cast<void*(__cdecl*)(unsigned)>(GetProcAddress(library, "nvapi_QueryInterface"));
        if (!query) throw std::runtime_error("No NVAPI query interface");
        auto initialize = reinterpret_cast<NvAPI_Status(__cdecl*)()>(query(0x0150e828));
        if (!initialize) throw std::runtime_error("No NVAPI initialization");
        event("initialize_call");
        const auto initialized = initialize();
        event("initialize_result", {{"status", int(initialized)}});
        if (initialized != NVAPI_OK) return 2;
        if (std::wstring(argv[1]) == L"--safe-module") {
            phoenix_kernel::require_native_module_capability(sizeof(void*), [&] {
                auto supported = reinterpret_cast<decltype(&NvAPI_D3D12_IsFatbinPTXSupported)>(query(0x70c07832));
                bool value = false;
                event("safe_capability_call");
                return supported && supported(device.Get(), &value) == NVAPI_OK && value;
            });
        }
        if (std::wstring(argv[1]) == L"--capability") {
            auto supported = reinterpret_cast<decltype(&NvAPI_D3D12_IsFatbinPTXSupported)>(query(0x70c07832));
            if (!supported) throw std::runtime_error("No capability entry point");
            bool value = false;
            event("capability_call");
            const auto status = supported(device.Get(), &value);
            event("capability_result", {{"status", int(status)}, {"supported", value}});
            return status == NVAPI_OK ? 0 : 2;
        }
        if (shader || arithmetic) {
            auto create = reinterpret_cast<decltype(&NvAPI_D3D12_CreateCubinComputeShaderWithName)>(query(0x1dc7261f));
            auto destroy = reinterpret_cast<decltype(&NvAPI_D3D12_DestroyCubinComputeShader)>(query(0x7fb785ba));
            if (!create || !destroy) throw std::runtime_error("No legacy cubin shader entry points");
            const std::wstring wide_entry(arithmetic?L"phoenix_arithmetic":argv[3]);
            const std::string entry(wide_entry.begin(), wide_entry.end());
            NVDX_ObjectHandle handle = nullptr;
            event("create_shader_call", {{"blob_bytes", blob.size()}});
            const auto status = create(device.Get(), blob.data(), NvU32(blob.size()),
                                       arithmetic?64u:NvU32(std::stoul(argv[4])), arithmetic?1u:NvU32(std::stoul(argv[5])),
                                       arithmetic?1u:NvU32(std::stoul(argv[6])), entry.c_str(), &handle);
            event("create_shader_result", {{"status", int(status)}, {"non_null_handle", handle != nullptr}});
            if (status != NVAPI_OK || !handle) return 2;
            if(arithmetic) {
                auto launch=reinterpret_cast<decltype(&NvAPI_D3D12_LaunchCubinShader)>(query(0x5c52bb86));
                if(!launch) throw std::runtime_error("No legacy kernel launch API");
                ComPtr<ID3D12CommandAllocator> allocator;
                ComPtr<ID3D12GraphicsCommandList> list;
                ComPtr<ID3D12Fence> fence;
                checked(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)),"allocator");
                checked(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)),"list");
                checked(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"fence");
                D3D12_RESOURCE_DESC buffer{}; buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
                buffer.Width=4096*4; buffer.Height=buffer.DepthOrArraySize=buffer.MipLevels=buffer.SampleDesc.Count=1;
                buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                auto allocate=[&](D3D12_HEAP_TYPE type,D3D12_RESOURCE_STATES state) {
                    D3D12_HEAP_PROPERTIES heap{}; heap.Type=type;
                    buffer.Flags=type==D3D12_HEAP_TYPE_DEFAULT?D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS:D3D12_RESOURCE_FLAG_NONE;
                    ComPtr<ID3D12Resource> resource;
                    checked(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,state,nullptr,IID_PPV_ARGS(&resource)),"buffer");
                    return resource;
                };
                auto gpu=allocate(D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_COPY_DEST);
                auto upload=allocate(D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
                auto readback=allocate(D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
                D3D12_RANGE empty{}; void* mapped=nullptr;
                checked(upload->Map(0,&empty,&mapped),"upload map");
                for(unsigned i=0;i<4096;++i) static_cast<unsigned*>(mapped)[i]=0xdeadbeefu;
                upload->Unmap(0,nullptr);
                list->CopyBufferRegion(gpu.Get(),0,upload.Get(),0,4096*4);
                D3D12_RESOURCE_BARRIER barrier{}; barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Transition={gpu.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
                list->ResourceBarrier(1,&barrier);
                const uint64_t address=gpu->GetGPUVirtualAddress();
                const auto launched=launch(list.Get(),handle,64,1,1,&address,sizeof(address));
                event("arithmetic_launch",{{"status",int(launched)},{"gpu_address_bits",64}});
                if(launched!=NVAPI_OK) { destroy(device.Get(),handle); return 2; }
                barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
                list->ResourceBarrier(1,&barrier);
                list->CopyBufferRegion(readback.Get(),0,gpu.Get(),0,4096*4);
                checked(list->Close(),"close list"); ID3D12CommandList* lists[]{list.Get()};
                queue->ExecuteCommandLists(1,lists); checked(queue->Signal(fence.Get(),1),"signal");
                const auto completion=CreateEventW(nullptr,FALSE,FALSE,nullptr);
                if(!completion) throw std::runtime_error("No fence event");
                const auto registered=fence->SetEventOnCompletion(1,completion);
                const auto waited=SUCCEEDED(registered)?WaitForSingleObject(completion,10000):WAIT_FAILED;
                CloseHandle(completion); checked(registered,"register fence");
                if(waited!=WAIT_OBJECT_0) throw std::runtime_error("Fence timeout");
                checked(device->GetDeviceRemovedReason(),"removed device");
                D3D12_RANGE range{0,4096*4}; checked(readback->Map(0,&range,&mapped),"readback map");
                unsigned matches=0;
                for(unsigned i=0;i<4096;++i) matches+=static_cast<unsigned*>(mapped)[i]==i*17u+23u;
                readback->Unmap(0,&empty);
                event("arithmetic_readback",{{"matches",matches},{"expected",4096}});
                if(matches!=4096) { destroy(device.Get(),handle); return 2; }
            }
            event("destroy_shader_call");
            const auto destroyed = destroy(device.Get(), handle);
            event("destroy_shader_result", {{"status", int(destroyed)}});
            return destroyed == NVAPI_OK ? 0 : 2;
        }
        auto create = reinterpret_cast<decltype(&NvAPI_D3D12_CreateCuModule)>(query(0xad1a677d));
        auto destroy = reinterpret_cast<decltype(&NvAPI_D3D12_DestroyCuModule)>(query(0x41c65285));
        if (!create || !destroy) throw std::runtime_error("No module lifecycle entry points");
        NVDX_ObjectHandle module = nullptr;
        event("create_module_call", {{"blob_bytes", blob.size()}});
        const auto status = create(device.Get(), blob.data(), NvU32(blob.size()), &module);
        event("create_module_result", {{"status", int(status)}, {"non_null_handle", module != nullptr}});
        if (status != NVAPI_OK || !module) return 2;
        event("destroy_module_call");
        const auto destroyed = destroy(device.Get(), module);
        event("destroy_module_result", {{"status", int(destroyed)}});
        // Keep the library loaded until process exit; driver-owned device objects
        // are released after this scope and must not outlive its code.
        return destroyed == NVAPI_OK ? 0 : 2;
    } catch (const std::exception& error) {
        event("error", {{"message", error.what()}});
        return 1;
    }
}
