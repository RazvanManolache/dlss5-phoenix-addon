// Sample the calling benchmark process; no graphics device or inference is created.
#define NOMINMAX
#include <windows.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <psapi.h>
#include "../../third_party/nlohmann/json.hpp"
using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllexport) const char* phoenix_benchmark_memory(unsigned vendor) {
    static thread_local std::string text;
    nlohmann::json report;
    PROCESS_MEMORY_COUNTERS_EX process{};process.cb=sizeof(process);
    if(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&process),sizeof(process))) {
        report["host_private_bytes"]=process.PrivateUsage;
        report["host_working_set_bytes"]=process.WorkingSetSize;
        report["host_peak_working_set_bytes"]=process.PeakWorkingSetSize;
    }
    if(vendor) {
        ComPtr<IDXGIFactory6> factory;
        if(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
            for(unsigned index=0;;++index) {
                ComPtr<IDXGIAdapter1> adapter;
                if(factory->EnumAdapterByGpuPreference(index,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,IID_PPV_ARGS(&adapter))==DXGI_ERROR_NOT_FOUND)break;
                DXGI_ADAPTER_DESC1 desc{};
                if(!adapter || FAILED(adapter->GetDesc1(&desc)) || desc.VendorId!=vendor)continue;
                ComPtr<IDXGIAdapter3> measured;
                DXGI_QUERY_VIDEO_MEMORY_INFO local{},nonlocal{};
                if(SUCCEEDED(adapter.As(&measured)) && SUCCEEDED(measured->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&local)) &&
                   SUCCEEDED(measured->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,&nonlocal))) {
                    report["device_local_bytes"]=local.CurrentUsage;
                    report["device_nonlocal_bytes"]=nonlocal.CurrentUsage;
                    report["vendor_id"]=vendor;
                }
                break;
            }
    }
    report["scope"]="Current-process DXGI residency and process RAM, sampled while resources are alive; includes harness/driver caches";
    text=report.dump();return text.c_str();
}
