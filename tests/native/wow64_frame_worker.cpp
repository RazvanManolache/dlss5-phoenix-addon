// Headless real-model probe, loaded only after native TLS/thread setup.
#define NOMINMAX
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "../../native_runtime/runtime_api.h"
#include "../../third_party/nlohmann/json.hpp"
#include <fstream>
#include <chrono>
#include <cmath>
#include <iostream>
#include <roapi.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <cstdio>
#include "wow64_umd_probe.h"
using Json = nlohmann::json;
namespace {
wchar_t fault_path[1024]{};
const char* fault_stage="startup";
volatile LONG fault_count=0;
using MappedName=DWORD(WINAPI*)(HANDLE,void*,char*,DWORD);
MappedName mapped_name=nullptr;
LONG CALLBACK record_fault(EXCEPTION_POINTERS* exception) {
    if(exception->ExceptionRecord->ExceptionCode!=EXCEPTION_ACCESS_VIOLATION || InterlockedIncrement(&fault_count)>8)
        return EXCEPTION_CONTINUE_SEARCH;
    const auto error=GetLastError();
    MEMORY_BASIC_INFORMATION info{};
    const auto address=exception->ExceptionRecord->ExceptionAddress;
    VirtualQuery(address,&info,sizeof(info));
    char module[1024]{};
    GetModuleFileNameA(static_cast<HMODULE>(info.AllocationBase),module,sizeof(module));
    if(!module[0] && mapped_name) mapped_name(GetCurrentProcess(),info.AllocationBase,module,sizeof(module));
    module[sizeof(module)-1]=0;
    IMAGE_DOS_HEADER dos{}; struct Header { DWORD signature; IMAGE_FILE_HEADER file; } header{};
    SIZE_T read=0;
    if(ReadProcessMemory(GetCurrentProcess(),info.AllocationBase,&dos,sizeof(dos),&read) && read==sizeof(dos) &&
        dos.e_magic==IMAGE_DOS_SIGNATURE && dos.e_lfanew>0 && dos.e_lfanew<1024*1024)
        ReadProcessMemory(GetCurrentProcess(),static_cast<char*>(info.AllocationBase)+dos.e_lfanew,&header,sizeof(header),&read);
    unsigned char instruction[16]{};
    ReadProcessMemory(GetCurrentProcess(),address,instruction,sizeof(instruction),&read);
    char code[33]{}; const char hex[]="0123456789abcdef";
    for(unsigned i=0;i<16;++i) { code[i*2]=hex[instruction[i]>>4]; code[i*2+1]=hex[instruction[i]&15]; }
    const auto base=strrchr(module,'\\'); auto name=base?base+1:module;
    for(auto p=name;*p;++p) if(!((*p>='A'&&*p<='Z')||(*p>='a'&&*p<='z')||(*p>='0'&&*p<='9')||*p=='.'||*p=='_'||*p=='-')) *p='?';
    uint64_t caller=0;
    ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<void*>(exception->ContextRecord->Rsp),&caller,sizeof(caller),&read);
    MEMORY_BASIC_INFORMATION caller_info{};
    VirtualQuery(reinterpret_cast<void*>(caller),&caller_info,sizeof(caller_info));
    char caller_module[1024]{};
    if(mapped_name) mapped_name(GetCurrentProcess(),caller_info.AllocationBase,caller_module,sizeof(caller_module));
    if(!caller_module[0]) GetModuleFileNameA(static_cast<HMODULE>(caller_info.AllocationBase),caller_module,sizeof(caller_module));
    caller_module[sizeof(caller_module)-1]=0;
    const auto caller_base=strrchr(caller_module,'\\'); auto caller_name=caller_base?caller_base+1:caller_module;
    for(auto p=caller_name;*p;++p) if(!((*p>='A'&&*p<='Z')||(*p>='a'&&*p<='z')||(*p>='0'&&*p<='9')||*p=='.'||*p=='_'||*p=='-')) *p='?';
    char line[1400]{};
    const auto length=_snprintf_s(line,sizeof(line),_TRUNCATE,
        "{\"stage\":\"%s\",\"module\":\"%s\",\"instruction\":%llu,\"module_base\":%llu,\"machine\":%u,\"timestamp\":%u,\"code\":\"%s\",\"rax\":%llu,\"rcx\":%llu,\"rdx\":%llu,\"caller_module\":\"%s\",\"caller\":%llu,\"caller_base\":%llu,\"operation\":%llu,\"fault_address\":%llu}\n",
        fault_stage,name,uint64_t(address),uint64_t(info.AllocationBase),
        unsigned(header.file.Machine),unsigned(header.file.TimeDateStamp),code,
        uint64_t(exception->ContextRecord->Rax),uint64_t(exception->ContextRecord->Rcx),uint64_t(exception->ContextRecord->Rdx),
        caller_name,caller,uint64_t(caller_info.AllocationBase),
        uint64_t(exception->ExceptionRecord->NumberParameters>0?exception->ExceptionRecord->ExceptionInformation[0]:0),
        uint64_t(exception->ExceptionRecord->NumberParameters>1?exception->ExceptionRecord->ExceptionInformation[1]:0));
    auto file=CreateFileW(fault_path,FILE_APPEND_DATA,FILE_SHARE_READ,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(file!=INVALID_HANDLE_VALUE) { DWORD written=0; if(length>0) WriteFile(file,line,DWORD(length),&written,nullptr); CloseHandle(file); }
    SetLastError(error);
    return EXCEPTION_CONTINUE_SEARCH;
}
}
extern "C" __declspec(dllexport) void PhoenixSetGdiRpc64(PhoenixGdiRpc* rpc) { wow64_umd_probe::gdi = rpc; }
extern "C" __declspec(dllexport) int PhoenixPortableFrameProbe64(const wchar_t* path) {
    Json report{{"passed", false}, {"pointer_bits", 64}, {"frames", Json::array()}};
    Json job;
    void* fault_handler=nullptr;
    std::filesystem::path receipt;
    auto progress=[&](const char* stage) {
        fault_stage=stage;
        report["stage"]=stage;
        if(!receipt.empty()) { std::ofstream file(receipt); file<<report.dump(2)<<'\n'; }
    };
    HRESULT com = E_FAIL, winrt = E_FAIL;
    try {
        std::ifstream input{std::filesystem::path(path)};
        input >> job;
        receipt=std::filesystem::u8path(job.at("output").get<std::string>());
        const auto fault_file=receipt.parent_path()/L"frame-faults.jsonl";
        if(wcscpy_s(fault_path,fault_file.c_str())) throw std::runtime_error("Fault receipt path too long");
        mapped_name=reinterpret_cast<MappedName>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"),"K32GetMappedFileNameA"));
        fault_handler=AddVectoredExceptionHandler(1,&record_fault);
        if(!fault_handler) throw std::runtime_error("Fault observer registration failed");
        if(wow64_umd_probe::gdi && job.contains("appcompat_offsets")) {
            const auto fields=job.at("appcompat_offsets").get<std::vector<DWORD>>();
            if(fields.size()==4) {
                const DWORD offsets[]{fields[0],fields[1],fields[2],fields[3]};
                if(PhoenixValidNativeCompatOffsets(offsets)) {
                    const auto peb=__readgsqword(0x60);
                    report["native_appcompat_observed"]=Json::array();
                    for(auto offset:offsets) report["native_appcompat_observed"].push_back(*reinterpret_cast<const uint64_t*>(peb+offset));
                }
            }
        }
        if(job.value("native_appcompat_view",false) && wow64_umd_probe::gdi) {
            const auto fields=job.at("appcompat_offsets").get<std::vector<DWORD>>();
            if(fields.size()!=4) throw std::runtime_error("Native compatibility symbols unavailable");
            const DWORD offsets[]{fields[0],fields[1],fields[2],fields[3]};
            if(!wow64_umd_probe::install_native_compat_view(job.at("ntdll_timestamp"),offsets))
                throw std::runtime_error("Native compatibility view unavailable");
            report["native_appcompat_before"]=std::vector<uint64_t>(wow64_umd_probe::compat_before,wow64_umd_probe::compat_before+4);
            report["native_appcompat_view"]=true;
        }
        if(job.contains("vulkan_icd")) {
            auto icd=std::filesystem::u8path(job.at("vulkan_icd").get<std::string>());
            if(!std::filesystem::is_regular_file(icd)) throw std::runtime_error("Missing Vulkan ICD manifest");
            SetEnvironmentVariableW(L"VK_DRIVER_FILES",icd.c_str());
            SetEnvironmentVariableW(L"VK_ICD_FILENAMES",icd.c_str());
            SetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE",L"~implicit~");
            SetEnvironmentVariableW(L"VK_LOADER_DEBUG",L"error,warn,driver");
        }
        progress("native_dependencies");
        wow64_runtime_diagnostics::native_registry=job.value("native_registry",false);
        wow64_umd_probe::bridge_dc_creation=job.value("bridge_create_dc",false);
        wow64_umd_probe::bridge_display_gdi=job.value("bridge_display_gdi",false);
        if(wow64_umd_probe::bridge_display_gdi && wow64_umd_probe::gdi && !wow64_umd_probe::install_display_exports())
            throw std::runtime_error("Native display export bridge unavailable");
        wow64_umd_probe::force_native_identity=job.value("force_native_identity",false);
        if(job.value("early_native_identity",false) && wow64_umd_probe::gdi && !wow64_umd_probe::install_identity_exports())
            throw std::runtime_error("Native identity export unavailable");
        if(wow64_umd_probe::gdi && job.value("native_registry",false)) {
            wow64_umd_probe::owner=GetCurrentThreadId();
            if(!wow64_umd_probe::install_gdi_export()) throw std::runtime_error("Native GDI export unavailable");
            for(const auto* name:{L"combase.dll",L"WinTypes.dll"}) {
                auto image=LoadLibraryExW(name,nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
                if(!image) throw std::runtime_error("Native WinRT dependency load failed");
                wow64_umd_probe::patch(image);
            }
            auto root=std::filesystem::u8path(job.at("config").at("runtime_root").get<std::string>());
            for(const auto& library:{root/L"phoenix_runtime_cpp.dll",root/L"phoenix_vulkan_cpp.dll",
                root/L"vulkan/phoenix_vulkan.dll"}) {
                auto image=LoadLibraryW(library.c_str());
                if(!image) throw std::runtime_error("Native runtime dependency load failed");
                wow64_umd_probe::patch(image);
            }
        }
        const unsigned width = job.at("width"), height = job.at("height");
        if (!width || !height || width > 1920 || height > 1080) throw std::runtime_error("Probe extent refused");
        if (wow64_umd_probe::gdi) {
            com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
            winrt = RoInitialize(RO_INIT_MULTITHREADED);
            report["com_status"] = uint32_t(com);
            report["winrt_status"] = uint32_t(winrt);
        }
        if (job.contains("vulkan_icd")) {
            const auto icd = std::filesystem::u8path(job.at("vulkan_icd").get<std::string>());
            if (!std::filesystem::is_regular_file(icd)) throw std::runtime_error("Missing Vulkan ICD manifest");
            SetEnvironmentVariableW(L"VK_DRIVER_FILES", icd.c_str());
            SetEnvironmentVariableW(L"VK_ICD_FILENAMES", icd.c_str());
            SetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", L"~implicit~");
            if (wow64_umd_probe::gdi && !wow64_umd_probe::gdi_export && !wow64_umd_probe::install_gdi_export())
                throw std::runtime_error("Native GDI export bridge unavailable");
        }
        phoenix_cpp::RuntimeApi runtime;
        if(job.contains("vulkan_icd")) {
            progress("vulkan_instance");
            const auto direct=job.value("direct_vulkan_icd",false);
            HMODULE loader{};
            if(direct) {
                const auto manifest=std::filesystem::u8path(job.at("vulkan_icd").get<std::string>());
                std::ifstream file(manifest); Json description; file>>description;
                const auto driver=std::filesystem::weakly_canonical(manifest.parent_path()/
                    std::filesystem::u8path(description.at("ICD").at("library_path").get<std::string>()));
                if(!wow64_umd_probe::x64_image(driver.c_str())) throw std::runtime_error("ICD is not a native x64 image");
                loader=wow64_umd_probe::gdi?wow64_umd_probe::load_ex(driver.c_str(),nullptr,0):LoadLibraryW(driver.c_str());
            } else loader=LoadLibraryExW(L"vulkan-1.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
            if(!loader) throw std::runtime_error("Vulkan loader unavailable");
            if(wow64_umd_probe::gdi) wow64_umd_probe::patch(loader);
            auto get=reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(loader,direct?"vk_icdGetInstanceProcAddr":"vkGetInstanceProcAddr"));
            if(!get) throw std::runtime_error("Missing Vulkan instance procedure lookup");
            if(direct) {
                using Negotiate=VkResult(VKAPI_PTR*)(uint32_t*);
                auto negotiate=reinterpret_cast<Negotiate>(GetProcAddress(loader,"vk_icdNegotiateLoaderICDInterfaceVersion"));
                uint32_t version=7;
                if(!negotiate || negotiate(&version)!=VK_SUCCESS) throw std::runtime_error("ICD interface negotiation failed");
                report["vulkan_icd_interface"]=version;
            }
            report["direct_vulkan_icd"]=direct;
            auto create=reinterpret_cast<PFN_vkCreateInstance>(get(nullptr,"vkCreateInstance"));
            if(!create) throw std::runtime_error("Missing Vulkan instance API");
            VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_3;
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; info.pApplicationInfo=&app;
            VkInstance instance{}; auto code=create(&info,nullptr,&instance);
            report["vulkan_instance_status"]=int(code); progress("vulkan_instance_result");
            if(code!=VK_SUCCESS) throw std::runtime_error("Vulkan instance creation refused: "+std::to_string(int(code)));
            auto destroy=reinterpret_cast<PFN_vkDestroyInstance>(get(instance,"vkDestroyInstance"));
            auto enumerate=reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(get(instance,"vkEnumeratePhysicalDevices"));
            auto properties=reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(get(instance,"vkGetPhysicalDeviceProperties2"));
            if(!destroy) throw std::runtime_error("Missing Vulkan instance destruction API");
            if(!enumerate || !properties) { destroy(instance,nullptr); throw std::runtime_error("Missing Vulkan device API"); }
            if(direct) {
                using EnumAdapter=VkResult(VKAPI_PTR*)(VkInstance,LUID,uint32_t*,VkPhysicalDevice*);
                auto enum_adapter=reinterpret_cast<EnumAdapter>(get(nullptr,"vk_icdEnumerateAdapterPhysicalDevices"));
                report["vulkan_adapters"]=Json::array();
                Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
                if(enum_adapter && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
                    for(unsigned i=0;i<16;++i) {
                        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
                        if(FAILED(factory->EnumAdapters1(i,&adapter))) break;
                        DXGI_ADAPTER_DESC1 desc{};
                        if(FAILED(adapter->GetDesc1(&desc)) || desc.VendorId!=0x10de) continue;
                        uint32_t found=0;
                        const auto status=enum_adapter(instance,desc.AdapterLuid,&found,nullptr);
                        report["vulkan_adapters"].push_back({{"vendor",desc.VendorId},{"device",desc.DeviceId},
                            {"luid_low",desc.AdapterLuid.LowPart},{"luid_high",desc.AdapterLuid.HighPart},
                            {"status",int(status)},{"count",found}});
                    }
                }
                progress("vulkan_adapter_result");
            }
            uint32_t count=0;
            report["vulkan_enumerate_status"]=int(enumerate(instance,&count,nullptr));
            report["vulkan_device_count"]=count;
            report["vulkan_devices"]=Json::array();
            if(count>16) { destroy(instance,nullptr); throw std::runtime_error("Vulkan device count exceeds probe limit"); }
            if(count) {
                std::vector<VkPhysicalDevice> devices(count);
                const auto status=enumerate(instance,&count,devices.data());
                report["vulkan_devices_status"]=int(status);
                if(status==VK_SUCCESS) for(uint32_t i=0;i<count;++i) {
                    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
                    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; props.pNext=&id;
                    properties(devices[i],&props);
                    std::string uuid; const char hex[]="0123456789abcdef";
                    for(auto byte:id.deviceUUID) { uuid+=hex[byte>>4]; uuid+=hex[byte&15]; }
                    report["vulkan_devices"].push_back({{"name",props.properties.deviceName},
                        {"vendor",props.properties.vendorID},{"device",props.properties.deviceID},{"uuid",uuid}});
                }
            }
            progress("vulkan_devices_result");
            destroy(instance,nullptr);
        }
        const auto config = job.at("config").dump();
        progress("model_create");
        runtime.open(std::filesystem::u8path(job.at("config").at("runtime_root").get<std::string>()), config, width, height);
        std::vector<float> color(size_t(width)*height*3), output(color.size());
        for (size_t i = 0; i < color.size(); ++i) color[i] = float((i*7+13)%251)/256.f;
        const auto metadata = Json{{"reset", true}, {"has_motion", false}, {"has_depth", false},
            {"controls", {{"seed", 13}, {"intensity", 1.f}}}}.dump();
        for (unsigned frame = 0; frame < job.value("frames", 3u); ++frame) {
            const auto begin = std::chrono::steady_clock::now();
            progress("model_frame");
            if (runtime.frame(runtime.instance, color.data(), nullptr, nullptr, nullptr, metadata.c_str(), output.data()))
                throw std::runtime_error(runtime.error());
            const auto wall = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            size_t changed = 0;
            for (size_t i = 0; i < output.size(); ++i) {
                if (!std::isfinite(output[i])) throw std::runtime_error("Nonfinite output");
                changed += output[i] != color[i];
            }
            auto state = Json::parse(runtime.status_json());
            if (!changed || state.value("state", "") != "completed" || !state.value("command_count", 0u))
                throw std::runtime_error("Missing real neural output");
            report["frames"].push_back({{"wall_ms", wall}, {"changed", changed},
                {"gpu_operator_ms", state.value("operator_ms", 0.)}, {"backend", state.value("backend", "")},
                {"commands", state.value("command_count", 0u)}});
        }
        report["passed"] = true;
    } catch (const std::exception& error) { report["failed_stage"]=report.value("stage",""); report["error"] = error.what(); }
    report["registry_calls"]=LONG(wow64_runtime_diagnostics::registry_calls);
    report["identity_queries"]=wow64_umd_probe::identity_queries;
    report["gdi_requests"]=wow64_umd_probe::gdi?wow64_umd_probe::gdi->requests:0;
    report["adapter_queries"]=Json::array();
    for(unsigned i=0;i<unsigned(wow64_umd_probe::adapter_query_count) && i<128;++i) {
        const auto& value=wow64_umd_probe::adapter_queries[i];
        report["adapter_queries"].push_back({{"type",value.type},{"bytes",value.bytes},{"status",uint32_t(value.status)},
            {"words",std::vector<DWORD>(value.words,value.words+16)}});
    }
    wow64_umd_probe::restore();
    wow64_umd_probe::restore_gdi_export();
    wow64_umd_probe::restore_identity_exports();
    wow64_umd_probe::restore_display_exports();
    if (SUCCEEDED(winrt)) RoUninitialize();
    if (SUCCEEDED(com)) CoUninitialize();
    const auto compat_restored=wow64_umd_probe::restore_native_compat_view();
    if(report.value("native_appcompat_view",false)) report["native_appcompat_restored"]=compat_restored;
    if(fault_handler) RemoveVectoredExceptionHandler(fault_handler);
    if (job.contains("output")) {
        std::ofstream file(std::filesystem::u8path(job.at("output").get<std::string>()));
        file << report.dump(2) << '\n';
        if (!file) return 2;
    }
    return report.value("passed", false) ? 0 : 2;
}
