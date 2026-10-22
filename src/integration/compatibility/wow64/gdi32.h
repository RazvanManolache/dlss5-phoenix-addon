#pragma once
#include "probe_contract.h"
#include <windows.h>
#include <d3dkmthk.h>
#include <vector>
#include <stdexcept>
#pragma comment(lib,"gdi32.lib")
#ifndef _WIN64
namespace phoenix_wow64 {
inline void service_gdi(PhoenixGdiRpc& rpc,HMODULE extension) {
    using Read=BOOL(__cdecl*)(HANDLE,uint64_t,void*,SIZE_T,SIZE_T*);
    using Write=BOOL(__cdecl*)(HANDLE,uint64_t,const void*,SIZE_T,SIZE_T*);
    auto read=reinterpret_cast<Read>(GetProcAddress(extension,"ReadProcessMemory64"));
    auto write=reinterpret_cast<Write>(GetProcAddress(extension,"WriteProcessMemory64"));
    const auto operation=InterlockedExchange(reinterpret_cast<volatile LONG*>(&rpc.pending),0);
    rpc.result=0; SetLastError(ERROR_INVALID_PARAMETER);
    if(operation==1) rpc.result=DeleteDC(reinterpret_cast<HDC>(uintptr_t(rpc.handle)))!=0;
    else if(operation==2) {
        PhoenixGdiCreateDC request{};
        if(read&&read(GetCurrentProcess(),rpc.handle,&request,sizeof(request),nullptr)&&PhoenixValidGdiCreateDC(request,sizeof(DEVMODEA)))
            rpc.result=uint64_t(uintptr_t(CreateDCA(request.present&1?request.driver:nullptr,request.present&2?request.device:nullptr,
                request.present&4?request.output:nullptr,request.mode_bytes?reinterpret_cast<const DEVMODEA*>(request.mode):nullptr)));
    } else if(operation==3) rpc.result=uint32_t(GetDeviceCaps(reinterpret_cast<HDC>(uintptr_t(uint32_t(rpc.handle))),int(rpc.handle>>32)));
    else if(operation==4) rpc.result=uint32_t(uintptr_t(GetDC(reinterpret_cast<HWND>(uintptr_t(uint32_t(rpc.handle))))));
    else if(operation==5) rpc.result=uint32_t(ReleaseDC(reinterpret_cast<HWND>(uintptr_t(uint32_t(rpc.handle))),reinterpret_cast<HDC>(uintptr_t(uint32_t(rpc.handle>>32)))));
    else if(operation==6) {
        PhoenixGdiDisplayDevice request{}; DISPLAY_DEVICEA display{}; display.cb=sizeof(display);
        if(read&&write&&read(GetCurrentProcess(),rpc.handle,&request,sizeof(request),nullptr)&&PhoenixValidGdiDisplayDevice(request)) {
            const auto ok=EnumDisplayDevicesA(request.present?request.device:nullptr,request.index,&display,request.flags);
            const auto error=GetLastError();
            if(ok&&write(GetCurrentProcess(),rpc.handle+offsetof(PhoenixGdiDisplayDevice,display),&display,sizeof(display),nullptr)) rpc.result=1;
            SetLastError(error);
        }
    } else if(operation==7) {
        PhoenixKmtEscape request{}; rpc.result=0xc000000du;
        if(read&&write&&read(GetCurrentProcess(),rpc.handle,&request,sizeof(request),nullptr)&&PhoenixValidKmtEscape(request)) {
            std::vector<uint8_t> payload(request.bytes);
            if(!request.bytes||read(GetCurrentProcess(),request.data,payload.data(),request.bytes,nullptr)) {
                D3DKMT_ESCAPE escape{}; escape.hAdapter=request.adapter; escape.hDevice=request.device;
                escape.Type=static_cast<D3DKMT_ESCAPETYPE>(request.type); escape.Flags.Value=request.flags;
                escape.hContext=request.context; escape.PrivateDriverDataSize=request.bytes; escape.pPrivateDriverData=request.bytes?payload.data():nullptr;
                rpc.result=uint32_t(D3DKMTEscape(&escape));
                if(request.bytes&&!write(GetCurrentProcess(),request.data,payload.data(),request.bytes,nullptr)) rpc.result=0xc0000005u;
            }
        }
    } else if(operation==8) {
        rpc.result=TlsAlloc();
        SetLastError(rpc.result==TLS_OUT_OF_INDEXES?ERROR_NOT_ENOUGH_MEMORY:ERROR_SUCCESS);
    } else if(operation==9) {
        rpc.result=TlsFree(DWORD(rpc.handle));
        if(rpc.result)SetLastError(ERROR_SUCCESS);
    }
    rpc.error=GetLastError(); ++rpc.requests; SetEvent(reinterpret_cast<HANDLE>(uintptr_t(rpc.response_event)));
}

inline PhoenixGdiRpc& process_native_service(HMODULE extension) {
    // Native driver imports can retain callbacks after an inference worker is
    // retired. One pinned, process-owned x86 service keeps those targets valid,
    // and coordinates TLS indices with the game for every native worker.
    struct Service { PhoenixGdiRpc rpc{}; HMODULE extension=nullptr; };
    static Service* service=[&] {
        HMODULE pinned=nullptr;
        if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,
                reinterpret_cast<LPCWSTR>(&service_gdi),&pinned))
            throw std::runtime_error("Cannot retain native callback service module");
        auto* value=new Service;
        value->extension=extension;
        value->rpc.request_event=uintptr_t(CreateEventW(nullptr,FALSE,FALSE,nullptr));
        value->rpc.response_event=uintptr_t(CreateEventW(nullptr,FALSE,FALSE,nullptr));
        auto close=[&] {
            for(auto handle:{value->rpc.request_event,value->rpc.response_event})
                if(handle)CloseHandle(reinterpret_cast<HANDLE>(uintptr_t(handle)));
            delete value;
        };
        if(!value->rpc.request_event||!value->rpc.response_event) {
            close();throw std::runtime_error("Cannot create native callback service events");
        }
        auto thread=CreateThread(nullptr,256*1024,[](void* parameter)->DWORD {
            auto& value=*static_cast<Service*>(parameter);
            while(WaitForSingleObject(reinterpret_cast<HANDLE>(uintptr_t(value.rpc.request_event)),INFINITE)==WAIT_OBJECT_0) {
                try {service_gdi(value.rpc,value.extension);}
                catch(...) {
                    value.rpc.result=0;value.rpc.error=ERROR_NOT_ENOUGH_MEMORY;
                    SetEvent(reinterpret_cast<HANDLE>(uintptr_t(value.rpc.response_event)));
                }
            }
            return 0;
        },value,0,nullptr);
        if(!thread){close();throw std::runtime_error("Cannot start native callback service");}
        CloseHandle(thread);
        return value;
    }();
    return service->rpc;
}
}
#endif
