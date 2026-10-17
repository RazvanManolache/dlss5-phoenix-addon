// Native WOW64 diagnostics and guarded driver allocation adaptation.
#pragma once
#include <windows.h>
#include <stdint.h>
#include <intrin.h>
#include "probe_contract.h"
namespace wow64_runtime_diagnostics {
struct Failure { unsigned operation; uint64_t bytes; DWORD flags, error; };
inline Failure failures[32]{};
inline volatile LONG failure_count = 0;
inline volatile LONG heap_calls = 0;
struct Reservation { uint64_t bytes, requested, initial, retried; DWORD flags,protect,error,retry_error,parameters,method; bool verified; };
inline Reservation reservations[32]{};
inline volatile LONG reservation_count=0;
inline bool retry_native_reservation=false;
inline void* WINAPI virtual_alloc2(HANDLE process,void* address,SIZE_T bytes,ULONG flags,ULONG protect,
    MEM_EXTENDED_PARAMETER* parameters,ULONG count) {
    using Allocate=void*(WINAPI*)(HANDLE,void*,SIZE_T,ULONG,ULONG,MEM_EXTENDED_PARAMETER*,ULONG);
    auto original=reinterpret_cast<Allocate>(GetProcAddress(GetModuleHandleW(L"kernelbase.dll"),"VirtualAlloc2"));
    if(!original) { SetLastError(ERROR_PROC_NOT_FOUND); return nullptr; }
    auto value=original(process,address,bytes,flags,protect,parameters,count);
    const auto error=value?0u:GetLastError(); auto retry_error=error;
    void* retried=nullptr;
    DWORD method=0; bool verified=false;
    SYSTEM_INFO system{}; GetNativeSystemInfo(&system);
    const auto maximum=uint64_t(system.lpMaximumApplicationAddress);
    const auto current=!process || process==GetCurrentProcess();
    if(!value && PhoenixMayRetryNativeReservation(retry_native_reservation,current,uint64_t(address),bytes,
        flags,protect,parameters!=nullptr,count,maximum)) {
        constexpr auto lower=PhoenixNativeReservationLower;
        {
            MEM_ADDRESS_REQUIREMENTS requirements{};
            requirements.LowestStartingAddress=reinterpret_cast<void*>(lower);
            requirements.HighestEndingAddress=system.lpMaximumApplicationAddress;
            MEM_EXTENDED_PARAMETER parameter{};
            parameter.Type=MemExtendedParameterAddressRequirements; parameter.Pointer=&requirements;
            retried=original(process,nullptr,bytes,flags,protect,&parameter,1);
            method=1;
            retry_error=retried?0u:GetLastError();
        }
        if(!retried && bytes<=maximum-lower+1) for(unsigned i=0;i<4 && !retried;++i) {
            const auto candidate=lower*(i+1);
            if(candidate>maximum || bytes>maximum-candidate+1) break;
            retried=VirtualAlloc(reinterpret_cast<void*>(candidate),bytes,flags,protect);
            method=2;
            retry_error=retried?0u:GetLastError();
        }
        if(retried) {
            MEMORY_BASIC_INFORMATION info{};
            verified=VirtualQuery(retried,&info,sizeof(info))==sizeof(info) && info.AllocationBase==retried &&
                info.State==MEM_RESERVE && info.RegionSize>=bytes && info.AllocationProtect==PAGE_NOACCESS;
            if(!verified) { VirtualFree(retried,0,MEM_RELEASE); retried=nullptr; retry_error=ERROR_INVALID_ADDRESS; }
        }
    }
    const auto index=InterlockedIncrement(&reservation_count)-1;
    if(index<32) reservations[index]={bytes,uint64_t(address),uint64_t(value),uint64_t(retried),flags,protect,error,retry_error,count,method,verified};
    if(retried) value=retried;
    SetLastError(value?0u:error); return value;
}
struct SlotCall { unsigned operation; DWORD slot,error; BOOL success; };
inline SlotCall slot_calls[64]{};
inline volatile LONG slot_count=0;
inline void slot_result(unsigned operation,DWORD slot,BOOL success,DWORD error) {
    const auto i=InterlockedIncrement(&slot_count)-1;
    if(i<64) slot_calls[i]={operation,slot,error,success};
}
inline DWORD WINAPI tls_alloc() {
    const auto slot=TlsAlloc(),error=GetLastError();
    slot_result(1,slot,slot!=TLS_OUT_OF_INDEXES,error); SetLastError(error); return slot;
}
inline DWORD WINAPI fls_alloc(PFLS_CALLBACK_FUNCTION callback) {
    const auto slot=FlsAlloc(callback),error=GetLastError();
    slot_result(2,slot,slot!=FLS_OUT_OF_INDEXES,error); SetLastError(error); return slot;
}
inline BOOL WINAPI tls_set(DWORD slot,void* value) {
    const auto ok=TlsSetValue(slot,value); const auto error=GetLastError();
    slot_result(3,slot,ok,error); SetLastError(error); return ok;
}
inline BOOL WINAPI fls_set(DWORD slot,void* value) {
    const auto ok=FlsSetValue(slot,value); const auto error=GetLastError();
    slot_result(4,slot,ok,error); SetLastError(error); return ok;
}
inline HLOCAL WINAPI local_alloc(UINT flags,SIZE_T bytes) {
    auto value=LocalAlloc(flags,bytes); const auto error=GetLastError();
    InterlockedIncrement(&heap_calls);
    if(!value) {
        const auto index=InterlockedIncrement(&failure_count)-1;
        if(index<32) failures[index]={4,bytes,flags,error};
    }
    SetLastError(error); return value;
}
struct Thread {
    uint64_t start, stack, handle, instruction, entered_teb, tls_vector, tls_after;
    DWORD flags, error, context_error, exit_code, tls_status;
    WORD segment;
    volatile LONG entered;
};
inline Thread threads[32]{};
inline volatile LONG thread_count = 0;
inline bool inspect_threads = false;
inline bool wrap_threads = false;
using AllocateTls=LONG(NTAPI*)();
using FreeTls=void(NTAPI*)();
inline AllocateTls allocate_tls=nullptr;
inline FreeTls free_tls=nullptr;
inline bool repair_thread_tls=false;
struct Start { LPTHREAD_START_ROUTINE procedure; void* argument; LONG index; };
inline DWORD WINAPI thread_start(void* parameter) {
    const auto context=*static_cast<Start*>(parameter);
    HeapFree(GetProcessHeap(),0,parameter);
    const auto previous_tls=__readgsqword(0x58);
    if(context.index<32) {
        auto& row=threads[context.index];
        row.entered_teb=__readgsqword(0x30); row.tls_vector=__readgsqword(0x58);
        InterlockedExchange(&row.entered,1);
    }
    if(repair_thread_tls) {
        const auto status=allocate_tls();
        if(context.index<32) { threads[context.index].tls_status=ULONG(status); threads[context.index].tls_after=__readgsqword(0x58); }
        if(status<0) return ULONG(status);
    }
    const auto result=context.procedure(context.argument);
    if(repair_thread_tls) { free_tls(); __writegsqword(0x58,previous_tls); }
    if(context.index<32) threads[context.index].exit_code=result;
    return result;
}
inline bool native_registry = false;
inline volatile LONG registry_calls = 0;
inline LSTATUS WINAPI reg_open_w(HKEY key,const wchar_t* name,DWORD options,REGSAM access,HKEY* value) {
    InterlockedIncrement(&registry_calls);
    if(native_registry && !(access&(KEY_WOW64_32KEY|KEY_WOW64_64KEY))) access|=KEY_WOW64_64KEY;
    return RegOpenKeyExW(key,name,options,access,value);
}
inline LSTATUS WINAPI reg_open_a(HKEY key,const char* name,DWORD options,REGSAM access,HKEY* value) {
    InterlockedIncrement(&registry_calls);
    if(native_registry && !(access&(KEY_WOW64_32KEY|KEY_WOW64_64KEY))) access|=KEY_WOW64_64KEY;
    return RegOpenKeyExA(key,name,options,access,value);
}
inline LSTATUS WINAPI reg_get_w(HKEY key,const wchar_t* subkey,const wchar_t* name,DWORD flags,
    DWORD* type,void* data,DWORD* bytes) {
    InterlockedIncrement(&registry_calls);
    if(native_registry && !(flags&(RRF_SUBKEY_WOW6432KEY|RRF_SUBKEY_WOW6464KEY))) flags|=RRF_SUBKEY_WOW6464KEY;
    return RegGetValueW(key,subkey,name,flags,type,data,bytes);
}
struct Mapping { uint64_t bytes, requested, returned; DWORD error; };
inline Mapping mappings[32]{};
inline volatile LONG mapping_count=0;
inline void* WINAPI map_file_ex(HANDLE handle,DWORD access,DWORD high,DWORD low,SIZE_T bytes,void* address) {
    auto value=MapViewOfFileEx(handle,access,high,low,bytes,address); const auto error=GetLastError();
    const auto index=InterlockedIncrement(&mapping_count)-1;
    if(index<32) mappings[index]={bytes,uint64_t(address),uint64_t(value),value?0u:error};
    SetLastError(error); return value;
}
inline void* WINAPI map_file(HANDLE handle,DWORD access,DWORD high,DWORD low,SIZE_T bytes) {
    return map_file_ex(handle,access,high,low,bytes,nullptr);
}
struct GdiCall { unsigned operation; uint64_t handle; int argument, result; DWORD error; };
inline GdiCall gdi_calls[64]{};
inline volatile LONG gdi_count = 0;
inline HDC WINAPI create_dc(const wchar_t* driver,const wchar_t* device,const wchar_t* output,const DEVMODEW* mode) {
    const auto value=CreateDCW(driver,device,output,mode); const auto error=GetLastError();
    const auto index=InterlockedIncrement(&gdi_count)-1;
    if(index<64) gdi_calls[index]={1,uint64_t(value),0,value?1:0,error};
    SetLastError(error); return value;
}
inline HDC WINAPI create_dc_a(const char* driver,const char* device,const char* output,const DEVMODEA* mode) {
    const auto value=CreateDCA(driver,device,output,mode); const auto error=GetLastError();
    const auto index=InterlockedIncrement(&gdi_count)-1;
    if(index<64) gdi_calls[index]={3,uint64_t(value),0,value?1:0,error};
    SetLastError(error); return value;
}
inline int WINAPI device_caps(HDC handle,int index) {
    const auto value=GetDeviceCaps(handle,index); const auto error=GetLastError();
    const auto position=InterlockedIncrement(&gdi_count)-1;
    if(position<64) gdi_calls[position]={2,uint64_t(handle),index,value,error};
    SetLastError(error); return value;
}
inline void failure(unsigned operation, uint64_t bytes, DWORD flags, DWORD error) {
    const auto index = InterlockedIncrement(&failure_count)-1;
    if (index < 32) failures[index] = {operation, bytes, flags, error};
}
inline void* WINAPI heap_alloc(HANDLE heap, DWORD flags, SIZE_T bytes) {
    auto value = HeapAlloc(heap, flags, bytes);
    const auto error = GetLastError();
    InterlockedIncrement(&heap_calls);
    if (!value) failure(1, bytes, flags, error);
    SetLastError(error); return value;
}
inline void* WINAPI heap_realloc(HANDLE heap, DWORD flags, void* before, SIZE_T bytes) {
    auto value = HeapReAlloc(heap, flags, before, bytes);
    const auto error = GetLastError();
    InterlockedIncrement(&heap_calls);
    if (!value) failure(2, bytes, flags, error);
    SetLastError(error); return value;
}
inline HANDLE WINAPI heap_create(DWORD flags, SIZE_T initial, SIZE_T maximum) {
    auto value = HeapCreate(flags, initial, maximum);
    const auto error = GetLastError();
    InterlockedIncrement(&heap_calls);
    if (!value) failure(3, maximum, flags, error);
    SetLastError(error); return value;
}
inline HANDLE WINAPI create_thread(SECURITY_ATTRIBUTES* attributes, SIZE_T stack,
    LPTHREAD_START_ROUTINE start, void* argument, DWORD flags, DWORD* identifier) {
    const auto index = InterlockedIncrement(&thread_count)-1;
    Start* context=nullptr;
    if(wrap_threads) {
        context=static_cast<Start*>(HeapAlloc(GetProcessHeap(),0,sizeof(Start)));
        if(!context) { SetLastError(ERROR_NOT_ENOUGH_MEMORY); return nullptr; }
        *context={start,argument,index};
    }
    if(index<32) { threads[index].start=uint64_t(start); threads[index].stack=stack; threads[index].flags=flags; threads[index].exit_code=STILL_ACTIVE; }
    auto value = CreateThread(attributes, stack, context?&thread_start:start, context?context:argument,
        inspect_threads ? flags|CREATE_SUSPENDED : flags, identifier);
    const auto error = GetLastError();
    if(!value && context) HeapFree(GetProcessHeap(),0,context);
    CONTEXT registers{}; registers.ContextFlags = CONTEXT_CONTROL;
    DWORD context_error = 0;
    if (value && inspect_threads) {
        if (!GetThreadContext(value,&registers)) context_error=GetLastError();
        if (!(flags&CREATE_SUSPENDED) && ResumeThread(value)==DWORD(-1)) context_error=GetLastError();
    }
    if (index < 32) {
        auto& row=threads[index]; row.handle=uint64_t(value); row.instruction=registers.Rip;
        row.error=value?0u:error; row.context_error=context_error; row.segment=registers.SegCs;
    }
    SetLastError(error); return value;
}
inline void* replacement(const char* name) {
    if (!lstrcmpA(name,"VirtualAlloc2")) return reinterpret_cast<void*>(&virtual_alloc2);
    if (!lstrcmpA(name,"TlsAlloc")) return reinterpret_cast<void*>(&tls_alloc);
    if (!lstrcmpA(name,"FlsAlloc")) return reinterpret_cast<void*>(&fls_alloc);
    if (!lstrcmpA(name,"TlsSetValue")) return reinterpret_cast<void*>(&tls_set);
    if (!lstrcmpA(name,"FlsSetValue")) return reinterpret_cast<void*>(&fls_set);
    if (!lstrcmpA(name,"LocalAlloc")) return reinterpret_cast<void*>(&local_alloc);
    if (!lstrcmpA(name,"RegOpenKeyExW")) return reinterpret_cast<void*>(&reg_open_w);
    if (!lstrcmpA(name,"RegOpenKeyExA")) return reinterpret_cast<void*>(&reg_open_a);
    if (!lstrcmpA(name,"RegGetValueW")) return reinterpret_cast<void*>(&reg_get_w);
    if (!lstrcmpA(name,"MapViewOfFileEx")) return reinterpret_cast<void*>(&map_file_ex);
    if (!lstrcmpA(name,"MapViewOfFile")) return reinterpret_cast<void*>(&map_file);
    if (!lstrcmpA(name,"HeapAlloc") || !lstrcmpA(name,"RtlAllocateHeap")) return reinterpret_cast<void*>(&heap_alloc);
    if (!lstrcmpA(name,"HeapReAlloc") || !lstrcmpA(name,"RtlReAllocateHeap")) return reinterpret_cast<void*>(&heap_realloc);
    if (!lstrcmpA(name,"HeapCreate")) return reinterpret_cast<void*>(&heap_create);
    if (!lstrcmpA(name,"CreateThread")) return reinterpret_cast<void*>(&create_thread);
    if (!lstrcmpA(name,"CreateDCW")) return reinterpret_cast<void*>(&create_dc);
    if (!lstrcmpA(name,"CreateDCA")) return reinterpret_cast<void*>(&create_dc_a);
    if (!lstrcmpA(name,"GetDeviceCaps")) return reinterpret_cast<void*>(&device_caps);
    return nullptr;
}
}
