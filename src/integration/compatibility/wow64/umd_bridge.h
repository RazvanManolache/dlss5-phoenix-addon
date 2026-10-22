// Opt-in native WOW64 import adaptation. Scoped to the serialized native worker.
#pragma once
#include <windows.h>
#include "probe_contract.h"
#include "runtime_diagnostics.h"
static_assert(sizeof(void*) == 8, "Only the disposable native worker can use this probe");

namespace wow64_umd_probe {
inline DWORD owner = 0;
inline unsigned redirects = 0;
struct Slot { void** address; void* original; };
inline constexpr unsigned import_capacity = 1024;
inline Slot slots[import_capacity]{};
inline unsigned slot_count = 0;
inline PhoenixGdiRpc* gdi = nullptr;
struct Allocation { uint64_t requested, bytes, returned; DWORD flags, error; };
inline Allocation allocations[64]{};
inline unsigned allocation_count = 0;
inline unsigned identity_queries = 0;
inline bool force_native_identity = false;
inline bool bridge_dc_creation = false;
inline bool bridge_display_gdi = false;
inline uint64_t* compat_slots[4]{};
inline uint64_t compat_before[4]{};
inline bool install_native_compat_view(DWORD stamp,const DWORD (&offsets)[4]) {
    auto ntdll=GetModuleHandleW(L"ntdll.dll");
    auto base=reinterpret_cast<unsigned char*>(ntdll);
    auto dos=reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    auto pe=reinterpret_cast<IMAGE_NT_HEADERS64*>(base+dos->e_lfanew);
    if(pe->FileHeader.Machine!=IMAGE_FILE_MACHINE_AMD64 || pe->FileHeader.TimeDateStamp!=stamp) return false;
    const auto peb=__readgsqword(0x60);
    using Query=LONG(NTAPI*)(HANDLE,ULONG,void*,ULONG,ULONG*);
    auto query=reinterpret_cast<Query>(GetProcAddress(ntdll,"NtQueryInformationProcess"));
    uint64_t wow_peb=0;
    if(!peb || !query || query(GetCurrentProcess(),26,&wow_peb,sizeof(wow_peb),nullptr)<0 || !wow_peb || wow_peb==peb) return false;
    if(!PhoenixValidNativeCompatOffsets(offsets)) return false;
    // An inherited x86 compatibility blob is not a native x64 shim context.
    for(unsigned i=0;i<4;++i) {
        compat_slots[i]=reinterpret_cast<uint64_t*>(peb+offsets[i]);
        compat_before[i]=*compat_slots[i]; *compat_slots[i]=0;
    }
    return true;
}
inline bool restore_native_compat_view() {
    bool ok=true;
    for(unsigned i=0;i<4;++i) if(compat_slots[i]) {
        *compat_slots[i]=compat_before[i]; ok &= *compat_slots[i]==compat_before[i]; compat_slots[i]=nullptr;
    }
    return ok;
}
struct ExportPatch { DWORD* slot; DWORD before; void* thunk; void* thunk_target; };
inline ExportPatch gdi_export_patch{};
inline ExportPatch identity_exports[2]{};
inline ExportPatch display_exports[4]{};
inline BOOL WINAPI native_identity(HANDLE,BOOL*);
using Wow64Process2=BOOL(WINAPI*)(HANDLE,USHORT*,USHORT*);
inline Wow64Process2 original_identity2=nullptr;
inline BOOL WINAPI native_identity2(HANDLE process,USHORT* machine,USHORT* native) {
    ++identity_queries;
    if(machine && GetProcessId(process)==GetCurrentProcessId()) {
        *machine=IMAGE_FILE_MACHINE_UNKNOWN;
        if(native) *native=IMAGE_FILE_MACHINE_AMD64;
        return TRUE;
    }
    return original_identity2?original_identity2(process,machine,native):FALSE;
}
inline void* allocate_export_stub(uintptr_t image, SIZE_T image_bytes) {
    // An export RVA can address any byte in the next 4 GiB. The former scan
    // stopped after 128 MiB and below 2 GiB, rejecting valid native image
    // layouts in a fragmented WOW64 process (or an image above that ceiling).
    SYSTEM_INFO system{};
    GetNativeSystemInfo(&system);
    const auto granularity = uintptr_t(system.dwAllocationGranularity);
    const auto maximum = uintptr_t(system.lpMaximumApplicationAddress);
    if (!granularity || image > maximum || image_bytes > maximum-image) return nullptr;
    const auto rva_limit = maximum-image < UINT32_MAX ? maximum : image+UINT32_MAX;
    auto address = (image+image_bytes+granularity-1) & ~(granularity-1);
    while (address <= rva_limit && rva_limit-address >= 4095) {
        MEMORY_BASIC_INFORMATION region{};
        if (!VirtualQuery(reinterpret_cast<void*>(address), &region, sizeof(region))) break;
        const auto region_start = uintptr_t(region.BaseAddress);
        if (!region.RegionSize || region.RegionSize > maximum-region_start) break;
        const auto region_end = region_start+region.RegionSize;
        if (region.State == MEM_FREE && region_end-address >= 4096) {
            if (auto allocation = VirtualAlloc(reinterpret_cast<void*>(address), 4096,
                    MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE)) return allocation;
            // Another thread may have reserved the free range after the query.
            address += granularity;
        } else {
            address = (region_end+granularity-1) & ~(granularity-1);
        }
    }
    SetLastError(ERROR_NOT_ENOUGH_MEMORY);
    return nullptr;
}
inline bool redirect_export(HMODULE image,const char* name,void* destination,ExportPatch& patch) {
    auto base=reinterpret_cast<unsigned char*>(image);
    if(!base) return false;
    auto dos=reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    auto nt=reinterpret_cast<IMAGE_NT_HEADERS64*>(base+dos->e_lfanew);
    if(nt->FileHeader.Machine!=IMAGE_FILE_MACHINE_AMD64) return false;
    const auto directory=nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
    auto table=reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(base+directory.VirtualAddress);
    auto names=reinterpret_cast<DWORD*>(base+table->AddressOfNames);
    auto ordinals=reinterpret_cast<WORD*>(base+table->AddressOfNameOrdinals);
    auto functions=reinterpret_cast<DWORD*>(base+table->AddressOfFunctions);
    DWORD* slot=nullptr;
    for(unsigned i=0;i<table->NumberOfNames;++i)
        if(!lstrcmpA(reinterpret_cast<char*>(base+names[i]),name)) { slot=functions+ordinals[i]; break; }
    if(!slot) return false;
    auto target=uintptr_t(destination);
    if(target<=uintptr_t(base) || target-uintptr_t(base)>UINT32_MAX) {
        // Driver imports may retain this address after the export is restored.
        // Publish the stub once, then reuse it unchanged across worker restarts.
        // Rewriting its now-RX page caused an access violation on the second
        // initialization when ASLR required an export-relative trampoline.
        if(patch.thunk && patch.thunk_target!=destination) return false;
        if(!patch.thunk) {
            auto allocation=allocate_export_stub(uintptr_t(base),nt->OptionalHeader.SizeOfImage);
            if(!allocation) return false;
            auto code=static_cast<unsigned char*>(allocation); code[0]=0x48; code[1]=0xb8;
            for(unsigned i=0;i<8;++i) code[2+i]=static_cast<unsigned char>(target>>(i*8));
            code[10]=0xff; code[11]=0xe0; DWORD old=0;
            if(!VirtualProtect(code,4096,PAGE_EXECUTE_READ,&old)) {
                VirtualFree(allocation,0,MEM_RELEASE); return false;
            }
            FlushInstructionCache(GetCurrentProcess(),code,12);
            patch.thunk=allocation; patch.thunk_target=destination;
        }
        target=uintptr_t(patch.thunk);
    }
    const auto rva=target-uintptr_t(base);
    if(!rva || rva>UINT32_MAX || (rva>=directory.VirtualAddress && rva<directory.VirtualAddress+directory.Size)) return false;
    DWORD old=0,ignored=0;
    if(!VirtualProtect(slot,sizeof(*slot),PAGE_READWRITE,&old)) return false;
    patch.slot=slot; patch.before=*slot;
    InterlockedExchange(reinterpret_cast<volatile LONG*>(slot),LONG(rva));
    VirtualProtect(slot,sizeof(*slot),old,&ignored); return true;
}
inline bool install_identity_exports() {
    auto base=GetModuleHandleW(L"kernelbase.dll");
    original_identity2=reinterpret_cast<Wow64Process2>(GetProcAddress(base,"IsWow64Process2"));
    return redirect_export(base,"IsWow64Process",reinterpret_cast<void*>(&native_identity),identity_exports[0]) &&
        (!original_identity2 || redirect_export(base,"IsWow64Process2",reinterpret_cast<void*>(&native_identity2),identity_exports[1]));
}
inline void restore_identity_exports() {
    for(auto& patch:identity_exports) if(patch.slot) {
        DWORD old=0,ignored=0;
        if(VirtualProtect(patch.slot,sizeof(*patch.slot),PAGE_READWRITE,&old)) {
            InterlockedExchange(reinterpret_cast<volatile LONG*>(patch.slot),LONG(patch.before));
            VirtualProtect(patch.slot,sizeof(*patch.slot),old,&ignored);
        }
        patch.slot=nullptr;
        // Driver imports can cache these targets. Keep owned RX thunks until exit.
    }
}
inline void restore_display_exports() {
    for(auto& patch:display_exports) if(patch.slot) {
        DWORD old=0,ignored=0;
        if(VirtualProtect(patch.slot,sizeof(*patch.slot),PAGE_READWRITE,&old)) {
            InterlockedExchange(reinterpret_cast<volatile LONG*>(patch.slot),LONG(patch.before));
            VirtualProtect(patch.slot,sizeof(*patch.slot),old,&ignored);
        }
        patch.slot=nullptr;
    }
}
inline HMODULE native_main_image = nullptr;
inline volatile LONG main_image_queries = 0;
inline HMODULE WINAPI module_w(const wchar_t* name) {
    if(!name && native_main_image) { InterlockedIncrement(&main_image_queries); return native_main_image; }
    return GetModuleHandleW(name);
}
inline HMODULE WINAPI module_a(const char* name) {
    if(!name && native_main_image) { InterlockedIncrement(&main_image_queries); return native_main_image; }
    return GetModuleHandleA(name);
}
inline DWORD WINAPI filename_w(HMODULE image,wchar_t* output,DWORD size) {
    if(!image && native_main_image) { InterlockedIncrement(&main_image_queries); image=native_main_image; }
    return GetModuleFileNameW(image,output,size);
}
inline DWORD WINAPI filename_a(HMODULE image,char* output,DWORD size) {
    if(!image && native_main_image) { InterlockedIncrement(&main_image_queries); image=native_main_image; }
    return GetModuleFileNameA(image,output,size);
}
inline void* image_replacement(const char* name) {
    if(!native_main_image) return nullptr;
    if(!lstrcmpA(name,"GetModuleHandleW")) return reinterpret_cast<void*>(&module_w);
    if(!lstrcmpA(name,"GetModuleHandleA")) return reinterpret_cast<void*>(&module_a);
    if(!lstrcmpA(name,"GetModuleFileNameW")) return reinterpret_cast<void*>(&filename_w);
    if(!lstrcmpA(name,"GetModuleFileNameA")) return reinterpret_cast<void*>(&filename_a);
    return nullptr;
}
struct Ioctl { DWORD code, input_bytes, output_bytes, error; BOOL result; };
inline Ioctl ioctls[64]{};
inline unsigned ioctl_count = 0;
struct Packet {
    DWORD adapter,device,flags,code, bytes, words, before[16], after[16];
};
inline Packet packets[16]{};
inline volatile LONG packet_count=0;
inline void snapshot_words(const void* data, DWORD bytes, DWORD (&words)[16]) {
    SIZE_T read=0;
    ReadProcessMemory(GetCurrentProcess(),data,words,bytes<sizeof(words)?bytes:sizeof(words),&read);
}
using KmtFunction=LONG(WINAPI*)(void*);
inline constexpr const char* kmt_names[]{"D3DKMTCreateDevice","D3DKMTCreateContextVirtual",
    "D3DKMTCreateAllocation","D3DKMTCreateAllocation2","D3DKMTReserveGpuVirtualAddress",
    "D3DKMTMapGpuVirtualAddress","D3DKMTEscape","D3DKMTCreatePagingQueue",
    "D3DKMTOpenAdapterFromLuid","D3DKMTQueryAdapterInfo","D3DKMTMakeResident"};
inline KmtFunction kmt_originals[11]{};
struct KmtCall { unsigned function; LONG status; };
inline KmtCall kmt_calls[128]{};
inline volatile LONG kmt_count=0;
inline volatile LONG kmt_observed[11]{};
struct AdapterQuery { DWORD type,bytes; LONG status; DWORD words[16]; };
inline AdapterQuery adapter_queries[128]{};
inline volatile LONG adapter_query_count=0;
inline bool bridge_kmt_escape=false;
inline volatile LONG bridged_escape_count=0;
inline uint64_t gdi_request(LONG operation,uint64_t value);
template<unsigned N> inline LONG WINAPI kmt_call(void* data) {
    LONG packet_index=-1;
    struct Escape { DWORD adapter,device,type,flags; void* data; DWORD bytes,context; };
    static_assert(sizeof(Escape)==32);
    if constexpr(N==6) {
        const auto before_error=GetLastError();
        packet_index=InterlockedIncrement(&packet_count)-1;
        if(packet_index<16) {
            const auto* escape=static_cast<const Escape*>(data);
            auto& packet=packets[packet_index]; packet.code=escape->type; packet.bytes=escape->bytes;
            packet.adapter=escape->adapter; packet.device=escape->device; packet.flags=escape->flags;
            packet.words=(escape->bytes<64?escape->bytes:64)/4;
            snapshot_words(escape->data,escape->bytes,packet.before);
        }
        SetLastError(before_error);
    }
    LONG result;
    if constexpr(N==6) {
        if(bridge_kmt_escape && gdi) {
            const auto* escape=static_cast<const Escape*>(data);
            const PhoenixKmtEscape request{escape->adapter,escape->device,escape->type,escape->flags,
                uint64_t(escape->data),escape->bytes,escape->context};
            if(!PhoenixValidKmtEscape(request)) return LONG(0xc000000du);
            result=LONG(uint32_t(gdi_request(7,uint64_t(&request))));
            InterlockedIncrement(&bridged_escape_count);
        } else result=kmt_originals[N](data);
    } else result=kmt_originals[N](data);
    if constexpr(N==9) {
        const auto error=GetLastError();
        struct Query { DWORD adapter,type; void* data; DWORD bytes; };
        static_assert(sizeof(Query)==24);
        const auto index=InterlockedIncrement(&adapter_query_count)-1;
        if(index<128) {
            const auto* query=static_cast<const Query*>(data);
            auto& row=adapter_queries[index]; row.type=query->type; row.bytes=query->bytes; row.status=result;
            snapshot_words(query->data,query->bytes,row.words);
        }
        SetLastError(error);
    }
    if constexpr(N==6) if(packet_index>=0 && packet_index<16) {
        const auto error=GetLastError();
        const auto* escape=static_cast<const Escape*>(data);
        snapshot_words(escape->data,escape->bytes,packets[packet_index].after);
        SetLastError(error);
    }
    InterlockedIncrement(&kmt_observed[N]);
    if(N!=9 || result<0) {
        const auto index=InterlockedIncrement(&kmt_count)-1;
        if(index<128) kmt_calls[index]={N,result};
    }
    return result;
}
inline KmtFunction kmt_wrappers[]{&kmt_call<0>,&kmt_call<1>,&kmt_call<2>,&kmt_call<3>,
    &kmt_call<4>,&kmt_call<5>,&kmt_call<6>,&kmt_call<7>,&kmt_call<8>,&kmt_call<9>,&kmt_call<10>};
inline BOOL WINAPI device_io(HANDLE handle, DWORD code, void* input, DWORD input_bytes,
    void* output, DWORD output_bytes, DWORD* returned, OVERLAPPED* overlapped) {
    const auto ok=DeviceIoControl(handle,code,input,input_bytes,output,output_bytes,returned,overlapped);
    const auto error=GetLastError();
    if(GetCurrentThreadId()==owner && ioctl_count<64)
        ioctls[ioctl_count++]={code,input_bytes,output_bytes,ok?0u:error,ok};
    SetLastError(error); return ok;
}
inline void* WINAPI virtual_alloc(void* address, SIZE_T bytes, DWORD flags, DWORD protection) {
    auto result = VirtualAlloc(address, bytes, flags, protection);
    const auto error = GetLastError();
    if (GetCurrentThreadId() == owner && allocation_count < 64)
        allocations[allocation_count++] = {uint64_t(address), bytes, uint64_t(result), flags, result ? 0u : error};
    SetLastError(error);
    return result;
}
inline void patch(HMODULE image);
inline HDC WINAPI create_dc_a(const char*,const char*,const char*,const DEVMODEA*);
inline void* display_replacement(const char*);
struct LibraryLoad { wchar_t path[512]; uint64_t module; DWORD flags,error; };
inline LibraryLoad library_loads[96]{};
inline volatile LONG library_load_count=0;
inline void record_library(const wchar_t* path,HMODULE module,DWORD flags,DWORD error) {
    const auto index=InterlockedIncrement(&library_load_count)-1;
    if(index>=96) return;
    auto& row=library_loads[index]; row.module=uint64_t(module); row.flags=flags; row.error=module?0:error;
    if(path) for(unsigned i=0;i<511 && path[i];++i) row.path[i]=path[i];
}
inline void patch_driver(HMODULE image) {
    wchar_t path[1024]{};
    if(!image || !GetModuleFileNameW(image,path,1024)) return;
    const auto base=wcsrchr(path,L'\\');
    const auto name=base?base+1:path;
    if(!lstrcmpiW(name,L"nvapi64.dll") || !lstrcmpiW(name,L"nvapi64_impl.dll") || !lstrcmpiW(name,L"nvcuda64.dll") ||
        !lstrcmpiW(name,L"WinTypes.dll") || !lstrcmpiW(name,L"combase.dll") || !lstrcmpiW(name,L"vulkan-1.dll") ||
        !lstrcmpiW(name,L"nvoglv64.dll")) patch(image);
}
inline bool is_optional_driver_client(const wchar_t* path) {
    if (!path || GetCurrentThreadId() != owner) return false;
    const wchar_t* base = path;
    for (const auto* p = path; *p; ++p) if (*p == L'\\' || *p == L'/') base = p+1;
    // Overlay capture and diagnostic clients are not required for inference.
    // The diagnostic client's thread callbacks faulted during mixed-bitness
    // worker restarts in WARP testing; omit it from this experimental host.
    return lstrcmpiW(base, L"nvspcap64.dll") == 0 || lstrcmpiW(base, L"nvdiagclt64.dll") == 0;
}
inline bool x64_image(const wchar_t* path) {
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    IMAGE_DOS_HEADER dos{};
    struct Header { DWORD signature; IMAGE_FILE_HEADER file; } header{};
    DWORD read = 0;
    bool valid = ReadFile(file, &dos, sizeof(dos), &read, nullptr) && read == sizeof(dos) &&
        dos.e_magic == IMAGE_DOS_SIGNATURE && dos.e_lfanew > 0 && dos.e_lfanew < 1024*1024;
    if (valid) valid = SetFilePointer(file, dos.e_lfanew, nullptr, FILE_BEGIN) != INVALID_SET_FILE_POINTER &&
        ReadFile(file, &header, sizeof(header), &read, nullptr) && read == sizeof(header) &&
        header.signature == IMAGE_NT_SIGNATURE && header.file.Machine == IMAGE_FILE_MACHINE_AMD64;
    CloseHandle(file);
    return valid;
}

inline const wchar_t* native_path(const wchar_t* path, wchar_t (&buffer)[1024]) {
    if (!path || GetCurrentThreadId() != owner) return path;
    size_t length = 0, base = 0;
    for (; path[length] && length < 1000; ++length)
        if (path[length] == L'\\' || path[length] == L'/') base = length+1;
    if (!base || length >= 1000) return path;
    const wchar_t* replacement = nullptr;
    if (lstrcmpiW(path+base, L"nvldumd.dll") == 0) replacement = L"nvldumdx.dll";
    else if (lstrcmpiW(path+base, L"nvwgf2um.dll") == 0) replacement = L"nvwgf2umx.dll";
    if (!replacement) return path;
    for (size_t i = 0; i < base; ++i) buffer[i] = path[i];
    size_t i = 0;
    do { buffer[base+i] = replacement[i]; } while (replacement[i++]);
    if (!x64_image(buffer)) return path;
    ++redirects;
    OutputDebugStringW(L"PHX_UMD_REDIRECT");
    OutputDebugStringW(buffer);
    return buffer;
}
inline HMODULE WINAPI load_ex(const wchar_t* path, HANDLE file, DWORD flags) {
    if (is_optional_driver_client(path)) { SetLastError(ERROR_MOD_NOT_FOUND); return nullptr; }
    wchar_t buffer[1024]{};
    const auto selected = native_path(path, buffer);
    auto loaded = LoadLibraryExW(selected, file, flags);
    const auto error=GetLastError(); record_library(selected,loaded,flags,error);
    patch_driver(loaded);
    patch_driver(GetModuleHandleW(L"nvapi64_impl.dll"));
    if (selected != path) {
        patch(loaded);
        patch(GetModuleHandleW(L"nvwgf2umx.dll"));
        patch(GetModuleHandleW(L"nvspcap64.dll"));
    }
    SetLastError(error); return loaded;
}
inline HMODULE WINAPI load_w(const wchar_t* path) {
    if (is_optional_driver_client(path)) { SetLastError(ERROR_MOD_NOT_FOUND); return nullptr; }
    wchar_t buffer[1024]{};
    const auto selected=native_path(path, buffer);
    auto loaded=LoadLibraryW(selected);
    const auto error=GetLastError(); record_library(selected,loaded,0,error);
    patch_driver(loaded); patch_driver(GetModuleHandleW(L"nvapi64_impl.dll"));
    SetLastError(error); return loaded;
}
inline HMODULE WINAPI load_a(const char* path) {
    wchar_t wide[1024]{}, buffer[1024]{};
    if (!path || !MultiByteToWideChar(CP_ACP, 0, path, -1, wide, 1024)) return LoadLibraryA(path);
    if (is_optional_driver_client(wide)) { SetLastError(ERROR_MOD_NOT_FOUND); return nullptr; }
    const auto selected = native_path(wide, buffer);
    auto loaded=selected == wide ? LoadLibraryA(path) : LoadLibraryW(selected);
    const auto error=GetLastError(); record_library(selected,loaded,0,error);
    patch_driver(loaded); patch_driver(GetModuleHandleW(L"nvapi64_impl.dll"));
    SetLastError(error); return loaded;
}
inline BOOL WINAPI native_identity(HANDLE process, BOOL* wow64) {
    ++identity_queries;
    if (wow64 && GetProcessId(process) == GetCurrentProcessId()) { *wow64 = FALSE; return TRUE; }
    return IsWow64Process(process, wow64);
}
inline LONG NTAPI query_process(HANDLE process, ULONG information, void* buffer, ULONG size, ULONG* returned) {
    using Query=LONG(NTAPI*)(HANDLE,ULONG,void*,ULONG,ULONG*);
    auto original=reinterpret_cast<Query>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"),"NtQueryInformationProcess"));
    auto status=original(process,information,buffer,size,returned);
    if(information==26 && status>=0 && buffer && size>=8 && GetProcessId(process)==GetCurrentProcessId()) {
        ++identity_queries;
        if(force_native_identity) *static_cast<uint64_t*>(buffer)=0;
    }
    return status;
}
inline FARPROC WINAPI get_proc(HMODULE library, const char* name) {
    if(uintptr_t(name)>0xffff) if(auto replacement=display_replacement(name)) return reinterpret_cast<FARPROC>(replacement);
    if(uintptr_t(name)>0xffff) if(auto replacement=image_replacement(name)) return reinterpret_cast<FARPROC>(replacement);
    if(bridge_dc_creation && gdi && uintptr_t(name)>0xffff && !lstrcmpA(name,"CreateDCA"))
        return reinterpret_cast<FARPROC>(&create_dc_a);
    if(GetCurrentThreadId()==owner && uintptr_t(name)>0xffff)
        if(auto replacement=wow64_runtime_diagnostics::replacement(name)) return reinterpret_cast<FARPROC>(replacement);
    if(GetCurrentThreadId()==owner && uintptr_t(name)>0xffff) {
        for(unsigned i=0;i<11;++i) if(lstrcmpA(name,kmt_names[i])==0) {
            auto original=GetProcAddress(library,name);
            if(!original) return nullptr;
            kmt_originals[i]=reinterpret_cast<KmtFunction>(original);
            return reinterpret_cast<FARPROC>(kmt_wrappers[i]);
        }
    }
    if (GetCurrentThreadId() == owner && uintptr_t(name) > 0xffff &&
        (lstrcmpA(name,"NtQueryInformationProcess")==0 || lstrcmpA(name,"ZwQueryInformationProcess")==0))
        return reinterpret_cast<FARPROC>(&query_process);
    if (GetCurrentThreadId() == owner && uintptr_t(name) > 0xffff && lstrcmpA(name, "VirtualAlloc") == 0)
        return reinterpret_cast<FARPROC>(&virtual_alloc);
    if (GetCurrentThreadId() == owner && uintptr_t(name) > 0xffff && lstrcmpA(name, "IsWow64Process") == 0)
        return reinterpret_cast<FARPROC>(&native_identity);
    if (GetCurrentThreadId() == owner && uintptr_t(name) > 0xffff &&
        lstrcmpA(name, "QueryShadowPlayDdiShimInterface") == 0) {
        SetLastError(ERROR_PROC_NOT_FOUND);
        OutputDebugStringW(L"PHX_OPTIONAL_CAPTURE_INTERFACE_SKIPPED\n");
        return nullptr;
    }
    return GetProcAddress(library, name);
}
inline HBITMAP WINAPI unavailable_bitmap(HINSTANCE, const char*) {
    SetLastError(ERROR_RESOURCE_NAME_NOT_FOUND);
    OutputDebugStringW(L"PHX_OPTIONAL_CAPTURE_BITMAP_UNAVAILABLE\n");
    return nullptr;
}
inline uint64_t gdi_request(LONG operation,uint64_t value) {
    if (!gdi) { SetLastError(ERROR_INVALID_HANDLE); return 0; }
    // All native modules share this packet, including callbacks retained by
    // the driver between model sessions. A lock in each DLL cannot serialize it.
    while(InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(&gdi->busy),1,0)!=0) Sleep(1);
    gdi->handle = value;
    InterlockedExchange(reinterpret_cast<volatile LONG*>(&gdi->pending), operation);
    if (!SetEvent(reinterpret_cast<HANDLE>(gdi->request_event))) {
        InterlockedExchange(reinterpret_cast<volatile LONG*>(&gdi->pending),0);
        const auto error=GetLastError();InterlockedExchange(reinterpret_cast<volatile LONG*>(&gdi->busy),0);SetLastError(error);return 0;
    }
    // Requests reference this worker's stack. The owning x86 worker services
    // callbacks until retirement; never terminate the game or abandon a live
    // packet on a timer. Neither side blocks the presentation thread.
    WaitForSingleObject(reinterpret_cast<HANDLE>(gdi->response_event),INFINITE);
    const auto error = gdi->error;
    const auto result = gdi->result;
    InterlockedExchange(reinterpret_cast<volatile LONG*>(&gdi->busy),0);
    SetLastError(error);
    return result;
}
inline DWORD WINAPI shared_tls_alloc() {
    // The native and WOW64 TLS bitmaps are distinct, but allocating an index
    // can clear that index in the game's existing threads. Reserve indices
    // through the x86 allocator so a native DLL cannot erase game/ICD TLS.
    const auto slot=DWORD(gdi_request(8,0));
    if(GetLastError()!=ERROR_SUCCESS||slot==TLS_OUT_OF_INDEXES) return TLS_OUT_OF_INDEXES;
    TlsSetValue(slot,nullptr);
    return slot;
}
inline BOOL WINAPI shared_tls_free(DWORD slot) {
    const auto result=BOOL(gdi_request(9,slot));
    if(result)TlsSetValue(slot,nullptr);
    return result;
}
inline ExportPatch tls_exports[4]{};
inline bool install_tls_exports() {
    const auto kernel=GetModuleHandleW(L"kernelbase.dll");
    const auto facade=GetModuleHandleW(L"kernel32.dll");
    return gdi && redirect_export(kernel,"TlsAlloc",reinterpret_cast<void*>(&shared_tls_alloc),tls_exports[0]) &&
        redirect_export(kernel,"TlsFree",reinterpret_cast<void*>(&shared_tls_free),tls_exports[1]) &&
        redirect_export(facade,"TlsAlloc",reinterpret_cast<void*>(&shared_tls_alloc),tls_exports[2]) &&
        redirect_export(facade,"TlsFree",reinterpret_cast<void*>(&shared_tls_free),tls_exports[3]);
}
inline BOOL WINAPI delete_dc(HDC dc) {
    if ((uint64_t(dc) >> 32) != 0 && (uint64_t(dc) >> 32) != 0xffffffffu) { SetLastError(ERROR_INVALID_HANDLE); return FALSE; }
    return BOOL(gdi_request(1,uint32_t(uint64_t(dc))));
}
inline int WINAPI device_caps(HDC dc,int index) {
    return int(uint32_t(gdi_request(3,uint32_t(uint64_t(dc))|(uint64_t(uint32_t(index))<<32))));
}
inline HDC WINAPI display_dc(HWND window) { return reinterpret_cast<HDC>(gdi_request(4,uint32_t(uint64_t(window)))); }
inline int WINAPI release_dc(HWND window,HDC dc) {
    return int(uint32_t(gdi_request(5,uint32_t(uint64_t(window))|(uint64_t(uint32_t(uint64_t(dc)))<<32))));
}
inline BOOL WINAPI enum_display(const char* device,DWORD index,DISPLAY_DEVICEA* display,DWORD flags) {
    if(!display || display->cb!=sizeof(*display)) { SetLastError(ERROR_INVALID_PARAMETER); return FALSE; }
    PhoenixGdiDisplayDevice request{}; request.index=index; request.flags=flags;
    if(device) {
        request.present=1;
        unsigned i=0;
        for(;i<sizeof(request.device);++i) { request.device[i]=device[i]; if(!device[i]) break; }
        if(i==sizeof(request.device)) { SetLastError(ERROR_INSUFFICIENT_BUFFER); return FALSE; }
    }
    const auto ok=BOOL(gdi_request(6,uint64_t(&request)));
    if(ok) for(unsigned i=0;i<sizeof(*display);++i) reinterpret_cast<unsigned char*>(display)[i]=request.display[i];
    return ok;
}
inline void* display_replacement(const char* name) {
    if(!bridge_display_gdi || !gdi) return nullptr;
    if(!lstrcmpA(name,"GetDeviceCaps")) return reinterpret_cast<void*>(&device_caps);
    if(!lstrcmpA(name,"GetDC")) return reinterpret_cast<void*>(&display_dc);
    if(!lstrcmpA(name,"ReleaseDC")) return reinterpret_cast<void*>(&release_dc);
    if(!lstrcmpA(name,"EnumDisplayDevicesA")) return reinterpret_cast<void*>(&enum_display);
    return nullptr;
}
inline bool install_display_exports() {
    auto gdi_image=LoadLibraryExW(L"gdi32.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    auto user_image=LoadLibraryExW(L"user32.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    return redirect_export(gdi_image,"GetDeviceCaps",reinterpret_cast<void*>(&device_caps),display_exports[0]) &&
        redirect_export(user_image,"GetDC",reinterpret_cast<void*>(&display_dc),display_exports[1]) &&
        redirect_export(user_image,"ReleaseDC",reinterpret_cast<void*>(&release_dc),display_exports[2]) &&
        redirect_export(user_image,"EnumDisplayDevicesA",reinterpret_cast<void*>(&enum_display),display_exports[3]);
}
inline HDC WINAPI create_dc_a(const char* driver,const char* device,const char* output,const DEVMODEA* mode) {
    PhoenixGdiCreateDC request{};
    auto copy=[&](const char* text,char (&target)[128],unsigned flag) {
        if(!text) return true;
        request.present|=flag;
        for(unsigned i=0;i<128;++i) { target[i]=text[i]; if(!text[i]) return true; }
        return false;
    };
    if(!copy(driver,request.driver,1) || !copy(device,request.device,2) || !copy(output,request.output,4)) {
        SetLastError(ERROR_INSUFFICIENT_BUFFER); return nullptr;
    }
    if(mode) {
        const auto bytes=unsigned(mode->dmSize)+unsigned(mode->dmDriverExtra);
        if(mode->dmSize<sizeof(DEVMODEA) || bytes>sizeof(request.mode)) { SetLastError(ERROR_INVALID_PARAMETER); return nullptr; }
        request.mode_bytes=bytes;
        for(unsigned i=0;i<bytes;++i) request.mode[i]=reinterpret_cast<const unsigned char*>(mode)[i];
    }
    return reinterpret_cast<HDC>(gdi_request(2,uint64_t(&request)));
}
inline bool install_gdi_export() {
    if (!gdi) return false;
    auto image = LoadLibraryExW(L"gdi32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    // Use the same immutable-stub lifecycle as identity and display exports.
    return image && redirect_export(image, "DeleteDC", reinterpret_cast<void*>(&delete_dc),
        gdi_export_patch);
}
inline void restore_gdi_export() {
    auto& patch = gdi_export_patch;
    if (!patch.slot) return;
    DWORD previous = 0, ignored = 0;
    if (VirtualProtect(patch.slot, sizeof(*patch.slot), PAGE_READWRITE, &previous)) {
        InterlockedExchange(reinterpret_cast<volatile LONG*>(patch.slot), LONG(patch.before));
        VirtualProtect(patch.slot, sizeof(*patch.slot), previous, &ignored);
    }
    patch.slot = nullptr;
}
inline void patch(HMODULE image) {
    if (!image) return;
    auto bytes = reinterpret_cast<unsigned char*>(image);
    auto dos = reinterpret_cast<IMAGE_DOS_HEADER*>(bytes);
    auto nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(bytes+dos->e_lfanew);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || nt->Signature != IMAGE_NT_SIGNATURE ||
        nt->FileHeader.Machine != IMAGE_FILE_MACHINE_AMD64) return;
    auto dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress || dir.VirtualAddress+dir.Size > nt->OptionalHeader.SizeOfImage) return;
    auto imports = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(bytes+dir.VirtualAddress);
    for (unsigned m = 0; m < dir.Size/sizeof(*imports) && imports[m].Name; ++m) {
        if (!imports[m].OriginalFirstThunk) continue;
        auto names = reinterpret_cast<IMAGE_THUNK_DATA64*>(bytes+imports[m].OriginalFirstThunk);
        auto table = reinterpret_cast<IMAGE_THUNK_DATA64*>(bytes+imports[m].FirstThunk);
        for (unsigned i = 0; names[i].u1.AddressOfData && i < 4096; ++i) {
            if (IMAGE_SNAP_BY_ORDINAL64(names[i].u1.Ordinal)) continue;
            auto name = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(bytes+names[i].u1.AddressOfData)->Name;
            void* replacement = wow64_runtime_diagnostics::replacement(reinterpret_cast<const char*>(name));
            if(auto display_proc=display_replacement(reinterpret_cast<const char*>(name))) replacement=display_proc;
            if(auto image_proc=image_replacement(reinterpret_cast<const char*>(name))) replacement=image_proc;
            if(bridge_dc_creation && gdi && !lstrcmpA(reinterpret_cast<const char*>(name),"CreateDCA"))
                replacement=reinterpret_cast<void*>(&create_dc_a);
            if (lstrcmpA(name, "LoadLibraryExW") == 0) replacement = reinterpret_cast<void*>(&load_ex);
            if (lstrcmpA(name, "LoadLibraryW") == 0) replacement = reinterpret_cast<void*>(&load_w);
            if (lstrcmpA(name, "LoadLibraryA") == 0) replacement = reinterpret_cast<void*>(&load_a);
            if (lstrcmpA(name, "GetProcAddress") == 0) replacement = reinterpret_cast<void*>(&get_proc);
            if (image == GetModuleHandleW(L"nvspcap64.dll") && lstrcmpA(name, "LoadBitmapA") == 0)
                replacement = reinterpret_cast<void*>(&unavailable_bitmap);
            if (gdi && image == GetModuleHandleW(L"nvapi64_impl.dll") && lstrcmpA(name, "DeleteDC") == 0)
                replacement = reinterpret_cast<void*>(&delete_dc);
            if (lstrcmpA(name, "VirtualAlloc") == 0)
                replacement = reinterpret_cast<void*>(&virtual_alloc);
            if (lstrcmpA(name,"DeviceIoControl")==0) replacement=reinterpret_cast<void*>(&device_io);
            if (lstrcmpA(name,"NtQueryInformationProcess")==0) replacement=reinterpret_cast<void*>(&query_process);
            if (force_native_identity && lstrcmpA(name,"IsWow64Process")==0)
                replacement=reinterpret_cast<void*>(&native_identity);
            if (!replacement || slot_count == import_capacity) continue;
            auto address = reinterpret_cast<void**>(&table[i].u1.Function);
            bool already = false;
            for (unsigned s = 0; s < slot_count; ++s) already |= slots[s].address == address;
            if (already) continue;
            DWORD previous = 0, ignored = 0;
            if (!VirtualProtect(address, sizeof(*address), PAGE_READWRITE, &previous)) continue;
            slots[slot_count++] = {address, InterlockedExchangePointer(address, replacement)};
            VirtualProtect(address, sizeof(*address), previous, &ignored);
        }
    }
}
inline unsigned install() {
    owner = GetCurrentThreadId();
    // Adapt an existing capture client, but do not load an optional overlay
    // into a game merely to run inference. Driver requests are excluded above.
    patch(GetModuleHandleW(L"nvspcap64.dll"));
    auto core = LoadLibraryExW(L"D3D12Core.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    patch(core);
    patch(GetModuleHandleW(L"dxgi.dll"));
    patch(GetModuleHandleW(L"dxcore.dll"));
    return slot_count;
}
inline void restore() {
    for (unsigned i = slot_count; i; --i) {
        auto& slot = slots[i-1];
        DWORD previous = 0, ignored = 0;
        if (!VirtualProtect(slot.address, sizeof(void*), PAGE_READWRITE, &previous)) continue;
        InterlockedExchangePointer(slot.address, slot.original);
        VirtualProtect(slot.address, sizeof(void*), previous, &ignored);
    }
    slot_count = 0;
    owner = 0;
}
}
