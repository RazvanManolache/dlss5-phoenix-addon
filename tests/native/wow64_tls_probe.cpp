#define NOMINMAX
#include <windows.h>
#include "../../third_party/nlohmann/json.hpp"
#include "wow64_tls_contract.h"
#include "wow64_probe_contract.h"
#include "../../native_runtime/phoenix_native.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <intrin.h>
#include <d3dkmthk.h>
#include <vector>
#ifdef PHX_X86_HOST_PROBE
#include "wow64_x86_host.h"
#endif
using Json = nlohmann::json;
void emit(const char* event, Json fields = Json::object()) {
    fields["event"] = event;
    std::cout << fields.dump() << std::endl;
}
void report(const PhoenixTlsProbe& r) {
    emit("tls_result", {{"stage", r.stage}, {"teb", r.teb}, {"vector_before", r.vector_before},
        {"vector_after", r.vector_after}, {"slot", r.slot}, {"index", r.index},
        {"callbacks", r.callbacks}, {"template_ok", r.template_ok != 0}, {"zero_fill_ok", r.zero_fill_ok != 0},
        {"handler_status", uint32_t(r.handler_status)}, {"lookup_status", uint32_t(r.lookup_status)},
        {"lock_status", uint32_t(r.lock_status)}, {"initializer_status", uint32_t(r.initializer_status)}});
}
int wmain(int argc, wchar_t** argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    if (argc < 2) return 1;
#ifdef _WIN64
#ifdef PHX_X86_HOST_PROBE
    if(argc==4&&std::wstring(argv[1])==L"--x86-host-control") return PhoenixRunNativeHostControl(argv[2],argv[3]);
#endif
    if (argc == 4 && std::wstring(argv[1]) == L"--frame-library") {
        auto library = LoadLibraryW(argv[2]);
        if (!library) return 2;
        auto run = reinterpret_cast<int(*)(const wchar_t*)>(GetProcAddress(library, "PhoenixPortableFrameProbe64"));
        return run ? run(argv[3]) : 2;
    }
    if (argc == 3 && std::wstring(argv[1]) == L"--native-api") {
        auto library = LoadLibraryW(argv[2]);
        if (!library) return 2;
        auto defaults = reinterpret_cast<decltype(&phoenix_native_default_controls)>(GetProcAddress(library, "phoenix_native_default_controls"));
        auto frame = reinterpret_cast<decltype(&phoenix_native_frame_host)>(GetProcAddress(library, "phoenix_native_frame_host"));
        auto error = reinterpret_cast<decltype(&phoenix_native_error)>(GetProcAddress(library, "phoenix_native_error"));
        if (!defaults || !frame || !error) return 2;
        PhoenixNativeControls controls{};
        defaults(&controls);
        auto status = frame(nullptr, nullptr, 0, 0, nullptr, 0, 0, nullptr, 0, 0, &controls);
        const std::string text(error());
        emit("native_phoenix_error_control", {{"status", status}, {"message", text}});
        return status == -1 && text == "Null native session" ? 0 : 2;
    }
    if (argc != 3 || std::wstring(argv[1]) != L"--tls") return 1;
    HMODULE library = LoadLibraryW(argv[2]);
    if (!library) throw std::runtime_error("Native TLS control failed to load");
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(library);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(reinterpret_cast<char*>(library) + dos->e_lfanew);
    auto* tls = reinterpret_cast<IMAGE_TLS_DIRECTORY64*>(reinterpret_cast<char*>(library) + nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_TLS].VirtualAddress);
    emit("native_tls_directory", {{"module", uint64_t(library)}, {"start", tls->StartAddressOfRawData},
        {"end", tls->EndAddressOfRawData}, {"index", tls->AddressOfIndex}, {"callbacks", tls->AddressOfCallBacks}});
    using FindEntry = LONG(NTAPI*)(void*, void**);
    auto find_entry = reinterpret_cast<FindEntry>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "LdrFindEntryForAddress"));
    void* entry = nullptr;
    if (find_entry && find_entry(library, &entry) >= 0 && entry)
        emit("native_loader_entry", {{"load_reason", *reinterpret_cast<unsigned*>(reinterpret_cast<char*>(entry)+0x10c)},
            {"tls_index", *reinterpret_cast<unsigned short*>(reinterpret_cast<char*>(entry)+0x6e)}});
    auto inspect = reinterpret_cast<PhoenixTlsProbeProc>(GetProcAddress(library, "PhoenixTlsInspect64"));
    if (!inspect) return 2;
    PhoenixTlsProbe result{};
    result.module = uint64_t(library);
    const auto code = inspect(&result);
    report(result);
    return code || !result.template_ok || !result.zero_fill_ok ? 2 : 0;
#else
    using Call = uint64_t(__cdecl*)(uint64_t, int, ...);
    using Module = uint64_t(__cdecl*)(const wchar_t*);
    using Proc = uint64_t(__cdecl*)(uint64_t, const char*);
    HMODULE extension = LoadLibraryW(L"wow64ext.dll");
    if (!extension) throw std::runtime_error("WOW64Ext unavailable");
    const auto call = reinterpret_cast<Call>(GetProcAddress(extension, "X64Call"));
    const auto module = reinterpret_cast<Module>(GetProcAddress(extension, "GetModuleHandle64"));
    const auto proc = reinterpret_cast<Proc>(GetProcAddress(extension, "GetProcAddress64"));
    if (!call || !module || !proc) return 2;
    const auto ntdll = module(L"ntdll.dll");
    auto address = [](const void* value) { return uint64_t(uintptr_t(value)); };
    struct NativeString { uint16_t length, capacity; uint32_t padding; uint64_t buffer; };
    const auto loader = proc(ntdll, "LdrLoadDll");
    if (!loader) return 2;
    const std::wstring mode(argv[1]);
    if (mode == L"--dxgi") {
        wchar_t windows[MAX_PATH]{};
        GetWindowsDirectoryW(windows, MAX_PATH);
        const std::wstring path = std::wstring(windows) + L"\\System32\\dxgi.dll";
        NativeString name{uint16_t(path.size()*2), uint16_t((path.size()+1)*2), 0, address(path.c_str())};
        uint64_t library = 0;
        emit("dxgi_load_start");
        const auto status = uint32_t(call(loader, 4, uint64_t(0), uint64_t(0), address(&name), address(&library)));
        emit("dxgi_load_result", {{"status", status}, {"loaded", library != 0}});
        return int32_t(status) < 0 || !library ? 2 : 0;
    }
    if (argc != 3 && argc != 5 && argc != 6 && argc != 7 && argc != 8) return 1;
    const bool host_mode = mode == L"--thread-x86-host";
    const bool task_mode = mode == L"--thread-native" || mode == L"--thread-gpu" || mode == L"--thread-frame" || host_mode;
    const bool initialize = mode == L"--initialize" || mode == L"--thread-initialize" || mode == L"--initialize-dxgi" || mode == L"--initialize-native" || mode == L"--initialize-gpu" || task_mode;
    const bool repair = mode == L"--repair" || mode == L"--thread-repair" || initialize;
    const bool thread = mode == L"--thread" || mode == L"--thread-repair" || mode == L"--thread-initialize" || task_mode;
    if (mode != L"--tls" && !repair && !thread) return 1;
    const std::wstring path(argv[2]);
    NativeString name{uint16_t(path.size()*2), uint16_t((path.size()+1)*2), 0, address(path.c_str())};
    uint64_t library = 0;
    emit("tls_load_start");
    const auto status = uint32_t(call(loader, 4, uint64_t(0), uint64_t(0), address(&name), address(&library)));
    emit("tls_load_result", {{"status", status}, {"loaded", library != 0}});
    if (int32_t(status) < 0 || !library) return 2;
    PhoenixTlsProbe result{};
    result.module = library;
    if (repair) {
        if (argc != 5 && argc != 6 && argc != 7 && argc != 8) return 1;
        // These values are resolved from this system image's symbols, not a
        // hardcoded Windows-version offset. Reading uses the cross-mode helper.
        using Read64 = BOOL(__cdecl*)(HANDLE, uint64_t, void*, SIZE_T, SIZE_T*);
        const auto read = reinterpret_cast<Read64>(GetProcAddress(extension, "ReadProcessMemory64"));
        const auto rva = std::stoull(argv[3], nullptr, 16);
        const auto stamp = std::stoul(argv[4], nullptr, 16);
        IMAGE_DOS_HEADER dos{};
        IMAGE_NT_HEADERS64 nt{};
        if (!read || !read(GetCurrentProcess(), ntdll, &dos, sizeof(dos), nullptr) || dos.e_magic != IMAGE_DOS_SIGNATURE ||
            !read(GetCurrentProcess(), ntdll + dos.e_lfanew, &nt, sizeof(nt), nullptr) || nt.Signature != IMAGE_NT_SIGNATURE ||
            nt.FileHeader.TimeDateStamp != stamp || rva >= nt.OptionalHeader.SizeOfImage || !rva)
            throw std::runtime_error("Native TLS handler image identity mismatch");
        result.handler = ntdll + rva;
        if (initialize) { result.initializer = result.handler; result.handler = 0; }
        result.finder = proc(ntdll, "LdrFindEntryForAddress");
        result.loader_lock = proc(ntdll, "LdrLockLoaderLock");
        result.loader_unlock = proc(ntdll, "LdrUnlockLoaderLock");
    }
    const auto entry = proc(library, task_mode ? "PhoenixTlsRunTask64" : thread ? "PhoenixTlsThread64" : "PhoenixTlsInspect64");
    if (!entry) return 2;
    emit("tls_inspect_start", {{"repair", repair}, {"native_thread", thread}});
    if (thread) {
        PhoenixTlsTask task{};
        task.tls = result;
#ifdef PHX_X86_HOST_PROBE
        std::unique_ptr<PhoenixX86HostProbe> host;
        if(host_mode) {
            if(argc!=7) return 1;
            host=std::make_unique<PhoenixX86HostProbe>(argv[6]);
        }
#else
        if(host_mode) return 1;
#endif
        if (task_mode) {
            if ((mode == L"--thread-native" && argc != 6) ||
                (mode == L"--thread-frame" && argc != 7) || (mode == L"--thread-gpu" && argc != 7 && argc != 8)) return 1;
            task.kind = host_mode ? 4 : mode == L"--thread-gpu" ? 1 : mode == L"--thread-frame" ? 3 : 2;
            task.path = address(argv[5]);
            if (argc >= 7) task.kernel = address(argv[6]);
#ifdef PHX_X86_HOST_PROBE
            if(host) task.kernel=address(&host->channel);
#endif
            if (argc == 8) task.cuda_library = address(argv[7]);
            task.gdi.request_event = address(CreateEventW(nullptr, FALSE, FALSE, nullptr));
            task.gdi.response_event = address(CreateEventW(nullptr, FALSE, FALSE, nullptr));
            if (!task.gdi.request_event || !task.gdi.response_event) return 2;
        }
        const auto create = proc(ntdll, "NtCreateThreadEx");
        uint64_t handle = 0;
        const auto created = uint32_t(call(create, 11, address(&handle), uint64_t(0x1fffff), uint64_t(0),
            uint64_t(-1), entry, task_mode ? address(&task) : address(&result), uint64_t(2), uint64_t(0), uint64_t(0), uint64_t(0), uint64_t(0)));
        if (int32_t(created) < 0 || !handle) return 2;
        const auto h = reinterpret_cast<HANDLE>(uintptr_t(handle));
        DWORD waited = WAIT_TIMEOUT;
        const auto deadline = GetTickCount64()+60000;
        do {
            HANDLE handles[]{h, reinterpret_cast<HANDLE>(uintptr_t(task.gdi.request_event)),nullptr};
#ifdef PHX_X86_HOST_PROBE
            if(host) handles[2]=reinterpret_cast<HANDLE>(uintptr_t(host->channel.response_event));
#endif
            waited = task_mode ? WaitForMultipleObjects(host_mode?3:2, handles, FALSE, 100) : WaitForSingleObject(h, 100);
#ifdef PHX_X86_HOST_PROBE
            if(host&&waited==WAIT_OBJECT_0+2) {
                try { host->advance(); } catch(const std::exception& error) {
                    host->fail(error.what()); TerminateProcess(GetCurrentProcess(),2);
                }
            }
#endif
            if (task_mode && waited == WAIT_OBJECT_0+1) {
                const auto operation=InterlockedExchange(reinterpret_cast<volatile LONG*>(&task.gdi.pending),0);
                if(operation==1) task.gdi.result = DeleteDC(reinterpret_cast<HDC>(uintptr_t(task.gdi.handle))) != 0;
                else if(operation==2) {
                    using Read64=BOOL(__cdecl*)(HANDLE,uint64_t,void*,SIZE_T,SIZE_T*);
                    const auto read=reinterpret_cast<Read64>(GetProcAddress(extension,"ReadProcessMemory64"));
                    PhoenixGdiCreateDC request{};
                    if(read && read(GetCurrentProcess(),task.gdi.handle,&request,sizeof(request),nullptr) &&
                        PhoenixValidGdiCreateDC(request,sizeof(DEVMODEA))) {
                        task.gdi.result=uint64_t(uintptr_t(CreateDCA(request.present&1?request.driver:nullptr,
                            request.present&2?request.device:nullptr,request.present&4?request.output:nullptr,
                            request.mode_bytes?reinterpret_cast<const DEVMODEA*>(request.mode):nullptr)));
                    } else { task.gdi.result=0; SetLastError(ERROR_INVALID_PARAMETER); }
                } else if(operation==3) {
                    task.gdi.result=uint32_t(GetDeviceCaps(reinterpret_cast<HDC>(uintptr_t(uint32_t(task.gdi.handle))),int(task.gdi.handle>>32)));
                } else if(operation==4) {
                    task.gdi.result=uint32_t(uintptr_t(GetDC(reinterpret_cast<HWND>(uintptr_t(uint32_t(task.gdi.handle))))));
                } else if(operation==5) {
                    task.gdi.result=uint32_t(ReleaseDC(reinterpret_cast<HWND>(uintptr_t(uint32_t(task.gdi.handle))),
                        reinterpret_cast<HDC>(uintptr_t(uint32_t(task.gdi.handle>>32)))));
                } else if(operation==6) {
                    using Read64=BOOL(__cdecl*)(HANDLE,uint64_t,void*,SIZE_T,SIZE_T*);
                    using Write64=BOOL(__cdecl*)(HANDLE,uint64_t,const void*,SIZE_T,SIZE_T*);
                    auto read=reinterpret_cast<Read64>(GetProcAddress(extension,"ReadProcessMemory64"));
                    auto write=reinterpret_cast<Write64>(GetProcAddress(extension,"WriteProcessMemory64"));
                    PhoenixGdiDisplayDevice request{}; DISPLAY_DEVICEA display{}; display.cb=sizeof(display);
                    static_assert(sizeof(display)==sizeof(request.display));
                    task.gdi.result=0;
                    if(read && write && read(GetCurrentProcess(),task.gdi.handle,&request,sizeof(request),nullptr) &&
                        PhoenixValidGdiDisplayDevice(request)) {
                        const auto ok=EnumDisplayDevicesA(request.present?request.device:nullptr,request.index,&display,request.flags);
                        const auto error=GetLastError();
                        if(!ok) SetLastError(error);
                        else if(write(GetCurrentProcess(),task.gdi.handle+offsetof(PhoenixGdiDisplayDevice,display),&display,sizeof(display),nullptr)) {
                            task.gdi.result=1; SetLastError(error);
                        }
                    } else SetLastError(ERROR_INVALID_PARAMETER);
                } else if(operation==7) {
                    using Read64=BOOL(__cdecl*)(HANDLE,uint64_t,void*,SIZE_T,SIZE_T*);
                    using Write64=BOOL(__cdecl*)(HANDLE,uint64_t,const void*,SIZE_T,SIZE_T*);
                    auto read=reinterpret_cast<Read64>(GetProcAddress(extension,"ReadProcessMemory64"));
                    auto write=reinterpret_cast<Write64>(GetProcAddress(extension,"WriteProcessMemory64"));
                    PhoenixKmtEscape request{};
                    task.gdi.result=0xc000000du;
                    if(read && write && read(GetCurrentProcess(),task.gdi.handle,&request,sizeof(request),nullptr) &&
                        PhoenixValidKmtEscape(request)) {
                        std::vector<unsigned char> payload(request.bytes);
                        if(!request.bytes || read(GetCurrentProcess(),request.data,payload.data(),request.bytes,nullptr)) {
                            D3DKMT_ESCAPE escape{};
                            escape.hAdapter=request.adapter; escape.hDevice=request.device;
                            escape.Type=static_cast<D3DKMT_ESCAPETYPE>(request.type);
                            escape.Flags.Value=request.flags; escape.hContext=request.context;
                            escape.pPrivateDriverData=request.bytes?payload.data():nullptr;
                            escape.PrivateDriverDataSize=request.bytes;
                            const auto status=D3DKMTEscape(&escape);
                            task.gdi.result=uint32_t(status);
                            if(request.bytes && !write(GetCurrentProcess(),request.data,payload.data(),request.bytes,nullptr))
                                task.gdi.result=0xc0000005u;
                        }
                    }
                } else { task.gdi.result=0; SetLastError(ERROR_INVALID_FUNCTION); }
                task.gdi.error = GetLastError();
                ++task.gdi.requests;
                SetEvent(reinterpret_cast<HANDLE>(uintptr_t(task.gdi.response_event)));
            }
        } while (waited != WAIT_OBJECT_0 && waited != WAIT_FAILED && GetTickCount64() < deadline);
        DWORD code = 1;
        if (waited == WAIT_OBJECT_0) GetExitCodeThread(h, &code);
        CloseHandle(h);
        // Never unwind the task's storage while a native worker still owns it.
        if (waited != WAIT_OBJECT_0) TerminateProcess(GetCurrentProcess(), 3);
        report(task_mode ? task.tls : result);
        if (task_mode) {
            CloseHandle(reinterpret_cast<HANDLE>(uintptr_t(task.gdi.request_event)));
            CloseHandle(reinterpret_cast<HANDLE>(uintptr_t(task.gdi.response_event)));
            emit("native_thread_task", {{"completed", task.completed != 0}, {"loaded", task.loaded != 0},
                {"pid",GetCurrentProcessId()},{"caller_pointer_bits",32},
                {"umd_redirects", task.umd_redirects},
                {"gdi32_requests", task.gdi.requests},
                {"status", task.status}, {"load_error", task.load_error}, {"defaults_ok", task.defaults_ok != 0},
                {"message", task.message}, {"stack_address", task.stack_address}, {"stack_base", task.stack_base},
                {"stack_limit", task.stack_limit}});
            if (task.kind == 1)
                emit("gpu_worker_result", {{"worker_stage", task.gpu.stage}, {"status", task.gpu.status},
                    {"module_created", task.gpu.module_created}, {"fatbin_supported", task.gpu.fatbin_supported},
                    {"worker_code", task.status}});
#ifdef PHX_X86_HOST_PROBE
            if(host) {
                if(host->channel.operation==phoenix_driver_probe::shutdown&&host->channel.status==0) host->advance();
                const auto passed=host->finish(); return !code&&task.completed&&task.loaded&&!task.status&&passed?0:2;
            }
#endif
            const bool ok = task.completed && task.loaded && (task.kind == 3 ? task.status == 0 : task.kind == 1 ? task.status == 0 &&
                task.gpu.stage == 12 && task.gpu.module_created && task.gpu.status == 0 : task.defaults_ok &&
                task.status == -1 && std::string(task.message) == "Null native session");
            return !code && ok ? 0 : 2;
        }
        return code ? 2 : 0;
    }
    const auto tls32_before = __readfsdword(0x2c);
    const auto code = uint32_t(call(entry, 1, address(&result)));
    if (__readfsdword(0x2c) != tls32_before) {
        const char error[] = "x86 TLS vector changed unexpectedly\n";
        DWORD written = 0;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), error, sizeof(error)-1, &written, nullptr);
        TerminateProcess(GetCurrentProcess(), 4);
    }
    report(result);
    if (mode == L"--initialize-gpu" && argc == 7 && result.template_ok && result.zero_fill_ok) {
        const std::wstring path(argv[5]);
        NativeString target{uint16_t(path.size()*2), uint16_t((path.size()+1)*2), 0, address(path.c_str())};
        uint64_t loaded = 0;
        emit("gpu_worker_load_start");
        const auto loaded_status = uint32_t(call(loader, 4, uint64_t(0), uint64_t(0), address(&target), address(&loaded)));
        emit("gpu_worker_load_result", {{"status", loaded_status}, {"loaded", loaded != 0}});
        if (int32_t(loaded_status) < 0 || !loaded) return 2;
        const auto run = proc(loaded, "PhoenixWow64Probe64");
        if (!run) return 2;
        PhoenixWow64ProbeResult gpu{};
        emit("gpu_worker_start");
        const auto gpu_code = uint32_t(call(run, 2, address(argv[6]), address(&gpu)));
        emit("gpu_worker_result", {{"worker_stage", gpu.stage}, {"status", gpu.status},
            {"win32_error", gpu.win32_error}, {"fatbin_supported", gpu.fatbin_supported},
            {"module_created", gpu.module_created}, {"blob_bytes", gpu.blob_bytes}, {"worker_code", gpu_code}});
        return gpu_code || gpu.stage != 12 || gpu.status || !gpu.module_created ? 2 : 0;
    }
    if (mode == L"--initialize-dxgi" && result.template_ok && result.zero_fill_ok) {
        wchar_t windows[MAX_PATH]{};
        GetWindowsDirectoryW(windows, MAX_PATH);
        const std::wstring dxgi = std::wstring(windows) + L"\\System32\\dxgi.dll";
        NativeString target{uint16_t(dxgi.size()*2), uint16_t((dxgi.size()+1)*2), 0, address(dxgi.c_str())};
        uint64_t loaded = 0;
        emit("initialized_dxgi_load_start");
        const auto loaded_status = uint32_t(call(loader, 4, uint64_t(0), uint64_t(0), address(&target), address(&loaded)));
        emit("initialized_dxgi_load_result", {{"status", loaded_status}, {"loaded", loaded != 0}});
    }
    if (mode == L"--initialize-native" && argc == 6 && result.template_ok && result.zero_fill_ok) {
        const std::wstring path(argv[5]);
        NativeString target{uint16_t(path.size()*2), uint16_t((path.size()+1)*2), 0, address(path.c_str())};
        uint64_t loaded = 0;
        emit("phoenix_load_start");
        const auto loaded_status = uint32_t(call(loader, 4, uint64_t(0), uint64_t(0), address(&target), address(&loaded)));
        emit("phoenix_load_result", {{"status", loaded_status}, {"loaded", loaded != 0}});
        if (int32_t(loaded_status) < 0 || !loaded) return 2;
        const auto defaults = proc(loaded, "phoenix_native_default_controls");
        const auto frame = proc(loaded, "phoenix_native_frame_host");
        const auto error = proc(loaded, "phoenix_native_error");
        if (!defaults || !frame || !error) return 2;
        struct Controls { uint32_t bytes, reset; float mx, my, tone, structure, skin; uint32_t mask, inverted; } controls{};
        static_assert(sizeof(controls) == 36);
        call(defaults, 1, address(&controls));
        const bool controls_ok = controls.bytes == 36 && controls.mx == 1 && controls.my == 1 && controls.tone == 1 &&
            controls.structure == 1 && controls.skin == -1 && controls.mask == 0 && controls.inverted == 1;
        emit("phoenix_defaults", {{"matched", controls_ok}});
        // A null-session failure exercises the real x64 C++ error/TLS path;
        // it cannot create a GPU device or perform model inference.
        const auto frame_status = int32_t(call(frame, 11, uint64_t(0), uint64_t(0), uint64_t(0), uint64_t(0),
            uint64_t(0), uint64_t(0), uint64_t(0), uint64_t(0), uint64_t(0), uint64_t(0), address(&controls)));
        const auto text = call(error, 0);
        using Read64 = BOOL(__cdecl*)(HANDLE, uint64_t, void*, SIZE_T, SIZE_T*);
        const auto read = reinterpret_cast<Read64>(GetProcAddress(extension, "ReadProcessMemory64"));
        char message[256]{};
        bool copied = read && text;
        for (unsigned i = 0; copied && i < sizeof(message)-1; ++i) {
            copied = read(GetCurrentProcess(), text+i, message+i, 1, nullptr) != 0;
            if (!message[i]) break;
        }
        const bool tls32_preserved = __readfsdword(0x2c) == tls32_before;
        emit("phoenix_error_contract", {{"frame_status", frame_status}, {"message", copied ? message : "unreadable"},
            {"x86_tls_preserved", tls32_preserved}});
        return controls_ok && frame_status == -1 && copied && std::string(message) == "Null native session" && tls32_preserved ? 0 : 2;
    }
    return code ? 2 : 0;
#endif
} catch (const std::exception& error) { emit("error", {{"message", error.what()}}); return 1; }
