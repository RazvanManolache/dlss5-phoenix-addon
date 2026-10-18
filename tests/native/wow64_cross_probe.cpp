// All cross-mode calls remain inside this disposable process.
#define NOMINMAX
#include <windows.h>
#include "../../third_party/nlohmann/json.hpp"
#include "wow64_probe_contract.h"
#include <iostream>
#include <string>
#include <stdexcept>

using Json = nlohmann::json;
static void event(const char* stage, Json fields = Json::object()) {
    fields["stage"] = stage;
    std::cout << fields.dump() << std::endl;
}
static Json describe(const PhoenixWow64ProbeResult& r) {
    return {{"worker_stage", r.stage}, {"status", r.status}, {"win32_error", r.win32_error},
            {"fatbin_supported", r.fatbin_supported}, {"module_created", r.module_created},
            {"blob_bytes", r.blob_bytes}};
}

#ifndef _WIN64
using Call64 = uint64_t(__cdecl*)(uint64_t, int, ...);
using Module64 = uint64_t(__cdecl*)(const wchar_t*);
using Proc64 = uint64_t(__cdecl*)(uint64_t, const char*);
static uint64_t address(const void* p) { return uint64_t(uintptr_t(p)); }
struct Unicode64 {
    uint16_t length, maximum;
    uint32_t padding;
    uint64_t buffer;
};
static_assert(sizeof(Unicode64) == 16);
#endif

int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    try {
#ifdef _WIN64
        if (argc != 4 || std::wstring(argv[1]) != L"--worker") return 1;
        event("native_worker_load");
        auto library = LoadLibraryW(argv[2]);
        if (!library) { event("load_result", {{"win32_error", GetLastError()}}); return 2; }
        auto worker = reinterpret_cast<PhoenixWow64ProbeProc>(GetProcAddress(library, "PhoenixWow64Probe64"));
        if (!worker) return 2;
        PhoenixWow64ProbeResult result{};
        event("native_worker_call");
        const auto code = worker(argv[3], &result);
        event("worker_result", describe(result));
        return code;
#else
        if (argc < 2) return 1;
        BOOL wow64 = FALSE;
        if (!IsWow64Process(GetCurrentProcess(), &wow64) || !wow64) return 1;
        auto extension = LoadLibraryW(L"wow64ext.dll");
        if (!extension) { event("extension_error", {{"win32_error", GetLastError()}}); return 1; }
        auto call = reinterpret_cast<Call64>(GetProcAddress(extension, "X64Call"));
        auto module = reinterpret_cast<Module64>(GetProcAddress(extension, "GetModuleHandle64"));
        auto proc = reinterpret_cast<Proc64>(GetProcAddress(extension, "GetProcAddress64"));
        if (!call || !module || !proc) throw std::runtime_error("Missing cross-mode exports");
        const auto ntdll = module(L"ntdll.dll");
        const auto crc = proc(ntdll, "RtlComputeCrc32");
        if (!ntdll || !crc) throw std::runtime_error("Missing native NTDLL function");
        const char input[] = "Phoenix DLSS5 cross-bitness probe";
        using Crc32 = ULONG(NTAPI*)(ULONG, const void*, ULONG);
        auto crc32 = reinterpret_cast<Crc32>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlComputeCrc32"));
        if (!crc32) return 1;
        event("mode_switch_call");
        const auto actual = uint32_t(call(crc, 3, uint64_t(0), address(input), uint64_t(sizeof(input)-1)));
        const auto expected = crc32(0, input, sizeof(input)-1);
        event("mode_switch_result", {{"matched", actual == expected}, {"crc", actual}});
        if (actual != expected) return 2;
        const std::wstring mode(argv[1]);
        if (mode == L"--mode-switch") return 0;
        std::wstring path;
        const bool activation = mode == L"--activation" || mode == L"--thread-activation";
        const bool native_thread = (mode == L"--thread" || mode == L"--thread-noattach" || mode == L"--thread-activation") && (argc == 4 || argc == 5);
        if ((mode == L"--worker" && argc == 4) || native_thread || (activation && (argc == 4 || argc == 5))) path = argv[2];
        else if (mode == L"--nvapi" || mode == L"--d3d12" || mode == L"--kernel32" || mode == L"--kernelbase") {
            wchar_t windows[MAX_PATH]{};
            if (!GetWindowsDirectoryW(windows, MAX_PATH)) return 1;
            const auto library = mode == L"--nvapi" ? L"nvapi64.dll" : mode == L"--d3d12" ? L"d3d12.dll" :
                                 mode == L"--kernel32" ? L"kernel32.dll" : L"kernelbase.dll";
            path = std::wstring(windows)+L"\\System32\\"+library;
        } else return 1;
        const auto load = proc(ntdll, "LdrLoadDll");
        if (!load || path.size() > 30000) return 1;
        Unicode64 name{uint16_t(path.size()*2), uint16_t((path.size()+1)*2), 0, address(path.c_str())};
        uint64_t loaded = 0;
        const auto filename = path.substr(path.find_last_of(L"/\\")+1);
        event("native_loader_call", {{"library", std::string(filename.begin(), filename.end())}});
        if (IsDebuggerPresent() && mode == L"--kernel32") DebugBreak();
        const auto status = uint32_t(call(load, 4, uint64_t(0), uint64_t(0), address(&name), address(&loaded)));
        event("native_loader_result", {{"ntstatus", status}, {"loaded", loaded != 0}});
        if (int32_t(status) < 0 || !loaded) return 2;
        if (native_thread || activation) {
            const auto start = proc(loaded, "PhoenixWow64Thread64");
            const auto create = proc(ntdll, "NtCreateThreadEx");
            const auto lookup = proc(ntdll, "LdrGetProcedureAddress");
            if (!start || !create || !lookup) return 2;
            PhoenixWow64ThreadContext context{};
            context.loader = load;
            context.procedure_lookup = lookup;
            context.library_path = address(argv[3]);
            context.kernel_path = argc == 5 ? address(argv[4]) : 0;
            if (activation) {
                context.activation_allocate = proc(ntdll, "RtlAllocateActivationContextStack");
                context.activation_free = proc(ntdll, "RtlFreeActivationContextStack");
                if (!context.activation_allocate || !context.activation_free) return 2;
                if (!native_thread) {
                    event("activation_loader_call");
                    const auto code = int32_t(call(start, 1, address(&context)));
                    auto details = describe(context.result);
                    details["activation_prepared"] = context.activation_prepared;
                    details["activation_status"] = context.activation_status;
                    details["ntstatus"] = uint32_t(context.loader_status);
                    details["library_loaded"] = context.library != 0;
                    event("activation_loader_result", details);
                    return code;
                }
            }
            uint64_t thread_handle = 0;
            const auto flags = mode == L"--thread-noattach" || mode == L"--thread-activation" ? uint64_t(2) : uint64_t(0);
            event("native_thread_create_call", {{"creation_flags", flags}});
            const auto created = uint32_t(call(create, 11, address(&thread_handle), uint64_t(0x1fffff), uint64_t(0),
                uint64_t(-1), start, address(&context), flags, uint64_t(0), uint64_t(0), uint64_t(0), uint64_t(0)));
            event("native_thread_create_result", {{"ntstatus", created}, {"handle_created", thread_handle != 0}});
            if (int32_t(created) < 0 || !thread_handle) return 2;
            const auto handle = reinterpret_cast<HANDLE>(uintptr_t(thread_handle));
            const auto waited = WaitForSingleObject(handle, 20000);
            DWORD code = 1;
            if (waited == WAIT_OBJECT_0) GetExitCodeThread(handle, &code);
            CloseHandle(handle);
            auto details = describe(context.result);
            details["thread_stage"] = context.stage;
            details["ntstatus"] = uint32_t(context.loader_status);
            details["library_loaded"] = context.library != 0;
            details["wait_result"] = waited;
            details["thread_exit_code"] = code;
            details["activation_prepared"] = context.activation_prepared;
            details["activation_status"] = context.activation_status;
            event("native_thread_result", details);
            return waited == WAIT_OBJECT_0 ? int(code) : 3;
        }
        if (mode == L"--nvapi") {
            const auto query = proc(loaded, "nvapi_QueryInterface");
            if (!query) return 2;
            const auto initialize = call(query, 1, uint64_t(0x0150e828));
            if (!initialize) return 2;
            event("native_nvapi_initialize_call");
            const auto code = int32_t(call(initialize, 0));
            event("native_nvapi_initialize_result", {{"status", code}});
            return code == 0 ? 0 : 2;
        }
        if (mode == L"--worker") {
            const auto worker = proc(loaded, "PhoenixWow64Probe64");
            if (!worker) return 2;
            PhoenixWow64ProbeResult result{};
            event("cross_worker_call");
            const auto code = int32_t(call(worker, 2, address(argv[3]), address(&result)));
            event("worker_result", describe(result));
            return code;
        }
        return 0;
#endif
    } catch (const std::exception& error) {
        event("error", {{"message", error.what()}});
        return 1;
    }
}
