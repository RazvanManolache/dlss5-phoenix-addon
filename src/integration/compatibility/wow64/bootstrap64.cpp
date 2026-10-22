// Minimal PE static-TLS template. Only the disposable caller's native state
// is inspected. No driver, registry, system file, or game is patched.
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#include "bootstrap_contract.h"
#include "umd_bridge.h"

#pragma section(".tls", read, write)
#pragma section(".tls$ZZZ", read, write)
extern "C" {
int _fltused = 0;
volatile unsigned long _tls_index = 0;
__declspec(allocate(".tls")) unsigned char tls_start = 0;
__declspec(thread) volatile uint64_t tls_cookie = 0x13579bdf2468ace0ull;
__declspec(thread) volatile unsigned char tls_zero[16]{};
__declspec(allocate(".tls$ZZZ")) unsigned char tls_end = 0;
}
static volatile uint32_t callback_count = 0;
static void* callback_dispatch = nullptr;
static void** callback_table = nullptr;
alignas(16) static uint64_t gui_peb_before[0x7d8/8]{},gui_peb_shadow[0x7d8/8]{};
static bool bridge_callbacks(void** slot,void* original,void* native_table=nullptr) {
    if(!native_table)native_table=*slot;
    if(!original||native_table==original)return true;
    // Kernel GUI callbacks use buffer arguments. These tail thunks preserve
    // every argument/register except volatile R11 and retain the original stack.
    // Native threads have loader-managed native TLS; ordinary WOW64 game
    // threads retain their original callback dispatcher. No system image is
    // modified. The process-owned RX thunks remain valid until process exit.
    constexpr unsigned entries=256,stride=48;
    if(!callback_dispatch) {
        callback_dispatch=VirtualAlloc(nullptr,entries*stride,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
        callback_table=static_cast<void**>(VirtualAlloc(nullptr,entries*sizeof(void*),MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
        if(!callback_dispatch||!callback_table)return false;
        auto* native=static_cast<void**>(native_table);
        auto* wow=static_cast<void**>(original);
        for(unsigned i=0;i<entries;++i) {
            auto* code=static_cast<unsigned char*>(callback_dispatch)+i*stride;
            // mov r11,gs:[58h]; test r11,r11; jz wow64; mov r11,target; jmp r11
            const unsigned char head[]{0x65,0x4c,0x8b,0x1c,0x25,0x58,0,0,0,0x4d,0x85,0xdb,0x74,0x0d,0x49,0xbb};
            for(unsigned j=0;j<sizeof(head);++j)code[j]=head[j];
            const auto a=uint64_t(native[i]),b=uint64_t(wow[i]);
            for(unsigned j=0;j<8;++j)code[16+j]=static_cast<unsigned char>(a>>(8*j));
            code[24]=0x41;code[25]=0xff;code[26]=0xe3;
            code[27]=0x49;code[28]=0xbb;
            for(unsigned j=0;j<8;++j)code[29+j]=static_cast<unsigned char>(b>>(8*j));
            code[37]=0x41;code[38]=0xff;code[39]=0xe3;
            callback_table[i]=code;
        }
        DWORD old=0;
        if(!VirtualProtect(callback_dispatch,entries*stride,PAGE_EXECUTE_READ,&old))return false;
        FlushInstructionCache(GetCurrentProcess(),callback_dispatch,entries*stride);
    }
    _InterlockedExchangePointer(slot,callback_table);
    return true;
}
static bool initialize_native_gui(PhoenixTlsTask& task,void** callbacks,void* game_callbacks) {
    if(GetModuleHandleW(L"user32.dll"))return bridge_callbacks(callbacks,game_callbacks);
    // USER32 sets its callback table during DllMain. A live x86 DXGI monitor
    // callback can run before LoadLibrary returns, so restoring afterward is
    // too late. Isolate that one initialization through a native thread-local
    // PEB view. The loader/heap pointers stay shared; only scalar/pointer fields
    // changed by initialization are merged back, excluding the callback table.
    // The PEB size/offset are verified for the hash-bound ntdll profile. Keep
    // this view process-owned in case a native dependency retains its address.
    auto* original=reinterpret_cast<volatile uint64_t*>(__readgsqword(0x60));
    for(unsigned i=0;i<sizeof(gui_peb_before)/8;++i)gui_peb_before[i]=gui_peb_shadow[i]=original[i];
    __writegsqword(0x60,uint64_t(gui_peb_shadow));
    const auto library=LoadLibraryExW(L"user32.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto error=library?0:GetLastError();
    __writegsqword(0x60,uint64_t(original));
    if(!library) {task.load_error=error;return false;}
    for(unsigned i=0;i<sizeof(gui_peb_before)/8;++i) {
        if(i==0x58/8||gui_peb_before[i]==gui_peb_shadow[i])continue;
        _InterlockedCompareExchange64(reinterpret_cast<volatile long long*>(original+i),
            static_cast<long long>(gui_peb_shadow[i]),static_cast<long long>(gui_peb_before[i]));
    }
    return bridge_callbacks(callbacks,game_callbacks,reinterpret_cast<void*>(gui_peb_shadow[0x58/8]));
}
static void NTAPI callback(void*, DWORD reason, void*) {
    if (reason == DLL_PROCESS_ATTACH || reason == DLL_THREAD_ATTACH) ++callback_count;
}
static const PIMAGE_TLS_CALLBACK callbacks[]{callback, nullptr};
#pragma section(".rdata$T", read)
extern "C" {
__declspec(allocate(".rdata$T")) extern const IMAGE_TLS_DIRECTORY64 _tls_used = {
    (ULONGLONG)&tls_start, (ULONGLONG)&tls_end, (ULONGLONG)&_tls_index,
    (ULONGLONG)callbacks, 0, 0};
}
#pragma comment(linker, "/INCLUDE:_tls_used")

extern "C" __declspec(dllexport) unsigned __cdecl PhoenixTlsInspect64(PhoenixTlsProbe* result) {
    if (!result) return 1;
    result->stage = 1;
    result->teb = __readgsqword(0x30);
    result->vector_before = __readgsqword(0x58);
    result->index = _tls_index;
    result->callbacks = callback_count;
    if (result->initializer && !result->vector_before) {
        using Lock = int32_t(NTAPI*)(unsigned, unsigned*, uint64_t*);
        using Unlock = int32_t(NTAPI*)(unsigned, uint64_t);
        using Initialize = int32_t(NTAPI*)();
        if (!result->loader_lock || !result->loader_unlock) return 2;
        uint64_t cookie = 0;
        unsigned disposition = 0;
        result->lock_status = reinterpret_cast<Lock>(result->loader_lock)(0, &disposition, &cookie);
        if (result->lock_status < 0) return 2;
        result->stage = 6;
        result->initializer_status = reinterpret_cast<Initialize>(result->initializer)();
        result->stage = 7;
        reinterpret_cast<Unlock>(result->loader_unlock)(0, cookie);
    }
    if (result->handler && !result->vector_before) {
        using Finder = int32_t(NTAPI*)(void*, uint64_t*);
        using Lock = int32_t(NTAPI*)(unsigned, unsigned*, uint64_t*);
        using Unlock = int32_t(NTAPI*)(unsigned, uint64_t);
        using Handler = int32_t(NTAPI*)(uint64_t);
        if (!result->finder || !result->loader_lock || !result->loader_unlock) return 2;
        uint64_t entry = 0, cookie = 0;
        unsigned disposition = 0;
        result->lock_status = reinterpret_cast<Lock>(result->loader_lock)(0, &disposition, &cookie);
        if (result->lock_status < 0) return 2;
        result->lookup_status = reinterpret_cast<Finder>(result->finder)(reinterpret_cast<void*>(result->module), &entry);
        result->stage = 2;
        if (result->lookup_status >= 0 && entry) {
            result->stage = 3;
            result->handler_status = reinterpret_cast<Handler>(result->handler)(entry);
            result->stage = 4;
        }
        reinterpret_cast<Unlock>(result->loader_unlock)(0, cookie);
    }
    result->vector_after = __readgsqword(0x58);
    result->index = _tls_index;
    if (!result->vector_after || result->index >= 4096) return 0;
    result->slot = reinterpret_cast<uint64_t*>(result->vector_after)[result->index];
    if (!result->slot) return 0;
    result->cookie = tls_cookie;
    result->template_ok = result->cookie == 0x13579bdf2468ace0ull;
    result->zero_fill_ok = 1;
    for (unsigned i = 0; i < 16; ++i) result->zero_fill_ok &= tls_zero[i] == 0;
    result->stage = 5;
    return 0;
}
extern "C" __declspec(dllexport) unsigned NTAPI PhoenixTlsThread64(void* context) {
    return PhoenixTlsInspect64(static_cast<PhoenixTlsProbe*>(context));
}
extern "C" __declspec(dllexport) unsigned NTAPI PhoenixTlsRunTask64(void* context) {
    auto* task = static_cast<PhoenixTlsTask*>(context);
    if (!task || PhoenixTlsInspect64(&task->tls)) return 2;
    // Process TLS initialization is one-shot. Later native workers need their
    // own loader-managed slots for every already-loaded native image.
    if(!task->tls.vector_after&&task->thread_allocator) {
        using Allocate=int32_t(NTAPI*)();
        task->tls.initializer_status=reinterpret_cast<Allocate>(task->thread_allocator)();
        if(task->tls.initializer_status<0||PhoenixTlsInspect64(&task->tls)) return 2;
    }
    if(!task->tls.template_ok||!task->tls.zero_fill_ok) return 2;
    task->stack_base = __readgsqword(8);
    task->stack_limit = __readgsqword(16);
    task->stack_address = uint64_t(&task);
    // The game's application directory normally contains an x86 ReShade proxy.
    // Native worker imports must resolve against its own folder and System32,
    // not that x86 dxgi.dll/d3d12.dll or the game's current directory.
    // Native USER32 initialization replaces the process callback table. Route
    // callbacks by thread rather than entering an x86 WndProc in long mode or
    // sending a native driver's worker through an uninitialized x86 context.
    auto** callbacks=reinterpret_cast<void**>(__readgsqword(0x60)+0x58);
    void* game_callbacks=*callbacks;
    auto* rpc=task->shared_rpc?reinterpret_cast<PhoenixGdiRpc*>(task->shared_rpc):&task->gdi;
    if(task->kind==4||task->kind==6) {
        // Install before USER32 and the renderer DLL initialize their TLS.
        // The shared x86 service remains alive after this native worker exits.
        if(!task->shared_rpc)return 2;
        wow64_umd_probe::gdi=rpc;
        if(!wow64_umd_probe::tls_exports[0].slot&&!wow64_umd_probe::install_tls_exports())return 2;
        // Install mixed-bitness callbacks before DXGI/driver initializers can
        // send messages to existing game windows during model-library loading.
        if(!initialize_native_gui(*task,callbacks,game_callbacks))return 2;
        game_callbacks=*callbacks;
    }
    auto library = task->kind==4||task->kind==6 ?
        LoadLibraryExW(reinterpret_cast<const wchar_t*>(task->path),nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32) :
        LoadLibraryW(reinterpret_cast<const wchar_t*>(task->path));
    const auto load_error=library?0:GetLastError();
    if(!bridge_callbacks(callbacks,game_callbacks))return 2;
    task->loaded = library != nullptr;
    if (!library) { task->load_error = load_error; return 2; }
    if (task->kind == 1) {
        using Run = int(__cdecl*)(const wchar_t*, PhoenixWow64ProbeResult*);
        auto run = reinterpret_cast<Run>(GetProcAddress(library, "PhoenixWow64Probe64"));
        if (!run) return 2;
        auto configure = reinterpret_cast<void(*)(PhoenixGdiRpc*)>(GetProcAddress(library, "PhoenixSetGdiRpc64"));
        if (configure) configure(rpc);
        if(task->cuda_library) {
            auto cuda_path=reinterpret_cast<void(*)(const wchar_t*)>(GetProcAddress(library,"PhoenixSetCudaLibrary64"));
            if(!cuda_path) return 2;
            cuda_path(reinterpret_cast<const wchar_t*>(task->cuda_library));
        }
        task->status = run(reinterpret_cast<const wchar_t*>(task->kernel), &task->gpu);
        auto redirects = reinterpret_cast<unsigned(*)()>(GetProcAddress(library, "PhoenixUmdRedirectCount64"));
        if (redirects) task->umd_redirects = redirects();
    } else if (task->kind == 3) {
        auto run = reinterpret_cast<int(*)(const wchar_t*)>(GetProcAddress(library, "PhoenixPortableFrameProbe64"));
        if (!run) return 2;
        auto configure = reinterpret_cast<void(*)(PhoenixGdiRpc*)>(GetProcAddress(library, "PhoenixSetGdiRpc64"));
        if (configure) configure(rpc);
        task->status = run(reinterpret_cast<const wchar_t*>(task->kernel));
    } else if (task->kind == 4 || task->kind == 6) {
        auto run = reinterpret_cast<int(*)(void*)>(GetProcAddress(library,task->kind==4?"PhoenixDriverServe64":"PhoenixModelServe64"));
        auto configure = reinterpret_cast<void(*)(PhoenixGdiRpc*)>(GetProcAddress(library,"PhoenixSetGdiRpc64"));
        if(!run||!configure) return 2;
        configure(rpc);
        task->status=run(reinterpret_cast<void*>(task->kernel));
    } else if (task->kind == 2) {
        struct Controls { uint32_t bytes, reset; float mx, my, tone, structure, skin; uint32_t mask, inverted; } controls{};
        using Defaults = void(__cdecl*)(Controls*);
        using Frame = int(__cdecl*)(void*, const void*, uint64_t, uint32_t, const void*, uint64_t, uint32_t,
            void*, uint64_t, uint32_t, const Controls*);
        using Error = const char*(__cdecl*)();
        auto defaults = reinterpret_cast<Defaults>(GetProcAddress(library, "phoenix_native_default_controls"));
        auto frame = reinterpret_cast<Frame>(GetProcAddress(library, "phoenix_native_frame_host"));
        auto error = reinterpret_cast<Error>(GetProcAddress(library, "phoenix_native_error"));
        if (!defaults || !frame || !error) return 2;
        defaults(&controls);
        task->defaults_ok = controls.bytes == 36 && controls.mx == 1 && controls.my == 1 && controls.tone == 1 &&
            controls.structure == 1 && controls.skin == -1 && controls.mask == 0 && controls.inverted == 1;
        task->status = frame(nullptr, nullptr, 0, 0, nullptr, 0, 0, nullptr, 0, 0, &controls);
        const auto* message = error();
        if (!message) return 2;
        unsigned i = 0;
        for (; i < sizeof(task->message)-1 && message[i]; ++i) task->message[i] = message[i];
        task->message[i] = 0;
    } else return 2;
    task->completed = 1;
    return 0;
}
static volatile DWORD attached_pid = 0;
extern "C" BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) attached_pid = GetCurrentProcessId();
    return TRUE;
}
