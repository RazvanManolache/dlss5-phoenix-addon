// Import-free control: proves x64 DLL execution, not graphics compatibility.
#include "wow64_probe_contract.h"
#include <intrin.h>

extern "C" __declspec(dllexport) int __cdecl PhoenixWow64Probe64(
    const wchar_t*, PhoenixWow64ProbeResult* result) {
    if (!result) return 1;
    result->stage = 100;
    result->status = 0;
    result->win32_error = 0;
    result->fatbin_supported = 0;
    result->module_created = 0;
    result->blob_bytes = 0;
    return 0;
}

extern "C" int __stdcall DllMain(void*, unsigned, void*) { return 1; }

struct NativeString {
    uint16_t length, maximum;
    uint32_t padding;
    uint64_t buffer;
};
static_assert(sizeof(NativeString) == 16);

static unsigned invoke(PhoenixWow64ThreadContext* context) {
    using Load = int32_t(__stdcall*)(void*, void*, NativeString*, uint64_t*);
    using Lookup = int32_t(__stdcall*)(uint64_t, NativeString*, unsigned, uint64_t*);
    auto load = reinterpret_cast<Load>(context->loader);
    auto lookup = reinterpret_cast<Lookup>(context->procedure_lookup);
    auto* path = reinterpret_cast<const wchar_t*>(context->library_path);
    unsigned characters = 0;
    while (characters < 30000 && path[characters]) ++characters;
    if (characters == 30000) return 1;
    NativeString name{uint16_t(characters*2), uint16_t((characters+1)*2), 0, context->library_path};
    context->stage = 1;
    context->loader_status = load(nullptr, nullptr, &name, &context->library);
    if (context->loader_status < 0 || !context->library) return 2;
    context->stage = 2;
    if (!context->kernel_path) return 0;
    const char entry[] = "PhoenixWow64Probe64";
    NativeString symbol{sizeof(entry)-1, sizeof(entry), 0, reinterpret_cast<uint64_t>(entry)};
    uint64_t address = 0;
    context->loader_status = lookup(context->library, &symbol, 0, &address);
    if (context->loader_status < 0 || !address) return 2;
    context->stage = 3;
    const auto code = reinterpret_cast<PhoenixWow64ProbeProc>(address)(
        reinterpret_cast<const wchar_t*>(context->kernel_path), &context->result);
    context->stage = 4;
    return unsigned(code);
}

extern "C" __declspec(dllexport) unsigned __stdcall PhoenixWow64Thread64(void* argument) {
    auto* context = static_cast<PhoenixWow64ThreadContext*>(argument);
    if (!context) return 1;
    uint64_t* slot = nullptr;
    uint64_t original = 0, allocated = 0;
    if (context->activation_allocate && context->activation_free) {
        // Verified against this machine's ntdll _TEB symbols. Experimental only;
        // this private offset is not a portable Windows ABI or product code.
        slot = reinterpret_cast<uint64_t*>(__readgsqword(0x30) + 0x2c8);
        original = *slot;
        using Allocate = int32_t(__stdcall*)(uint64_t*);
        context->activation_status = reinterpret_cast<Allocate>(context->activation_allocate)(&allocated);
        if (context->activation_status < 0 || !allocated) return 3;
        *slot = allocated;
        context->activation_prepared = 1;
    }
    const auto code = invoke(context);
    if (allocated) {
        *slot = original;
        using Free = void(__stdcall*)(uint64_t);
        reinterpret_cast<Free>(context->activation_free)(allocated);
    }
    return code;
}
