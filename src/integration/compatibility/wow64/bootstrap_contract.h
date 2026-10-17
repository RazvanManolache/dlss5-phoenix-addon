#pragma once
#include <stdint.h>
#include "probe_contract.h"
struct PhoenixTlsProbe {
    uint64_t module, finder, handler, loader_lock, loader_unlock;
    uint64_t teb, vector_before, vector_after, slot, cookie;
    uint32_t stage, index, callbacks, template_ok, zero_fill_ok;
    int32_t lookup_status, lock_status, handler_status;
    uint64_t initializer;
    int32_t initializer_status;
};
static_assert(sizeof(PhoenixTlsProbe) == 128);
using PhoenixTlsProbeProc = unsigned(__cdecl*)(PhoenixTlsProbe*);
struct PhoenixTlsTask {
    PhoenixTlsProbe tls;
    uint64_t path, kernel, cuda_library;
    uint64_t stack_base, stack_limit, stack_address;
    uint32_t kind, loaded, defaults_ok, completed;
    int32_t status;
    uint32_t load_error;
    PhoenixWow64ProbeResult gpu;
    char message[256];
    uint32_t umd_redirects;
    PhoenixGdiRpc gdi;
    uint64_t thread_allocator;
};
static_assert(sizeof(PhoenixTlsTask) == 544);
