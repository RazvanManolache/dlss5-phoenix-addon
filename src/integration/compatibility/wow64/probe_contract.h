#pragma once
#include <stdint.h>

struct PhoenixWow64ProbeResult {
    uint32_t stage;
    int32_t status;
    uint32_t win32_error;
    uint32_t fatbin_supported;
    uint32_t module_created;
    uint32_t blob_bytes;
};
using PhoenixWow64ProbeProc = int(__cdecl*)(const wchar_t*, PhoenixWow64ProbeResult*);
static_assert(sizeof(PhoenixWow64ProbeResult) == 24);
struct PhoenixGdiRpc {
    uint64_t request_event, response_event, handle, result;
    volatile int32_t pending;
    uint32_t requests, error;
    volatile int32_t busy;
};
static_assert(sizeof(PhoenixGdiRpc) == 48);
struct PhoenixGdiCreateDC {
    uint32_t present, mode_bytes;
    char driver[128], device[128], output[128];
    unsigned char mode[512];
};
static_assert(sizeof(PhoenixGdiCreateDC) == 904);
struct PhoenixGdiDisplayDevice {
    uint32_t index, flags, present;
    char device[128];
    unsigned char display[424];
};
static_assert(sizeof(PhoenixGdiDisplayDevice) == 564);
struct PhoenixKmtEscape {
    uint32_t adapter, device, type, flags;
    uint64_t data;
    uint32_t bytes, context;
};
static_assert(sizeof(PhoenixKmtEscape) == 32);
inline constexpr uint64_t PhoenixNativeReservationLower=0x10000000000ull;
constexpr bool PhoenixMayRetryNativeReservation(bool enabled,bool current,uint64_t address,uint64_t bytes,
    uint32_t flags,uint32_t protect,bool parameters,uint32_t count,uint64_t maximum) {
    return enabled && current && !address && bytes && flags==0x2000 && protect==1 && !parameters && !count &&
        maximum>=PhoenixNativeReservationLower && bytes<=maximum-PhoenixNativeReservationLower+1;
}
constexpr bool PhoenixValidKmtEscape(const PhoenixKmtEscape& value) {
    return value.bytes <= 1024 * 1024 && (!value.bytes ||
        (value.data && value.data <= UINT64_MAX - value.bytes));
}
constexpr bool PhoenixValidGdiCreateDC(const PhoenixGdiCreateDC& value,uint32_t minimum_mode_bytes) {
    return !(value.present&~7u) && value.mode_bytes<=sizeof(value.mode) &&
        (!value.mode_bytes || value.mode_bytes>=minimum_mode_bytes) &&
        !value.driver[127] && !value.device[127] && !value.output[127];
}
constexpr bool PhoenixValidGdiDisplayDevice(const PhoenixGdiDisplayDevice& value) {
    return value.present<=1 && !value.device[127];
}
template<class T> constexpr bool PhoenixValidNativeCompatOffsets(const T (&offsets)[4]) {
    static_assert(sizeof(T)==4);
    for(unsigned i=0;i<4;++i) {
        if(!offsets[i] || offsets[i]>=0x1000 || (offsets[i]&7)) return false;
        for(unsigned j=0;j<i;++j) if(offsets[j]==offsets[i]) return false;
    }
    return true;
}

struct PhoenixWow64ThreadContext {
    uint64_t loader, procedure_lookup, library_path, kernel_path, library;
    int32_t loader_status;
    uint32_t stage;
    PhoenixWow64ProbeResult result;
    uint64_t activation_allocate, activation_free;
    int32_t activation_status;
    uint32_t activation_prepared;
};
static_assert(sizeof(PhoenixWow64ThreadContext) == 96);
