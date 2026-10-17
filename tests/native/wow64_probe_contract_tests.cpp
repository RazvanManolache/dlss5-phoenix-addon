#include "wow64_probe_contract.h"
#include "wow64_driver_contract.h"
#include <windows.h>
#include <cstdio>
#include <initializer_list>

int main() {
    namespace bridge=phoenix_driver_probe;
    if(bridge::valid_host_range(0x100000000ull,16,32)||bridge::valid_host_range(UINT32_MAX,2,32)||
        bridge::valid_host_range(UINT64_MAX,2,64)||bridge::valid_host_range(0,16,32)||
        !bridge::valid_host_range(0x100000000ull,16,64)||!bridge::valid_host_range(UINT32_MAX,1,32)) return 2;
    bridge::Launch launch{}; launch.bytes=264;
    for(unsigned i=0;i<3;++i) launch.grid[i]=launch.block[i]=1;
    if(!bridge::valid_launch(launch)) return 2;
    launch.kernel=41; if(bridge::valid_launch(launch)) return 2; launch.kernel=0;
    launch.bytes=265; if(bridge::valid_launch(launch)) return 2; launch.bytes=264;
    launch.block[0]=1024; launch.block[1]=2;
    if(bridge::valid_launch(launch)) return 2;
    launch.block[0]=launch.block[1]=1; launch.grid[2]=0;
    if(bridge::valid_launch(launch)) return 2;
    bridge::Channel channel{};
    channel.caller_bits=32; channel.width=640; channel.height=384;
    channel.request_event=channel.response_event=channel.weights=channel.color=channel.output=channel.cuda_library=1;
    channel.weight_bytes=4096; channel.workspace_bytes=8192; channel.clear_words=16; channel.plane_bytes=uint64_t(640)*384*8;
    if(!bridge::valid_initialize(channel)) return 2;
    channel.struct_bytes--; if(bridge::valid_initialize(channel)) return 2; channel.struct_bytes++;
    channel.width=UINT32_MAX; if(bridge::valid_initialize(channel)) return 2; channel.width=640;
    channel.frame_flags=4; if(bridge::valid_initialize(channel)) return 2; channel.frame_flags=3;
    if(!bridge::valid_initialize(channel)) return 2; channel.frame_flags=0;
    channel.reserved=1; if(bridge::valid_initialize(channel)) return 2; channel.reserved=0;
    channel.clear_words=UINT32_MAX; if(bridge::valid_initialize(channel)) return 2;
    std::printf("{\"event\":\"driver_packet_contract\",\"channel_bytes\":%u,\"launch_offset\":%u,\"pointer_bits\":%u}\n",
        unsigned(sizeof(bridge::Channel)),unsigned(offsetof(bridge::Channel,launches)),unsigned(sizeof(void*)*8));
    static_assert(MEM_RESERVE==0x2000 && PAGE_NOACCESS==1);
    constexpr uint64_t maximum=0x7ffffffeffffull, bytes=8657043456ull;
    if(!PhoenixMayRetryNativeReservation(true,true,0,bytes,MEM_RESERVE,PAGE_NOACCESS,false,0,maximum)) return 2;
    if(PhoenixMayRetryNativeReservation(false,true,0,bytes,MEM_RESERVE,PAGE_NOACCESS,false,0,maximum) ||
       PhoenixMayRetryNativeReservation(true,false,0,bytes,MEM_RESERVE,PAGE_NOACCESS,false,0,maximum) ||
       PhoenixMayRetryNativeReservation(true,true,1,bytes,MEM_RESERVE,PAGE_NOACCESS,false,0,maximum) ||
       PhoenixMayRetryNativeReservation(true,true,0,0,MEM_RESERVE,PAGE_NOACCESS,false,0,maximum) ||
       PhoenixMayRetryNativeReservation(true,true,0,bytes,MEM_COMMIT|MEM_RESERVE,PAGE_NOACCESS,false,0,maximum) ||
       PhoenixMayRetryNativeReservation(true,true,0,bytes,MEM_RESERVE,PAGE_READWRITE,false,0,maximum) ||
       PhoenixMayRetryNativeReservation(true,true,0,bytes,MEM_RESERVE,PAGE_NOACCESS,true,0,maximum) ||
       PhoenixMayRetryNativeReservation(true,true,0,bytes,MEM_RESERVE,PAGE_NOACCESS,false,1,maximum) ||
       PhoenixMayRetryNativeReservation(true,true,0,bytes,MEM_RESERVE,PAGE_NOACCESS,false,0,UINT32_MAX) ||
       PhoenixMayRetryNativeReservation(true,true,0,UINT64_MAX,MEM_RESERVE,PAGE_NOACCESS,false,0,maximum)) return 2;
    PhoenixGdiCreateDC create{};
    if(!PhoenixValidGdiCreateDC(create,sizeof(DEVMODEA))) return 2;
    create.present=8;
    if(PhoenixValidGdiCreateDC(create,sizeof(DEVMODEA))) return 2;
    create.present=7; create.mode_bytes=sizeof(DEVMODEA);
    if(!PhoenixValidGdiCreateDC(create,sizeof(DEVMODEA))) return 2;
    create.mode_bytes=sizeof(DEVMODEA)-1;
    if(PhoenixValidGdiCreateDC(create,sizeof(DEVMODEA))) return 2;
    create.mode_bytes=sizeof(create.mode)+1;
    if(PhoenixValidGdiCreateDC(create,sizeof(DEVMODEA))) return 2;
    create.mode_bytes=sizeof(create.mode);
    if(!PhoenixValidGdiCreateDC(create,sizeof(DEVMODEA))) return 2;
    for(auto field:{create.driver,create.device,create.output}) {
        field[127]='x';
        if(PhoenixValidGdiCreateDC(create,sizeof(DEVMODEA))) return 2;
        field[127]=0;
    }
    PhoenixGdiDisplayDevice display{};
    if(!PhoenixValidGdiDisplayDevice(display)) return 2;
    display.present=2;
    if(PhoenixValidGdiDisplayDevice(display)) return 2;
    display.present=1; display.device[127]='x';
    if(PhoenixValidGdiDisplayDevice(display)) return 2;
    display.device[127]=0;
    if(!PhoenixValidGdiDisplayDevice(display)) return 2;
    static_assert(sizeof(DISPLAY_DEVICEA)==sizeof(display.display));
    PhoenixKmtEscape escape{};
    if(!PhoenixValidKmtEscape(escape)) return 2;
    escape.bytes=1;
    if(PhoenixValidKmtEscape(escape)) return 2;
    escape.data=UINT64_MAX;
    if(PhoenixValidKmtEscape(escape)) return 2;
    escape.data=0x100000000ull; escape.bytes=1024*1024;
    if(!PhoenixValidKmtEscape(escape)) return 2;
    ++escape.bytes;
    if(PhoenixValidKmtEscape(escape)) return 2;
    uint32_t offsets[]{0x2c8,0x2d0,0x2d8,0x2e0};
    if(!PhoenixValidNativeCompatOffsets(offsets)) return 2;
    for(auto invalid:{0u,0x2c9u,0x1000u,0x2d0u}) {
        offsets[0]=invalid;
        if(PhoenixValidNativeCompatOffsets(offsets)) return 2;
    }
    std::printf("{\"event\":\"gdi_contract\",\"passed\":true,\"pointer_bits\":%u}\n",unsigned(sizeof(void*)*8));
    return 0;
}
