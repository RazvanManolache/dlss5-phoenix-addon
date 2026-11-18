#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Experimental same-process packets. No C++/COM object or native pointer-sized
// field crosses the architecture boundary; driver addresses stay 64-bit values.
namespace phoenix_driver_probe {
constexpr uint32_t version=3, kernel_count=41, launch_count=154;
constexpr uint32_t skip_readback=2, gpu_output_seed=4, packed_rgba8=8, reuse_refinement=16;
// A packed batch carries two equally sized launch lists, one for each parity.
// Its second list uses the host pointer fields otherwise read at startup.
// The pass count occupies otherwise unused flag bits in the v3 packet.
constexpr uint32_t packed_pass_shift=8, packed_pass_mask=15u<<packed_pass_shift;
enum Operation : int32_t { initialize=1, frame=2, shutdown=3, check_compatibility=4 };
struct CompatibilityEvidence {
    uint32_t abi=1,bytes=sizeof(CompatibilityEvidence),cuda_version=0,device_count=0;
    uint32_t cuda_initialized=0,adapter_matched=0,d3d12_copy_verified=0,nvapi_entries=0;
};
struct Kernel {
    uint64_t bytes, size, entry;
};
struct Launch {
    uint32_t kernel, bytes, grid[3], block[3], barrier_after, reserved;
    uint8_t arguments[264];
};
struct alignas(8) Channel {
    uint32_t abi=version, struct_bytes=sizeof(Channel), width=0, height=0;
    uint64_t request_event=0, response_event=0;
    int32_t operation=initialize, status=0;
    uint32_t response=0, caller_bits=0;
    union { uint64_t weights=0; uint64_t alternate_launches; };
    union { uint64_t weight_bytes=0; uint64_t alternate_launch_count; };
    uint64_t workspace_bytes=0;
    uint32_t clear_words=0, target=0;
    uint32_t frame_flags=0, launch_total=0;
    uint64_t color=0, output=0, plane_bytes=0, cuda_library=0;
    uint32_t ntdll_timestamp=0, appcompat_offsets[4]{};
    uint32_t driver_pid=0, driver_bits=0, phoenix64_loaded=0, helper_pid=0;
    uint32_t functions_destroyed=0, modules_destroyed=0, cleanup_failures=0, native_compat_restored=0;
    uint64_t workspace_address=0, weights_address=0;
    uint64_t color_texture=0, motion_texture=0, history_texture[2]{}, surface[2]{};
    uint64_t aux_texture=0;
    uint64_t completed_frames=0, dispatched_kernels=0;
    double gpu_ms=0;
    Kernel kernels[kernel_count]{};
    Launch launches[launch_count]{};
    char error[256]{};
};
static_assert(sizeof(Kernel)==24 && sizeof(Launch)==304);
static_assert(sizeof(Channel)==48328 && offsetof(Channel,launches)==1256);
static_assert(std::is_standard_layout_v<Channel> && std::is_trivially_copyable_v<Channel>);
inline bool valid_host_range(uint64_t address,uint64_t bytes,uint32_t bits) {
    const uint64_t maximum=bits==32?UINT32_MAX:UINT64_MAX;
    return (bits==32||bits==64) && address && bytes && address<=maximum && bytes-1<=maximum-address;
}
inline bool valid_launch(const Launch& value) {
    if(value.kernel>=kernel_count || !value.bytes || value.bytes>264 || value.bytes%4 ||
        value.barrier_after>1 || value.reserved) return false;
    uint64_t threads=1;
    for(unsigned axis=0;axis<3;++axis) {
        if(!value.grid[axis] || !value.block[axis] || value.block[axis]>1024) return false;
        threads*=value.block[axis];
    }
    return threads<=1024;
}
inline bool valid_pixel_planes(const Channel& value, Operation operation) {
    const uint64_t pixels=uint64_t(value.width)*value.height;
    const auto flags=value.frame_flags;
    if(!(flags&packed_rgba8))
        return flags<=7 && value.plane_bytes==pixels*8 &&
            valid_host_range(value.color,value.plane_bytes,value.caller_bits) &&
            valid_host_range(value.output,value.plane_bytes,value.caller_bits);
    if(operation==initialize)
        return flags==packed_rgba8 && !value.color && !value.output && !value.plane_bytes;
    const auto passes=(flags&packed_pass_mask)>>packed_pass_shift;
    if(passes && (passes<2 || passes>10 || (flags&(reuse_refinement|skip_readback))))return false;
    return !(flags&~(packed_rgba8|reuse_refinement|skip_readback|gpu_output_seed|packed_pass_mask)) &&
        (flags&gpu_output_seed) && value.plane_bytes==pixels*4 &&
        ((flags&reuse_refinement)?value.color==0:valid_host_range(value.color,value.plane_bytes,value.caller_bits)) &&
        ((flags&skip_readback)?value.output==0:valid_host_range(value.output,value.plane_bytes,value.caller_bits));
}
inline bool valid_program_storage(const Channel& value,Operation operation) {
    if(operation==frame && (value.frame_flags&packed_pass_mask)) {
        return value.alternate_launch_count==value.launch_total &&
            valid_host_range(value.alternate_launches,value.alternate_launch_count*sizeof(Launch),value.caller_bits);
    }
    return valid_host_range(value.weights,value.weight_bytes,value.caller_bits) && value.weight_bytes<=512ull*1024*1024;
}
inline bool valid_packet(const Channel& value, Operation operation) {
    return value.abi==version && value.struct_bytes==sizeof(Channel) && value.operation==operation && value.launch_total<=launch_count &&
        (value.caller_bits==32 || value.caller_bits==64) && value.width && value.height &&
        value.width<=8192 && value.height<=8192 && value.request_event && value.response_event &&
        valid_program_storage(value,operation) &&
        value.workspace_bytes && value.workspace_bytes<=4ull*1024*1024*1024 &&
        value.clear_words && uint64_t(value.clear_words)*4<value.workspace_bytes &&
        valid_pixel_planes(value,operation) && valid_host_range(value.cuda_library,2,value.caller_bits);
}
inline bool valid_initialize(const Channel& value) { return valid_packet(value,initialize); }
inline bool valid_frame(const Channel& value) { return valid_packet(value,frame) && value.launch_total>=1; }
inline bool valid_compatibility_check(const Channel& value) {
    return value.abi==version&&value.struct_bytes==sizeof(Channel)&&value.caller_bits==32&&
        value.operation==check_compatibility&&value.request_event&&value.response_event&&
        valid_host_range(value.cuda_library,2,32)&&valid_host_range(value.output,sizeof(CompatibilityEvidence),32)&&!value.launch_total;
}
}
