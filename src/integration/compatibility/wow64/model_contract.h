#pragma once
#include "driver_contract.h"
namespace phoenix_wow64 {
struct alignas(8) ModelChannel {
    phoenix_driver_probe::Channel base;
    uint64_t config_json=0, parameters_json=0;
    uint32_t backend=3, bytes=0;
    uint32_t nvidia_create_result=0,nvidia_evaluate_result=0;
    uint64_t nvidia_creates=0,nvidia_evaluates=0;
    // Optional bounded caller-owned telemetry. Legacy packets end before this
    // extension; both peers check bytes before accessing it.
    uint64_t status_json=0;
    // Former reserved word: nonzero is used only after hash-bound bootstrap
    // capability negotiation. Old peers continue to receive RGBA16F (zero).
    uint32_t status_capacity=0, pixel_transport=0;
};
inline constexpr unsigned model_channel_legacy_bytes = offsetof(ModelChannel,status_json);
inline constexpr unsigned library_frame_services_version = 1;
inline bool model_channel_has_frame_services(unsigned bytes) { return bytes==sizeof(ModelChannel); }
inline bool valid_model_channel_bytes(unsigned bytes) {
    return bytes==model_channel_legacy_bytes||model_channel_has_frame_services(bytes);
}
inline bool declared_library_frame_services(unsigned version,unsigned bytes) {
    return version==library_frame_services_version&&model_channel_has_frame_services(bytes);
}
inline constexpr unsigned rgba8_gpu_transport_version=1;
inline bool declared_rgba8_gpu_transport(unsigned version,unsigned services,unsigned bytes) {
    return version==rgba8_gpu_transport_version&&declared_library_frame_services(services,bytes);
}
inline bool valid_model_frame_buffers(const ModelChannel& packet) {
    if(!valid_model_channel_bytes(packet.bytes))return false;
    const auto& c=packet.base;
    const unsigned transport=model_channel_has_frame_services(packet.bytes)?packet.pixel_transport:0;
    const bool compact=transport==rgba8_gpu_transport_version;
    if(transport>rgba8_gpu_transport_version||
       (compact&&(!model_channel_has_frame_services(packet.bytes)||packet.backend!=3)))return false;
    const auto bytes=uint64_t(c.width)*c.height*(compact?4:8);
    return c.width&&c.height&&c.width<=8192&&c.height<=8192&&c.plane_bytes==bytes&&
        phoenix_driver_probe::valid_host_range(c.color,bytes,32)&&phoenix_driver_probe::valid_host_range(c.output,bytes,32)&&
        (c.color+bytes<=c.output||c.output+bytes<=c.color);
}
// A negotiated bootstrap can defer CPU image storage until the first frame.
// No pixels are read during device creation. Frame requests still require the
// complete, nonoverlapping caller buffers checked by valid_model_frame_buffers.
inline bool valid_model_bootstrap_buffers(const ModelChannel& packet) {
    if(model_channel_has_frame_services(packet.bytes) &&
       packet.pixel_transport==rgba8_gpu_transport_version && packet.backend==3 &&
       packet.base.color==0 && packet.base.output==0) {
        const auto& c=packet.base;
        return c.width&&c.height&&c.width<=8192&&c.height<=8192&&
            c.plane_bytes==uint64_t(c.width)*c.height*4;
    }
    return (!model_channel_has_frame_services(packet.bytes)||packet.pixel_transport==0)&&
        valid_model_frame_buffers(packet);
}
static_assert(model_channel_legacy_bytes==48376 && sizeof(ModelChannel)==48392);
static_assert(std::is_standard_layout_v<ModelChannel> && std::is_trivially_copyable_v<ModelChannel>);
}
