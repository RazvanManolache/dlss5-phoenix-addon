#pragma once
#include "driver_contract.h"
namespace phoenix_wow64 {
struct alignas(8) ModelChannel {
    phoenix_driver_probe::Channel base;
    uint64_t config_json=0, parameters_json=0;
    uint32_t backend=3, bytes=0;
    uint32_t nvidia_create_result=0,nvidia_evaluate_result=0;
    uint64_t nvidia_creates=0,nvidia_evaluates=0;
};
static_assert(std::is_standard_layout_v<ModelChannel> && std::is_trivially_copyable_v<ModelChannel>);
}
