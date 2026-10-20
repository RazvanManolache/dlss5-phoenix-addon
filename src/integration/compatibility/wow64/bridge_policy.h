#pragma once
#include "../session_policy.h"
#include <filesystem>
namespace phoenix_compat {
struct NativeOptions { std::filesystem::path root; bool driver_host=false; };
inline void validate_native_settings(const Settings& settings,bool driver_host) {
    phoenix_addon::check(settings.execution_device=="native",
        "Native x86 mechanisms require the Native RTX executor; use portable or helper processing for another executor.");
    phoenix_addon::check(settings.resolution_mode==0&&!settings.host_unjitter&&!settings.automatic_budget,
        "Experimental in-process x86 requires normal resolution, no unjitter and fixed budget; use the helper for these options.");
    if(driver_host) {
        phoenix_addon::check(settings.model_id != "small-fp8",
            "The experimental x86 driver host uses the Original launch ABI; select the x64-library or helper route for Lightweight FP8.");
        phoenix_addon::check(!needs_nvidia_dll(settings),"The x86 driver host supports Phoenix DLSS5 only; NVIDIA DLSS5 requires the x64-library route or helper.");
        phoenix_addon::check(settings.blocks_for(settings.mode)==8&&std::all_of(settings.block_repeats.begin(),settings.block_repeats.end(),[](int count){return count==1;}),
            "Experimental x86 driver host currently requires all eight blocks once; use the library route or helper for block overrides.");
    }
}
}
