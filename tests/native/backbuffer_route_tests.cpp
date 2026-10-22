#include "../../src/integration/compatibility/backbuffer_route.h"
#include "../../src/integration/compatibility/session_policy.h"
#include "../../src/integration/compatibility/wow64/bridge_policy.h"
#include <iostream>

int main() try {
    using namespace phoenix_compat;
    using A = reshade::api::device_api;
    using F = reshade::api::format;
    unsigned checks = 0;
    auto verify = [&](bool value) { ++checks; if (!value) throw std::runtime_error("Backbuffer route contract failed at check " + std::to_string(checks)); };
    using C=reshade::api::color_space;
    for(auto format:{F::r8g8b8a8_unorm,F::b8g8r8a8_unorm,F::b8g8r8x8_unorm})
        verify(sdr_backbuffer(C::unknown,format));
    for(auto format:{F::r16g16b16a16_float,F::r10g10b10a2_unorm,F::unknown})
        verify(!sdr_backbuffer(C::unknown,format));
    for(auto space:{C::scrgb,C::hdr10_pq,C::hdr10_hlg})
        verify(!sdr_backbuffer(space,F::r8g8b8a8_unorm));
    const auto automatic = ExecutionPreference::Automatic;
    BackbufferCapabilities base;
    base.bits = 64; base.width = 2560; base.height = 1080;
    base.local_api = true; base.format = F::r8g8b8a8_unorm;
    base.native_library_available = base.native_driver_available = true;
    for(auto api:{A::d3d9,A::d3d10,A::opengl}) {
        auto legacy=base;legacy.api=api;legacy.bits=32;legacy.format=F::b8g8r8x8_unorm;
        legacy.sdr=sdr_backbuffer(C::unknown,legacy.format);
        verify(choose_backbuffer_route(legacy,ExecutionPreference::NativeLibrary).route==
            (api==A::opengl?BackbufferRoute::Unsupported:BackbufferRoute::NativeLibrary));
        verify(choose_backbuffer_route(legacy,ExecutionPreference::NativeDriver).route==
            (api==A::opengl?BackbufferRoute::Unsupported:BackbufferRoute::NativeDriver));
        legacy.sdr=sdr_backbuffer(C::hdr10_pq,legacy.format);
        verify(choose_backbuffer_route(legacy,ExecutionPreference::NativeDriver).route==BackbufferRoute::Unsupported);
    }
    verify(execution_preference("auto") == automatic);
    verify(execution_preference("bogus") == ExecutionPreference::Invalid);
    for (auto api : {A::d3d10, A::d3d11, A::d3d12}) {
        auto software = base;
        software.api = api;
        software.software_device = true;
        verify(choose_backbuffer_route(software, automatic).route == BackbufferRoute::Helper);
        verify(choose_backbuffer_route(software, ExecutionPreference::InProcess).route == BackbufferRoute::Unsupported);
    }
    for (auto format : {F::r8g8b8a8_unorm, F::r8g8b8a8_unorm_srgb, F::r8g8b8a8_typeless,
                       F::b8g8r8a8_unorm, F::b8g8r8a8_unorm_srgb, F::b8g8r8a8_typeless,
                       F::r10g10b10a2_unorm, F::r10g10b10a2_typeless}) {
        auto caps = base; caps.format = format;
        verify(choose_backbuffer_route(caps, automatic).route == BackbufferRoute::InProcess);
    }
    for (auto api : {A::d3d9, A::d3d10, A::d3d11, A::d3d12, A::opengl, A::vulkan}) {
        auto caps = base; caps.api = api; caps.bits = 32;
        verify(choose_backbuffer_route(caps, automatic).route ==
               (api == A::opengl ? BackbufferRoute::Helper : BackbufferRoute::NativeLibrary));
        verify(choose_backbuffer_route(caps, ExecutionPreference::InProcess).route ==
               BackbufferRoute::Unsupported);
        caps.bits = 64;
        verify(choose_backbuffer_route(caps, automatic).route ==
               (api == A::d3d12 ? BackbufferRoute::InProcess : BackbufferRoute::Helper));
    }
    auto unvalidated32 = base;
    unvalidated32.bits = 32; unvalidated32.native_library_available = unvalidated32.native_driver_available = false;
    verify(choose_backbuffer_route(unvalidated32,automatic).route == BackbufferRoute::Helper);
    verify(choose_backbuffer_route(unvalidated32,ExecutionPreference::InProcess).route == BackbufferRoute::Unsupported);
    for (int mode : {phoenix_addon::phoenix::nvidia, phoenix_addon::phoenix::split, phoenix_addon::phoenix::ours}) {
        for (const char* executor : {"native", "cpu"}) {
            Settings requested; requested.mode = mode; requested.execution_device = executor;
            auto effective = in_process_settings(requested);
            verify(requested.execution_device == executor);
            verify(effective.execution_device == (mode == phoenix_addon::phoenix::nvidia ? "native" : executor));
            for (int source : {0, 1, 2}) {
                requested.source_a = source;
                auto caps = base;
                caps.nvidia_required = needs_nvidia_dll(requested);
                caps.native_executor = in_process_settings(requested).execution_device == "native";
                const bool local = mode == phoenix_addon::phoenix::nvidia || requested.execution_device == "native";
                verify(choose_backbuffer_route(caps, automatic).route ==
                       (local ? BackbufferRoute::InProcess : BackbufferRoute::Helper));
                verify(choose_backbuffer_route(caps, ExecutionPreference::Helper).route == BackbufferRoute::Helper);
                verify(choose_backbuffer_route(caps, ExecutionPreference::InProcess).route ==
                       (local ? BackbufferRoute::InProcess : BackbufferRoute::Unsupported));
                caps.bits = 32;
                verify(choose_backbuffer_route(caps, automatic).route ==
                       (local && !caps.nvidia_required ? BackbufferRoute::NativeLibrary : BackbufferRoute::Helper));
                verify(choose_backbuffer_route(caps, ExecutionPreference::InProcess).route ==
                       BackbufferRoute::Unsupported);
            }
        }
    }
    for (int a : {0, 1, 2}) for (int b : {0, 1, 2}) {
        Settings comparison;
        comparison.mode = phoenix_addon::phoenix::split;
        comparison.source_a = a; comparison.source_b = b;
        verify(needs_nvidia_dll(comparison) == (a == 1 || b == 1));
        auto caps = base; caps.bits = 32;
        caps.nvidia_required = needs_nvidia_dll(comparison);
        verify(choose_backbuffer_route(caps, automatic).route ==
               (caps.nvidia_required ? BackbufferRoute::Helper : BackbufferRoute::NativeLibrary));
    }
    for (unsigned failure = 0; failure < 2; ++failure) {
        auto caps = base;
        if (failure == 0) caps.local_api = false;
        if (failure == 1) caps.native_executor = false;
        auto decision = choose_backbuffer_route(caps, automatic);
        verify(decision.route == BackbufferRoute::Helper && !decision.reason.empty());
        verify(choose_backbuffer_route(caps, ExecutionPreference::InProcess).route == BackbufferRoute::Unsupported);
    }
    for (unsigned failure = 0; failure < 10; ++failure) {
        auto caps = base;
        if (failure == 0) caps.sdr = false;
        if (failure == 1) caps.samples = 4;
        if (failure == 2) caps.layers = 2;
        if (failure == 3) caps.levels = 2;
        if (failure == 4) caps.format = F::r16g16b16a16_float;
        if (failure == 5) caps.width = 0;
        if (failure == 6) caps.height = 16385;
        if (failure == 7) caps.protected_resource = true;
        if (failure == 8) caps.queue_matches = false;
        if (failure == 9) caps.bits = 0;
        verify(choose_backbuffer_route(caps, automatic).route == BackbufferRoute::Unsupported);
        verify(choose_backbuffer_route(caps, ExecutionPreference::Helper).route == BackbufferRoute::Unsupported);
    }
    auto large = base; large.width = 12000;
    verify(choose_backbuffer_route(large, automatic).route == BackbufferRoute::InProcess);
    large.local_api = false;
    verify(choose_backbuffer_route(large, automatic).route == BackbufferRoute::Unsupported);
    verify(choose_backbuffer_route(base, ExecutionPreference::Helper).route == BackbufferRoute::Helper);
    verify(choose_backbuffer_route(base, ExecutionPreference::Invalid).route == BackbufferRoute::Unsupported);
    verify(may_fallback_to_helper(automatic, true));
    verify(!may_fallback_to_helper(automatic, false));
    verify(!may_fallback_to_helper(ExecutionPreference::InProcess, true));
    verify(execution_preference("portable") == ExecutionPreference::Portable);
    verify(execution_preference("wow64_library")==ExecutionPreference::NativeLibrary);
    verify(execution_preference("wow64_driver")==ExecutionPreference::NativeDriver);
    for(auto api:{A::d3d9,A::d3d10,A::d3d11,A::d3d12,A::opengl,A::vulkan}) {
        auto caps=base;caps.bits=32;caps.api=api;
        for(auto preference:{ExecutionPreference::NativeLibrary,ExecutionPreference::NativeDriver}) {
            const auto expected=api==A::opengl?BackbufferRoute::Unsupported:
                preference==ExecutionPreference::NativeLibrary?BackbufferRoute::NativeLibrary:BackbufferRoute::NativeDriver;
            verify(choose_backbuffer_route(caps,preference).route==expected);
            caps.nvidia_required=true;
            verify(choose_backbuffer_route(caps,preference).route==
                (api==A::opengl || preference==ExecutionPreference::NativeLibrary ?
                 BackbufferRoute::Unsupported:BackbufferRoute::Helper));
            caps.nvidia_required=false;
            for(unsigned failure=0;failure<6;++failure) {
                auto invalid=caps;
                if(failure==0) invalid.bits=64;
                if(failure==1) invalid.sdr=false;
                if(failure==2) invalid.width=8193;
                if(failure==3) invalid.layers=2;
                if(failure==4) invalid.protected_resource=true;
                if(failure==5) invalid.queue_matches=false;
                verify(choose_backbuffer_route(invalid,preference).route==BackbufferRoute::Unsupported);
            }
        }
    }
    for (unsigned bits : {32u, 64u})
        for (auto api : {A::d3d9, A::d3d10, A::d3d11, A::d3d12, A::opengl, A::vulkan}) {
            auto caps = base; caps.bits = bits; caps.api = api; caps.portable_available = true;
            verify(choose_backbuffer_route(caps, ExecutionPreference::Portable).route == BackbufferRoute::Portable);
            caps.nvidia_required = true;
            verify(choose_backbuffer_route(caps, ExecutionPreference::Portable).route == BackbufferRoute::Helper);
            caps.nvidia_required = false; caps.width = 8193;
            verify(choose_backbuffer_route(caps, ExecutionPreference::Portable).route == BackbufferRoute::Unsupported);
        }
    auto ordered = base; ordered.bits = 32; ordered.portable_available = true;
    verify(choose_backbuffer_route(ordered, automatic).route == BackbufferRoute::NativeLibrary);
    ordered.native_library_available = false;
    verify(choose_backbuffer_route(ordered, automatic).route == BackbufferRoute::NativeDriver);
    ordered.native_driver_available = false;
    verify(choose_backbuffer_route(ordered, automatic).route == BackbufferRoute::Portable);
    ordered.portable_available = false;
    verify(choose_backbuffer_route(ordered, automatic).route == BackbufferRoute::Helper);
    const std::string exhausted="Low 32-bit address space: 415 MiB free; use the helper.";
    ordered.native_library_reason=ordered.native_driver_reason=ordered.portable_reason=exhausted;
    for(auto forced:{ExecutionPreference::NativeLibrary,ExecutionPreference::NativeDriver,ExecutionPreference::Portable}) {
        const auto refusal=choose_backbuffer_route(ordered,forced);
        verify(refusal.route==BackbufferRoute::Unsupported);
        verify(refusal.reason==exhausted);
        const auto choices=available_execution_preferences(ordered);
        verify(std::find(choices.begin(),choices.end(),forced)==choices.end());
    }
    verify(choose_backbuffer_route(ordered,automatic).route==BackbufferRoute::Helper);
    const auto blocked=choose_backbuffer_route(ordered,ExecutionPreference::NativeLibrary);
    verify(terminal_route_refusal(ordered,blocked));
    ordered.discovery_pending=true;
    verify(!terminal_route_refusal(ordered,blocked));
    ordered.discovery_pending=false;
    verify(!terminal_route_refusal(ordered,choose_backbuffer_route(ordered,automatic)));
    auto opengl = base; opengl.bits = 32; opengl.api = A::opengl;
    auto coordinated_gl=opengl;coordinated_gl.native_tls_coordinated=true;
    verify(choose_backbuffer_route(coordinated_gl,automatic).route==BackbufferRoute::NativeLibrary);
    verify(choose_backbuffer_route(coordinated_gl,ExecutionPreference::NativeLibrary).route==BackbufferRoute::NativeLibrary);
    verify(choose_backbuffer_route(coordinated_gl,ExecutionPreference::NativeDriver).route==BackbufferRoute::NativeDriver);
    verify(choose_backbuffer_route(opengl, automatic).route == BackbufferRoute::Helper);
    opengl.portable_available = true;
    verify(choose_backbuffer_route(opengl, automatic).route == BackbufferRoute::Portable);
    const auto gl_choices = available_execution_preferences(opengl);
    verify(std::find(gl_choices.begin(), gl_choices.end(), ExecutionPreference::NativeLibrary) == gl_choices.end());
    verify(std::find(gl_choices.begin(), gl_choices.end(), ExecutionPreference::NativeDriver) == gl_choices.end());
    ordered.discovery_pending = true;
    verify(choose_backbuffer_route(ordered, automatic).route == BackbufferRoute::Unsupported);
    verify(choose_backbuffer_route(ordered, ExecutionPreference::Helper).route == BackbufferRoute::Helper);
    for (auto bits : {32u, 64u}) {
        auto caps = base; caps.bits = bits;
        const auto choices = available_execution_preferences(caps);
        verify(choices.front() == ExecutionPreference::Automatic);
        verify(choices.back() == ExecutionPreference::Helper);
        verify((std::find(choices.begin(), choices.end(), ExecutionPreference::InProcess) != choices.end()) == (bits == 64));
        verify((std::find(choices.begin(), choices.end(), ExecutionPreference::NativeLibrary) != choices.end()) == (bits == 32));
        verify(std::find(choices.begin(), choices.end(), ExecutionPreference::Portable) == choices.end());
    }
    for(auto api:{A::d3d9,A::d3d10,A::d3d11,A::d3d12,A::opengl,A::vulkan}) {
        auto vendor=base;vendor.bits=32;vendor.api=api;
        vendor.native_tls_coordinated=true;vendor.nvidia_required=true;vendor.portable_available=true;
        const auto choices=available_execution_preferences(vendor);
        verify(choices==std::vector<ExecutionPreference>{ExecutionPreference::Automatic,ExecutionPreference::Helper});
        verify(choose_backbuffer_route(vendor,automatic).route==BackbufferRoute::Helper);
        const auto refused=choose_backbuffer_route(vendor,ExecutionPreference::NativeLibrary);
        verify(refused.route==BackbufferRoute::Unsupported && refused.reason.find("x64 helper")!=std::string::npos);
        vendor.bits=64;
        verify(choose_backbuffer_route(vendor,automatic).route==
            (api==A::d3d12?BackbufferRoute::InProcess:BackbufferRoute::Helper));
    }
    Settings native; native.mode = phoenix_addon::phoenix::ours;
    verify(!*native_settings_unavailable_reason(native, false));
    verify(!*native_settings_unavailable_reason(native, true));
    native.model_id = "small-fp8";
    verify(!*native_settings_unavailable_reason(native, false));
    verify(*native_settings_unavailable_reason(native, true));
    native.model_id = "original"; native.block_repeats[0] = 2;
    verify(!*native_settings_unavailable_reason(native, false));
    verify(*native_settings_unavailable_reason(native, true));
    native.block_repeats[0] = 1; native.mode = phoenix_addon::phoenix::nvidia;
    verify(*native_settings_unavailable_reason(native, false));
    verify(*native_settings_unavailable_reason(native, true));
    native.mode = phoenix_addon::phoenix::ours; native.execution_device = "cpu";
    verify(*native_settings_unavailable_reason(native, false));
    native.execution_device = "native"; native.automatic_budget = true;
    verify(*native_settings_unavailable_reason(native, false));
    native.automatic_budget=false;
    for(bool driver:{false,true}){
        native.pipeline.chain_enabled=true;verify(*native_settings_unavailable_reason(native,driver));native.pipeline={};
        native.pipeline.motion_enabled=true;verify(*native_settings_unavailable_reason(native,driver));native.pipeline={};
        native.pipeline.scene_detection=true;verify(*native_settings_unavailable_reason(native,driver));native.pipeline={};
        native.pipeline.monitoring=1;verify(*native_settings_unavailable_reason(native,driver));native.pipeline={};
        verify(!*native_settings_unavailable_reason(native,driver));
    }
    std::cout << "{\"passed\":true,\"cpu_only\":true,\"checks\":" << checks << "}\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
