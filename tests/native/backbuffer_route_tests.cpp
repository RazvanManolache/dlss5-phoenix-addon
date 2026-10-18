#include "../../src/integration/compatibility/backbuffer_route.h"
#include "../../src/integration/compatibility/session_policy.h"
#include <iostream>

int main() {
    using namespace phoenix_compat;
    using A = reshade::api::device_api;
    using F = reshade::api::format;
    unsigned checks = 0;
    auto verify = [&](bool value) { ++checks; if (!value) throw std::runtime_error("Backbuffer route contract failed"); };
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
    base.x86_native_validated = true;
    for(auto api:{A::d3d9,A::d3d10,A::opengl}) {
        auto legacy=base;legacy.api=api;legacy.bits=32;legacy.format=F::b8g8r8x8_unorm;
        legacy.sdr=sdr_backbuffer(C::unknown,legacy.format);
        verify(choose_backbuffer_route(legacy,ExecutionPreference::NativeLibrary).route==BackbufferRoute::NativeLibrary);
        verify(choose_backbuffer_route(legacy,ExecutionPreference::NativeDriver).route==BackbufferRoute::NativeDriver);
        legacy.sdr=sdr_backbuffer(C::hdr10_pq,legacy.format);
        verify(choose_backbuffer_route(legacy,ExecutionPreference::NativeDriver).route==BackbufferRoute::Unsupported);
    }
    verify(execution_preference("auto") == automatic);
    verify(execution_preference("bogus") == ExecutionPreference::Invalid);
    for (auto format : {F::r8g8b8a8_unorm, F::r8g8b8a8_unorm_srgb, F::r8g8b8a8_typeless,
                       F::b8g8r8a8_unorm, F::b8g8r8a8_unorm_srgb, F::b8g8r8a8_typeless,
                       F::r10g10b10a2_unorm, F::r10g10b10a2_typeless}) {
        auto caps = base; caps.format = format;
        verify(choose_backbuffer_route(caps, automatic).route == BackbufferRoute::InProcess);
    }
    for (auto api : {A::d3d9, A::d3d10, A::d3d11, A::d3d12, A::opengl, A::vulkan}) {
        auto caps = base; caps.api = api; caps.bits = 32;
        verify(choose_backbuffer_route(caps, automatic).route ==
               (api == A::d3d12 ? BackbufferRoute::InProcess : BackbufferRoute::Helper));
        verify(choose_backbuffer_route(caps, ExecutionPreference::InProcess).route ==
               (api == A::d3d12 ? BackbufferRoute::InProcess : BackbufferRoute::Unsupported));
        caps.bits = 64;
        verify(choose_backbuffer_route(caps, automatic).route ==
               (api == A::d3d12 ? BackbufferRoute::InProcess : BackbufferRoute::Helper));
    }
    auto unvalidated32 = base;
    unvalidated32.bits = 32; unvalidated32.x86_native_validated = false;
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
                const bool local32 = local && !caps.nvidia_required;
                verify(choose_backbuffer_route(caps, automatic).route ==
                       (local32 ? BackbufferRoute::InProcess : BackbufferRoute::Helper));
                verify(choose_backbuffer_route(caps, ExecutionPreference::InProcess).route ==
                       (local32 ? BackbufferRoute::InProcess : BackbufferRoute::Unsupported));
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
               (a == 1 || b == 1 ? BackbufferRoute::Helper : BackbufferRoute::InProcess));
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
        auto caps=base;caps.bits=32;caps.api=api;caps.x86_native_validated=false;
        for(auto preference:{ExecutionPreference::NativeLibrary,ExecutionPreference::NativeDriver}) {
            const auto expected=preference==ExecutionPreference::NativeLibrary?BackbufferRoute::NativeLibrary:BackbufferRoute::NativeDriver;
            verify(choose_backbuffer_route(caps,preference).route==expected);
            caps.nvidia_required=true;
            verify(choose_backbuffer_route(caps,preference).route==
                (preference==ExecutionPreference::NativeLibrary?BackbufferRoute::NativeLibrary:BackbufferRoute::Helper));
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
            auto caps = base; caps.bits = bits; caps.api = api; caps.x86_native_validated = false;
            verify(choose_backbuffer_route(caps, ExecutionPreference::Portable).route == BackbufferRoute::Portable);
            caps.nvidia_required = true;
            verify(choose_backbuffer_route(caps, ExecutionPreference::Portable).route == BackbufferRoute::Helper);
            caps.nvidia_required = false; caps.width = 8193;
            verify(choose_backbuffer_route(caps, ExecutionPreference::Portable).route == BackbufferRoute::Unsupported);
        }
    std::cout << "{\"passed\":true,\"cpu_only\":true,\"checks\":" << checks << "}\n";
}
