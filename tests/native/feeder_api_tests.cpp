#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "../../src/integration/reshade/feeder_parameters.h"
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    unsigned checks = 0;
    try {
        const auto expect = [&](bool condition) {
            if (!condition) throw std::runtime_error("feeder API contract failure");
            ++checks;
        };
        PhoenixFeederFrameV1 frame;
        PhoenixFeederResultV1 result;
        expect(!PhoenixFeederContractV1(frame, result));
        frame.command_list = reinterpret_cast<ID3D12GraphicsCommandList*>(uintptr_t(1));
        frame.queue = reinterpret_cast<ID3D12CommandQueue*>(uintptr_t(2));
        frame.output = reinterpret_cast<ID3D12Resource*>(uintptr_t(3));
        frame.motion = reinterpret_cast<ID3D12Resource*>(uintptr_t(4));
        expect(PhoenixFeederContractV1(frame, result));
        const auto invalid = [&](const std::function<void(PhoenixFeederFrameV1&)>& change) {
            auto altered = frame;
            change(altered);
            expect(!PhoenixFeederContractV1(altered, result));
        };
        invalid([](auto& value) { --value.size; });
        invalid([](auto& value) { ++value.version; });
        invalid([](auto& value) { value.command_list = nullptr; });
        invalid([](auto& value) { value.queue = nullptr; });
        invalid([](auto& value) { value.output = nullptr; });
        invalid([](auto& value) { value.motion = nullptr; });
        invalid([](auto& value) { value.motion = value.output; });
        invalid([](auto& value) { value.depth = value.output; });
        invalid([](auto& value) { value.depth = value.motion; });
        invalid([](auto& value) { value.stream_id = 3; });
        invalid([](auto& value) { value.reset = 2; });
        invalid([](auto& value) { value.depth_inverted = 2; });
        invalid([](auto& value) { value.reserved = 1; });
        invalid([](auto& value) { value.motion_scale_x = std::numeric_limits<float>::infinity(); });
        invalid([](auto& value) { value.motion_scale_y = std::numeric_limits<float>::quiet_NaN(); });
        for (uint32_t stream = 0; stream <= 2; ++stream) {
            frame.stream_id = stream;
            expect(PhoenixFeederContractV1(frame, result));
        }
        result.version = 2;
        expect(!PhoenixFeederContractV1(frame, result));
        result.version = 1;
        --result.size;
        expect(!PhoenixFeederContractV1(frame, result));
        result.size = sizeof(result);
        result.reserved = 1;
        expect(!PhoenixFeederContractV1(frame, result));
        result.reserved = 0;
        frame.depth = reinterpret_cast<ID3D12Resource*>(uintptr_t(5));
        frame.reset = frame.depth_inverted = 1;
        frame.motion_scale_x = -4;
        frame.motion_scale_y = 2.5f;
        phoenix_addon::FeederParameters parameters(frame);
        ID3D12Resource* resource = nullptr;
        expect(NVSDK_NGX_SUCCEED(parameters.Get(NVSDK_NGX_Parameter_Output, &resource)) && resource == frame.output);
        expect(NVSDK_NGX_SUCCEED(parameters.Get(NVSDK_NGX_Parameter_MotionVectors, &resource)) && resource == frame.motion);
        void* pointer = nullptr;
        expect(NVSDK_NGX_SUCCEED(parameters.Get(NVSDK_NGX_Parameter_Depth, &pointer)) && pointer == frame.depth);
        int reset = 0;
        unsigned flags = 0;
        float x = 0, y = 0;
        expect(NVSDK_NGX_SUCCEED(parameters.Get(NVSDK_NGX_Parameter_Reset, &reset)) && reset == 1);
        expect(NVSDK_NGX_SUCCEED(parameters.Get("DLSS.Feature.Create.Flags", &flags)) && flags == 8);
        expect(NVSDK_NGX_SUCCEED(parameters.Get(NVSDK_NGX_Parameter_MV_Scale_X, &x)) && x == -4);
        expect(NVSDK_NGX_SUCCEED(parameters.Get(NVSDK_NGX_Parameter_MV_Scale_Y, &y)) && y == 2.5f);
        float unchanged = 17;
        expect(NVSDK_NGX_FAILED(parameters.Get("missing", &unchanged)) && unchanged == 17);
        parameters.Set(NVSDK_NGX_Parameter_Reset, 0);
        parameters.Reset();
        expect(frame.reset == 1);
        PhoenixFeederFrameV2 extended;extended.frame=frame;
        extended.control_mask=reinterpret_cast<ID3D12Resource*>(uintptr_t(6));
        extended.intensity=.5f;extended.appearance_generation=7;
        expect(PhoenixFeederContractV2(extended,result));
        phoenix_addon::active_feeder_v2()=&extended;
        phoenix_addon::FeederParameters developer(extended.frame);
        expect(NVSDK_NGX_SUCCEED(developer.Get("DLSSNR.ControlMask",&resource))&&resource==extended.control_mask);
        expect(NVSDK_NGX_SUCCEED(developer.Get("DLSSNR.Intensity",&x))&&x==.5f);
        expect(NVSDK_NGX_SUCCEED(developer.Get("Phoenix.Appearance.Contract",&flags))&&flags==1);
        expect(NVSDK_NGX_SUCCEED(developer.Get("Phoenix.Appearance.Generation",&flags))&&flags==7);
        expect(NVSDK_NGX_FAILED(parameters.Get("Phoenix.Appearance.Contract",&flags)));
        phoenix_addon::active_feeder_v2()=nullptr;
        auto altered=extended;altered.version=3;expect(!PhoenixFeederContractV2(altered,result));
        altered=extended;altered.intensity=std::numeric_limits<float>::quiet_NaN();expect(!PhoenixFeederContractV2(altered,result));
        altered=extended;altered.ui=altered.control_mask;altered.ui_alpha=altered.control_mask;expect(!PhoenixFeederContractV2(altered,result));
        std::cout << "{\"cpu_only\":true,\"passed\":true,\"cases\":" << checks << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
