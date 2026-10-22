#include "phoenix_native.h"
#include "../../third_party/nlohmann/json.hpp"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <d3d12sdklayers.h>
#include <wrl/client.h>
using Json = nlohmann::json;
static void expect(bool success, const char* what) {
    if (!success) throw std::runtime_error(std::string(what) + ": " + phoenix_native_error());
}
static Json status(PhoenixNativeSession* session) {
    std::string text(phoenix_native_status(session, nullptr, 0), '\0');
    expect(phoenix_native_status(session, text.data(), text.size()) == text.size(), "Status");
    return Json::parse(text.c_str());
}
int main(int argc, char** argv) {
    try {
        const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
        const bool chain_motion=argc>1&&std::string(argv[1])=="--chain-motion";
        const bool hardware=gpu||chain_motion;
        const bool nis = argc > 1 && std::string(argv[1]) == "--nis";
        const unsigned width = hardware ? 320 : nis ? 71 : 64, height = hardware ? 180 : nis ? 41 : 36;
        const unsigned output_width = nis ? width * 2 : width, output_height = nis ? height * 2 : height;
        const Json pipeline{{"PipelineMonitoring", 0}, {"MotionEstimation", hardware},
                            {"MotionProvider", 2}, {"SceneDetection", true},
                            {"ProcessingChain", nis||chain_motion}, {"MotionDiagnosticView",chain_motion?1:0},{"AdditionalScaler", nis ? 1 : 0}};
        Json config{{"width", width}, {"height", height}, {"warp", !hardware},
                          {"spatial_only", true}, {"pipeline", pipeline}, {"debug", true},
                          {"output_width", output_width}, {"output_height", output_height},
                          {"engine_motion_valid", false}};
        if(chain_motion)config["tuning"]={{"network_passes",2}};
        auto* session = phoenix_native_create(config.dump().c_str(), nullptr, nullptr);
        expect(session != nullptr, "Create pipeline session");
        std::vector<uint16_t> color(size_t(width) * height * 4), output(size_t(output_width) * output_height * 4);
        for (unsigned frame = 0; frame != 4; ++frame) {
            for (unsigned y = 0; y != height; ++y) for (unsigned x = 0; x != width; ++x) {
                const auto i = (size_t(y) * width + x) * 4;
                color[i] = nis && frame < 2 ? 0x3800 : uint16_t(0x3000 + (((x + frame * 4) / 8 + y / 8) % 2) * 0x900);
                color[i + 1] = nis && frame < 2 ? 0x3800 : uint16_t(0x3000 + (y * 9) % 0x900);
                color[i + 2] = uint16_t(0x3400); color[i + 3] = 0x3c00;
            }
            const auto processed = phoenix_native_frame_host(session, color.data(), color.size() * 2, width * 8,
                nullptr, 0, 0, output.data(), output.size() * 2, output_width * 8, nullptr);
            if (processed) {
                std::cerr << "Failed frame " << frame << '\n';
                Microsoft::WRL::ComPtr<ID3D12Device> device;
                phoenix_native_device(session, &device, nullptr);
                Microsoft::WRL::ComPtr<ID3D12InfoQueue> messages;
                if (device && SUCCEEDED(device.As(&messages))) {
                    for (UINT64 i = 0; i < messages->GetNumStoredMessages(); ++i) {
                        SIZE_T bytes = 0; messages->GetMessage(i, nullptr, &bytes);
                        std::vector<uint8_t> storage(bytes);
                        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                        if (SUCCEEDED(messages->GetMessage(i, message, &bytes)))
                            std::cerr << message->ID << ": " << message->pDescription << '\n';
                    }
                }
                throw std::runtime_error("Process pipeline frame failed; see graphics diagnostics");
            }
            if (!nis&&!chain_motion) expect(output == color, "Spatial bypass must preserve pixels");
            else if(chain_motion){if(frame>1)expect(output!=color,"Motion diagnostic was bypassed by the chain");}
            else for (size_t pixel = 0; pixel < output.size(); pixel += 4) {
                expect(output[pixel + 3] == 0x3c00, "NIS alpha coverage");
                for (unsigned channel = 0; channel != 3; ++channel)
                    expect((output[pixel + channel] & 0x7c00) != 0x7c00, "NIS finite output");
                if (frame < 2) {
                    expect(output[pixel] >= 0x37ff && output[pixel] <= 0x3801, "NIS constant red");
                    expect(output[pixel + 1] >= 0x37ff && output[pixel + 1] <= 0x3801, "NIS constant green");
                }
            }
        }
        const auto report = status(session);
        expect(report.at("gpu_monitoring_enabled") == false, "Monitoring disabled");
        expect(report.at("complete_call_ms")==0,"Disabled monitoring must not time host transfers");
        if (hardware) expect(report.at("motion_provider") == "NVIDIA Optical Flow", "Hardware motion actually active");
        if (nis) expect(report.at("additional_scaler") == "NVIDIA Image Scaling", "NIS actually active");
        expect(phoenix_native_destroy(session) == 0, "Retire pipeline session");
        std::cout << Json{{"passed",true},{"gpu_used",hardware},{"frames",4},
            {"exact_bypass",!nis&&!chain_motion},{"motion_diagnostic_displayed",chain_motion},{"nis",nis},{"monitoring_off",true},{"status",report}}.dump() << '\n';
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
