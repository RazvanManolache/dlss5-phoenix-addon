// Exercises the exported DLL interface, not its private implementation.
#include "phoenix_native.h"
#include "../../src/native/runtime/texture_binding.h"
#include "../../third_party/nlohmann/json.hpp"
#include <iostream>
#include <thread>
using Json = nlohmann::json;
static void expect(bool success, const char* message) {
    if (!success)
        throw std::runtime_error(std::string(message) + ": " + phoenix_native_error());
}
static Json status(PhoenixNativeSession* session) {
    std::string text(phoenix_native_status(session, nullptr, 0), '\0');
    expect(!text.empty(), "status size");
    expect(phoenix_native_status(session, text.data(), text.size()) == text.size(), "status value");
    return Json::parse(text.c_str());
}
int main() {
    try {
        ComPtr<ID3D12Debug> debug;
        check(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
        debug->EnableDebugLayer();
        unsigned cases = 0;
        for (const auto dimensions :
             {std::array<unsigned, 4>{31, 19, 31, 19}, {31, 19, 13, 11}, {1, 1, 1, 1}}) {
            const auto [width, height, neural_width, neural_height] = dimensions;
            const Json config = {{"width", width},
                                 {"height", height},
                                 {"neural_width", neural_width},
                                 {"neural_height", neural_height},
                                 {"spatial_only", true},
                                 {"warp", true}};
            auto* session = phoenix_native_create(config.dump().c_str(), nullptr, nullptr);
            expect(session != nullptr, "owned WARP session");
            const size_t stride = size_t(width) * 8 + 24;
            std::vector<uint16_t> color(stride * height / 2, 0x3555),
                output(stride * height / 2, 0x7bff);
            for (unsigned row = 0; row < height; ++row)
                for (unsigned column = 0; column < width; ++column) {
                    auto* pixel = &color[(row * stride + column * 8) / 2];
                    pixel[0] = 0x3400;
                    pixel[1] = 0x3800;
                    pixel[2] = 0x3a00;
                    pixel[3] = uint16_t(0x3000 + (row + column) % 100);
                }
            for (unsigned frame = 0; frame < 3; ++frame) {
                expect(phoenix_native_frame_host(session, color.data(), color.size() * 2, stride,
                                                 nullptr, 0, 0, output.data(), output.size() * 2,
                                                 stride, nullptr) == 0,
                       "spatial host frame");
                for (unsigned row = 0; row < height; ++row) {
                    expect(std::memcmp(reinterpret_cast<char*>(color.data()) + row * stride,
                                       reinterpret_cast<char*>(output.data()) + row * stride,
                                       width * 8) == 0,
                           "constant RGB and original alpha");
                    for (size_t offset = width * 8 / 2; offset < stride / 2; ++offset)
                        expect(output[row * stride / 2 + offset] == 0x7bff,
                               "output row padding untouched");
                }
            }
            expect(phoenix_native_frame_host(session, color.data(), 1, stride, nullptr, 0, 0,
                                             output.data(), output.size() * 2, stride,
                                             nullptr) == -1,
                   "short input rejected");
            std::thread wrong_thread([&] {
                expect(phoenix_native_status(session, nullptr, 0) == 0, "thread ownership");
            });
            wrong_thread.join();
            ComPtr<ID3D12Device> device;
            ComPtr<ID3D12CommandQueue> queue;
            expect(phoenix_native_device(session, &device, &queue) == 0, "public device access");
            auto* borrowed =
                phoenix_native_create(config.dump().c_str(), device.Get(), queue.Get());
            expect(borrowed != nullptr, "borrowed device/queue");
            expect(phoenix_native_destroy(borrowed) == 0, "borrowed destruction");
            auto color_texture = texture(device.Get(), width, height, 16, false);
            expect(phoenix_native_frame_d3d12(
                       session, color_texture.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                       nullptr, D3D12_RESOURCE_STATE_COMMON, color_texture.resource.Get(),
                       D3D12_RESOURCE_STATE_COPY_DEST, nullptr) == -1,
                   "aliased textures rejected before recording");
            const auto report = status(session);
            expect(report.at("frames") == 3, "successful spatial frame count");
            char guard[2] = {'x', 'y'};
            expect(phoenix_native_status(session, guard, 1) > 1 && guard[0] == 'x' &&
                       guard[1] == 'y',
                   "short status buffer untouched");
            expect(!phoenix_native_create(config.dump().c_str(), nullptr, queue.Get()),
                   "unpaired device/queue rejected");
            expect(report.at("backend") == "spatial_d3d12" && !report.at("failed").get<bool>(),
                   "honest path status");
            ComPtr<ID3D12InfoQueue> messages;
            check(device.As(&messages));
            for (UINT64 index = 0; index < messages->GetNumStoredMessages(); ++index) {
                SIZE_T bytes = 0;
                check(messages->GetMessage(index, nullptr, &bytes));
                std::vector<char> storage(bytes);
                auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                check(messages->GetMessage(index, message, &bytes));
                expect(message->Severity > D3D12_MESSAGE_SEVERITY_ERROR, message->pDescription);
            }
            expect(phoenix_native_destroy(session) == 0, "session drain/destruction");
            ++cases;
        }
        expect(!phoenix_native_create("{\"width\":0,\"height\":1}", nullptr, nullptr),
               "zero extent rejected");
        {
            const char* config =
                "{\"width\":2,\"height\":2,\"output_width\":1,\"output_height\":1,\"neural_width\":1,\"neural_height\":1,\"spatial_only\":true,\"warp\":true}";
            auto* session = phoenix_native_create(config, nullptr, nullptr);
            expect(session != nullptr, "gradient session");
            uint16_t pixels[16] = {0,      0,      0,      0,      0x3400, 0x3400, 0x3400, 0x3400,
                                   0x3800, 0x3800, 0x3800, 0x3800, 0x3a00, 0x3a00, 0x3a00, 0x3a00};
            uint16_t output[4]{};
            expect(phoenix_native_frame_host(session, pixels, sizeof(pixels), 16, nullptr, 0, 0,
                                             output, sizeof(output), 8, nullptr) == 0,
                   "gradient reduction");
            for (auto value : output)
                expect(value == 0x3600, "RGB/alpha mean of four gradient pixels");
            expect(phoenix_native_destroy(session) == 0, "gradient destruction");
            ++cases;
        }
        for (const auto output_extent : {std::array<unsigned, 2>{62, 38}, {17, 9}}) {
            Json config = {{"width", 31},
                           {"height", 19},
                           {"output_width", output_extent[0]},
                           {"output_height", output_extent[1]},
                           {"neural_width", 13},
                           {"neural_height", 11},
                           {"spatial_only", true},
                           {"warp", true}};
            auto* session = phoenix_native_create(config.dump().c_str(), nullptr, nullptr);
            expect(session != nullptr, "separate output extent");
            std::vector<uint16_t> color(31 * 19 * 4, 0x3800),
                output(output_extent[0] * output_extent[1] * 4, 0x7bff);
            expect(phoenix_native_frame_host(session, color.data(), color.size() * 2, 31 * 8,
                                             nullptr, 0, 0, output.data(), output.size() * 2,
                                             output_extent[0] * 8, nullptr) == 0,
                   "true spatial upscale/downscale");
            for (auto value : output)
                expect(value == 0x3800, "resized constant RGB and alpha");
            expect(phoenix_native_destroy(session) == 0, "resized destruction");
            ++cases;
        }
        std::cout << Json{{"passed", true}, {"spatial_cases", cases}, {"debug_layer", true}}.dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
