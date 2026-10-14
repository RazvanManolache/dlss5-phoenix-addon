// Public API benchmark. Readback is outside the timed GPU-texture calls.
#include "phoenix_native.h"
#include "../../src/native/runtime/texture_binding.h"
#include "../../third_party/nlohmann/json.hpp"
#include <bcrypt.h>
#include <iostream>
#include <chrono>
using Json = nlohmann::json;
static void expect(bool success, const char* message) {
    if (!success)
        throw std::runtime_error(std::string(message) + ": " + phoenix_native_error());
}
static Json status(PhoenixNativeSession* session) {
    std::string text(phoenix_native_status(session, nullptr, 0), '\0');
    expect(!text.empty(), "status");
    phoenix_native_status(session, text.data(), text.size());
    return Json::parse(text.c_str());
}
static std::string digest(const std::vector<uint8_t>& bytes) {
    unsigned char hash[32];
    char encoded[65]{};
    expect(BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0, const_cast<PUCHAR>(bytes.data()),
                      ULONG(bytes.size()), hash, 32) >= 0,
           "SHA-256");
    for (unsigned index = 0; index < 32; ++index)
        sprintf_s(encoded + 2 * index, 3, "%02x", unsigned(hash[index]));
    return encoded;
}
int main(int argc, char** argv) {
    PhoenixNativeSession* session = nullptr;
    try {
        expect(argc >= 5, "CONFIG_JSON CAPTURE_PREFIX RESULT_JSON gpu|host [frames]");
        Json config;
        std::ifstream(argv[1]) >> config;
        const unsigned width = config.at("width"), height = config.at("height");
        const unsigned output_width = config.value("output_width", width),
                       output_height = config.value("output_height", height);
        const unsigned frames = argc > 5 ? unsigned(std::stoul(argv[5])) : 128;
        expect(frames > 1 && frames <= 4096, "frame count");
        const bool host = std::string(argv[4]) == "host";
        expect(host || std::string(argv[4]) == "gpu", "execution mode");
        auto input = read(std::string(argv[2]) + ".proxy.bin"),
             motion = read(std::string(argv[2]) + ".motion.bin");
        expect(input.size() == size_t(width) * height * 8 &&
                   motion.size() == size_t(width) * height * 4,
               "fixture extent");
        ComPtr<ID3D12Debug> debug;
        check(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
        debug->EnableDebugLayer();
        session = phoenix_native_create(config.dump().c_str(), nullptr, nullptr);
        expect(session != nullptr, "native create");
        ComPtr<ID3D12Device> device;
        ComPtr<ID3D12CommandQueue> queue;
        expect(phoenix_native_device(session, &device, &queue) == 0, "get device");
        auto color = texture(device.Get(), width, height, 16, false);
        auto vectors = texture(device.Get(), width, height, 16, false, 2);
        auto output = texture(device.Get(), output_width, output_height, 16, true);
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> command;
        ComPtr<ID3D12Fence> fence;
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                             IID_PPV_ARGS(&allocator)));
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                        IID_PPV_ARGS(&command)));
        check(command->Close());
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        uint64_t serial = 0;
        auto submit = [&](auto record) {
            check(allocator->Reset());
            check(command->Reset(allocator.Get(), nullptr));
            record();
            check(command->Close());
            ID3D12CommandList* lists[]{command.Get()};
            queue->ExecuteCommandLists(1, lists);
            check(queue->Signal(fence.Get(), ++serial));
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            expect(event != nullptr, "test fence event");
            check(fence->SetEventOnCompletion(serial, event));
            const DWORD result = WaitForSingleObject(event, 30000);
            CloseHandle(event);
            expect(result == WAIT_OBJECT_0, "test fence wait");
        };
        if (!host)
            submit([&] {
                for (unsigned kind = 0; kind < 2; ++kind) {
                    auto& plane = kind ? vectors : color;
                    auto& bytes = kind ? motion : input;
                    const size_t row_bytes = size_t(width) * (kind ? 4 : 8);
                    void* mapped = nullptr;
                    D3D12_RANGE empty{};
                    check(plane.transfer->Map(0, &empty, &mapped));
                    for (unsigned row = 0; row < height; ++row)
                        std::memcpy(static_cast<char*>(mapped) +
                                        row * plane.footprint.Footprint.RowPitch,
                                    bytes.data() + row * row_bytes, row_bytes);
                    plane.transfer->Unmap(0, nullptr);
                    D3D12_TEXTURE_COPY_LOCATION from{plane.transfer.Get(),
                                                     D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                                     plane.footprint};
                    D3D12_TEXTURE_COPY_LOCATION to{plane.resource.Get(),
                                                   D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
                    command->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
                    barrier(command.Get(), plane.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                }
            });
        Json report = {{"passed", false},
                       {"mode", host ? "host" : "gpu"},
                       {"frames", Json::array()},
                       {"hashes", Json::object()},
                       {"debug_errors", 0}};
        std::vector<uint8_t> pixels(size_t(output_width) * output_height * 8);
        PhoenixNativeControls controls;
        phoenix_native_default_controls(&controls);
        controls.auto_mask = 1;
        controls.inverted_depth = 0;
        for (unsigned frame = 0; frame < frames; ++frame) {
            controls.reset = frame == 0;
            if (config.value("vary_controls", false)) {
                controls.reset = (frame % 8) == 0;
                controls.local_tone = frame % 2 ? 0.5f : 1.f;
                controls.local_structure = frame % 3 ? 1.f : 0.75f;
            }
            int result;
            if (host)
                result = phoenix_native_frame_host(session, input.data(), input.size(),
                                                   size_t(width) * 8, motion.data(), motion.size(),
                                                   size_t(width) * 4, pixels.data(), pixels.size(),
                                                   size_t(output_width) * 8, &controls);
            else
                result = phoenix_native_frame_d3d12(
                    session, color.resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    vectors.resource.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    output.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &controls);
            expect(result == 0, "native frame");
            report["frames"].push_back(status(session));
            if (frame == 0 || frame == frames - 1 || config.value("save_every_frame", false)) {
                if (!host) {
                    submit([&] {
                        barrier(command.Get(), output.resource.Get(),
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_COPY_SOURCE);
                        D3D12_TEXTURE_COPY_LOCATION from{
                            output.resource.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, 0};
                        D3D12_TEXTURE_COPY_LOCATION to{output.transfer.Get(),
                                                       D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                                       output.footprint};
                        command->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
                        barrier(command.Get(), output.resource.Get(),
                                D3D12_RESOURCE_STATE_COPY_SOURCE,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    });
                    void* mapped = nullptr;
                    D3D12_RANGE range{0, size_t(output.bytes)}, empty{};
                    check(output.transfer->Map(0, &range, &mapped));
                    for (unsigned row = 0; row < output_height; ++row)
                        std::memcpy(pixels.data() + size_t(row) * output_width * 8,
                                    static_cast<const char*>(mapped) +
                                        row * output.footprint.Footprint.RowPitch,
                                    size_t(output_width) * 8);
                    output.transfer->Unmap(0, &empty);
                }
                report["hashes"][std::to_string(frame)] = digest(pixels);
                std::ofstream(std::string(argv[3]) + "." + std::to_string(frame) + ".bin",
                              std::ios::binary)
                    .write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
            }
        }
        expect(phoenix_native_destroy(session) == 0, "native destroy");
        session = nullptr;
        // Release the fixture's objects before asking the debug layer whether
        // the DLL retained any model, scratch, descriptor or command resources.
        color = {};
        vectors = {};
        output = {};
        allocator.Reset();
        command.Reset();
        fence.Reset();
        queue.Reset();
        ComPtr<ID3D12DebugDevice> debug_device;
        check(device.As(&debug_device));
        check(debug_device->ReportLiveDeviceObjects(
            D3D12_RLDO_FLAGS(D3D12_RLDO_DETAIL | D3D12_RLDO_IGNORE_INTERNAL)));
        ComPtr<ID3D12InfoQueue> messages;
        check(device.As(&messages));
        for (UINT64 index = 0; index < messages->GetNumStoredMessages(); ++index) {
            SIZE_T bytes = 0;
            check(messages->GetMessage(index, nullptr, &bytes));
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(messages->GetMessage(index, message, &bytes));
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                report["debug_errors"] = report["debug_errors"].get<int>() + 1;
                std::cerr << message->pDescription << '\n';
            }
            expect(std::string(message->pDescription).find("Live ID3D12Resource") ==
                       std::string::npos,
                   "resource retained after library destruction");
        }
        expect(report["debug_errors"] == 0, "D3D12 debug errors");
        report["passed"] = true;
        std::ofstream(argv[3]) << report.dump(2);
        std::cout << "Passed " << frames << " " << (host ? "host" : "GPU") << " frames\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (session)
            phoenix_native_destroy(session);
        return 1;
    }
}
