// Numerical resampling and state/lifetime checks on a software D3D12 device.
#include "../../src/native/support/direct.h"
#include <dxgi1_6.h>
#include "../../src/native/support/gpu_timing.h"
#include <d3d12sdklayers.h>
#include <iostream>
using namespace phoenix_addon;

int main() {
    try {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
            debug->EnableDebugLayer();
        ComPtr<IDXGIFactory4> factory;
        phoenix_native::hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter> adapter;
        phoenix_native::hr(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
        ComPtr<ID3D12Device> device;
        phoenix_native::hr(
            D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
        ComPtr<ID3D12InfoQueue> messages;
        device.As(&messages);
        ComPtr<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC description{};
        phoenix_native::hr(device->CreateCommandQueue(&description, IID_PPV_ARGS(&queue)));
        ComPtr<ID3D12CommandAllocator> allocator;
        phoenix_native::hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                          IID_PPV_ARGS(&allocator)));
        ComPtr<ID3D12GraphicsCommandList> command;
        phoenix_native::hr(device->CreateCommandList(
            0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&command)));
        phoenix::GpuTiming empty_timer(device.Get(), queue.Get());
        check(Json::parse(empty_timer.report().dump()).at("active_settings").is_object(),
              "An empty timing context must serialize as valid JSON");
        phoenix::DepthBudget timing_budget;
        timing_budget.configure(true, 8, 2, 10);
        empty_timer.budget(&timing_budget);
        empty_timer.settings("{\"test\":1}");
        empty_timer.upscaler_status("No upscaling");
        timing_budget.sample(8, 5);
        empty_timer.settings("{\"test\":1}");
        empty_timer.upscaler_status("No upscaling");
        check(timing_budget.ema() == 5 &&
                  empty_timer.report().at("active_settings").at("UpscalerStatus") == "No upscaling",
              "Refreshing a restored stream preserves resolved status and budget samples");
        empty_timer.settings("{\"test\":2}");
        check(timing_budget.ema() == 0,
              "A genuine settings change resets budget samples");
        empty_timer.budget(nullptr);
        empty_timer.begin(command.Get(), 3, 8, 96, 54);
        auto geometry = phoenix::fit_resolution(96, 54, 48, 32);
        phoenix::direct::ResolutionBridge bridge(device.Get(), geometry,
                                                 DXGI_FORMAT_R32G32B32A32_FLOAT, {});
        auto low = phoenix_native::image(device.Get(), 48, 32, DXGI_FORMAT_R32G32B32A32_FLOAT,
                                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        auto motion = phoenix_native::image(device.Get(), 96, 54, DXGI_FORMAT_R32G32_FLOAT,
                                            D3D12_RESOURCE_STATE_COPY_DEST);
        auto low_motion = phoenix_native::image(device.Get(), 48, 32, DXGI_FORMAT_R32G32_FLOAT,
                                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        std::vector<ComPtr<ID3D12Resource>> uploads;
        auto initialize = [&](ID3D12Resource* resource, unsigned components, bool vectors,
                              float offset = 0, bool uniform = false) {
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
            UINT64 bytes = 0;
            auto desc = resource->GetDesc();
            device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
            auto upload = phoenix_native::buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ);
            void* mapped = nullptr;
            phoenix_native::hr(upload->Map(0, nullptr, &mapped));
            std::memset(mapped, 0, size_t(bytes));
            for (unsigned row = 0; row < desc.Height; ++row)
                for (unsigned column = 0; column < desc.Width; ++column) {
                    auto* pixel =
                        reinterpret_cast<float*>(static_cast<uint8_t*>(mapped) + footprint.Offset +
                                                 row * footprint.Footprint.RowPitch) +
                        column * components;
                    if (uniform) {
                        pixel[0] = pixel[1] = pixel[2] = offset;
                        pixel[3] = 1.f;
                    } else if (vectors) {
                        pixel[0] = 4;
                        pixel[1] = -2;
                    } else {
                        pixel[0] = (column + .5f) / float(desc.Width) + offset;
                        pixel[1] = (row + .5f) / desc.Height;
                        pixel[2] = .5f;
                        pixel[3] = .25f + column * .001f + offset;
                    }
                }
            upload->Unmap(0, nullptr);
            D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
            source.pResource = upload.Get();
            source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            source.PlacedFootprint = footprint;
            destination.pResource = resource;
            destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            command->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
            phoenix_native::transition(command.Get(), resource, D3D12_RESOURCE_STATE_COPY_DEST,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            uploads.push_back(std::move(upload));
        };
        struct Readback {
            ComPtr<ID3D12Resource> buffer;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
        };
        auto read = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES state) {
            Readback result{};
            UINT64 bytes = 0;
            auto desc = resource->GetDesc();
            device->GetCopyableFootprints(&desc, 0, 1, 0, &result.footprint, nullptr, nullptr,
                                          &bytes);
            result.buffer = phoenix_native::buffer(device.Get(), bytes, D3D12_HEAP_TYPE_READBACK,
                                                   D3D12_RESOURCE_STATE_COPY_DEST);
            phoenix_native::transition(command.Get(), resource, state,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
            source.pResource = resource;
            source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            destination.pResource = result.buffer.Get();
            destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            destination.PlacedFootprint = result.footprint;
            command->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
            phoenix_native::transition(command.Get(), resource, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                       state);
            return result;
        };
        initialize(motion.Get(), 2, true);
        bridge.reduce(command.Get(), motion.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                      low_motion.Get(), true);
        auto vectors = read(low_motion.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        std::vector<ComPtr<ID3D12Resource>> games;
        std::vector<Readback> answers;
        std::vector<float> offsets;
        // Record more than the old 16-texture limit before any fence completes.
        // Distinct content catches accidental descriptor reuse within a batch.
        for (unsigned index = 0; index < 32; ++index) {
            auto game = phoenix_native::image(device.Get(), 96, 54, DXGI_FORMAT_R32G32B32A32_FLOAT,
                                              D3D12_RESOURCE_STATE_COPY_DEST);
            const float offset = index * .01f;
            initialize(game.Get(), 4, false, offset);
            offsets.push_back(offset);
            bridge.reduce(command.Get(), game.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                          low.Get(), false);
            phoenix_native::transition(command.Get(), low.Get(),
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            bridge.present(command.Get(), low.Get(), game.Get(),
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            phoenix_native::transition(command.Get(), low.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            answers.push_back(read(game.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
            games.push_back(std::move(game));
        }
        // Missing guides must fall back before loading a vendor runtime.
        phoenix::direct::TemporalUpscaler unavailable(device.Get(), geometry, 1);
        check(!unavailable.record(command.Get(), low.Get(), low_motion.Get(), nullptr, false, true,
                                  1, 1, 0, 0) &&
                  unavailable.status().find("depth and motion") != std::string::npos,
              "Missing temporal guides did not produce a visible spatial fallback");
        // Full-resolution vendor output uses a separate immutable descriptor
        // binding from spatial output, preserving the game's original alpha.
        auto reconstructed = phoenix_native::image(
            device.Get(), 96, 54, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_STATE_COPY_DEST);
        initialize(reconstructed.Get(), 4, false);
        phoenix_native::transition(command.Get(), reconstructed.Get(),
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        bridge.present_full(command.Get(), reconstructed.Get(), games.front().Get(),
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        answers.push_back(
            read(games.front().Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
        offsets.push_back(0);
        const auto pending_bindings = bridge.binding_count();
        check(pending_bindings == 66, "Unsubmitted frames reused live descriptors");
        empty_timer.upscaler_status("Spatial fallback: test guides unavailable");
        empty_timer.end();
        phoenix_native::hr(command->Close());
        ID3D12CommandList* lists[]{command.Get()};
        queue->ExecuteCommandLists(1, lists);
        empty_timer.submitted(queue.Get(), 1, lists);
        ComPtr<ID3D12Fence> fence;
        phoenix_native::hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        phoenix_native::hr(queue->Signal(fence.Get(), 1));
        auto deadline = GetTickCount64() + 10000;
        while (fence->GetCompletedValue() < 1 && GetTickCount64() < deadline)
            Sleep(1);
        check(fence->GetCompletedValue() == 1, "Resolution test fence timeout");
        // A new batch with entirely new texture identities must reuse the
        // retired descriptor tables, without growing the pool or the network.
        phoenix_native::hr(allocator->Reset());
        phoenix_native::hr(command->Reset(allocator.Get(), nullptr));
        bridge.begin_frame(1, 2);
        for (unsigned index = 0; index < 32; ++index) {
            auto game = phoenix_native::image(device.Get(), 96, 54, DXGI_FORMAT_R32G32B32A32_FLOAT,
                                              D3D12_RESOURCE_STATE_COPY_DEST);
            const float offset = .4f + index * .01f;
            initialize(game.Get(), 4, false, offset);
            bridge.reduce(command.Get(), game.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                          low.Get(), false);
            phoenix_native::transition(command.Get(), low.Get(),
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            bridge.present(command.Get(), low.Get(), game.Get(),
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            phoenix_native::transition(command.Get(), low.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            answers.push_back(read(game.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
            offsets.push_back(offset);
            games.push_back(std::move(game));
        }
        check(bridge.binding_count() == pending_bindings, "Retired descriptor pool kept growing");
        phoenix_native::hr(command->Close());
        queue->ExecuteCommandLists(1, lists);
        phoenix_native::hr(queue->Signal(fence.Get(), 2));
        deadline = GetTickCount64() + 10000;
        while (fence->GetCompletedValue() < 2 && GetTickCount64() < deadline)
            Sleep(1);
        check(fence->GetCompletedValue() == 2, "Reused descriptor fence timeout");
        empty_timer.poll();
        const auto timing = empty_timer.report();
        check(timing.at("rows").size() == 1 &&
                  timing.at("rows").at(0).at("settings").at("UpscalerStatus") ==
                      "Spatial fallback: test guides unavailable",
              "Fallback timing was incorrectly attributed to a requested vendor method");
        void* mapped = nullptr;
        phoenix_native::hr(vectors.buffer->Map(0, nullptr, &mapped));
        for (unsigned row = 0; row < 32; ++row)
            for (unsigned column = 0; column < 48; ++column) {
                auto* value = reinterpret_cast<float*>(static_cast<uint8_t*>(mapped) +
                                                       vectors.footprint.Offset +
                                                       row * vectors.footprint.Footprint.RowPitch) +
                              column * 2;
                const bool active =
                    row >= geometry.top && row < geometry.top + geometry.image_height;
                check(std::abs(value[0] - (active ? 2.f : 0.f)) < 1e-6f &&
                          std::abs(value[1] - (active ? -1.f : 0.f)) < 1e-6f,
                      "Motion scaling or padding differs");
            }
        vectors.buffer->Unmap(0, nullptr);
        double max_error = 0;
        for (size_t index = 0; index < answers.size(); ++index) {
            auto& answer = answers[index];
            const float offset = offsets[index];
            phoenix_native::hr(answer.buffer->Map(0, nullptr, &mapped));
            for (unsigned row = 0; row < 54; ++row)
                for (unsigned column = 0; column < 96; ++column) {
                    auto* value = reinterpret_cast<float*>(
                                      static_cast<uint8_t*>(mapped) + answer.footprint.Offset +
                                      row * answer.footprint.Footprint.RowPitch) +
                                  column * 4;
                    max_error = std::max(
                        max_error, double(std::abs(value[0] - ((column + .5f) / 96 + offset))));
                    max_error = std::max(max_error, double(std::abs(value[1] - (row + .5f) / 54)));
                    check(std::abs(value[2] - .5f) < 1e-6f &&
                              value[3] == .25f + column * .001f + offset,
                          "Output alpha or constant channel changed");
                }
            answer.buffer->Unmap(0, nullptr);
        }
        check(max_error < .02, "Spatial reconstruction error exceeds bound");
        // A grey image must meter identically with and without letterboxing.
        // Jittered taps at the image edge must never blend in storage padding.
        phoenix_native::hr(allocator->Reset());
        phoenix_native::hr(command->Reset(allocator.Get(), nullptr));
        auto grey = phoenix_native::image(device.Get(), 96, 54, DXGI_FORMAT_R32G32B32A32_FLOAT,
                                          D3D12_RESOURCE_STATE_COPY_DEST);
        initialize(grey.Get(), 4, false, .23f, true);
        std::vector<std::unique_ptr<phoenix::direct::ResolutionBridge>> meter_bridges;
        std::vector<std::unique_ptr<phoenix::direct::Pass>> meters;
        std::vector<ComPtr<ID3D12Resource>> meter_images, meter_outputs;
        std::vector<Readback> meter_answers;
        for (const auto extent : {std::array<unsigned, 2>{96, 54}, {48, 36}, {32, 48}}) {
            const auto fitted = phoenix::fit_resolution(96, 54, extent[0], extent[1]);
            auto image = phoenix_native::image(device.Get(), extent[0], extent[1],
                                               DXGI_FORMAT_R32G32B32A32_FLOAT,
                                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            auto resampler = std::make_unique<phoenix::direct::ResolutionBridge>(
                device.Get(), fitted, DXGI_FORMAT_R32G32B32A32_FLOAT, nullptr);
            resampler->reduce(command.Get(), grey.Get(),
                              D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, image.Get(), false);
            auto meter = std::make_unique<phoenix::direct::Pass>(device.Get(),
                                                                 phoenix::direct::meter_shader);
            auto output = phoenix_native::image(device.Get(), 1, 1, DXGI_FORMAT_R32_FLOAT,
                                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            phoenix::direct::Constants common{extent[0], extent[1], 1, 0, 0, 0, 0, 0, 0};
            const float jitter = 2.5f;
            memcpy(&common.jitter_x_bits, &jitter, sizeof(jitter));
            memcpy(&common.jitter_y_bits, &jitter, sizeof(jitter));
            meter->record(command.Get(), {image.Get(), nullptr, nullptr}, output.Get(),
                          phoenix::direct::exposure_constants(common, fitted), true);
            meter_answers.push_back(read(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
            meter_bridges.push_back(std::move(resampler));
            meters.push_back(std::move(meter));
            meter_images.push_back(std::move(image));
            meter_outputs.push_back(std::move(output));
        }
        phoenix_native::hr(command->Close());
        queue->ExecuteCommandLists(1, lists);
        phoenix_native::hr(queue->Signal(fence.Get(), 3));
        deadline = GetTickCount64() + 10000;
        while (fence->GetCompletedValue() < 3 && GetTickCount64() < deadline)
            Sleep(1);
        check(fence->GetCompletedValue() == 3, "Exposure regression fence timeout");
        for (auto& answer : meter_answers) {
            phoenix_native::hr(answer.buffer->Map(0, nullptr, &mapped));
            const float exposure =
                *reinterpret_cast<float*>(static_cast<uint8_t*>(mapped) + answer.footprint.Offset);
            answer.buffer->Unmap(0, nullptr);
            check(std::abs(exposure - 1.f) < .0001f,
                  "Letterbox padding changed exposure or jitter taps escaped the image");
        }
        // DLL-only construction needs neither a Phoenix model nor a worker,
        // even when a previously selected CPU runner remains in saved settings.
        phoenix::Settings dll;
        dll.mode = phoenix::nvidia;
        dll.execution_device = "cpu";
        phoenix::direct::Runner dll_only(device.Get(), queue.Get(), {{"width", 96}, {"height", 64}},
                                         {}, {}, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                         DXGI_FORMAT_R16G16_FLOAT, false, 1,
                                         {{1, 1, 1, 1, 1, 1, 1, 1}}, {}, &dll);
        check(dll_only.architecture_matches(dll) && dll_only.launches() == 0,
              "DLL-only route created a Phoenix executor");
        if (messages)
            for (UINT64 index = 0; index < messages->GetNumStoredMessages(); ++index) {
                SIZE_T bytes = 0;
                messages->GetMessage(index, nullptr, &bytes);
                std::vector<uint8_t> storage(bytes);
                auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                messages->GetMessage(index, message, &bytes);
                check(message->Severity != D3D12_MESSAGE_SEVERITY_ERROR &&
                          message->Severity != D3D12_MESSAGE_SEVERITY_CORRUPTION,
                      message->pDescription);
            }
        std::cout << Json({{"passed", true},
                           {"software_d3d12", "WARP"},
                           {"model_inference_executed", false},
                           {"max_gradient_error", max_error},
                           {"motion_scaling_exact", true},
                           {"alpha_preserved", true},
                           {"letterbox_exposure_invariant", true},
                           {"jittered_exposure_edges_clamped", true},
                           {"rotating_buffers", 64},
                           {"descriptor_pool_capacity", pending_bindings},
                           {"retired_descriptors_reused", true},
                           {"dll_only_no_phoenix_model", true},
                           {"debug_layer", bool(debug)}})
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
