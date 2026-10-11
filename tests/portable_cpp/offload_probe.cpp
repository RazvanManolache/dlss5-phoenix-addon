#include "../../src/native/support/offload.h"
#include <dxgi1_6.h>
#include <d3d12sdklayers.h>
#include <fstream>
#include <iostream>
using namespace phoenix_addon::phoenix;
using phoenix_native::hr;
using Microsoft::WRL::ComPtr;
constexpr unsigned image_width = 96, image_height = 64;
using Pixel = std::array<float, 4>;
int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 8)
            throw std::runtime_error(
                "Usage: RUNTIME_ROOT UNUSED WEIGHT_SOURCE_JSON WORKER_DEVICE ADAPTER ORACLE_BIN NEW_OUTPUT");
        const std::filesystem::path out = argv[7];
        if (std::filesystem::exists(out))
            throw std::runtime_error("Fresh output required");
        std::filesystem::create_directories(out);
        std::vector<float> oracle(image_width * image_height * 3);
        {
            std::ifstream file(argv[6], std::ios::binary);
            file.read(reinterpret_cast<char*>(oracle.data()), oracle.size() * sizeof(float));
            if (!file)
                throw std::runtime_error("Missing oracle");
        }
        ComPtr<ID3D12Debug> debug;
        hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
        debug->EnableDebugLayer();
        ComPtr<IDXGIFactory4> factory;
        hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
        if (std::wstring(argv[5]) == L"auto") {
            for (unsigned iteration_index = 0;; ++iteration_index) {
                ComPtr<IDXGIAdapter1> candidate;
                if (factory->EnumAdapters1(iteration_index, &candidate) == DXGI_ERROR_NOT_FOUND)
                    break;
                DXGI_ADAPTER_DESC1 description{};
                hr(candidate->GetDesc1(&description));
                if (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                    continue;
                if (!adapter || description.VendorId == 0x10de)
                    adapter = candidate;
                if (description.VendorId == 0x10de)
                    break;
            }
            if (!adapter)
                throw std::runtime_error("No physical D3D12 GPU");
        } else
            hr(factory->EnumAdapters1(unsigned(std::stoul(argv[5])), &adapter));
        DXGI_ADAPTER_DESC1 selected{};
        hr(adapter->GetDesc1(&selected));
        if (selected.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            throw std::runtime_error("Software adapter is not a GPU validation");
        ComPtr<ID3D12Device> device;
        hr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        ComPtr<ID3D12InfoQueue> messages;
        hr(device.As(&messages));
        ComPtr<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC queue_description{};
        hr(device->CreateCommandQueue(&queue_description, IID_PPV_ARGS(&queue)));
        ComPtr<ID3D12CommandAllocator> allocator;
        hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                          IID_PPV_ARGS(&allocator)));
        ComPtr<ID3D12GraphicsCommandList> cmd;
        hr(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
                                     IID_PPV_ARGS(&cmd)));
        ComPtr<ID3D12Fence> fence;
        hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event)
            throw std::runtime_error("Fence event");
        uint64_t gpu_serial = 0;
        auto execute = [&] {
            hr(cmd->Close());
            ID3D12CommandList* lists[]{cmd.Get()};
            queue->ExecuteCommandLists(1, lists);
            hr(queue->Signal(fence.Get(), ++gpu_serial));
            hr(fence->SetEventOnCompletion(gpu_serial, event));
            if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0)
                throw std::runtime_error("GPU timeout");
            hr(allocator->Reset());
            hr(cmd->Reset(allocator.Get(), nullptr));
        };
        auto create = [&](D3D12_RESOURCE_DESC description, D3D12_HEAP_TYPE heap,
                          D3D12_RESOURCE_STATES state) {
            D3D12_HEAP_PROPERTIES heap_properties{};
            heap_properties.Type = heap;
            ComPtr<ID3D12Resource> resource;
            hr(device->CreateCommittedResource(&heap_properties, D3D12_HEAP_FLAG_NONE, &description,
                                               state, nullptr, IID_PPV_ARGS(&resource)));
            return resource;
        };
        auto barrier = [&](ID3D12Resource* resource, D3D12_RESOURCE_STATES before_state,
                           D3D12_RESOURCE_STATES after_state) {
            D3D12_RESOURCE_BARRIER transition{};
            transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            transition.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                     before_state, after_state};
            cmd->ResourceBarrier(1, &transition);
        };
        constexpr auto shader_read_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        D3D12_RESOURCE_DESC texture_description{};
        texture_description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture_description.Width = image_width;
        texture_description.Height = image_height;
        texture_description.DepthOrArraySize = texture_description.MipLevels = 1;
        texture_description.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        texture_description.SampleDesc.Count = 1;
        auto color =
            create(texture_description, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        auto guides =
            create(texture_description, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 total = 0;
        device->GetCopyableFootprints(&texture_description, 0, 1, 0, &footprint, nullptr, nullptr,
                                      &total);
        D3D12_RESOURCE_DESC buffer_description{};
        buffer_description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer_description.Width = total;
        buffer_description.Height = 1;
        buffer_description.DepthOrArraySize = buffer_description.MipLevels = 1;
        buffer_description.SampleDesc.Count = 1;
        buffer_description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        auto upload =
            create(buffer_description, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto readback =
            create(buffer_description, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto copy = [&](ID3D12Resource* tex, ID3D12Resource* buffer, bool download) {
            D3D12_TEXTURE_COPY_LOCATION texture_location{}, buffer_location{};
            texture_location.pResource = tex;
            texture_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            buffer_location.pResource = buffer;
            buffer_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            buffer_location.PlacedFootprint = footprint;
            if (download)
                cmd->CopyTextureRegion(&buffer_location, 0, 0, 0, &texture_location, nullptr);
            else
                cmd->CopyTextureRegion(&texture_location, 0, 0, 0, &buffer_location, nullptr);
        };
        auto fill = [&](bool is_color, float motion = 0, float depth = 1) {
            void* mapped = nullptr;
            D3D12_RANGE empty{0, 0};
            hr(upload->Map(0, &empty, &mapped));
            for (unsigned row_index = 0; row_index < image_height; ++row_index)
                for (unsigned column_index = 0; column_index < image_width; ++column_index) {
                    Pixel pixel = is_color ? Pixel{float(column_index) / image_width,
                                                   float(row_index) / image_height, .3f, .8f}
                                           : Pixel{motion, 0, depth, 1};
                    std::memcpy(static_cast<char*>(mapped) +
                                    row_index * footprint.Footprint.RowPitch +
                                    column_index * sizeof(Pixel),
                                pixel.data(), sizeof(Pixel));
                }
            upload->Unmap(0, nullptr);
            auto* resource = is_color ? color.Get() : guides.Get();
            copy(resource, upload.Get(), false);
            barrier(resource, D3D12_RESOURCE_STATE_COPY_DEST, shader_read_state);
            execute();
        };
        fill(true);
        fill(false);
        // Extend age only for this intentionally slow CPU/reference correctness
        // probe. Production defaults remain four frames / 100 milliseconds.
        portable::configure_native_runtime(argv[1]);
        phoenix_model::WeightSource weights;
        weights = phoenix_model::WeightSource::config(phoenix_bundle::json_file(argv[3]));
        const bool slow_cpu = std::wstring(argv[4]) == L"sycl:cpu";
        const unsigned age_ms = slow_cpu ? 30000 : 5000;
        auto executor_owner = std::make_unique<portable::OffloadExecutor>(
            device.Get(), image_width, image_height, std::filesystem::path{},
            std::filesystem::path{}, std::filesystem::path{}, argv[4], out, 6000, age_ms,
            std::filesystem::path{}, L"auto", std::filesystem::path{}, std::filesystem::path{},
            weights);
        auto& executor = *executor_owner;
        phoenix_native::memory::Ledger owned;
        auto generation = owned.begin(image_width, image_height);
        executor.account(generation);
        ExecutorFrame frame;
        frame.command = cmd.Get();
        frame.color = color.Get();
        frame.guides = guides.Get();
        frame.controls.auto_mask = 1;
        unsigned held = 0;
        uint64_t capture = 0;
        bool fence_checked = false;
        double max_record_ms = 0;
        auto deadline = GetTickCount64() + 45000;
        std::vector<Pixel> actual(image_width * image_height);
        auto record = [&](bool hold = false) {
            frame.serial = gpu_serial + 1;
            frame.completed_serial = hold ? capture - 1 : gpu_serial;
            const auto start = std::chrono::steady_clock::now();
            auto* result = executor.record(frame);
            max_record_ms = std::max(max_record_ms, std::chrono::duration<double, std::milli>(
                                                        std::chrono::steady_clock::now() - start)
                                                        .count());
            barrier(result, shader_read_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
            copy(result, readback.Get(), true);
            barrier(result, D3D12_RESOURCE_STATE_COPY_SOURCE, shader_read_state);
            execute();
            void* data = nullptr;
            hr(readback->Map(0, nullptr, &data));
            for (unsigned row_index = 0; row_index < image_height; ++row_index)
                std::memcpy(actual.data() + row_index * image_width,
                            static_cast<char*>(data) + row_index * footprint.Footprint.RowPitch,
                            image_width * sizeof(Pixel));
            D3D12_RANGE empty{0, 0};
            readback->Unmap(0, &empty);
            return result;
        };
        while (!executor.status().at("delivered").get<unsigned>()) {
            if (GetTickCount64() > deadline)
                throw std::runtime_error("Offload timed out: " + executor.status().dump());
            const bool hold = capture && held < 4;
            record(hold);
            const auto execution_status = executor.status();
            if (execution_status.at("state") == "failed_input_preserved")
                throw std::runtime_error(execution_status.dump());
            if (!capture && execution_status.at("captures").get<unsigned>())
                capture = execution_status.at("capture_fence").get<uint64_t>();
            else if (hold) {
                if (execution_status.at("worker_submitted").get<bool>())
                    throw std::runtime_error("Readback consumed before capture fence");
                ++held;
                fence_checked = held == 4;
            }
            Sleep(10);
        }
        float max_error = 0;
        auto verify = [&](unsigned shift, bool unchanged) {
            for (unsigned row_index = 0; row_index < image_height; ++row_index)
                for (unsigned column_index = 0; column_index < image_width; ++column_index)
                    for (unsigned channel = 0; channel < 4; ++channel) {
                        const Pixel source{float(column_index) / image_width,
                                           float(row_index) / image_height, .3f, .8f};
                        float expected = source[channel];
                        if (channel < 3 && !unchanged && column_index + shift < image_width) {
                            const Pixel captured{float(column_index + shift) / image_width,
                                                 float(row_index) / image_height, .3f, .8f};
                            expected = std::clamp(
                                source[channel] +
                                    oracle[(row_index * image_width + column_index + shift) * 3 +
                                           channel] -
                                    captured[channel],
                                0.f, 1.f);
                        }
                        const float error = std::fabs(
                            actual[row_index * image_width + column_index][channel] - expected);
                        max_error = std::max(max_error, error);
                        if (!std::isfinite(
                                actual[row_index * image_width + column_index][channel]) ||
                            error > 2e-5f)
                            throw std::runtime_error("Offload pixel mismatch: " +
                                                     std::to_string(error));
                    }
        };
        verify(0, false);
        const auto first_frame = executor.status().at("ready_frame").get<uint64_t>();
        auto spatial_oracle = oracle;
        {
            std::ifstream file(std::filesystem::path(argv[6]).parent_path() / "temporal.bin",
                               std::ios::binary);
            file.read(reinterpret_cast<char*>(oracle.data()), oracle.size() * sizeof(float));
            if (!file)
                throw std::runtime_error("Missing temporal oracle");
        }
        deadline = GetTickCount64() + 30000;
        while (executor.status().at("delivered").get<unsigned>() < 2) {
            if (GetTickCount64() > deadline)
                throw std::runtime_error("Temporal offload timed out: " + executor.status().dump());
            record();
            Sleep(10);
        }
        verify(0, false);
        const auto temporal_result = executor.status().at("last_result");
        if (!temporal_result.at("history_used").get<bool>() ||
            temporal_result.at("motion_from_frame").get<uint64_t>() != first_frame ||
            temporal_result.at("frame_id").get<uint64_t>() <= first_frame + 1)
            throw std::runtime_error("Sparse inference did not retain the correct history origin");
        // Directly record the guide copy in the next display command list instead.
        auto prepare_guides = [&](float motion, float depth) {
            void* data = nullptr;
            D3D12_RANGE empty{0, 0};
            hr(upload->Map(0, &empty, &data));
            for (unsigned row_index = 0; row_index < image_height; ++row_index)
                for (unsigned column_index = 0; column_index < image_width; ++column_index) {
                    Pixel pixel{motion, 0, depth, 1};
                    std::memcpy(static_cast<char*>(data) +
                                    row_index * footprint.Footprint.RowPitch +
                                    column_index * sizeof(Pixel),
                                pixel.data(), sizeof(Pixel));
                }
            upload->Unmap(0, nullptr);
            barrier(guides.Get(), shader_read_state, D3D12_RESOURCE_STATE_COPY_DEST);
            copy(guides.Get(), upload.Get(), false);
            barrier(guides.Get(), D3D12_RESOURCE_STATE_COPY_DEST, shader_read_state);
        };
        prepare_guides(1, 1);
        record();
        verify(1, false);
        record();
        verify(2, false);
        prepare_guides(0, 2);
        record();
        verify(0, true);
        frame.controls.reset = 1;
        auto* reset_result = record();
        verify(0, true);
        if (reset_result != color.Get())
            throw std::runtime_error("Reset published old result");
        frame.controls.reset = 0;
        deadline = GetTickCount64() + 30000;
        oracle = spatial_oracle;
        while (executor.status().at("delivered").get<unsigned>() < 3) {
            if (GetTickCount64() > deadline)
                throw std::runtime_error("Offload recovery timed out: " + executor.status().dump());
            record();
            if (executor.status().at("delivered").get<unsigned>() < 3)
                verify(0, true);
            Sleep(10);
        }
        verify(0, false);
        if (!executor.status().at("discarded").get<unsigned>())
            throw std::runtime_error("Reset did not discard the in-flight generation");
        Sleep(age_ms + 100);
        record();
        verify(0, true);
        if (executor.status().at("ready_frame").get<uint64_t>())
            throw std::runtime_error("Expired residual stayed eligible");
        unsigned errors = 0;
        for (UINT64 iteration_index = 0; iteration_index < messages->GetNumStoredMessages();
             ++iteration_index) {
            SIZE_T size = 0;
            messages->GetMessage(iteration_index, nullptr, &size);
            std::vector<char> data(size);
            auto* debug_message = reinterpret_cast<D3D12_MESSAGE*>(data.data());
            hr(messages->GetMessage(iteration_index, debug_message, &size));
            if (debug_message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                ++errors;
                std::cerr << debug_message->pDescription << '\n';
            }
        }
        if (errors || !fence_checked)
            throw std::runtime_error("D3D12 validation or fence gate failure");
        hr(cmd->Close());
        CloseHandle(event);
        DXGI_ADAPTER_DESC1 desc{};
        hr(adapter->GetDesc1(&desc));
        char name[256]{};
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name), nullptr, nullptr);
        auto report = executor.status();
        report["owned_transport_before_shutdown"] = owned.snapshot();
        {
            phoenix_native::memory::SessionLifetime lifetime{generation};
            executor_owner.reset();
        }
        report["owned_transport_after_shutdown"] = owned.snapshot();
        if (!generation->reclaimable() || generation->total.allocated != generation->total.freed)
            throw std::runtime_error("Offload transport allocations did not retire");
        report["passed"] = true;
        report["adapter"] = name;
        report["max_pixel_error"] = max_error;
        report["max_record_wall_ms"] = max_record_ms;
        report["debug_errors"] = errors;
        report["capture_fence_gate"] = fence_checked;
        report["temporal_result"] = temporal_result;
        report["checks"] = {"GPU input through real worker to GPU output",
                            "sparse inference retains correct temporal history",
                            "two-frame accumulated motion",
                            "depth rejection",
                            "reset rejects stale output",
                            "in-flight generation discard",
                            "result recovers",
                            "expired output rejected",
                            "alpha preserved"};
        std::ofstream file(out / "report.json");
        file << report.dump(2);
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
