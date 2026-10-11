// Offline game-adapter regression: real refinement passes, captures and teardown.
// Harness uploads/readback are deliberately outside the owned adapter ledger.
#include "../../src/native/runtime/texture_binding.h"
#include "../../src/native/support/direct.h"
#include "../../src/native/support/gpu_timing.h"
#include <dxgi1_6.h>
#include <psapi.h>
#include <iostream>
using namespace phoenix_addon;
namespace fs = std::filesystem;
static phoenix::GpuTiming* active_timer = nullptr;
static std::string hash_bytes(const std::vector<uint8_t>& bytes) {
    unsigned char hash[32];
    char text[65]{};
    check(BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0, const_cast<PUCHAR>(bytes.data()),
                     ULONG(bytes.size()), hash, 32) >= 0,
          "Hash failed");
    for (unsigned index = 0; index < 32; ++index)
        sprintf_s(text + index * 2, 3, "%02x", unsigned(hash[index]));
    return text;
}
static Json process_memory(IDXGIAdapter3* adapter) {
    PROCESS_MEMORY_COUNTERS_EX memory{};
    memory.cb = sizeof(memory);
    check(GetProcessMemoryInfo(GetCurrentProcess(),
                               reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory)),
          "Process memory query failed");
    DXGI_QUERY_VIDEO_MEMORY_INFO local{}, nonlocal{};
    phoenix_native::hr(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local));
    phoenix_native::hr(
        adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal));
    return {{"private_commit_bytes", memory.PrivateUsage},
            {"working_set_bytes", memory.WorkingSetSize},
            {"device_local_bytes", local.CurrentUsage},
            {"device_nonlocal_bytes", nonlocal.CurrentUsage}};
}
struct Inputs : phoenix_device::Payload {
    Texture color, motion, depth, readback;
    Inputs(ID3D12Device* device, unsigned width, unsigned height, const std::vector<uint8_t>& rgb,
           const std::vector<uint8_t>& vectors, bool temporal = false) {
        auto make_input = [&](DXGI_FORMAT format) {
            Texture input;
            input.resource =
                phoenix_native::image(device, width, height, format,
                                      D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, false);
            const auto description = input.resource->GetDesc();
            device->GetCopyableFootprints(&description, 0, 1, 0, &input.footprint, nullptr, nullptr,
                                          &input.bytes);
            input.transfer = phoenix_native::buffer(device, input.bytes, D3D12_HEAP_TYPE_UPLOAD,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ);
            return input;
        };
        color = make_input(DXGI_FORMAT_R16G16B16A16_FLOAT);
        motion = make_input(DXGI_FORMAT_R16G16_FLOAT);
        if (temporal)
            depth = make_input(DXGI_FORMAT_R32_FLOAT);
        readback = texture(device, width, height, 16, true);
        for (unsigned kind = 0; kind < (temporal ? 3u : 2u); ++kind) {
            auto& target = kind == 2 ? depth : kind ? motion : color;
            void* mapped = nullptr;
            D3D12_RANGE empty{};
            phoenix_native::hr(target.transfer->Map(0, &empty, &mapped));
            memset(mapped, 0, size_t(target.bytes));
            for (unsigned row = 0; row < height; ++row) {
                auto* destination = static_cast<uint8_t*>(mapped) + target.footprint.Offset +
                                    row * target.footprint.Footprint.RowPitch;
                if (kind == 2) {
                    const float value = .8f;
                    for (unsigned column = 0; column < width; ++column)
                        std::memcpy(destination + column * 4, &value, 4);
                } else if (!kind)
                    memcpy(destination, rgb.data() + size_t(row) * width * 8, size_t(width) * 8);
                else
                    memcpy(destination, vectors.data() + size_t(row) * width * 4,
                           size_t(width) * 4);
            }
            target.transfer->Unmap(0, nullptr);
        }
    }
};
int wmain(int argc, wchar_t** argv) {
    Json result{{"passed", false}, {"cases", Json::array()}};
    fs::path destination;
    try {
        check(
            argc >= 5 && argc <= 10,
            "direct_memory_probe BUNDLE CAPTURE_PREFIX WEIGHT_CONFIG NEW_OUTPUT [--staged] [--super-resolution] [--xess|--dlss-sr --upscaler-runtime DIRECTORY]");
        bool mapped_weights = true, super_resolution = false;
        int upscaler_method = 0;
        fs::path upscaler_directory;
        for (int index = 5; index < argc; ++index) {
            const std::wstring flag = argv[index];
            if (flag == L"--staged")
                mapped_weights = false;
            else if (flag == L"--super-resolution")
                super_resolution = true;
            else if (flag == L"--xess" || flag == L"--dlss-sr") {
                upscaler_method = flag == L"--xess" ? 2 : 1;
                super_resolution = true;
            } else if (flag == L"--upscaler-runtime" && index + 1 < argc)
                upscaler_directory = argv[++index];
            else
                check(false, "Unknown probe option");
        }
        check(!upscaler_method || !upscaler_directory.empty(),
              "Vendor probe requires --upscaler-runtime DIRECTORY");
        result["upscaler"] = upscaler_method;
        result["synthetic_depth_fixture"] = upscaler_method != 0;
        result["allow_gpu_upload"] = mapped_weights;
        destination = argv[4];
        check(!fs::exists(destination), "Fresh output directory required");
        fs::create_directories(destination);
        Json configuration;
        std::ifstream(argv[3]) >> configuration;
        phoenix_bundle::Bundle bundle(
            argv[1], [](const fs::path& path) { return hash_bytes(read(path)); },
            phoenix_model::WeightSource::config(configuration));
        auto weights = bundle.load_weights();
        auto kernels = phoenix_kernel::load_catalog(phoenix_native::catalog(bundle));
        fs::path prefix = argv[2];
        auto metadata = read(prefix.string() + ".meta");
        phoenix::direct::CaptureMeta capture;
        check(phoenix::direct::capture_meta_parse(std::string(metadata.begin(), metadata.end()),
                                                  capture),
              "Capture metadata failed");
        const auto geometry = phoenix::choose_resolution(
            capture.width, capture.height, super_resolution ? 1 : 0, 0, 0, 16, 9, true);
        result["resolution"] = {{"width", geometry.width},
                                {"height", geometry.height},
                                {"output_width", capture.width},
                                {"output_height", capture.height},
                                {"fallback", geometry.fallback}};
        const Json* selected = nullptr;
        for (auto* plan :
             {&bundle.small_plan, &bundle.full_plan, &bundle.hd_plan, &bundle.resize_plan})
            if (plan->contains("frames"))
                for (const auto& frame : plan->at("frames"))
                    if (frame.at("width") == geometry.width &&
                        frame.at("height") == geometry.height &&
                        phoenix_bundle::has_barrier_policy(frame))
                        selected = &frame;
        check(selected != nullptr, "No production plan for capture extent");
        auto rgb = read(prefix.string() + ".proxy.bin");
        auto vectors = read(prefix.string() + ".motion.bin");
        check(rgb.size() == size_t(capture.width) * capture.height * 8 &&
                  vectors.size() == size_t(capture.width) * capture.height * 4,
              "Fixture sizes");
        ComPtr<ID3D12Debug> debug;
        phoenix_native::hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
        debug->EnableDebugLayer();
        ComPtr<IDXGIFactory6> factory;
        phoenix_native::hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> adapter;
        DXGI_ADAPTER_DESC1 description{};
        for (unsigned index = 0;; ++index) {
            phoenix_native::hr(factory->EnumAdapterByGpuPreference(
                index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)));
            adapter->GetDesc1(&description);
            if (description.VendorId == 0x10de && !(description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
                break;
            adapter.Reset();
        }
        ComPtr<IDXGIAdapter3> measured;
        phoenix_native::hr(adapter.As(&measured));
        phoenix_device::Owner owner;
        owner.initialize(description.AdapterLuid, {capture.width, capture.height},
                         [&](ID3D12Device* device, phoenix_device::Extent, uint64_t) {
                             return std::make_unique<Inputs>(device, capture.width, capture.height,
                                                             rgb, vectors, upscaler_method != 0);
                         });
        if (upscaler_method == 1) {
#ifdef PHOENIX_PROBE_NGX_SDK
            const auto folder = upscaler_directory.wstring();
            const wchar_t* search[]{folder.c_str()};
            NVSDK_NGX_FeatureCommonInfo info{};
            info.PathListInfo = {search, 1};
            auto cache = fs::temp_directory_path() / L"DLSS5Phoenix/ngx_probe";
            fs::create_directories(cache);
            check(NVSDK_NGX_SUCCEED(NVSDK_NGX_D3D12_Init_with_ProjectID(
                      "853cf9ad-19cb-4f49-96b6-7b3896477d20", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
                      cache.c_str(), owner.device(), &info)),
                  "Offline NGX SDK initialization failed");
#else
            check(false, "Build offline DLSS probe with --ngx-library nvsdk_ngx_d.lib");
#endif
        }
        auto* inputs = static_cast<Inputs*>(owner.payload());
        phoenix_native::memory::Ledger ledger;
        for (unsigned passes :
             super_resolution ? std::vector<unsigned>{1u} : std::vector<unsigned>{1u, 2u, 4u, 1u}) {
            Json row{{"passes", passes},
                     {"output_hashes", Json::array()},
                     {"before", process_memory(measured.Get())}};
            auto generation = ledger.begin(capture.width, capture.height);
            {
                phoenix::Settings settings;
                settings.refinement_passes = int(passes);
                settings.mode = phoenix::ours;
                settings.upscaler = upscaler_method;
                settings.resolution_mode = super_resolution ? 1 : 0;
                auto runner = std::make_unique<phoenix::direct::Runner>(
                    owner.device(), owner.queue(), *selected, weights, kernels,
                    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16_FLOAT, false, passes,
                    std::array<int, 8>{{1, 1, 1, 1, 1, 1, 1, 1}},
                    std::unique_ptr<phoenix::NeuralExecutor>{}, &settings, generation,
                    mapped_weights, capture.width, capture.height, upscaler_directory);
                phoenix::GpuTiming timer(owner.device(), owner.queue());
                active_timer = &timer;
                phoenix::direct::network_timing_begin = [] { active_timer->own_begin(); };
                phoenix::direct::network_timing_end = [] { active_timer->own_end(); };
                row["constructed"] = process_memory(measured.Get());
                row["owned_constructed"] = ledger.snapshot();
                // Both normal and capture requests traverse the adapter used by the addon.
                runner->arm_capture("memory_probe", 1);
                row["capture_armed"] = process_memory(measured.Get());
                row["owned_capture_armed"] = ledger.snapshot();
                for (unsigned frame_index = 0; frame_index < 12; ++frame_index) {
                    phoenix_device::Frame work;
                    check(owner.try_begin(work) == phoenix_device::Status::complete, "Frame slot");
                    for (auto* input : {&inputs->color, &inputs->motion, &inputs->depth}) {
                        if (!input->resource)
                            continue;
                        phoenix_native::transition(work.commands, input->resource.Get(),
                                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                   D3D12_RESOURCE_STATE_COPY_DEST);
                        D3D12_TEXTURE_COPY_LOCATION source{}, target{};
                        source.pResource = input->transfer.Get();
                        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                        source.PlacedFootprint = input->footprint;
                        target.pResource = input->resource.Get();
                        target.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                        work.commands->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
                        phoenix_native::transition(work.commands, input->resource.Get(),
                                                   D3D12_RESOURCE_STATE_COPY_DEST,
                                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    }
                    FrameEvidence evidence;
                    evidence.queue = owner.queue();
                    evidence.color_state = evidence.motion_state =
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    evidence.states_verified = true;
                    evidence.restore_bindings = [](void*, ID3D12GraphicsCommandList*) {};
                    Controls controls;
                    controls.reset = frame_index == 0 || frame_index == 7;
                    timer.begin(work.commands, 3, 8, capture.width, capture.height);
                    runner->record(work.commands, inputs->color.resource.Get(),
                                   inputs->motion.resource.Get(), evidence, controls, settings, 8,
                                   inputs->depth.resource.Get(),
                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    phoenix_native::transition(work.commands, inputs->color.resource.Get(),
                                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                               D3D12_RESOURCE_STATE_COPY_SOURCE);
                    D3D12_TEXTURE_COPY_LOCATION source{}, target{};
                    source.pResource = inputs->color.resource.Get();
                    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    target.pResource = inputs->readback.transfer.Get();
                    target.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                    target.PlacedFootprint = inputs->readback.footprint;
                    work.commands->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
                    phoenix_native::transition(work.commands, inputs->color.resource.Get(),
                                               D3D12_RESOURCE_STATE_COPY_SOURCE,
                                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    timer.upscaler_status(runner->upscale_status());
                    timer.end();
                    ID3D12CommandList* lists[]{work.commands};
                    auto ticket = owner.submit(work);
                    check(runner->after_submit(owner.queue(), 1, lists), "Adapter commit");
                    timer.submitted(owner.queue(), 1, lists);
                    check(owner.wait(ticket, 30000) == phoenix_device::Status::complete,
                          "GPU wait");
                    timer.poll();
                    runner->poll_exposure();
                    runner->poll_captures();
                    phoenix::direct::CaptureFrame retired_capture;
                    runner->take_capture(retired_capture);
                    void* mapped = nullptr;
                    phoenix_native::hr(inputs->readback.transfer->Map(0, nullptr, &mapped));
                    std::vector<uint8_t> image(size_t(capture.width) * capture.height * 8);
                    for (unsigned row_index = 0; row_index < capture.height; ++row_index)
                        memcpy(image.data() + size_t(row_index) * capture.width * 8,
                               static_cast<uint8_t*>(mapped) + inputs->readback.footprint.Offset +
                                   row_index * inputs->readback.footprint.Footprint.RowPitch,
                               size_t(capture.width) * 8);
                    inputs->readback.transfer->Unmap(0, nullptr);
                    for (size_t offset = 0; offset < image.size(); offset += 2) {
                        uint16_t value = 0;
                        std::memcpy(&value, image.data() + offset, 2);
                        check((value & 0x7c00u) != 0x7c00u, "Nonfinite output in direct adapter");
                        if (super_resolution && offset % 8 == 6)
                            check(image[offset] == rgb[offset] &&
                                      image[offset + 1] == rgb[offset + 1],
                                  "Upscale did not preserve original alpha");
                    }
                    row["output_hashes"].push_back(hash_bytes(image));
                }
                row["upscaler_status"] = runner->upscale_status();
                if (upscaler_method)
                    check(runner->upscale_status().find("active") != std::string::npos,
                          runner->upscale_status().c_str());
                row["gpu_timing"] = timer.report();
                active_timer = nullptr;
                phoenix::direct::network_timing_begin = nullptr;
                phoenix::direct::network_timing_end = nullptr;
                row["steady"] = process_memory(measured.Get());
                row["owned_steady"] = ledger.snapshot();
                row["capture_readback_bytes_after_retirement"] = runner->capture_readback_bytes();
                check(runner->idle(), "Adapter teardown requires retired GPU work");
            }
            row["after_shutdown"] = process_memory(measured.Get());
            row["owned_after_shutdown"] = ledger.snapshot();
            check(ledger.snapshot()["totals"]["live_allocation_info_bytes"] == 0,
                  "Adapter leaked owned allocation bytes");
            check(generation->reclaimable(), "Adapter generation did not retire completely");
            result["cases"].push_back(std::move(row));
        }
        // A rejected constructor must release even the bridge allocations
        // created before its refinement-count preflight fails.
        auto rejected = ledger.begin(capture.width, capture.height);
        bool refused = false;
        try {
            phoenix::direct::Runner invalid(owner.device(), owner.queue(), *selected, weights,
                                            kernels, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                            DXGI_FORMAT_R16G16_FLOAT, false, 5,
                                            {{1, 1, 1, 1, 1, 1, 1, 1}}, {}, nullptr, rejected);
        } catch (const std::exception&) {
            refused = true;
        }
        check(refused && rejected->reclaimable(), "Failed constructor retained owned objects");
        result["failed_constructor_retired"] = true;
        ComPtr<ID3D12InfoQueue> messages;
        phoenix_native::hr(owner.device()->QueryInterface(IID_PPV_ARGS(&messages)));
        unsigned errors = 0;
        for (UINT64 index = 0; index < messages->GetNumStoredMessages(); ++index) {
            SIZE_T size = 0;
            messages->GetMessage(index, nullptr, &size);
            std::vector<uint8_t> storage(size);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            messages->GetMessage(index, message, &size);
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::cerr << message->pDescription << '\n';
                ++errors;
            }
        }
        check(errors == 0, "D3D12 errors in adapter test");
        result["debug_errors"] = errors;
        result["passed"] = true;
        std::ofstream(destination / "result.json") << result.dump(2);
        return 0;
    } catch (const std::exception& failure) {
        result["error"] = failure.what();
        if (!destination.empty() && fs::exists(destination))
            std::ofstream(destination / "result.json") << result.dump(2);
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
