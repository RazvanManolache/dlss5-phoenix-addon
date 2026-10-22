// Explicit hardware test: the real production Runner, deterministic input,
// model changes and output copied back to an owned stand-in game texture.
#include "d3d12_test_context.h"
#include <chrono>
#include <iostream>
#include <numeric>

using phoenix_test::Gpu;
using phoenix_native::hr;
using phoenix_bundle::require;
using Json = nlohmann::json;
using Runner = phoenix_addon::phoenix::direct::Runner;
namespace fs = std::filesystem;

std::vector<uint16_t> pixels(Gpu& gpu, ID3D12Resource* image) {
    const auto desc = image->GetDesc();
    auto [buffer, footprint] = gpu.readback(image);
    gpu.submit();
    void* mapped = nullptr;
    hr(buffer->Map(0, nullptr, &mapped));
    std::vector<uint16_t> result(size_t(desc.Width) * desc.Height * 4);
    for (unsigned y = 0; y < desc.Height; ++y)
        memcpy(result.data() + y * desc.Width * 4,
               static_cast<const uint8_t*>(mapped) + footprint.Offset + y * footprint.Footprint.RowPitch,
               size_t(desc.Width) * 8);
    buffer->Unmap(0, nullptr);
    gpu.next();
    return result;
}

int main(int argc, char** argv) try {
    require(argc == 4, "Usage: model_switch_gpu_probe BUNDLE ORIGINAL_DLL SMALL_DLL");
    Gpu gpu(true);
    // Match the user's ultrawide extent; a downscaled generation is included
    // between full-size generations to reproduce the problematic transition.
    constexpr unsigned width = 2560, height = 1080;
    auto input = gpu.texture(width, height), motion = gpu.texture(width, height, true);
    auto baseline = pixels(gpu, input.Get());
    const auto paths = std::array{fs::path(argv[2]), fs::path(argv[3])};
    const auto digest = [](const auto& path) { return phoenix_model::mapped_file_digest(path); };
    phoenix_native::memory::Ledger ledger;
    std::unique_ptr<Runner> runner;
    std::unique_ptr<phoenix_bundle::Bundle> bundle;
    std::vector<phoenix_kernel::Binary> kernels;
    Json rows = Json::array();
    std::vector<uint16_t> original_output, lightweight_output;
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> queries;
    D3D12_QUERY_HEAP_DESC query_desc{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
    hr(gpu.device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&queries)));
    auto timestamps = phoenix_native::buffer(gpu.device.Get(), 16, D3D12_HEAP_TYPE_READBACK,
                                             D3D12_RESOURCE_STATE_COPY_DEST);
    uint64_t frequency = 0;
    hr(gpu.queue->GetTimestampFrequency(&frequency));
    const auto previous_group_marker = phoenix_native::group_marker;
    phoenix_native::group_marker = nullptr;
    unsigned retirements = 0;
    for (unsigned iteration = 0; iteration < 6; ++iteration) {
        const bool lightweight = iteration == 1 || iteration == 3;
        const bool scaled = iteration == 2;
        const auto& path = paths[lightweight ? 1 : 0];
        auto source = phoenix_model::WeightSource{path, digest(path)};
        source.model_id = lightweight ? "small-fp8" : "original";
        auto candidate_bundle = std::make_unique<phoenix_bundle::Bundle>(argv[1], digest, source, true);
        auto candidate_kernels = phoenix_kernel::load_catalog(phoenix_native::catalog(*candidate_bundle));
        phoenix_addon::phoenix::Settings settings;
        settings.mode = phoenix_addon::phoenix::ours;
        settings.model_id = source.model_id;
        settings.resolution_mode = scaled ? 1 : 0;
        const auto geometry = phoenix_addon::phoenix::choose_resolution(
            width, height, settings.resolution_mode, 0, 0, 16, 9, true);
        const auto& plan = candidate_bundle->select_frame(geometry.width, geometry.height,
            [&](const Json& frame) { phoenix_native::preflight(frame, candidate_kernels); });
        const auto weights = candidate_bundle->load_weight_view();
        // Exercise both supported transfer paths, including default-heap
        // staging after mapped GPU-upload generations.
        const bool gpu_upload = iteration < 4;
        require(phoenix_model_tools::replace_idle_generation(runner, [&] {
            return std::make_unique<Runner>(gpu.device.Get(), gpu.queue.Get(), plan, weights,
                candidate_kernels, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16_FLOAT,
                false, 1, settings.block_repeats, nullptr, &settings,
                ledger.begin(width, height), gpu_upload, width, height);
        }, [&](const Runner&) { ++retirements; }), "Previous runner failed to retire");
        bundle = std::move(candidate_bundle);
        kernels = std::move(candidate_kernels);
        std::vector<double> elapsed, gpu_elapsed;
        for (unsigned frame = 0; frame < 14; ++frame) {
            // Restore exactly the same input instead of repeatedly enhancing
            // the preceding output, which would invalidate the switch oracle.
            auto fresh = gpu.texture(width, height);
            phoenix_addon::FrameEvidence evidence;
            evidence.queue = gpu.queue.Get();
            evidence.states_verified = true;
            evidence.color_state = evidence.motion_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            evidence.restore_bindings = [](void*, ID3D12GraphicsCommandList*) {};
            phoenix_addon::Controls controls;
            controls.motion_x = width; controls.motion_y = height;
            const auto began = std::chrono::steady_clock::now();
            gpu.cmd->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            runner->record(gpu.cmd.Get(), fresh.Get(), motion.Get(), evidence, controls, settings, 8);
            gpu.cmd->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
            gpu.cmd->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, timestamps.Get(), 0);
            ID3D12CommandList* lists[]{gpu.cmd.Get()};
            runner->validate_submit(gpu.queue.Get(), 1, lists);
            gpu.submit(false);
            require(runner->after_submit(gpu.queue.Get(), 1, lists), "Frame submission unobserved");
            hr(gpu.queue->Signal(gpu.fence.Get(), ++gpu.serial));
            gpu.finish();
            void* mapped_times = nullptr;
            hr(timestamps->Map(0, nullptr, &mapped_times));
            uint64_t times[2]; memcpy(times, mapped_times, sizeof(times));
            timestamps->Unmap(0, nullptr);
            if (frame >= 4) gpu_elapsed.push_back(double(times[1] - times[0]) * 1000.0 / frequency);
            gpu.uploads.clear();
            gpu.retired_allocators.clear(); gpu.retired_lists.clear();
            if (frame >= 4) elapsed.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - began).count());
            gpu.next();
            if (frame == 13) {
                const auto output = pixels(gpu, fresh.Get());
                size_t changed = 0;
                for (size_t i = 0; i < output.size(); ++i) {
                    const float value = DirectX::PackedVector::XMConvertHalfToFloat(output[i]);
                    require(std::isfinite(value), "Nonfinite displayed output");
                    changed += output[i] != baseline[i];
                }
                require(changed > output.size() / 100, "Output writeback is effectively unchanged");
                if (iteration == 0) original_output = output;
                if (iteration == 1) lightweight_output = output;
                if (iteration == 5) require(output == original_output, "Switching back changed deterministic Original output");
                std::sort(elapsed.begin(), elapsed.end());
                std::sort(gpu_elapsed.begin(), gpu_elapsed.end());
                rows.push_back({{"model", source.model_id}, {"gpu_upload", gpu_upload},
                    {"scaled", scaled}, {"changed_components", changed},
                    {"wall_median_ms", elapsed[elapsed.size() / 2]},
                    {"gpu_median_ms", gpu_elapsed[gpu_elapsed.size() / 2]},
                    {"launches_per_frame", runner->launches() / 14}});
                std::cout << rows.back().dump() << '\n' << std::flush;
            }
        }
    }
    require(original_output != lightweight_output, "Models produced identical presented output");
    runner.reset(); bundle.reset(); kernels.clear();
    const auto accounting = ledger.snapshot();
    require(accounting.at("totals").at("live_allocation_info_bytes") == 0,
            "Retired model generations retain tracked buffers");
    const auto api_lifetime = phoenix_native::Api::lifetime_report();
    require(api_lifetime.at("initializations") == api_lifetime.at("unloads") &&
            api_lifetime.at("unload_failures") == 0, "NVAPI references leaked during model changes");
    gpu.validate();
    phoenix_native::group_marker = previous_group_marker;
    std::cout << Json({{"passed", true}, {"gpu_used", true}, {"rows", rows},
        {"retirements", retirements}, {"nvapi", api_lifetime}, {"owned_allocations", accounting},
        {"scope", "Production Runner model switches and owned game-output texture; no retail-game final-swapchain claim"}}).dump() << '\n';
    return 0;
} catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n'; return 1;
}
