// Real D3D12 allocation/recording probe. Does not submit inference or launch games.
#include "../../src/native/support/production_direct.h"
#include <iostream>
int main(int argc, char** argv) try {
    using namespace phoenix_native;
    using namespace phoenix_addon::phoenix::direct;
    const bool optimized = argc < 2 || std::string_view(argv[1]) != "before";
    ComPtr<IDXGIFactory6> factory; hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter;
    hr(factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)));
    ComPtr<ID3D12Device> device; hr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
    ComPtr<ID3D12CommandAllocator> allocator;
    hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
    ComPtr<ID3D12GraphicsCommandList> command;
    hr(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&command)));
    Json result = Json::array();
    for (auto extent : {std::array<unsigned,2>{1920,1080}, std::array<unsigned,2>{3840,2160}}) {
        memory::Ledger ledger; auto generation = ledger.begin(extent[0], extent[1]);
        auto snapshot = [&] { return generation->json().at("categories").at("bridge"); };
        {
            AppearanceStage stage(device.Get(), extent[0], extent[1], generation);
            const auto empty = snapshot();
            require(empty.at("live_objects") == (optimized ? 0 : 1), "Unused appearance allocation contract");
            auto input = image(device.Get(), extent[0], extent[1], DXGI_FORMAT_R16G16B16A16_FLOAT,
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            phoenix_addon::Controls controls; controls.style = 1;
            stage.prepare(command.Get(), input.Get(), nullptr, {}, controls, 0);
            stage.present(command.Get(), input.Get(), {}, controls, 0);
            const auto style = snapshot();
            require(style.at("live_objects") == (optimized ? 3 : 7), "Style allocation contract");
            stage.mask_target(); const auto mask = snapshot();
            require(mask.at("live_objects") == (optimized ? 4 : 7), "Lazy mask allocation contract");
            result.push_back({{"extent", extent}, {"unused", empty}, {"style", style}, {"with_mask", mask}});
            // Discard the unsubmitted recording while all referenced PSOs,
            // descriptor heaps and textures are still alive.
            hr(command->Close());
            hr(allocator->Reset());
            hr(command->Reset(allocator.Get(), nullptr));
        }
        require(snapshot().at("live_objects") == 0, "Appearance textures leaked");
    }
    hr(command->Close());
    std::cout << Json({{"passed",true},{"gpu_allocations",true},{"gpu_inference_executed",false},{"cases",result}}).dump() << '\n';
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
