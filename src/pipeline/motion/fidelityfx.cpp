#include "../motion_plugin.h"
#include <FidelityFX/host/ffx_opticalflow.h>
#include <FidelityFX/host/backends/dx12/ffx_dx12.h>
#include <memory>
#include <vector>

namespace {
struct Context {
    std::vector<unsigned char> scratch;
    FfxOpticalflowContext flow{};
    bool initialized = false;
    ~Context() { if (initialized) ffxOpticalflowContextDestroy(&flow); }
};
}

extern "C" __declspec(dllexport) void* PhoenixMotionCreateV1(ID3D12Device* device, unsigned width, unsigned height) {
    try {
        auto context = std::make_unique<Context>();
        context->scratch.resize(ffxGetScratchMemorySizeDX12(1));
        FfxOpticalflowContextDescription desc{};
        desc.resolution = {width, height};
        if (ffxGetInterfaceDX12(&desc.backendInterface, ffxGetDeviceDX12(device),
                context->scratch.data(), context->scratch.size(), 1) != FFX_OK ||
            ffxOpticalflowContextCreate(&context->flow, &desc) != FFX_OK) return nullptr;
        context->initialized = true;
        return context.release();
    } catch (...) { return nullptr; }
}

extern "C" __declspec(dllexport) void PhoenixMotionDestroyV1(void* opaque) {
    delete static_cast<Context*>(opaque);
}

extern "C" __declspec(dllexport) bool PhoenixMotionRunV1(void* opaque, ID3D12GraphicsCommandList* list,
    ID3D12Resource* color, ID3D12Resource* flow, ID3D12Resource* scene, bool reset) {
    auto input = [](ID3D12Resource* resource, FfxResourceStates state) {
        return ffxGetResourceDX12(resource, ffxGetResourceDescriptionDX12(resource), nullptr, state);
    };
    FfxOpticalflowDispatchDescription desc{};
    desc.commandList = ffxGetCommandListDX12(list);
    desc.color = input(color, FFX_RESOURCE_STATE_COMPUTE_READ);
    desc.opticalFlowVector = input(flow, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    desc.opticalFlowSCD = input(scene, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    desc.reset = reset;
    desc.backbufferTransferFunction = FFX_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
    return ffxOpticalflowContextDispatch(&static_cast<Context*>(opaque)->flow, &desc) == FFX_OK;
}
