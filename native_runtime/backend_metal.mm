// Initial Metal correctness backend. GPU tensors stay resident between graph
// commands; only the existing frame boundary uploads/downloads host planes.
#include "../third_party/nlohmann/json.hpp"
#include "backend.h"
#include "backend_options.h"
#include "metal_shader_source.h"
#include "platform.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <limits>
#include <unordered_map>
using namespace phoenix_cpp;
namespace {
using Json = nlohmann::json;
std::string identity(id<MTLDevice> device) { return "metal:" + std::to_string(device.registryID); }
std::string description(NSError *error) {
    return error ? std::string(error.localizedDescription.UTF8String) : "Unknown Metal failure";
}
struct DispatchScalars {
    uint32_t fields[12];
    float scale;
    uint32_t padding = 0;
    uint64_t count;
    uint64_t offsets[5];
};
static_assert(offsetof(DispatchScalars, count) == 56);
static_assert(offsetof(DispatchScalars, offsets) == 64);
class MetalBackend final : public Backend {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipeline_;
    id<MTLBuffer> unused_;
    std::unordered_map<uint64_t, id<MTLBuffer>> buffers_;
    uint64_t next_token_ = 1;
    static constexpr uint64_t offset_mask = (uint64_t(1) << 40) - 1;

    id<MTLBuffer> buffer(void *address, uint64_t &offset) const {
        if (!address) {
            offset = 0;
            return unused_;
        }
        const auto encoded = uint64_t(reinterpret_cast<uintptr_t>(address));
        const auto found = buffers_.find(encoded >> 40);
        demand(found != buffers_.end(), "Unknown Metal tensor allocation");
        offset = encoded & offset_mask;
        demand(offset < found->second.length, "Metal tensor offset bounds");
        return found->second;
    }
    id<MTLBuffer> transfer_buffer(Allocation allocation, uint64_t bytes, uint64_t &offset) const {
        auto result = buffer(allocation.address, offset);
        demand(allocation.address && bytes <= allocation.bytes && bytes <= result.length - offset,
               "Metal transfer bounds");
        return result;
    }

  public:
    explicit MetalBackend(const char *selected) {
        demand(selected != nullptr, "Metal device identity required");
        for (id<MTLDevice> device in MTLCopyAllDevices()) {
            if (identity(device) == selected ||
                (std::string(selected) == "metal:default" &&
                 device.registryID == MTLCreateSystemDefaultDevice().registryID)) {
                device_ = device;
                break;
            }
        }
        demand(device_ != nil, "Selected Metal device is unavailable");
        demand(device_.hasUnifiedMemory && [device_ supportsFamily:MTLGPUFamilyApple7],
               "Initial Metal backend requires Apple silicon unified memory");
        queue_ = [device_ newCommandQueue];
        demand(queue_ != nil, "Metal command queue creation failed");
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.fastMathEnabled = NO; // Keep explicit half/FP8 rounding and reduction order.
        NSError *error = nil;
        NSString *source = [[NSString alloc] initWithBytes:phoenix_metal_source
                                                    length:sizeof(phoenix_metal_source) - 1
                                                  encoding:NSUTF8StringEncoding];
        id<MTLLibrary> library = [device_ newLibraryWithSource:source options:options error:&error];
        demand(library != nil, description(error).c_str());
        id<MTLFunction> function = [library newFunctionWithName:@"phoenix_reference"];
        demand(function != nil, "Metal reference entry unavailable");
        pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&error];
        demand(pipeline_ != nil, description(error).c_str());
        unused_ = [device_ newBufferWithLength:256 options:MTLResourceStorageModeShared];
        demand(unused_ != nil, "Metal optional-operand buffer allocation failed");
        std::memset(unused_.contents, 0, 256);
    }
    const char *name() const override { return "metal-reference-cpp"; }
    Allocation allocate(uint64_t bytes) override {
        demand(bytes && bytes <= offset_mask && bytes <= device_.maxBufferLength &&
                   next_token_ < (1u << 24),
               "Metal allocation extent");
        auto resource = [device_ newBufferWithLength:NSUInteger(bytes)
                                             options:MTLResourceStorageModeShared];
        if (!resource)
            throw std::bad_alloc();
        std::memset(resource.contents, 0, size_t(bytes));
        const auto token = next_token_++;
        buffers_.emplace(token, resource);
        return {reinterpret_cast<void *>(token << 40), bytes, token};
    }
    void release(Allocation allocation) override {
        demand(buffers_.erase(allocation.token) == 1, "Unknown Metal allocation release");
    }
    void upload(Allocation allocation, const void *source, uint64_t bytes) override {
        uint64_t offset;
        auto resource = transfer_buffer(allocation, bytes, offset);
        std::memcpy(static_cast<uint8_t *>(resource.contents) + offset, source, size_t(bytes));
    }
    void download(Allocation allocation, void *destination, uint64_t bytes) override {
        uint64_t offset;
        auto resource = transfer_buffer(allocation, bytes, offset);
        std::memcpy(destination, static_cast<const uint8_t *>(resource.contents) + offset,
                    size_t(bytes));
    }
    void execute(const std::vector<Command> &commands, std::vector<Event> &events) override {
        if (commands.empty())
            return;
        @autoreleasepool {
            // One synchronous submission, separate encoders for tracked buffer
            // dependencies. No implicit CPU fallback and no per-operator readback.
            id<MTLCommandBuffer> submission = [queue_ commandBuffer];
            demand(submission != nil, "Metal command buffer creation failed");
            for (const auto &command : commands) {
                const auto &operation = command.operation;
                demand(operation.family <= phx::FrameFeatures &&
                           operation.family != phx::WindowAttention,
                       "Metal requires unfused attention commands");
                demand(command.count <= UINT32_MAX, "Metal dispatch extent");
                if (!command.count)
                    continue;
                DispatchScalars parameters{};
                std::memcpy(parameters.fields, &operation.family, sizeof(parameters.fields));
                parameters.scale = operation.scale;
                parameters.count = command.count;
                id<MTLComputeCommandEncoder> encoder = [submission computeCommandEncoder];
                demand(encoder != nil, "Metal compute encoder creation failed");
                [encoder setComputePipelineState:pipeline_];
                void *operands[] = {operation.left_input, operation.right_input, operation.output,
                                    operation.auxiliary, operation.secondary_output};
                for (unsigned index = 0; index < 5; ++index) {
                    auto resource = buffer(operands[index], parameters.offsets[index]);
                    [encoder setBuffer:resource offset:0 atIndex:index];
                }
                [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
                const auto threads =
                    std::min(NSUInteger(128), pipeline_.maxTotalThreadsPerThreadgroup);
                [encoder dispatchThreads:MTLSizeMake(NSUInteger(command.count), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                [encoder endEncoding];
            }
            [submission commit];
            [submission waitUntilCompleted];
            demand(submission.status == MTLCommandBufferStatusCompleted,
                   description(submission.error).c_str());
            // GPU timestamps cover the entire submission; do not invent layer
            // breakdowns or mix shader compilation/CPU encode time into GPU time.
            events.push_back({UINT32_MAX, 0,
                              (submission.GPUEndTime - submission.GPUStartTime) * 1000.0, true,
                              UINT32_MAX});
        }
    }
    void wait() override {} // execute() completes before returning.
};
thread_local std::string device_list;
} // namespace
extern "C" PHOENIX_EXPORT size_t phoenix_metal_devices(char *output, size_t capacity) {
    @autoreleasepool {
        Json devices = Json::array();
        for (id<MTLDevice> device in MTLCopyAllDevices())
            devices.push_back({{"id", identity(device)},
                               {"name", device.name.UTF8String},
                               {"unified_memory", bool(device.hasUnifiedMemory)},
                               {"supported", bool(device.hasUnifiedMemory &&
                                                  [device supportsFamily:MTLGPUFamilyApple7])},
                               {"inference_validated", false}});
        device_list = devices.dump();
        if (output && capacity > device_list.size())
            std::memcpy(output, device_list.c_str(), device_list.size() + 1);
        return device_list.size() + 1;
    }
}
extern "C" PHOENIX_EXPORT Backend *phoenix_backend_create(const char *selected, const wchar_t *,
                                                          char *error, size_t capacity) {
    @autoreleasepool {
        try {
            return new MetalBackend(selected);
        } catch (const std::exception &exception) {
            copy_error(error, capacity, exception.what());
            return nullptr;
        }
    }
}
extern "C" PHOENIX_EXPORT void phoenix_backend_destroy(Backend *backend) { delete backend; }
extern "C" PHOENIX_EXPORT uint32_t phoenix_backend_features(Backend *) {
    return DeviceFeaturePacking | FusedMatrixResidual | DeviceControlMaps;
}
