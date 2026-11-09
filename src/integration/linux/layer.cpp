// Linux presentation frontend. Written independently; vkBasalt was consulted
// as an architectural reference (see docs/LINUX.md). No vkBasalt code is bundled.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include "phoenix.h"
#include "phoenix_linux.h"
#include "pixels.h"
#include <dlfcn.h>
#include <cstdio>
#include <chrono>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define EXPORT extern "C" __attribute__((visibility("default")))
namespace {
using ProfileClock = std::chrono::steady_clock;
double profileMs(ProfileClock::time_point begin, ProfileClock::time_point end) {
    return std::chrono::duration<double,std::milli>(end-begin).count();
}
void log(const char* message) { std::fprintf(stderr, "[Phoenix Linux] %s\n", message); }
template<class T> void* key(T handle) { return handle ? *reinterpret_cast<void**>(handle) : nullptr; }
// Runtime-owned instances need the full driver/device list even when an app
// must see only its display GPU. Scope this by instance, never by process-wide
// environment mutation: other game threads may be creating instances too.
thread_local unsigned runtimeCallDepth{};
struct RuntimeCall {
    RuntimeCall() { ++runtimeCallDepth; }
    ~RuntimeCall() { --runtimeCallDepth; }
    RuntimeCall(const RuntimeCall&) = delete;
    RuntimeCall& operator=(const RuntimeCall&) = delete;
};
bool parseDevice(const char* value, std::array<uint8_t,VK_UUID_SIZE>& uuid) {
    if (!value || std::strlen(value)!=39 || std::strncmp(value,"vulkan:",7)) return false;
    auto digit=[](char c) -> int {
        if (c>='0' && c<='9') return c-'0';
        if (c>='a' && c<='f') return c-'a'+10;
        if (c>='A' && c<='F') return c-'A'+10;
        return -1;
    };
    for (size_t n=0;n<uuid.size();++n) {
        int high=digit(value[7+n*2]),low=digit(value[8+n*2]);
        if (high<0 || low<0) return false;
        uuid[n]=static_cast<uint8_t>((high<<4)|low);
    }
    return true;
}
struct Instance {
    VkInstance handle{};
    PFN_vkGetInstanceProcAddr gpa{};
    PFN_vkGetDeviceProcAddr gdpa{};
    bool filterDevices{}, runtimeOwned{};
    std::array<uint8_t,VK_UUID_SIZE> gameDevice{};
    std::atomic<bool> selectionLogged{};
};
#define DEVICE_FUNCTIONS(X) \
    X(DestroyDevice) X(CreateSwapchainKHR) X(DestroySwapchainKHR) X(GetSwapchainImagesKHR) \
    X(QueuePresentKHR) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) \
    X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) X(MapMemory) X(UnmapMemory) \
    X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(ResetCommandBuffer) \
    X(BeginCommandBuffer) X(EndCommandBuffer) X(CmdPipelineBarrier) X(CmdCopyImageToBuffer) \
    X(CmdCopyBufferToImage) X(CreateFence) X(DestroyFence) X(ResetFences) X(WaitForFences) \
    X(CreateSemaphore) X(DestroySemaphore) X(QueueSubmit) X(DeviceWaitIdle)
struct Device;
struct Swapchain {
    VkSwapchainKHR handle{};
    VkExtent2D extent{};
    bool bgra{}, eligible{}, ready{}, disabled{}, announced{};
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    void* mapped{};
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkFence fence{};
    uint32_t family = UINT32_MAX;
    std::vector<VkImage> images;
    std::vector<VkSemaphore> complete;
    std::vector<float> input, output;
    void* session{};
    uint64_t captured{}, processed{}, failed{}, changed{}, presented{};
    double inferenceMs{}, firstInferenceMs{};
    bool diagnostics{};
    bool profile{};
    std::array<std::array<double,8>,256> timingSamples{};
    uint64_t timingCount{};
    ProfileClock::time_point previousFrame{};
};
struct Device {
    VkDevice handle{};
    VkPhysicalDevice physical{};
    std::shared_ptr<Instance> instance;
    PFN_vkGetDeviceProcAddr gdpa{};
#define MEMBER(name) PFN_vk##name name{};
    DEVICE_FUNCTIONS(MEMBER)
#undef MEMBER
    std::unordered_map<VkSwapchainKHR, std::unique_ptr<Swapchain>> swaps;
    struct Queue { uint32_t family; bool graphics; };
    std::unordered_map<VkQueue, Queue> queues;
    std::vector<VkQueueFamilyProperties> families;
    VkPhysicalDeviceMemoryProperties memory{};
    PFN_vkSetDeviceLoaderData setLoaderData{};
    std::mutex mutex;
    void* runtime{};
    decltype(&phoenix_runtime_create) create{};
    decltype(&phoenix_runtime_destroy) destroy{};
    decltype(&phoenix_runtime_frame_v2) frame{};
    decltype(&phoenix_linux_frame_rgba8) frameRgba8{};
    decltype(&phoenix_runtime_error) error{};
    decltype(&phoenix_runtime_status) status{};
    std::string configuration;
    bool runtimeTried{};
};
std::mutex registryMutex;
std::unordered_map<void*, std::shared_ptr<Instance>> instances;
std::unordered_map<void*, std::shared_ptr<Device>> devices;
std::shared_ptr<Instance> instanceFor(VkInstance h) {
    std::lock_guard lock(registryMutex);
    auto i = instances.find(key(h)); return i == instances.end() ? nullptr : i->second;
}
std::shared_ptr<Device> deviceFor(VkDevice h) {
    std::lock_guard lock(registryMutex);
    auto i = devices.find(key(h)); return i == devices.end() ? nullptr : i->second;
}
void check(VkResult r) { if (r != VK_SUCCESS) throw std::runtime_error("Vulkan resource setup failed"); }
template<class T> T proc(const std::shared_ptr<Instance>& i, const char* name) {
    return reinterpret_cast<T>(i->gpa(i->handle, name));
}
bool enabled() { const char* v = std::getenv("PHOENIX_ENABLE"); return v && !std::strcmp(v,"1"); }
bool loadRuntime(Device& d) {
    if (d.runtimeTried) return d.runtime != nullptr;
    d.runtimeTried = true;
    const char* config = std::getenv("PHOENIX_CONFIG");
    const char* runtime = std::getenv("PHOENIX_RUNTIME_LIBRARY");
    if (!config || !runtime) { log("Missing launcher configuration; passing frames through."); return false; }
    std::ifstream file(config);
    if (!file) { log("Cannot open runtime configuration; passing frames through."); return false; }
    d.configuration.assign(std::istreambuf_iterator<char>(file), {});
    if (d.configuration.size() > 1024*1024) { log("Configuration exceeds 1 MiB."); return false; }
    d.runtime = dlopen(runtime, RTLD_NOW | RTLD_LOCAL);
    if (!d.runtime) { log(dlerror()); return false; }
    d.create = reinterpret_cast<decltype(d.create)>(dlsym(d.runtime, "phoenix_runtime_create"));
    d.destroy = reinterpret_cast<decltype(d.destroy)>(dlsym(d.runtime, "phoenix_runtime_destroy"));
    d.frame = reinterpret_cast<decltype(d.frame)>(dlsym(d.runtime, "phoenix_runtime_frame_v2"));
    d.frameRgba8 = reinterpret_cast<decltype(d.frameRgba8)>(dlsym(d.runtime, "phoenix_linux_frame_rgba8"));
    d.error = reinterpret_cast<decltype(d.error)>(dlsym(d.runtime, "phoenix_runtime_error"));
    d.status = reinterpret_cast<decltype(d.status)>(dlsym(d.runtime, "phoenix_runtime_status"));
    if (!d.create || !d.destroy || !d.frame || !d.error) {
        log("Incompatible runtime ABI; passing frames through.");
        dlclose(d.runtime); d.runtime = nullptr; return false;
    }
    return true;
}
void cleanup(Device& d, Swapchain& s) {
    if (s.timingCount) {
        const auto count=std::min<uint64_t>(s.timingCount,s.timingSamples.size());
        std::array<double,8> means{};
        for (uint64_t n=0;n<count;++n)
            for (size_t k=0;k<means.size();++k) means[k]+=s.timingSamples[n][k]/double(count);
        std::fprintf(stderr,"[Phoenix Linux] Timing summary: extent=%ux%u samples=%llu readback_ms=%.3f unpack_ms=%.3f inference_ms=%.3f pack_hash_ms=%.3f upload_ms=%.3f present_ms=%.3f hook_ms=%.3f present_interval_ms=%.3f present_fps=%.3f\n",
            s.extent.width,s.extent.height,static_cast<unsigned long long>(count),
            means[0],means[1],means[2],means[3],means[4],means[5],means[6],means[7],means[7]>0 ? 1000./means[7] : 0.);
        s.timingCount=0; s.previousFrame={};
    }
    if (s.captured) {
        const double count = static_cast<double>(s.processed + s.failed);
        std::fprintf(stderr,"[Phoenix Linux] Frame summary: extent=%ux%u captured=%llu processed=%llu failed=%llu presented=%llu changed=%llu pixel_hashes=%s inference_ms_mean=%.3f first_inference_ms=%.3f\n",
            s.extent.width,s.extent.height,
            static_cast<unsigned long long>(s.captured),static_cast<unsigned long long>(s.processed),
            static_cast<unsigned long long>(s.failed),static_cast<unsigned long long>(s.presented),
            static_cast<unsigned long long>(s.changed),s.diagnostics ? "enabled" : "disabled",count ? s.inferenceMs/count : 0.,s.firstInferenceMs);
        s.captured=s.processed=s.failed=s.changed=s.presented=0;
        s.inferenceMs=s.firstInferenceMs=0;
    }
    if (s.session) { RuntimeCall call; d.destroy(s.session); s.session = nullptr; }
    for (auto semaphore : s.complete) if (semaphore) d.DestroySemaphore(d.handle, semaphore, nullptr);
    s.complete.clear();
    if (s.fence) d.DestroyFence(d.handle, s.fence, nullptr);
    if (s.pool) d.DestroyCommandPool(d.handle, s.pool, nullptr);
    if (s.mapped) d.UnmapMemory(d.handle, s.memory);
    if (s.buffer) d.DestroyBuffer(d.handle, s.buffer, nullptr);
    if (s.memory) d.FreeMemory(d.handle, s.memory, nullptr);
    s.fence = {}; s.pool = {}; s.command = {}; s.mapped = nullptr; s.buffer = {}; s.memory = {};
    s.ready = false;
}
// Readback is CPU-read intensive. The first coherent type may be uncached
// device-local/write-combined memory even on an iGPU; prefer cached memory.
uint32_t transferMemoryType(const VkPhysicalDeviceMemoryProperties& memory, uint32_t bits) {
    constexpr auto required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t fallback = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        const auto flags = memory.memoryTypes[i].propertyFlags;
        if (!(bits & (1u << i)) || (flags & required) != required) continue;
        if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) return i;
        if (fallback == UINT32_MAX) fallback = i;
    }
    return fallback;
}
bool setup(Device& d, Swapchain& s, uint32_t family) {
    if (s.disabled) return false;
    if (s.ready) return s.family == family;
    s.disabled = true; // One attempt; no recurring allocations/errors on the present path.
    try {
        if (!loadRuntime(d)) return false;
        const char* diagnostics=std::getenv("PHOENIX_DIAGNOSTICS");
        s.diagnostics=diagnostics && !std::strcmp(diagnostics,"1");
        const char* profile=std::getenv("PHOENIX_PROFILE");
        s.profile=profile && !std::strcmp(profile,"1");
        uint32_t count = 0;
        check(d.GetSwapchainImagesKHR(d.handle, s.handle, &count, nullptr));
        if (!count) throw std::runtime_error("Swapchain contains no images");
        s.images.resize(count);
        check(d.GetSwapchainImagesKHR(d.handle, s.handle, &count, s.images.data()));
        s.images.resize(count);
        const size_t pixels = size_t(s.extent.width)*s.extent.height;
        if (!d.frameRgba8) { s.input.resize(pixels*3); s.output.resize(pixels*3); }
        // The current Vulkan provider permits one active model session. Extra
        // swapchains fail setup explicitly and retain normal presentation.
        { RuntimeCall call; s.session = d.create(d.configuration.c_str(), s.extent.width, s.extent.height); }
        if (!s.session) throw std::runtime_error(d.error());
        // Status identifies the actual runtime/backend/device instead of
        // inferring compute placement from the game's presentation GPU.
        if (d.status && s.diagnostics) {
            const size_t bytes=d.status(s.session,nullptr,0);
            if (bytes && bytes<=1024*1024) {
                std::vector<char> status(bytes);
                if (d.status(s.session,status.data(),status.size())==bytes)
                    std::fprintf(stderr,"[Phoenix Linux] Runtime status: %s\n",status.data());
            }
        }
        VkBufferCreateInfo b{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        b.size = pixels*4; b.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        b.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(d.CreateBuffer(d.handle, &b, nullptr, &s.buffer));
        VkMemoryRequirements req{}; d.GetBufferMemoryRequirements(d.handle, s.buffer, &req);
        const uint32_t type = transferMemoryType(d.memory, req.memoryTypeBits);
        if (type == UINT32_MAX) throw std::runtime_error("Host-coherent transfer memory unavailable");
        if (s.profile || s.diagnostics)
            std::fprintf(stderr,"[Phoenix Linux] Transfer memory: type=%u host_cached=%s\n",type,
                (d.memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "yes" : "no");
        VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; a.allocationSize = req.size; a.memoryTypeIndex = type;
        check(d.AllocateMemory(d.handle, &a, nullptr, &s.memory));
        check(d.BindBufferMemory(d.handle, s.buffer, s.memory, 0));
        check(d.MapMemory(d.handle, s.memory, 0, VK_WHOLE_SIZE, 0, &s.mapped));
        VkCommandPoolCreateInfo p{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        p.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; p.queueFamilyIndex = family;
        check(d.CreateCommandPool(d.handle, &p, nullptr, &s.pool));
        VkCommandBufferAllocateInfo c{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        c.commandPool = s.pool; c.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; c.commandBufferCount = 1;
        check(d.AllocateCommandBuffers(d.handle, &c, &s.command));
        // The loader must initialize the dispatch pointer of layer-created
        // dispatchable objects before other layers see those objects.
        if (!d.setLoaderData) throw std::runtime_error("Loader command-buffer callback unavailable");
        check(d.setLoaderData(d.handle, s.command));
        VkFenceCreateInfo f{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; check(d.CreateFence(d.handle, &f, nullptr, &s.fence));
        s.complete.resize(count);
        VkSemaphoreCreateInfo sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (auto& h : s.complete) check(d.CreateSemaphore(d.handle, &sem, nullptr, &h));
        s.family = family; s.ready = true; s.disabled = false;
        log("Processing SDR frames with synchronous host transport; HUD is included, temporal reuse is disabled.");
        return true;
    } catch (const std::exception& e) { log(e.what()); cleanup(d,s); return false; }
}
VkResult record(Device& d, Swapchain& s, uint32_t image, bool upload) {
    VkResult r = d.ResetCommandBuffer(s.command, 0); if (r != VK_SUCCESS) return r;
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = d.BeginCommandBuffer(s.command, &begin); if (r != VK_SUCCESS) return r;
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = upload ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.newLayout = upload ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = s.images[image]; barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    d.CmdPipelineBarrier(s.command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&barrier);
    VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
    region.imageExtent = {s.extent.width,s.extent.height,1};
    if (upload) {
        VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT; host.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host.buffer = s.buffer; host.size = VK_WHOLE_SIZE;
        d.CmdPipelineBarrier(s.command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&host,0,nullptr);
        d.CmdCopyBufferToImage(s.command,s.buffer,s.images[image],barrier.newLayout,1,&region);
    } else {
        d.CmdCopyImageToBuffer(s.command,s.images[image],barrier.newLayout,s.buffer,1,&region);
        VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host.buffer = s.buffer; host.size = VK_WHOLE_SIZE;
        d.CmdPipelineBarrier(s.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&host,0,nullptr);
    }
    barrier.srcAccessMask = upload ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = 0; barrier.oldLayout = barrier.newLayout; barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    d.CmdPipelineBarrier(s.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,nullptr,0,nullptr,1,&barrier);
    return d.EndCommandBuffer(s.command);
}
VkResult submit(Device& d, Swapchain& s, VkQueue queue, uint32_t waitCount, const VkSemaphore* waits, VkSemaphore signal) {
    // Allocation failure must not unwind through the Vulkan C ABI. In the
    // upload phase the application waits may already have been consumed.
    std::vector<VkPipelineStageFlags> stages;
    try { stages.assign(waitCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT); }
    catch (...) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO}; info.waitSemaphoreCount = waitCount;
    info.pWaitSemaphores = waits; info.pWaitDstStageMask = stages.data();
    info.commandBufferCount = 1; info.pCommandBuffers = &s.command;
    info.signalSemaphoreCount = signal ? 1 : 0; info.pSignalSemaphores = &signal;
    VkResult r = d.ResetFences(d.handle,1,&s.fence); if (r != VK_SUCCESS) return r;
    r = d.QueueSubmit(queue,1,&info,s.fence); if (r != VK_SUCCESS) return r;
    return d.WaitForFences(d.handle,1,&s.fence,VK_TRUE,UINT64_MAX);
}
uint64_t pixelHash(const void* data, size_t bytes) {
    uint64_t hash=14695981039346656037ull;
    const auto* values=static_cast<const uint8_t*>(data);
    for (size_t i=0;i<bytes;++i) { hash^=values[i]; hash*=1099511628211ull; }
    return hash;
}
VkResult selectedDevice(const std::shared_ptr<Instance>& i, VkPhysicalDevice& selected) {
    auto enumerate=proc<PFN_vkEnumeratePhysicalDevices>(i,"vkEnumeratePhysicalDevices");
    auto properties=proc<PFN_vkGetPhysicalDeviceProperties2>(i,"vkGetPhysicalDeviceProperties2");
    if (!properties) properties=reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(i->gpa(i->handle,"vkGetPhysicalDeviceProperties2KHR"));
    if (!enumerate || !properties) { log("Game GPU selection requires physical-device UUID queries (Vulkan 1.1+)."); return VK_ERROR_INITIALIZATION_FAILED; }
    try {
        for (unsigned attempt=0;attempt<3;++attempt) {
            uint32_t count=0;
            VkResult result=enumerate(i->handle,&count,nullptr); if (result!=VK_SUCCESS) return result;
            std::vector<VkPhysicalDevice> available(count);
            if (count) {
                result=enumerate(i->handle,&count,available.data());
                if (result==VK_INCOMPLETE) continue;
                if (result!=VK_SUCCESS) return result;
            }
            available.resize(count);
            for (auto physical : available) {
                VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
                VkPhysicalDeviceProperties2 info{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; info.pNext=&id;
                properties(physical,&info);
                if (!std::memcmp(id.deviceUUID,i->gameDevice.data(),VK_UUID_SIZE)) {
                    selected=physical;
                    if (!i->selectionLogged.exchange(true))
                        std::fprintf(stderr,"[Phoenix Linux] Game GPU visibility: %s only; runtime-owned instances retain all available GPUs.\n",info.properties.deviceName);
                    return VK_SUCCESS;
                }
            }
            log("Requested game GPU UUID is unavailable; refusing to substitute another GPU.");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        return VK_INCOMPLETE;
    } catch (...) { return VK_ERROR_OUT_OF_HOST_MEMORY; }
}
}

EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance, const char*);
EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice, const char*);

EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo* info, const VkAllocationCallbacks* allocator, VkInstance* out) {
    const bool runtimeOwned=runtimeCallDepth!=0;
    const char* requested=runtimeOwned ? nullptr : std::getenv("PHOENIX_GAME_DEVICE");
    std::array<uint8_t,VK_UUID_SIZE> requestedUUID{};
    if (requested && !parseDevice(requested,requestedUUID)) {
        log("PHOENIX_GAME_DEVICE must be vulkan:<32 hexadecimal UUID digits>.");
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (requested && (!info->pApplicationInfo || info->pApplicationInfo->apiVersion<VK_API_VERSION_1_1)) {
        log("Game GPU selection requires an application Vulkan instance version of at least 1.1.");
        return VK_ERROR_INCOMPATIBLE_DRIVER;
    }
    auto* chain = const_cast<VkLayerInstanceCreateInfo*>(reinterpret_cast<const VkLayerInstanceCreateInfo*>(info->pNext));
    while (chain && (chain->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO || chain->function != VK_LAYER_LINK_INFO))
        chain = reinterpret_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(chain->pNext));
    if (!chain || !chain->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;
    auto next = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    auto create = reinterpret_cast<PFN_vkCreateInstance>(next(nullptr,"vkCreateInstance"));
    VkResult r = create(info,allocator,out); if (r != VK_SUCCESS) return r;
    try {
        auto i = std::make_shared<Instance>(); i->handle = *out; i->gpa = next;
        i->runtimeOwned=runtimeOwned; i->filterDevices=requested!=nullptr; i->gameDevice=requestedUUID;
        std::lock_guard lock(registryMutex); instances[key(*out)] = i;
    } catch (...) {
        reinterpret_cast<PFN_vkDestroyInstance>(next(*out,"vkDestroyInstance"))(*out,allocator);
        *out = VK_NULL_HANDLE; return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return r;
}
EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(VkInstance instance, uint32_t* count, VkPhysicalDevice* physical) {
    auto i=instanceFor(instance); if (!i) return VK_ERROR_INITIALIZATION_FAILED;
    if (!i->filterDevices) return proc<PFN_vkEnumeratePhysicalDevices>(i,"vkEnumeratePhysicalDevices")(instance,count,physical);
    VkPhysicalDevice selected{};
    const VkResult result=selectedDevice(i,selected);
    if (result!=VK_SUCCESS) { *count=0; return result; }
    if (!physical) { *count=1; return VK_SUCCESS; }
    if (!*count) return VK_INCOMPLETE;
    physical[0]=selected; *count=1; return VK_SUCCESS;
}
namespace {
VkResult enumerateGroups(VkInstance instance, uint32_t* count, VkPhysicalDeviceGroupProperties* groups, const char* name) {
    auto i=instanceFor(instance); if (!i) return VK_ERROR_INITIALIZATION_FAILED;
    auto next=proc<PFN_vkEnumeratePhysicalDeviceGroups>(i,name);
    if (!next) return VK_ERROR_EXTENSION_NOT_PRESENT;
    if (!i->filterDevices) return next(instance,count,groups);
    VkPhysicalDevice selected{};
    const VkResult result=selectedDevice(i,selected);
    if (result!=VK_SUCCESS) { *count=0; return result; }
    if (!groups) { *count=1; return VK_SUCCESS; }
    if (!*count) return VK_INCOMPLETE;
    // A singleton subset is always a valid logical-device group. Keep the
    // caller's sType/pNext intact and do not leak hidden linked-group members.
    groups[0].physicalDeviceCount=1;
    std::fill(std::begin(groups[0].physicalDevices),std::end(groups[0].physicalDevices),VK_NULL_HANDLE);
    groups[0].physicalDevices[0]=selected;
    groups[0].subsetAllocation=VK_FALSE;
    *count=1; return VK_SUCCESS;
}
}
EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDeviceGroups(VkInstance instance, uint32_t* count, VkPhysicalDeviceGroupProperties* groups) {
    return enumerateGroups(instance,count,groups,"vkEnumeratePhysicalDeviceGroups");
}
EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDeviceGroupsKHR(VkInstance instance, uint32_t* count, VkPhysicalDeviceGroupProperties* groups) {
    return enumerateGroups(instance,count,groups,"vkEnumeratePhysicalDeviceGroupsKHR");
}
EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance instance, const VkAllocationCallbacks* allocator) {
    auto i = instanceFor(instance); if (!i) return;
    { std::lock_guard lock(registryMutex); instances.erase(key(instance)); }
    proc<PFN_vkDestroyInstance>(i,"vkDestroyInstance")(instance,allocator);
}
EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* info, const VkAllocationCallbacks* allocator, VkDevice* out) {
    auto i = instanceFor(reinterpret_cast<VkInstance>(physical)); if (!i) return VK_ERROR_INITIALIZATION_FAILED;
    auto* chain = const_cast<VkLayerDeviceCreateInfo*>(reinterpret_cast<const VkLayerDeviceCreateInfo*>(info->pNext));
    PFN_vkSetDeviceLoaderData loaderData = nullptr;
    for (auto* p = chain; p; p = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(p->pNext)))
        if (p->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO && p->function == VK_LOADER_DATA_CALLBACK)
            loaderData = p->u.pfnSetDeviceLoaderData;
    while (chain && (chain->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO || chain->function != VK_LAYER_LINK_INFO))
        chain = reinterpret_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(chain->pNext));
    if (!chain || !chain->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;
    auto gpa = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    auto gdpa = chain->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    auto create = reinterpret_cast<PFN_vkCreateDevice>(gpa(i->handle,"vkCreateDevice"));
    VkResult r = create(physical,info,allocator,out); if (r != VK_SUCCESS) return r;
    try {
    auto d = std::make_shared<Device>(); d->handle = *out; d->physical = physical; d->instance = i; d->gdpa = gdpa; d->setLoaderData = loaderData;
#define LOAD(name) d->name = reinterpret_cast<PFN_vk##name>(gdpa(*out,"vk" #name));
    DEVICE_FUNCTIONS(LOAD)
#undef LOAD
    auto families = proc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(i,"vkGetPhysicalDeviceQueueFamilyProperties");
    uint32_t count = 0; families(physical,&count,nullptr); d->families.resize(count); families(physical,&count,d->families.data());
    proc<PFN_vkGetPhysicalDeviceMemoryProperties>(i,"vkGetPhysicalDeviceMemoryProperties")(physical,&d->memory);
    auto getQueue = reinterpret_cast<PFN_vkGetDeviceQueue>(gdpa(*out,"vkGetDeviceQueue"));
    auto getQueue2 = reinterpret_cast<PFN_vkGetDeviceQueue2>(gdpa(*out,"vkGetDeviceQueue2"));
    for (uint32_t n=0;n<info->queueCreateInfoCount;++n) {
        const auto& q = info->pQueueCreateInfos[n];
        if (q.queueFamilyIndex >= d->families.size()) continue;
        for (uint32_t index=0;index<q.queueCount;++index) {
            VkQueue queue{};
            if (q.flags && getQueue2) {
                VkDeviceQueueInfo2 qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2}; qi.flags=q.flags; qi.queueFamilyIndex=q.queueFamilyIndex; qi.queueIndex=index;
                getQueue2(*out,&qi,&queue);
            } else if (!q.flags) getQueue(*out,q.queueFamilyIndex,index,&queue);
            if (queue) d->queues[queue] = {q.queueFamilyIndex, !q.flags && bool(d->families[q.queueFamilyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT)};
        }
    }
    { std::lock_guard lock(registryMutex); devices[key(*out)] = d; }
    return r;
    } catch (...) {
        reinterpret_cast<PFN_vkDestroyDevice>(gdpa(*out,"vkDestroyDevice"))(*out,allocator);
        *out = VK_NULL_HANDLE; return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
}
EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice device, const VkAllocationCallbacks* allocator) {
    auto d = deviceFor(device); if (!d) return;
    { std::lock_guard lock(d->mutex);
      if (!d->swaps.empty()) d->DeviceWaitIdle(device);
      for (auto& [h,s] : d->swaps) { (void)h; cleanup(*d,*s); }
      d->swaps.clear();
      if (d->runtime) dlclose(d->runtime);
    }
    { std::lock_guard lock(registryMutex); devices.erase(key(device)); }
    d->DestroyDevice(device,allocator);
}
EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* info, const VkAllocationCallbacks* allocator, VkSwapchainKHR* out) {
    auto d = deviceFor(device); if (!d || !d->CreateSwapchainKHR) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(d->mutex);
    auto modified = *info;
    bool eligible = enabled() && !info->pNext && info->imageArrayLayers == 1 && info->imageExtent.width && info->imageExtent.height &&
        info->imageExtent.width <= 8192 && info->imageExtent.height <= 8192 &&
        info->imageColorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR && !(info->flags & VK_SWAPCHAIN_CREATE_PROTECTED_BIT_KHR) &&
        (info->imageFormat == VK_FORMAT_R8G8B8A8_UNORM || info->imageFormat == VK_FORMAT_R8G8B8A8_SRGB ||
         info->imageFormat == VK_FORMAT_B8G8R8A8_UNORM || info->imageFormat == VK_FORMAT_B8G8R8A8_SRGB);
    VkSurfaceCapabilitiesKHR caps{};
    auto capabilities = proc<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(d->instance,"vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    const auto usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    eligible = eligible && capabilities && capabilities(d->physical,info->surface,&caps) == VK_SUCCESS && (caps.supportedUsageFlags & usage) == usage;
    if (eligible) modified.imageUsage |= usage;
    VkResult r = d->CreateSwapchainKHR(device,&modified,allocator,out);
    // A layer must never make an otherwise supported game fail to create its
    // swapchain just because extra usage flags were rejected by the driver.
    if (r != VK_SUCCESS && eligible) { eligible=false; r=d->CreateSwapchainKHR(device,info,allocator,out); }
    if (r != VK_SUCCESS) return r;
    try {
    auto s = std::make_unique<Swapchain>(); s->handle=*out; s->extent=info->imageExtent; s->eligible=eligible;
    s->bgra=info->imageFormat==VK_FORMAT_B8G8R8A8_UNORM || info->imageFormat==VK_FORMAT_B8G8R8A8_SRGB;
    // Replacement is the normal resize path. Retire the old session before
    // initializing the new one, including the single-session Vulkan provider.
    if (info->oldSwapchain) {
        auto old=d->swaps.find(info->oldSwapchain);
        if (old!=d->swaps.end()) { d->DeviceWaitIdle(device); cleanup(*d,*old->second); old->second->disabled=true; }
    }
    d->swaps[*out]=std::move(s);
    if (!eligible && enabled()) log("Unsupported swapchain format, flags, extension chain or transfer usage; passing through.");
    return r;
    } catch (...) {
        d->DestroySwapchainKHR(device,*out,allocator); *out=VK_NULL_HANDLE;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
}
EXPORT VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(VkDevice device, VkSwapchainKHR swap, const VkAllocationCallbacks* allocator) {
    auto d=deviceFor(device); if (!d) return;
    std::lock_guard lock(d->mutex);
    auto s=d->swaps.find(swap);
    if (s!=d->swaps.end()) { if (s->second->ready) d->DeviceWaitIdle(device); cleanup(*d,*s->second); d->swaps.erase(s); }
    d->DestroySwapchainKHR(device,swap,allocator);
}
EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* info) {
    auto d=deviceFor(reinterpret_cast<VkDevice>(queue)); if (!d || !d->QueuePresentKHR) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(d->mutex);
    auto q=d->queues.find(queue);
    // Extension chains can include device groups, present IDs/timing and other
    // semantics. Preserve them untouched; unsupported paths are pass-through.
    if (!enabled() || info->swapchainCount!=1 || info->pNext || q==d->queues.end() || !q->second.graphics)
        return d->QueuePresentKHR(queue,info);
    auto found=d->swaps.find(info->pSwapchains[0]);
    if (found==d->swaps.end() || !found->second->eligible) return d->QueuePresentKHR(queue,info);
    auto& s=*found->second;
    if (!setup(*d,s,q->second.family)) return d->QueuePresentKHR(queue,info);
    uint32_t image=info->pImageIndices[0];
    if (image>=s.images.size()) return d->QueuePresentKHR(queue,info);
    const auto captureStarted=s.profile ? ProfileClock::now() : ProfileClock::time_point{};
    VkResult r=record(*d,s,image,false);
    if (r!=VK_SUCCESS) return d->QueuePresentKHR(queue,info); // No waits consumed yet.
    r=submit(*d,s,queue,info->waitSemaphoreCount,info->pWaitSemaphores,{});
    if (r!=VK_SUCCESS) return r; // Never wait a second time on consumed binary semaphores.
    ++s.captured;
    const auto captureFinished=s.profile ? ProfileClock::now() : ProfileClock::time_point{};
    const size_t pixels=size_t(s.extent.width)*s.extent.height;
    const auto before=s.diagnostics ? pixelHash(s.mapped,pixels*4) : 0;
    if (!d->frameRgba8)
        phoenix_linux::unpack(static_cast<const uint8_t*>(s.mapped),s.input.data(),pixels,s.bgra);
    const auto unpackFinished=s.profile ? ProfileClock::now() : ProfileClock::time_point{};
    phoenix_frame_v2 frame{}; frame.struct_bytes=sizeof(frame); frame.version=PHOENIX_FRAME_VERSION_2;
    frame.color={s.input.data(),s.input.size()*sizeof(float),0,s.extent.width,s.extent.height,3,0};
    frame.output={s.output.data(),s.output.size()*sizeof(float),0,s.extent.width,s.extent.height,3,0};
    // A final-color hook supplies no engine motion or validity. Reset every
    // frame instead of pretending that identity motion is a temporal guide.
    frame.metadata_json="{\"reset\":true}";
    const auto started=std::chrono::steady_clock::now();
    int inferenceResult;
    {
        RuntimeCall call;
        if (d->frameRgba8) {
            const phoenix_rgba8_frame packed{sizeof(phoenix_rgba8_frame),1,s.extent.width,
                s.extent.height,s.mapped,uint64_t(pixels)*4,0,s.bgra?1u:0u,1};
            inferenceResult=d->frameRgba8(s.session,&packed);
        } else inferenceResult=d->frame(s.session,&frame);
    }
    const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    s.inferenceMs+=elapsed;
    if (s.captured==1) s.firstInferenceMs=elapsed;
    if (inferenceResult==0) {
        ++s.processed;
        if (!d->frameRgba8)
            phoenix_linux::pack(s.output.data(),static_cast<uint8_t*>(s.mapped),pixels,s.bgra);
        if (s.diagnostics && before!=pixelHash(s.mapped,pixels*4)) ++s.changed;
    } else {
        ++s.failed;
        if (!s.announced) { log(d->error()); log("Inference failed; retaining original pixels."); s.announced=true; }
    }
    const auto packFinished=s.profile ? ProfileClock::now() : ProfileClock::time_point{};
    r=record(*d,s,image,true); if (r!=VK_SUCCESS) return r;
    r=submit(*d,s,queue,0,nullptr,s.complete[image]); if (r!=VK_SUCCESS) return r;
    const auto uploadFinished=s.profile ? ProfileClock::now() : ProfileClock::time_point{};
    VkPresentInfoKHR modified=*info; modified.waitSemaphoreCount=1; modified.pWaitSemaphores=&s.complete[image];
    const VkResult presented=d->QueuePresentKHR(queue,&modified);
    if (presented==VK_SUCCESS || presented==VK_SUBOPTIMAL_KHR) ++s.presented;
    if (s.profile) {
        const auto finished=ProfileClock::now();
        if (s.captured>3 && s.previousFrame!=ProfileClock::time_point{} && inferenceResult==0 &&
            (presented==VK_SUCCESS || presented==VK_SUBOPTIMAL_KHR)) {
            s.timingSamples[s.timingCount++%s.timingSamples.size()]={
                profileMs(captureStarted,captureFinished),profileMs(captureFinished,unpackFinished),elapsed,
                profileMs(started,packFinished)-elapsed,profileMs(packFinished,uploadFinished),
                profileMs(uploadFinished,finished),profileMs(captureStarted,finished),
                profileMs(s.previousFrame,captureStarted)};
        }
        s.previousFrame=captureStarted;
    }
    return presented;
}

namespace {
PFN_vkVoidFunction hook(const char* name) {
    if (!name) return nullptr;
#define HOOK(fn) if (!std::strcmp(name,#fn)) return reinterpret_cast<PFN_vkVoidFunction>(fn)
    HOOK(vkGetInstanceProcAddr); HOOK(vkGetDeviceProcAddr); HOOK(vkCreateInstance); HOOK(vkDestroyInstance);
    HOOK(vkCreateDevice); HOOK(vkDestroyDevice); HOOK(vkCreateSwapchainKHR); HOOK(vkDestroySwapchainKHR); HOOK(vkQueuePresentKHR);
    HOOK(vkEnumeratePhysicalDevices); HOOK(vkEnumeratePhysicalDeviceGroups); HOOK(vkEnumeratePhysicalDeviceGroupsKHR);
#undef HOOK
    return nullptr;
}
}
EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* name) {
    if (!name) return nullptr;
    if (!instance) {
        if (!std::strcmp(name,"vkCreateInstance") || !std::strcmp(name,"vkGetInstanceProcAddr")) return hook(name);
        return nullptr;
    }
    auto i=instanceFor(instance);
    if ((!std::strcmp(name,"vkEnumeratePhysicalDeviceGroups") || !std::strcmp(name,"vkEnumeratePhysicalDeviceGroupsKHR")) &&
        (!i || !i->gpa(instance,name))) return nullptr;
    if (auto p=hook(name)) return p;
    return i ? i->gpa(instance,name) : nullptr;
}
EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* name) {
    auto d=deviceFor(device); if (!d || !name) return nullptr;
    // Do not advertise unenabled device extensions or instance-only commands.
    auto next=d->gdpa(device,name); if (!next) return nullptr;
    if (auto p=hook(name)) return p;
    return next;
}
EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* version) {
    if (!version || version->sType!=LAYER_NEGOTIATE_INTERFACE_STRUCT || version->loaderLayerInterfaceVersion<2)
        return VK_ERROR_INITIALIZATION_FAILED;
    version->loaderLayerInterfaceVersion=2;
    version->pfnGetInstanceProcAddr=vkGetInstanceProcAddr;
    version->pfnGetDeviceProcAddr=vkGetDeviceProcAddr;
    version->pfnGetPhysicalDeviceProcAddr=nullptr;
    return VK_SUCCESS;
}
