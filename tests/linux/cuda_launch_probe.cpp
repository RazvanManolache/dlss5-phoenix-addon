// Explicit GPU capability test; never registered with CTest.
// Launches only the embedded marker kernel, not Phoenix model inference.
#define VK_ENABLE_BETA_EXTENSIONS
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
struct State {
    VkInstance instance{};
    VkDevice device{};
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    void* mapped{};
    VkCommandPool pool{};
    VkFence fence{};
    VkCudaModuleNV module{};
    VkCudaFunctionNV function{};
    PFN_vkDestroyCudaModuleNV destroyModule{};
    PFN_vkDestroyCudaFunctionNV destroyFunction{};
    ~State() {
        if (device) {
            vkDeviceWaitIdle(device);
            if (function)
                destroyFunction(device, function, nullptr);
            if (module)
                destroyModule(device, module, nullptr);
            if (fence)
                vkDestroyFence(device, fence, nullptr);
            if (pool)
                vkDestroyCommandPool(device, pool, nullptr);
            if (mapped)
                vkUnmapMemory(device, memory);
            if (buffer)
                vkDestroyBuffer(device, buffer, nullptr);
            if (memory)
                vkFreeMemory(device, memory, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance)
            vkDestroyInstance(instance, nullptr);
    }
};
void check(VkResult result, const char* stage) {
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "%s: VkResult %d\n", stage, int(result));
        throw std::runtime_error(stage);
    }
}
int main(int argc, char** argv) {
    try {
        if (argc > 2)
            throw std::runtime_error("usage: phoenix_cuda_launch_probe [GPU name substring]");
        const char* requested = argc == 2 ? argv[1] : nullptr;
        State s;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Phoenix direct CUDA capability probe";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        check(vkCreateInstance(&ici, nullptr, &s.instance), "instance");
        uint32_t count{};
        check(vkEnumeratePhysicalDevices(s.instance, &count, nullptr), "device count");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(s.instance, &count, devices.data()), "devices");
        VkPhysicalDevice gpu{};
        for (auto candidate : devices) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(candidate, &p);
            if (p.vendorID == 0x10de && (!requested || std::strstr(p.deviceName, requested))) {
                if (gpu)
                    throw std::runtime_error("multiple NVIDIA GPUs match; provide a "
                                             "unique GPU name substring");
                gpu = candidate;
                std::printf("GPU: %s\n", p.deviceName);
            }
        }
        if (!gpu)
            throw std::runtime_error("requested NVIDIA GPU unavailable");
        uint32_t extensionCount{};
        check(vkEnumerateDeviceExtensionProperties(gpu, nullptr, &extensionCount, nullptr),
              "extension count");
        std::vector<VkExtensionProperties> extensions(extensionCount);
        check(
            vkEnumerateDeviceExtensionProperties(gpu, nullptr, &extensionCount, extensions.data()),
            "extensions");
        bool supported = false;
        for (const auto& extension : extensions)
            if (!std::strcmp(extension.extensionName, VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME))
                supported = true;
        if (!supported)
            throw std::runtime_error("VK_NV_cuda_kernel_launch unavailable");
        VkPhysicalDeviceCudaKernelLaunchFeaturesNV cuda{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV};
        VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        v12.pNext = &cuda;
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &v12;
        vkGetPhysicalDeviceFeatures2(gpu, &features);
        if (!cuda.cudaKernelLaunchFeatures || !v12.bufferDeviceAddress)
            throw std::runtime_error("CUDA launch or device address feature unavailable");
        v12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        v12.bufferDeviceAddress = VK_TRUE;
        v12.pNext = &cuda;
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());
        uint32_t family = UINT32_MAX;
        for (uint32_t n = 0; n < count; ++n)
            if (families[n].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                family = n;
                break;
            }
        if (family == UINT32_MAX)
            throw std::runtime_error("compute queue unavailable");
        float priority = 1;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        const char* extension = VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &v12;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qi;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = &extension;
        check(vkCreateDevice(gpu, &dci, nullptr, &s.device), "device");
        VkQueue queue{};
        vkGetDeviceQueue(s.device, family, 0, &queue);
        auto createModule = reinterpret_cast<PFN_vkCreateCudaModuleNV>(
            vkGetDeviceProcAddr(s.device, "vkCreateCudaModuleNV"));
        auto createFunction = reinterpret_cast<PFN_vkCreateCudaFunctionNV>(
            vkGetDeviceProcAddr(s.device, "vkCreateCudaFunctionNV"));
        auto launch = reinterpret_cast<PFN_vkCmdCudaLaunchKernelNV>(
            vkGetDeviceProcAddr(s.device, "vkCmdCudaLaunchKernelNV"));
        s.destroyModule = reinterpret_cast<PFN_vkDestroyCudaModuleNV>(
            vkGetDeviceProcAddr(s.device, "vkDestroyCudaModuleNV"));
        s.destroyFunction = reinterpret_cast<PFN_vkDestroyCudaFunctionNV>(
            vkGetDeviceProcAddr(s.device, "vkDestroyCudaFunctionNV"));
        if (!createModule || !createFunction || !launch || !s.destroyModule || !s.destroyFunction)
            throw std::runtime_error("CUDA extension entry points unavailable");
        const char ptx[] = R"(.version 7.0
.target sm_75
.address_size 64
.visible .entry phoenix_direct_probe(.param .u64 out) {
.reg .b64 %rd;
.reg .b32 %value;
ld.param.u64 %rd, [out];
mov.u32 %value, 0x50485831;
st.global.u32 [%rd], %value;
ret;
}
)";
        VkCudaModuleCreateInfoNV mi{VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV};
        mi.dataSize = sizeof(ptx);
        mi.pData = ptx;
        check(createModule(s.device, &mi, nullptr, &s.module), "CUDA module");
        VkCudaFunctionCreateInfoNV fi{VK_STRUCTURE_TYPE_CUDA_FUNCTION_CREATE_INFO_NV};
        fi.module = s.module;
        fi.pName = "phoenix_direct_probe";
        check(createFunction(s.device, &fi, nullptr, &s.function), "CUDA function");
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = 256;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(s.device, &bi, nullptr, &s.buffer), "buffer");
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(s.device, s.buffer, &req);
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(gpu, &memory);
        uint32_t type = UINT32_MAX;
        for (uint32_t n = 0; n < memory.memoryTypeCount; ++n)
            if ((req.memoryTypeBits & (1u << n)) &&
                (memory.memoryTypes[n].propertyFlags &
                 (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                type = n;
                break;
            }
        if (type == UINT32_MAX)
            throw std::runtime_error("host memory unavailable");
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.pNext = &flags;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        check(vkAllocateMemory(s.device, &ai, nullptr, &s.memory), "allocate");
        check(vkBindBufferMemory(s.device, s.buffer, s.memory, 0), "bind");
        check(vkMapMemory(s.device, s.memory, 0, VK_WHOLE_SIZE, 0, &s.mapped), "map");
        std::memset(s.mapped, 0, 256);
        VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        address.buffer = s.buffer;
        VkDeviceAddress pointer = vkGetBufferDeviceAddress(s.device, &address);
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = family;
        check(vkCreateCommandPool(s.device, &pi, nullptr, &s.pool), "pool");
        VkCommandBufferAllocateInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ci.commandPool = s.pool;
        ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ci.commandBufferCount = 1;
        VkCommandBuffer cmd{};
        check(vkAllocateCommandBuffers(s.device, &ci, &cmd), "commands");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(cmd, &begin), "begin");
        const void* params[] = {&pointer};
        VkCudaLaunchInfoNV info{VK_STRUCTURE_TYPE_CUDA_LAUNCH_INFO_NV};
        info.function = s.function;
        info.gridDimX = info.gridDimY = info.gridDimZ = 1;
        info.blockDimX = info.blockDimY = info.blockDimZ = 1;
        info.paramCount = 1;
        info.pParams = params;
        launch(cmd, &info);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                             1, &barrier, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(cmd), "end");
        VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vkCreateFence(s.device, &fence, nullptr, &s.fence), "fence");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        check(vkQueueSubmit(queue, 1, &submit, s.fence), "submit");
        check(vkWaitForFences(s.device, 1, &s.fence, VK_TRUE, 5000000000ull), "wait");
        uint32_t result{};
        std::memcpy(&result, s.mapped, sizeof(result));
        if (result != 0x50485831)
            throw std::runtime_error("CUDA kernel output mismatch");
        std::printf("PASS: direct CUDA PTX launch through Vulkan wrote the expected buffer "
                    "value. No CUDA toolkit or Windows NVAPI used.\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Direct CUDA probe failed: %s\n", e.what());
        return 1;
    }
}
