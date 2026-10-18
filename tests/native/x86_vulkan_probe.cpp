// Disposable GPU feasibility probe, not a model runner or game integration.
#define NOMINMAX
#define VK_NO_PROTOTYPES
#define VK_ENABLE_BETA_EXTENSIONS
#include <windows.h>
#include <vulkan/vulkan.h>
#include "../../third_party/nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>

static void event(const char* stage, nlohmann::json fields = nlohmann::json::object()) {
    fields["stage"] = stage;
    fields["pointer_bits"] = sizeof(void*) * 8;
    std::cout << fields.dump() << std::endl;
}
static void check(VkResult result) {
    if (result != VK_SUCCESS) throw std::runtime_error("Vulkan status " + std::to_string(result));
}

static const char ptx[] = R"ptx(.version 7.0
.target sm_52
.address_size 64
.visible .entry phoenix_probe(.param .u64 output_buffer) {
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd0, [output_buffer];
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    mad.lo.u32 %r4, %r3, 3, 7;
    mul.wide.u32 %rd1, %r3, 4;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r4;
    ret;
}
)ptx";

int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
    try {
        if (argc != 2) return 1;
        auto library = LoadLibraryExW(L"vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!library) throw std::runtime_error("System Vulkan loader unavailable");
        auto get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(library, "vkGetInstanceProcAddr"));
        if (!get) throw std::runtime_error("Missing Vulkan loader export");
        auto vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(get(nullptr, "vkCreateInstance"));
        if (!vkCreateInstance) throw std::runtime_error("Missing instance creation");
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Phoenix DLSS5 x86 feasibility probe";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        VkInstance instance{};
        event("instance_create_call");
        check(vkCreateInstance(&ici, nullptr, &instance));
#define INST(name) auto name = reinterpret_cast<PFN_##name>(get(instance, #name)); \
        if (!name) throw std::runtime_error("Missing " #name)
        INST(vkEnumeratePhysicalDevices);
        INST(vkGetPhysicalDeviceProperties);
        INST(vkGetPhysicalDeviceQueueFamilyProperties);
        INST(vkEnumerateDeviceExtensionProperties);
        INST(vkGetPhysicalDeviceFeatures2);
        INST(vkGetPhysicalDeviceMemoryProperties);
        INST(vkCreateDevice);
        INST(vkGetDeviceProcAddr);
        INST(vkDestroyInstance);
        uint32_t count{};
        check(vkEnumeratePhysicalDevices(instance, &count, nullptr));
        std::vector<VkPhysicalDevice> physical(count);
        check(vkEnumeratePhysicalDevices(instance, &count, physical.data()));
        VkPhysicalDevice gpu{};
        for (auto candidate : physical) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(candidate, &props);
            event("adapter", {{"name", props.deviceName}, {"vendor_id", props.vendorID}});
            if (props.vendorID == 0x10de) gpu = candidate;
        }
        if (!gpu) throw std::runtime_error("No NVIDIA Vulkan device");
        check(vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, nullptr));
        std::vector<VkExtensionProperties> extensions(count);
        check(vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, extensions.data()));
        bool has_cuda = false;
        for (const auto& ext : extensions)
            if (!strcmp(ext.extensionName, VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME)) has_cuda = true;
        VkPhysicalDeviceCudaKernelLaunchFeaturesNV cuda{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUDA_KERNEL_LAUNCH_FEATURES_NV};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        if (has_cuda) f12.pNext = &cuda;
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.pNext = &f12;
        vkGetPhysicalDeviceFeatures2(gpu, &features);
        event("capabilities", {{"cuda_extension", has_cuda}, {"cuda_feature", bool(cuda.cudaKernelLaunchFeatures)},
                              {"buffer_device_address", bool(f12.bufferDeviceAddress)},
                              {"shader_int64", bool(features.features.shaderInt64)}});
        if (!f12.bufferDeviceAddress || !features.features.shaderInt64) return 2;
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());
        uint32_t family = UINT32_MAX;
        for (uint32_t i = 0; i < count; ++i)
            if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) { family = i; break; }
        if (family == UINT32_MAX) throw std::runtime_error("No compute queue");
        float priority = 1;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = family; qci.queueCount = 1; qci.pQueuePriorities = &priority;
        const char* cuda_name = VK_NV_CUDA_KERNEL_LAUNCH_EXTENSION_NAME;
        const bool use_cuda = has_cuda && cuda.cudaKernelLaunchFeatures;
        f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f12.bufferDeviceAddress = VK_TRUE;
        if (use_cuda) f12.pNext = &cuda;
        VkPhysicalDeviceFeatures enabled{};
        enabled.shaderInt64 = VK_TRUE;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &f12; dci.pEnabledFeatures = &enabled;
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        if (use_cuda) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &cuda_name; }
        VkDevice device{};
        check(vkCreateDevice(gpu, &dci, nullptr, &device));
        event("device_created");
#define DEV(name) auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device, #name)); \
        if (!name) throw std::runtime_error("Missing " #name)
        DEV(vkGetDeviceQueue); DEV(vkCreateBuffer); DEV(vkGetBufferMemoryRequirements);
        DEV(vkAllocateMemory); DEV(vkBindBufferMemory); DEV(vkMapMemory); DEV(vkUnmapMemory);
        DEV(vkGetBufferDeviceAddress); DEV(vkCreateShaderModule); DEV(vkCreatePipelineLayout);
        DEV(vkCreateComputePipelines); DEV(vkCreateCommandPool); DEV(vkAllocateCommandBuffers);
        DEV(vkBeginCommandBuffer); DEV(vkEndCommandBuffer); DEV(vkCmdBindPipeline);
        DEV(vkCmdPushConstants); DEV(vkCmdDispatch); DEV(vkCmdPipelineBarrier);
        DEV(vkQueueSubmit); DEV(vkCreateFence); DEV(vkWaitForFences); DEV(vkResetFences);
        DEV(vkResetCommandPool); DEV(vkDestroyFence); DEV(vkDestroyCommandPool);
        DEV(vkDestroyPipeline); DEV(vkDestroyPipelineLayout); DEV(vkDestroyShaderModule);
        DEV(vkDestroyBuffer); DEV(vkFreeMemory); DEV(vkDestroyDevice);
        constexpr uint32_t values = 4096;
        constexpr VkDeviceSize bytes = values * sizeof(uint32_t);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkBuffer buffer{}; check(vkCreateBuffer(device, &bci, nullptr, &buffer));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        VkPhysicalDeviceMemoryProperties memories{};
        vkGetPhysicalDeviceMemoryProperties(gpu, &memories);
        uint32_t memory_index = UINT32_MAX;
        constexpr VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < memories.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) && (memories.memoryTypes[i].propertyFlags & wanted) == wanted) { memory_index = i; break; }
        if (memory_index == UINT32_MAX) throw std::runtime_error("No coherent host memory");
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &flags; mai.allocationSize = requirements.size; mai.memoryTypeIndex = memory_index;
        VkDeviceMemory memory{}; check(vkAllocateMemory(device, &mai, nullptr, &memory));
        check(vkBindBufferMemory(device, buffer, memory, 0));
        void* mapped{}; check(vkMapMemory(device, memory, 0, bytes, 0, &mapped));
        memset(mapped, 0, size_t(bytes));
        VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        address_info.buffer = buffer;
        auto gpu_address = vkGetBufferDeviceAddress(device, &address_info);
        if (!gpu_address) throw std::runtime_error("Null GPU address");
        std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Missing SPIR-V");
        auto size = file.tellg();
        if (size <= 0 || size > 1024 * 1024 || size % 4) throw std::runtime_error("Invalid SPIR-V size");
        std::vector<uint32_t> code(size_t(size) / 4);
        file.seekg(0); file.read(reinterpret_cast<char*>(code.data()), size);
        if (!file) throw std::runtime_error("Truncated SPIR-V");
        VkShaderModuleCreateInfo sci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sci.codeSize = size_t(size); sci.pCode = code.data();
        VkShaderModule shader{}; check(vkCreateShaderModule(device, &sci, nullptr, &shader));
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpu_address)};
        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.pushConstantRangeCount = 1; lci.pPushConstantRanges = &range;
        VkPipelineLayout layout{}; check(vkCreatePipelineLayout(device, &lci, nullptr, &layout));
        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pci.layout = layout; pci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pci.stage.module = shader; pci.stage.pName = "main";
        VkPipeline pipeline{}; check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline));
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pool_info.queueFamilyIndex = family;
        VkCommandPool pool{}; check(vkCreateCommandPool(device, &pool_info, nullptr, &pool));
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = pool; allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocate.commandBufferCount = 1;
        VkCommandBuffer command{}; check(vkAllocateCommandBuffers(device, &allocate, &command));
        VkQueue queue{}; vkGetDeviceQueue(device, family, 0, &queue);
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence{}; check(vkCreateFence(device, &fence_info, nullptr, &fence));
        auto begin = [&] { VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; check(vkBeginCommandBuffer(command, &info)); };
        auto submit_and_check = [&] {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            check(vkEndCommandBuffer(command));
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
            check(vkQueueSubmit(queue, 1, &submit, fence));
            check(vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull));
            uint32_t matched = 0;
            for (uint32_t i = 0; i < values; ++i) if (static_cast<uint32_t*>(mapped)[i] == i * 3 + 7) ++matched;
            return matched;
        };
        begin();
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpu_address), &gpu_address);
        vkCmdDispatch(command, values / 64, 1, 1);
        auto matched = submit_and_check();
        event("spirv_result", {{"matched", matched}, {"expected", values}, {"passed", matched == values}});
        if (matched != values) return 2;
        if (use_cuda) {
            DEV(vkCreateCudaModuleNV); DEV(vkCreateCudaFunctionNV); DEV(vkCmdCudaLaunchKernelNV);
            DEV(vkDestroyCudaFunctionNV); DEV(vkDestroyCudaModuleNV);
            VkCudaModuleCreateInfoNV module_info{VK_STRUCTURE_TYPE_CUDA_MODULE_CREATE_INFO_NV};
            module_info.dataSize = sizeof(ptx); module_info.pData = ptx;
            VkCudaModuleNV module{};
            event("cuda_module_create_call");
            check(vkCreateCudaModuleNV(device, &module_info, nullptr, &module));
            VkCudaFunctionCreateInfoNV function_info{VK_STRUCTURE_TYPE_CUDA_FUNCTION_CREATE_INFO_NV};
            function_info.module = module; function_info.pName = "phoenix_probe";
            VkCudaFunctionNV function{}; check(vkCreateCudaFunctionNV(device, &function_info, nullptr, &function));
            check(vkResetCommandPool(device, pool, 0)); check(vkResetFences(device, 1, &fence));
            memset(mapped, 0, size_t(bytes));
            begin();
            const void* params[] = {&gpu_address};
            VkCudaLaunchInfoNV launch{VK_STRUCTURE_TYPE_CUDA_LAUNCH_INFO_NV};
            launch.function = function; launch.gridDimX = values / 64; launch.gridDimY = launch.gridDimZ = 1;
            launch.blockDimX = 64; launch.blockDimY = launch.blockDimZ = 1;
            launch.paramCount = 1; launch.pParams = params;
            vkCmdCudaLaunchKernelNV(command, &launch);
            matched = submit_and_check();
            event("cuda_result", {{"matched", matched}, {"expected", values}, {"passed", matched == values}});
            vkDestroyCudaFunctionNV(device, function, nullptr); vkDestroyCudaModuleNV(device, module, nullptr);
        }
        vkDestroyFence(device, fence, nullptr); vkDestroyCommandPool(device, pool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr); vkDestroyPipelineLayout(device, layout, nullptr);
        vkDestroyShaderModule(device, shader, nullptr); vkUnmapMemory(device, memory);
        vkDestroyBuffer(device, buffer, nullptr); vkFreeMemory(device, memory, nullptr);
        vkDestroyDevice(device, nullptr); vkDestroyInstance(instance, nullptr);
        return matched == values ? 0 : 2;
    } catch (const std::exception& error) {
        event("error", {{"message", error.what()}});
        return 1;
    }
}
