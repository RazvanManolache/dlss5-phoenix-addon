// Task-local explicit ReShade Vulkan layer. No registry or global environment
// changes: the test launcher supplies a manifest only to this child process.
#pragma once
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
class VulkanFixture {
    PFN_vkGetInstanceProcAddr instance_proc_;
    PFN_vkGetDeviceProcAddr device_proc_;
    VkInstance instance_ = nullptr;
    VkDevice device_ = nullptr;
    VkSurfaceKHR surface_ = 0;
    VkSwapchainKHR swapchain_ = 0;
    VkQueue queue_ = nullptr;
    VkQueue present_queue_ = nullptr;
    VkSemaphore acquired_ = 0, rendered_ = 0;
    VkCommandPool pool_ = 0;
    VkCommandBuffer command_ = nullptr;
    std::vector<VkImage> images_;
    std::vector<bool> initialized_;
    uint32_t image_ = 0;
    bool ready_ = false;
    static void require(VkResult result) {
        if (result < 0)
            throw std::runtime_error("Vulkan test error " + std::to_string(result));
    }
    template <class Function> Function instance_function(const char* name) {
        auto value = reinterpret_cast<Function>(instance_proc_(instance_, name));
        if (!value)
            throw std::runtime_error(name);
        return value;
    }
    template <class Function> Function device_function(const char* name) {
        auto value = reinterpret_cast<Function>(device_proc_(device_, name));
        if (!value)
            throw std::runtime_error(name);
        return value;
    }

  public:
    VulkanFixture(HWND window, unsigned width, unsigned height, bool secondary_queue = false, bool compute_present = false) {
        auto loader = LoadLibraryW(L"vulkan-1.dll");
        if (!loader)
            throw std::runtime_error("Vulkan loader unavailable");
        instance_proc_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            GetProcAddress(loader, "vkGetInstanceProcAddr"));
        if (!instance_proc_)
            throw std::runtime_error("Vulkan entry point unavailable");
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.apiVersion = VK_API_VERSION_1_3;
        const char* extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME,
                                    VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
        const char* layer = "VK_LAYER_phoenix_integration_test";
        VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        create.pApplicationInfo = &application;
        create.enabledExtensionCount = 2;
        create.ppEnabledExtensionNames = extensions;
        create.enabledLayerCount = 1;
        create.ppEnabledLayerNames = &layer;
        require(instance_function<PFN_vkCreateInstance>("vkCreateInstance")(&create, nullptr,
                                                                            &instance_));
        VkWin32SurfaceCreateInfoKHR surface{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        surface.hinstance = GetModuleHandleW(nullptr);
        surface.hwnd = window;
        require(instance_function<PFN_vkCreateWin32SurfaceKHR>("vkCreateWin32SurfaceKHR")(
            instance_, &surface, nullptr, &surface_));
        uint32_t count = 0;
        auto enumerate =
            instance_function<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
        require(enumerate(instance_, &count, nullptr));
        std::vector<VkPhysicalDevice> devices(count);
        require(enumerate(instance_, &count, devices.data()));
        if (!count)
            throw std::runtime_error("No Vulkan device");
        VkPhysicalDevice selected = devices[0];
        for (auto candidate : devices) {
            VkPhysicalDeviceProperties properties{};
            instance_function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(
                candidate, &properties);
            if (properties.vendorID == 0x10de)
                selected = candidate;
        }
        auto families = instance_function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            "vkGetPhysicalDeviceQueueFamilyProperties");
        families(selected, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        families(selected, &count, queues.data());
        uint32_t family = UINT32_MAX;
        for (uint32_t index = 0; index < count; ++index) {
            VkBool32 supports = VK_FALSE;
            require(instance_function<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
                "vkGetPhysicalDeviceSurfaceSupportKHR")(selected, index, surface_, &supports));
            if (supports && (queues[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                family = index;
                break;
            }
        }
        if (family == UINT32_MAX)
            throw std::runtime_error("No graphics/presentation queue");
        uint32_t presentation_family = family;
        if (compute_present) {
            presentation_family = UINT32_MAX;
            for (uint32_t index = 0; index < count; ++index) {
                VkBool32 supports = VK_FALSE;
                require(instance_function<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
                    "vkGetPhysicalDeviceSurfaceSupportKHR")(selected, index, surface_, &supports));
                if (supports && (queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(queues[index].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
                    presentation_family = index;
                    break;
                }
            }
            if (presentation_family == UINT32_MAX) throw std::runtime_error("No compute-only presentation family");
        }
        if (secondary_queue && queues[family].queueCount < 2)
            throw std::runtime_error("Separate-queue Vulkan fixture requires two graphics queues");
        float priorities[] = {1, 1};
        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueFamilyIndex = family;
        queue.queueCount = secondary_queue ? 2 : 1;
        queue.pQueuePriorities = priorities;
        std::vector<VkDeviceQueueCreateInfo> requested_queues{queue};
        if (compute_present) {
            auto presentation = queue;
            presentation.queueFamilyIndex = presentation_family;
            presentation.queueCount = 1;
            requested_queues.push_back(presentation);
        }
        VkPhysicalDeviceVulkan12Features features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        features.timelineSemaphore = VK_TRUE;
        const char* swap_extension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device.pNext = &features;
        device.queueCreateInfoCount = uint32_t(requested_queues.size());
        device.pQueueCreateInfos = requested_queues.data();
        device.enabledExtensionCount = 1;
        device.ppEnabledExtensionNames = &swap_extension;
        require(instance_function<PFN_vkCreateDevice>("vkCreateDevice")(selected, &device, nullptr,
                                                                        &device_));
        device_proc_ = instance_function<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
        device_function<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(device_, family, 0, &queue_);
        device_function<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(device_, presentation_family, secondary_queue ? 1 : 0, &present_queue_);
        VkSurfaceCapabilitiesKHR capabilities{};
        require(instance_function<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")(selected, surface_, &capabilities));
        auto surface_formats = instance_function<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
            "vkGetPhysicalDeviceSurfaceFormatsKHR");
        require(surface_formats(selected, surface_, &count, nullptr));
        std::vector<VkSurfaceFormatKHR> formats(count);
        require(surface_formats(selected, surface_, &count, formats.data()));
        auto format = formats[0];
        for (auto candidate : formats)
            if (candidate.format == VK_FORMAT_R8G8B8A8_UNORM ||
                candidate.format == VK_FORMAT_B8G8R8A8_UNORM) {
                format = candidate;
                if (candidate.format == VK_FORMAT_R8G8B8A8_UNORM)
                    break;
            }
        VkSwapchainCreateInfoKHR swap{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        swap.surface = surface_;
        swap.minImageCount = (std::max)(2u, capabilities.minImageCount);
        if (capabilities.maxImageCount)
            swap.minImageCount = (std::min)(swap.minImageCount, capabilities.maxImageCount);
        swap.imageFormat = format.format;
        swap.imageColorSpace = format.colorSpace;
        swap.imageExtent = {width, height};
        swap.imageArrayLayers = 1;
        // The addon must request the transfer-destination usage it needs.
        swap.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        swap.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        const uint32_t image_families[] = {family, presentation_family};
        if (compute_present) {
            swap.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            swap.queueFamilyIndexCount = 2;
            swap.pQueueFamilyIndices = image_families;
        }
        swap.preTransform = capabilities.currentTransform;
        swap.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        swap.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        swap.clipped = VK_TRUE;
        require(device_function<PFN_vkCreateSwapchainKHR>("vkCreateSwapchainKHR")(
            device_, &swap, nullptr, &swapchain_));
        VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        require(device_function<PFN_vkCreateSemaphore>("vkCreateSemaphore")(device_, &semaphore,
                                                                            nullptr, &acquired_));
        require(device_function<PFN_vkCreateSemaphore>("vkCreateSemaphore")(device_, &semaphore,
                                                                            nullptr, &rendered_));
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = family;
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        require(device_function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(device_, &pool,
                                                                                nullptr, &pool_));
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = pool_;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        require(device_function<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(
            device_, &allocate, &command_));
        auto get_images = device_function<PFN_vkGetSwapchainImagesKHR>("vkGetSwapchainImagesKHR");
        require(get_images(device_, swapchain_, &count, nullptr));
        images_.resize(count);
        initialized_.resize(count);
        require(get_images(device_, swapchain_, &count, images_.data()));
    }
    void acquire() {
        if (ready_)
            return;
        require(device_function<PFN_vkAcquireNextImageKHR>("vkAcquireNextImageKHR")(
            device_, swapchain_, UINT64_MAX, acquired_, VK_NULL_HANDLE, &image_));
        ready_ = true;
        // A real game submits work that waits for acquisition before Present.
        // Model that order even though the test fills pixels through ReShade.
        require(device_function<PFN_vkResetCommandBuffer>("vkResetCommandBuffer")(command_, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        require(
            device_function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(command_, &begin));
        if (!initialized_[image_]) {
            VkImageMemoryBarrier transition{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            transition.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            transition.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            transition.srcQueueFamilyIndex = transition.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            transition.image = images_[image_];
            transition.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            device_function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(
                command_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0, 0, nullptr, 0, nullptr, 1, &transition);
            initialized_[image_] = true;
        }
        require(device_function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(command_));
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &acquired_;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &rendered_;
        require(device_function<PFN_vkQueueSubmit>("vkQueueSubmit")(queue_, 1, &submit,
                                                                    VK_NULL_HANDLE));
    }
    void present() {
        acquire();
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &rendered_;
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain_;
        present.pImageIndices = &image_;
        require(device_function<PFN_vkQueuePresentKHR>("vkQueuePresentKHR")(present_queue_, &present));
        // The harness serializes acquisition only to reuse its one binary
        // semaphore; the addon's own transport stays asynchronous.
        require(device_function<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(present_queue_));
        ready_ = false;
    }
    ~VulkanFixture() {
        if (!device_)
            return;
        device_function<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(device_);
        device_function<PFN_vkDestroySemaphore>("vkDestroySemaphore")(device_, acquired_, nullptr);
        device_function<PFN_vkDestroySemaphore>("vkDestroySemaphore")(device_, rendered_, nullptr);
        device_function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(device_, pool_, nullptr);
        device_function<PFN_vkDestroySwapchainKHR>("vkDestroySwapchainKHR")(device_, swapchain_,
                                                                            nullptr);
        device_function<PFN_vkDestroyDevice>("vkDestroyDevice")(device_, nullptr);
        instance_function<PFN_vkDestroySurfaceKHR>("vkDestroySurfaceKHR")(instance_, surface_,
                                                                          nullptr);
        instance_function<PFN_vkDestroyInstance>("vkDestroyInstance")(instance_, nullptr);
    }
};
