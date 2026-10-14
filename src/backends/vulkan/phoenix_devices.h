// DLSS5 Phoenix-owned device discovery. No implicit first-adapter fallback.
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// Bit 0: 8x16x16, bit 1: 16x16x16; FP16 operands, FP32 accumulator.
// Extension presence alone does not establish a usable matrix configuration.
// Require both the cooperative-matrix feature and supported operand/result
// types. An extension name alone does not prove a usable matrix tile shape.
static unsigned phx_matrix_shapes(VkInstance instance, VkPhysicalDevice physical_device) {
    uint32_t extension_count = 0;
    unsigned supported = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, NULL, &extension_count, NULL) !=
        VK_SUCCESS)
        return 0;
    VkExtensionProperties* ext = calloc(extension_count ? extension_count : 1, sizeof(*ext));
    if (!ext)
        return 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, NULL, &extension_count, ext) ==
        VK_SUCCESS)
        for (unsigned iteration_index = 0; iteration_index < extension_count; iteration_index++)
            if (!strcmp(ext[iteration_index].extensionName,
                        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME))
                supported = 1;
    free(ext);
    if (!supported)
        return 0;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR matrix_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                          .pNext = &matrix_features};
    vkGetPhysicalDeviceFeatures2(physical_device, &features);
    if (!matrix_features.cooperativeMatrix)
        return 0;
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 properties = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &props};
    vkGetPhysicalDeviceProperties2(physical_device, &properties);
    if (!(props.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT))
        return 0;
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR query =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
            instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!query || query(physical_device, &extension_count, NULL) != VK_SUCCESS ||
        extension_count > 4096)
        return 0;
    VkCooperativeMatrixPropertiesKHR* configs =
        calloc(extension_count ? extension_count : 1, sizeof(*configs));
    if (!configs)
        return 0;
    for (unsigned iteration_index = 0; iteration_index < extension_count; iteration_index++)
        configs[iteration_index].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    supported = 0;
    if (query(physical_device, &extension_count, configs) == VK_SUCCESS)
        for (unsigned iteration_index = 0; iteration_index < extension_count; iteration_index++) {
            VkCooperativeMatrixPropertiesKHR matrix_shape = configs[iteration_index];
            if (matrix_shape.scope == VK_SCOPE_SUBGROUP_KHR && matrix_shape.NSize == 16 &&
                matrix_shape.KSize == 16 && matrix_shape.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                matrix_shape.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                matrix_shape.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                matrix_shape.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                !matrix_shape.saturatingAccumulation)
                supported |= matrix_shape.MSize == 8 ? 1u : matrix_shape.MSize == 16 ? 2u : 0u;
        }
    free(configs);
    return supported;
}
int phoenix_matrix_shapes(unsigned index) {
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo instance_create_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                                 .pApplicationInfo = &app};
    VkInstance instance;
    if (vkCreateInstance(&instance_create_info, NULL, &instance))
        return -1;
    uint32_t device_count = 0;
    VkPhysicalDevice devices[32];
    int result = -1;
    if (vkEnumeratePhysicalDevices(instance, &device_count, NULL) == VK_SUCCESS &&
        device_count <= 32 && index < device_count &&
        vkEnumeratePhysicalDevices(instance, &device_count, devices) == VK_SUCCESS)
        result = (int)phx_matrix_shapes(instance, devices[index]);
    vkDestroyInstance(instance, NULL);
    return result;
}
// FP8 requires its own feature and matrix-shape query; vendor/name is insufficient.
// Bits 0/1: extension and scalar/storage features. Bits 2/3/4: 16x16 K=16/32/64 FP8 tiles.
static unsigned phx_fp8_caps(VkInstance instance, VkPhysicalDevice physical_device) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physical_device, NULL, &count, NULL);
    VkExtensionProperties* ext = calloc(count ? count : 1, sizeof(*ext));
    if (!ext)
        return 0;
    unsigned caps = 0;
    if (vkEnumerateDeviceExtensionProperties(physical_device, NULL, &count, ext) == VK_SUCCESS)
        for (unsigned iteration_index = 0; iteration_index < count; ++iteration_index)
            if (!strcmp(ext[iteration_index].extensionName, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME))
                caps |= 1;
    free(ext);
    if (!caps)
        return 0;
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f8};
    VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                          .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(physical_device, &features);
    if (f8.shaderFloat8 && f8.shaderFloat8CooperativeMatrix && f12.storageBuffer8BitAccess &&
        f12.shaderInt8)
        caps |= 2;
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR query =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
            instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!query || query(physical_device, &count, NULL) != VK_SUCCESS || count > 4096)
        return caps;
    VkCooperativeMatrixPropertiesKHR* configs = calloc(count ? count : 1, sizeof(*configs));
    if (!configs)
        return caps;
    for (unsigned iteration_index = 0; iteration_index < count; ++iteration_index)
        configs[iteration_index].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    if (query(physical_device, &count, configs) == VK_SUCCESS)
        for (unsigned iteration_index = 0; iteration_index < count; ++iteration_index) {
            const VkCooperativeMatrixPropertiesKHR matrix_shape = configs[iteration_index];
            if (matrix_shape.scope == VK_SCOPE_SUBGROUP_KHR && matrix_shape.MSize == 16 &&
                matrix_shape.NSize == 16 &&
                matrix_shape.AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
                matrix_shape.BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
                matrix_shape.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                matrix_shape.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                !matrix_shape.saturatingAccumulation)
                caps |= matrix_shape.KSize == 16   ? 4u
                        : matrix_shape.KSize == 32 ? 8u
                        : matrix_shape.KSize == 64 ? 16u
                                                   : 0u;
        }
    free(configs);
    return caps;
}
int phoenix_fp8_capabilities(unsigned index) {
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo instance_create_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                                 .pApplicationInfo = &app};
    VkInstance instance;
    if (vkCreateInstance(&instance_create_info, NULL, &instance))
        return -1;
    uint32_t count = 0;
    VkPhysicalDevice physical_device[32];
    int result = -1;
    if (vkEnumeratePhysicalDevices(instance, &count, NULL) == VK_SUCCESS && count <= 32 &&
        index < count &&
        vkEnumeratePhysicalDevices(instance, &count, physical_device) == VK_SUCCESS)
        result = (int)phx_fp8_caps(instance, physical_device[index]);
    vkDestroyInstance(instance, NULL);
    return result;
}
static void phx_uuid(VkPhysicalDevice physical_device, char text[33]) {
    VkPhysicalDeviceIDProperties identity = {.sType =
                                                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                         .pNext = &identity};
    vkGetPhysicalDeviceProperties2(physical_device, &props);
    for (unsigned iteration_index = 0; iteration_index < VK_UUID_SIZE; iteration_index++)
        sprintf(text + iteration_index * 2, "%02x", identity.deviceUUID[iteration_index]);
}
static VkPhysicalDevice phx_select(VkPhysicalDevice* devices, unsigned count, const char* uuid) {
    if (!uuid || strlen(uuid) != 32)
        return VK_NULL_HANDLE;
    VkPhysicalDevice found = VK_NULL_HANDLE;
    for (unsigned iteration_index = 0; iteration_index < count; iteration_index++) {
        char value[33];
        phx_uuid(devices[iteration_index], value);
        if (!strcmp(uuid, value)) {
            if (found)
                return VK_NULL_HANDLE;
            found = devices[iteration_index];
        }
    }
    return found;
}
// Capability results describe the minimum portable path, not Intel/RDNA4 kernels.
int phoenix_device_count(void) {
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo instance_create_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                                 .pApplicationInfo = &app};
    VkInstance instance;
    if (vkCreateInstance(&instance_create_info, NULL, &instance))
        return -1;
    uint32_t device_count = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance, &device_count, NULL);
    vkDestroyInstance(instance, NULL);
    return result ? -1 : (int)device_count;
}
int phoenix_device_info(unsigned index, char* uuid, char* name, unsigned* vendor,
                        unsigned* portable) {
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo instance_create_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                                 .pApplicationInfo = &app};
    VkInstance instance;
    if (vkCreateInstance(&instance_create_info, NULL, &instance))
        return -1;
    uint32_t device_count = 0;
    VkPhysicalDevice devices[32];
    if (vkEnumeratePhysicalDevices(instance, &device_count, NULL) || device_count > 32 ||
        index >= device_count) {
        vkDestroyInstance(instance, NULL);
        return -1;
    }
    if (vkEnumeratePhysicalDevices(instance, &device_count, devices)) {
        vkDestroyInstance(instance, NULL);
        return -1;
    }
    VkPhysicalDevice physical_device = devices[index];
    phx_uuid(physical_device, uuid);
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f13};
    VkPhysicalDeviceVulkan11Features f11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12};
    VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                          .pNext = &f11};
    VkPhysicalDeviceVulkan13Properties sizes = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceProperties2 props = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                         .pNext = &sizes};
    vkGetPhysicalDeviceFeatures2(physical_device, &features);
    vkGetPhysicalDeviceProperties2(physical_device, &props);
    snprintf(name, 256, "%s", props.properties.deviceName);
    *vendor = props.properties.vendorID;
    *portable =
        props.properties.apiVersion >= VK_API_VERSION_1_3 && features.features.shaderInt64 &&
        f11.storageBuffer16BitAccess && f12.shaderFloat16 && f12.bufferDeviceAddress &&
        f12.scalarBlockLayout && f12.vulkanMemoryModel && f12.vulkanMemoryModelDeviceScope &&
        f13.subgroupSizeControl && sizes.minSubgroupSize <= 32 && sizes.maxSubgroupSize >= 32 &&
        (sizes.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT);
    vkDestroyInstance(instance, NULL);
    return 0;
}
#include "phoenix_matrix2.h"
