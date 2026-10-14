// Optional NVIDIA workgroup FP8 path. Unsupported devices keep the existing
// portable engines; extension names alone never enable a shader.
static unsigned phx_matrix2_caps(VkInstance instance, VkPhysicalDevice device) {
    VkPhysicalDeviceProperties ordinary;
    vkGetPhysicalDeviceProperties(device, &ordinary);
    if (ordinary.vendorID != 0x10de)
        return 0; // Measured policy is limited to NVIDIA; other vendors are unchanged.
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, NULL, &count, NULL) != VK_SUCCESS ||
        count > 4096)
        return 0;
    VkExtensionProperties* extensions = calloc(count ? count : 1, sizeof(*extensions));
    if (!extensions)
        return 0;
    unsigned names = 0;
    if (vkEnumerateDeviceExtensionProperties(device, NULL, &count, extensions) == VK_SUCCESS)
        for (unsigned index = 0; index < count; ++index) {
            if (!strcmp(extensions[index].extensionName, VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME))
                names |= 1;
            if (!strcmp(extensions[index].extensionName, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME))
                names |= 2;
        }
    free(extensions);
    if (names != 3 || (phx_fp8_caps(instance, device) & 3) != 3)
        return 0;
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV matrix = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
    VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                          .pNext = &matrix};
    vkGetPhysicalDeviceFeatures2(device, &features);
    if (!matrix.cooperativeMatrixFlexibleDimensions || !matrix.cooperativeMatrixWorkgroupScope ||
        !matrix.cooperativeMatrixTensorAddressing || !matrix.cooperativeMatrixConversions ||
        !matrix.cooperativeMatrixPerElementOperations)
        return 0;
    VkPhysicalDeviceCooperativeMatrix2PropertiesNV limits = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV};
    VkPhysicalDeviceProperties2 properties = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &limits};
    vkGetPhysicalDeviceProperties2(device, &properties);
    if (limits.cooperativeMatrixWorkgroupScopeMaxWorkgroupSize < 128 ||
        limits.cooperativeMatrixFlexibleDimensionsMaxDimension < 128)
        return 0;
    PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV query =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV)
            vkGetInstanceProcAddr(
                instance, "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");
    if (!query || query(device, &count, NULL) != VK_SUCCESS || count > 4096)
        return 0;
    VkCooperativeMatrixFlexibleDimensionsPropertiesNV* shapes =
        calloc(count ? count : 1, sizeof(*shapes));
    if (!shapes)
        return 0;
    for (unsigned index = 0; index < count; ++index)
        shapes[index].sType =
            VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
    unsigned supported = 0;
    if (query(device, &count, shapes) == VK_SUCCESS)
        for (unsigned index = 0; index < count; ++index) {
            const VkCooperativeMatrixFlexibleDimensionsPropertiesNV* shape = &shapes[index];
            if (shape->scope == VK_SCOPE_WORKGROUP_KHR && shape->workgroupInvocations == 128 &&
                shape->MGranularity && 32 % shape->MGranularity == 0 && shape->NGranularity &&
                16 % shape->NGranularity == 0 && shape->KGranularity &&
                32 % shape->KGranularity == 0 && shape->CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                shape->ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
                !shape->saturatingAccumulation) {
                if (shape->AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
                    shape->BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT)
                    supported |= 1;
                if (shape->AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                    shape->BType == VK_COMPONENT_TYPE_FLOAT16_KHR)
                    supported |= 2;
            }
        }
    free(shapes);
    return supported == 3;
}
