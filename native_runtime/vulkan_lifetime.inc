// Vulkan buffer/graph lifetime helpers included into the generated provider translation unit.
// Keep fence completion and reference ownership paired before releasing GPU-visible allocations.

/* Explicit C++ initialization avoids process environment and CRT snapshots. */
static VkPipeline phx_features_pipeline;
// Offsets count stored elements in each operand, not bytes. The generated
// resident provider adds these view offsets when resolving the five buffer slots.
void phoenix_native_operand_offsets(uint64_t left_offset, uint64_t right_offset,
                                    uint64_t output_offset, uint64_t auxiliary_offset,
                                    uint64_t secondary_output_offset) {
    phx_native_view_offsets[0] = left_offset;
    phx_native_view_offsets[1] = right_offset;
    phx_native_view_offsets[2] = output_offset;
    phx_native_view_offsets[3] = auxiliary_offset;
    phx_native_view_offsets[4] = secondary_output_offset;
}
unsigned phoenix_native_vendor(void) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(g.pd, &properties);
    return properties.vendorID;
}
// Discover matrix support on the explicitly selected UUID before device creation.
// No process environment is changed, and devices without a supported shape keep
// the ordered scalar shader. FP8 weights still decode exactly to FP16 operands.
int phoenix_native_init_configured(const char* uuid, const char* gemm, const char* matrix8,
                                   const char* matrix16, const char* unary, const char* row,
                                   const char* history, unsigned options) {
    if (!uuid || strlen(uuid) != 32 || g.ready)
        return -1;
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
    VkInstance instance;
    if (vkCreateInstance(&ci, NULL, &instance) != VK_SUCCESS)
        return -1;
    unsigned count = 0, shapes = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, NULL) == VK_SUCCESS && count && count <= 256) {
        VkPhysicalDevice* devices = calloc(count, sizeof(*devices));
        if (devices) {
            if (vkEnumeratePhysicalDevices(instance, &count, devices) == VK_SUCCESS) {
                VkPhysicalDevice selected = phx_select(devices, count, uuid);
                if (selected && !(options & 1u))
                    shapes = phx_matrix_shapes(instance, selected);
            }
            free(devices);
        }
    }
    vkDestroyInstance(instance, NULL);
    phx_native_matrix_rows = (shapes & 2u) ? 16u : (shapes & 1u) ? 8u : 0u;
    const char* selected_shader = phx_native_matrix_rows == 16  ? matrix16
                                  : phx_native_matrix_rows == 8 ? matrix8
                                                                : gemm;
    memcpy(phx_native_uuid, uuid, 33);
    phx_native_configured = 1;
    int result = xmx_init(selected_shader);
    if (result < 0)
        return result;
    return xmx_res_init(selected_shader, unary, row, history, selected_shader, selected_shader);
}
void phoenix_native_matrix_residual(int skip, int gain, uint64_t skip_offset,
                                    uint64_t gain_offset) {
    phx_native_skip = addr_of(skip) + skip_offset;
    phx_native_gain = addr_of(gain) + gain_offset;
}
unsigned phoenix_native_matrix(void) {
    return phx_native_matrix_rows;
}
// Preserve the original factory for external clients of the resident ABI.
int phoenix_native_init(const char* uuid, const char* gemm, const char* unary, const char* row,
                        const char* history) {
    return phoenix_native_init_configured(uuid, gemm, gemm, gemm, unary, row, history, 1u);
}
// The five operands can be views into the same scratch arena. Apply their byte
// offsets before recording, exactly as the scalar GEMM/unary adapters do.
int phoenix_native_window_attention(int query, int key, int value, int bias, int output,
                                    unsigned batches, unsigned heads) {
    if (!g.recording || !g.rwindow[1] || !batches || !heads || batches % heads)
        FAIL("invalid fused attention recording", 0);
    struct push packet = {.a = addr_of(query) + phx_native_view_offsets[0],
                          .b = addr_of(key) + phx_native_view_offsets[1],
                          .c = addr_of(output) + phx_native_view_offsets[2],
                          .d = addr_of(value) + phx_native_view_offsets[3],
                          .n = heads,
                          .batch = batches,
                          .flags = 1,
                          .residual_cos = addr_of(bias) + phx_native_view_offsets[4]};
    memset(phx_native_view_offsets, 0, sizeof(phx_native_view_offsets));
    vkCmdBindPipeline(g.rcb, VK_PIPELINE_BIND_POINT_COMPUTE, g.rwindow[1]);
    vkCmdPushConstants(g.rcb, g.rpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof packet, &packet);
    vkCmdDispatch(g.rcb, 1, batches < 65535u ? batches : 65535u, (batches + 65534u) / 65535u);
    barrier();
    stamp(PK_ROW, 3);
    ++g.recorded;
    return 0;
}
int phoenix_native_features_init(const char* path) {
    return build_pipeline(path, g.rpl, &phx_features_pipeline);
}
int phoenix_native_features(int color, int prior, int noise, int validity, int output,
                            unsigned height, unsigned width, unsigned padded_width,
                            unsigned padded_height, unsigned style, unsigned tone_bits,
                            unsigned structure_bits, unsigned skin_bits, unsigned auto_mask) {
    if (!g.recording || !phx_features_pipeline)
        FAIL("feature packing not initialized", 0);
    struct push packet = {.a = addr_of(color) + phx_native_view_offsets[0],
                          .b = addr_of(prior) + phx_native_view_offsets[1],
                          .c = addr_of(output) + phx_native_view_offsets[2],
                          .d = addr_of(noise) + phx_native_view_offsets[3],
                          .residual_cos = addr_of(validity) + phx_native_view_offsets[4],
                          .m = height,
                          .n = width,
                          .k = padded_width,
                          .batch = padded_height,
                          .sa = style,
                          .sb = tone_bits,
                          .sc = structure_bits,
                          .lda = skin_bits,
                          .flags = auto_mask};
    memset(phx_native_view_offsets, 0, sizeof(phx_native_view_offsets));
    vkCmdBindPipeline(g.rcb, VK_PIPELINE_BIND_POINT_COMPUTE, phx_features_pipeline);
    vkCmdPushConstants(g.rcb, g.rpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof packet, &packet);
    unsigned groups = (padded_width * padded_height + 255u) / 256u;
    unsigned x = groups < 65535u ? groups : 65535u;
    vkCmdDispatch(g.rcb, x, 1, (groups + x - 1u) / x);
    barrier();
    ++g.recorded;
    return 0;
}
void xmx_shutdown(void) {
    if (g.dev) {
        vkDeviceWaitIdle(g.dev);
        if (phx_features_pipeline)
            vkDestroyPipeline(g.dev, phx_features_pipeline, NULL);
        phx_features_pipeline = VK_NULL_HANDLE;
        for (unsigned iteration_index = 0; iteration_index < MAX_RBUF; ++iteration_index)
            xmx_buf_destroy((int)iteration_index);
        struct buf* buffers[] = {&g.A, &g.B, &g.C, &g.stage};
        for (unsigned iteration_index = 0; iteration_index < 4; ++iteration_index) {
            if (buffers[iteration_index]->p)
                vkUnmapMemory(g.dev, buffers[iteration_index]->m);
            if (buffers[iteration_index]->b)
                vkDestroyBuffer(g.dev, buffers[iteration_index]->b, NULL);
            if (buffers[iteration_index]->m)
                vkFreeMemory(g.dev, buffers[iteration_index]->m, NULL);
        }
        for (unsigned iteration_index = 0; iteration_index < specialized_count; ++iteration_index)
            if (specialized[iteration_index].pipeline)
                vkDestroyPipeline(g.dev, specialized[iteration_index].pipeline, NULL);
        VkPipeline pipelines[] = {
            g.pipe,         g.pipeb,    g.rgemm,      g.rtiled,     g.rstaged, g.runary,
            g.rrow,         g.rhistory, g.rwindow[0], g.rwindow[1], g.rffn,    g.rstaged32[0],
            g.rstaged32[1], g.rrows,    g.rint8,      g.rblock,     g.rglobal};
        for (unsigned iteration_index = 0;
             iteration_index < sizeof(pipelines) / sizeof(pipelines[0]); ++iteration_index)
            if (pipelines[iteration_index])
                vkDestroyPipeline(g.dev, pipelines[iteration_index], NULL);
        if (g.qpool)
            vkDestroyQueryPool(g.dev, g.qpool, NULL);
        if (g.cpool)
            vkDestroyCommandPool(g.dev, g.cpool, NULL);
        if (g.fence)
            vkDestroyFence(g.dev, g.fence, NULL);
        if (g.rfence)
            vkDestroyFence(g.dev, g.rfence, NULL);
        if (g.tfence)
            vkDestroyFence(g.dev, g.tfence, NULL);
        if (g.dpool)
            vkDestroyDescriptorPool(g.dev, g.dpool, NULL);
        if (g.pl)
            vkDestroyPipelineLayout(g.dev, g.pl, NULL);
        if (g.plb)
            vkDestroyPipelineLayout(g.dev, g.plb, NULL);
        if (g.rpl)
            vkDestroyPipelineLayout(g.dev, g.rpl, NULL);
        if (g.dsl)
            vkDestroyDescriptorSetLayout(g.dev, g.dsl, NULL);
        vkDestroyDevice(g.dev, NULL);
    }
    if (g.inst)
        vkDestroyInstance(g.inst, NULL);
    for (unsigned iteration_index = 0; iteration_index < 12; ++iteration_index)
        free(g.rpaths[iteration_index]);
    for (unsigned iteration_index = 0; iteration_index < MAX_GRAPHS; ++iteration_index)
        free(graphs[iteration_index].kinds);
    if (g.stats)
        fclose(g.stats);
    memset(&g, 0, sizeof(g));
    memset(graphs, 0, sizeof(graphs));
    memset(rbufs, 0, sizeof(rbufs));
    memset(specialized, 0, sizeof(specialized));
    specialized_count = 0;
    xmx_profile_reset();
    phx_native_configured = 0;
    phx_native_matrix_rows = 0;
    memset(phx_native_uuid, 0, sizeof(phx_native_uuid));
    memset(phx_native_view_offsets, 0, sizeof(phx_native_view_offsets));
}
