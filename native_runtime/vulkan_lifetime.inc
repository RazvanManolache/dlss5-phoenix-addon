// Vulkan buffer/graph lifetime helpers included into the generated provider translation unit.
// Keep fence completion and reference ownership paired before releasing GPU-visible allocations.

/* Explicit C++ initialization avoids process environment and CRT snapshots. */
static VkPipeline phx_features_pipeline, phx_qkv_pipeline;
static VkPipeline phx_feed_forward_pipeline, phx_project_qkv_pipeline;

unsigned phoenix_native_matrix2(void) {
    return phx_native_matrix2;
}
int xmx_matrix2_init(unsigned kind, const char* path) {
    if (!phx_native_matrix2 || kind > 2)
        FAIL("workgroup FP8 engine unavailable", 0);
    VkPipeline* target = kind == 0   ? &phx_matrix2_pipeline
                         : kind == 1 ? &phx_feed_forward_pipeline
                                     : &phx_project_qkv_pipeline;
    if (*target)
        return 0;
    if (build_pipeline(path, g.rpl, target))
        return -1;
    if (kind == 0 && phx_native_matrix2_shapes) {
        char wide_path[1024];
        if (strlen(path) >= sizeof wide_path)
            FAIL("matrix shader path too long", 0);
        snprintf(wide_path, sizeof wide_path, "%s", path);
        char* separator = strrchr(wide_path, '/');
        char* backslash = strrchr(wide_path, '\\');
        if (!separator || (backslash && backslash > separator))
            separator = backslash;
        if (!separator)
            FAIL("matrix shader folder missing", 0);
        separator[1] = 0;
        if (strlen(wide_path) + strlen("phoenix_matrix2_wide.spv") >= sizeof wide_path)
            FAIL("wide matrix shader path too long", 0);
        strcat(wide_path, "phoenix_matrix2_wide.spv");
        return build_pipeline(wide_path, g.rpl, &phx_matrix2_wide_pipeline);
    }
    return 0;
}
// Six logical tensor addresses use the same buffer-ID/byte-offset encoding as
// the adapter's ordinary operations. Hidden feed-forward and projected QKV
// tensors stay inside their dispatch, so no additional device allocation exists.
int xmx_matrix2_fused_record(unsigned kind, const uint64_t* arguments) {
    VkPipeline pipeline = kind == 1 ? phx_feed_forward_pipeline : phx_project_qkv_pipeline;
    if (!g.recording || !phx_native_matrix2 || (kind != 1 && kind != 2) || !pipeline ||
        !arguments[6])
        FAIL("invalid fused workgroup recording", 0);
    struct push packet = {0};
    uint64_t* feed_forward_addresses[] = {
        &packet.a, &packet.b, &packet.c, &packet.d, &packet.residual_cos, &packet.qkv_scale};
    uint64_t* qkv_addresses[] = {&packet.residual_cos, &packet.a, &packet.b, &packet.c, &packet.d,
                                 &packet.qkv_scale};
    for (unsigned index = 0; index < 6; ++index) {
        uint64_t logical = arguments[index];
        uint64_t physical =
            logical ? addr_of((int)(logical >> 40)) + (logical & ((1ull << 40) - 1ull)) : 0;
        *(kind == 1 ? feed_forward_addresses[index] : qkv_addresses[index]) = physical;
    }
    packet.m = (unsigned)arguments[6];
    packet.n = kind == 1 ? (unsigned)arguments[7] : 64u;
    packet.k = kind == 1 ? (unsigned)arguments[8] : 0;
    packet.batch = kind == 1 ? (unsigned)arguments[9] : 1u;
    unsigned groups = (packet.m + 31u) / 32u, x = groups < 65535u ? groups : 65535u;
    vkCmdBindPipeline(g.rcb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(g.rcb, g.rpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof packet, &packet);
    if (phx_dispatch(g.rcb, x, 1, (groups + x - 1u) / x))
        return -1;
    barrier();
    stamp(PK_GEMM, 0);
    ++g.recorded;
    return 0;
}

// One dispatch normalizes local Q/K heads and publishes V. The five operands
// can be views of one scratch buffer; the adapter supplies byte offsets.
int xmx_qkv_init(const char* path) {
    return build_pipeline(path, g.rpl, &phx_qkv_pipeline);
}
int xmx_rec_qkv_prepare(int projected, int query, int key, int value, int scale, unsigned rows,
                        unsigned tokens, unsigned heads) {
    if (!g.recording || !phx_qkv_pipeline || !rows || tokens != 64u || !heads || heads == 16u)
        FAIL("invalid local QKV preparation", 0);
    struct push packet = {.a = addr_of(projected) + phx_native_view_offsets[0],
                          .b = addr_of(query) + phx_native_view_offsets[1],
                          .c = addr_of(key) + phx_native_view_offsets[2],
                          .d = addr_of(value) + phx_native_view_offsets[3],
                          .qkv_scale = addr_of(scale) + phx_native_view_offsets[4],
                          .m = rows,
                          .n = tokens,
                          .batch = heads};
    memset(phx_native_view_offsets, 0, sizeof(phx_native_view_offsets));
    vkCmdBindPipeline(g.rcb, VK_PIPELINE_BIND_POINT_COMPUTE, phx_qkv_pipeline);
    vkCmdPushConstants(g.rcb, g.rpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof packet, &packet);
    unsigned groups = (rows + 15u) / 16u, x = groups < 65535u ? groups : 65535u;
    if (phx_dispatch(g.rcb, x, 1, (groups + x - 1u) / x))
        return -1;
    barrier();
    stamp(PK_ROW, 0);
    ++g.recorded;
    return 0;
}
int phoenix_native_coalesced_init(const char* path) {
    if (!g.rready || g.matrix_rows)
        FAIL("invalid coalesced matrix initialization", 0);
    if (phx_coalesced_pipeline)
        return 0;
    free(g.rpaths[13]);
    if (!(g.rpaths[13] = strdup(path)))
        FAIL("coalesced matrix path allocation", 0);
    return build_pipeline(path, g.rpl, &phx_coalesced_pipeline);
}
int phoenix_native_wide_init(const char* path, unsigned columns) {
    if (!g.rready || (columns != 32 && columns != 64))
        FAIL("invalid wide matrix initialization", 0);
    if (g.matrix_rows != 16 || g.fp8_k)
        return 0; // A capability fallback keeps the original matrix engine.
    free(g.rpaths[12]);
    if (!(g.rpaths[12] = strdup(path)))
        FAIL("wide matrix path allocation", 0);
    if (build_pipeline(path, g.rpl, &phx_wide_pipeline))
        return -1;
    phx_wide_columns = columns;
    return 0;
}
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
unsigned phoenix_native_device_id(void) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(g.pd, &properties);
    return properties.deviceID;
}
// Discover matrix support on the explicitly selected UUID before device creation.
// No process environment is changed, and devices without a supported shape keep
// the ordered scalar shader. FP8 weights still decode exactly to FP16 operands.
int phoenix_native_init_configured(const char* uuid, const char* gemm, const char* matrix8,
                                   const char* matrix16, const char* unary, const char* row,
                                   const char* history, unsigned options, const char* fp8_matrix,
                                   const char* wide_matrix) {
    if (!uuid || strlen(uuid) != 32 || g.ready)
        return -1;
    phx_native_workgroup_compatibility = (options & (1u << 30)) != 0;
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
    VkInstance instance;
    if (vkCreateInstance(&ci, NULL, &instance) != VK_SUCCESS)
        return -1;
    unsigned count = 0, shapes = 0, fp8_caps = 0, selected_vendor = 0;
    phx_native_matrix2 = 0;
    phx_native_matrix2_shapes = 0;
    int selected_integrated = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, NULL) == VK_SUCCESS && count && count <= 256) {
        VkPhysicalDevice* devices = calloc(count, sizeof(*devices));
        if (devices) {
            if (vkEnumeratePhysicalDevices(instance, &count, devices) == VK_SUCCESS) {
                VkPhysicalDevice selected = phx_select(devices, count, uuid);
                if (selected) {
                    VkPhysicalDeviceProperties properties;
                    vkGetPhysicalDeviceProperties(selected, &properties);
                    selected_vendor = properties.vendorID;
                    selected_integrated =
                        properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
                }
                if (selected && !(options & 1u) && !phx_native_workgroup_compatibility) {
                    shapes = phx_matrix_shapes(instance, selected);
                    if ((options & (1u << 26)) && (shapes & 2))
                        phx_native_matrix2 = phx_matrix2_caps(instance, selected);
                    phx_native_matrix2_shapes = phx_native_matrix2 && (options & (1u << 29));
                    if (options & 8u)
                        fp8_caps = phx_fp8_caps(instance, selected);
                }
            }
            free(devices);
        }
    }
    vkDestroyInstance(instance, NULL);
    phx_native_matrix_rows = (shapes & 2u) ? 16u : (shapes & 1u) ? 8u : 0u;
    phx_native_scalar_columns = !phx_native_matrix_rows && (options & (1u << 21)) ? 32u : 16u;
    // A tile uses FP8 only when both operands are exactly representable. True
    // half-valued boundaries take the shader's original FP16 fallback.
    phx_native_fp8_k =
        phx_native_matrix_rows == 16 && (fp8_caps & 11u) == 11u && fp8_matrix ? 32u : 0u;
    // Four subgroups share decoded weights on the measured NVIDIA and AMD iGPU paths.
    // Other vendors keep the original workgroup unless explicitly requested;
    // the one-subgroup override also remains available for reproducible A/B runs.
    const int wide_workgroup =
        (options & 32u) ||
        (!(options & 16u) &&
         (selected_vendor == 0x10de || (selected_vendor == 0x1002 && selected_integrated)));
    phx_native_workgroup_rows =
        !phx_native_fp8_k && phx_native_matrix_rows == 16 && wide_workgroup && wide_matrix ? 64u
        : !phx_native_matrix_rows && (options & (1u << 21))                                ? 32u
                                                            : phx_native_matrix_rows;
    const char* selected_shader = !phx_native_matrix_rows           ? gemm
                                  : phx_native_fp8_k                ? fp8_matrix
                                  : phx_native_workgroup_rows == 64 ? wide_matrix
                                  : phx_native_matrix_rows == 16    ? matrix16
                                  : phx_native_matrix_rows == 8     ? matrix8
                                                                    : gemm;
    memcpy(phx_native_uuid, uuid, 33);
    // The RTX driver passed the complete finite conversion and temporal suites.
    // The available AMD iGPU failed a half-rounding contract; keep its fallback.
    phx_native_hardware_half = (options & (1u << 19)) != 0 && selected_vendor == 0x10de;
    phx_shape_matrices = (options & (1u << 18)) != 0;
    phx_wide_generic = (options & (1u << 20)) != 0;
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
    return phoenix_native_init_configured(uuid, gemm, gemm, gemm, unary, row, history, 1u, NULL,
                                          NULL);
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
    vkCmdDispatch(g.rcb, g.matrix_rows ? 1u : 4u, batches < 65535u ? batches : 65535u,
                  (batches + 65534u) / 65535u);
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
        VkPipeline matrix2_pipelines[] = {phx_matrix2_wide_pipeline, phx_matrix2_pipeline,
                                          phx_feed_forward_pipeline, phx_project_qkv_pipeline};
        for (unsigned index = 0; index < 4; ++index)
            if (matrix2_pipelines[index])
                vkDestroyPipeline(g.dev, matrix2_pipelines[index], NULL);
        phx_matrix2_wide_pipeline = phx_matrix2_pipeline = phx_feed_forward_pipeline =
            phx_project_qkv_pipeline = VK_NULL_HANDLE;
        if (phx_qkv_pipeline)
            vkDestroyPipeline(g.dev, phx_qkv_pipeline, NULL);
        phx_qkv_pipeline = VK_NULL_HANDLE;
        if (phx_features_pipeline)
            vkDestroyPipeline(g.dev, phx_features_pipeline, NULL);
        phx_features_pipeline = VK_NULL_HANDLE;
        if (phx_wide_pipeline)
            vkDestroyPipeline(g.dev, phx_wide_pipeline, NULL);
        phx_wide_pipeline = VK_NULL_HANDLE;
        if (phx_coalesced_pipeline)
            vkDestroyPipeline(g.dev, phx_coalesced_pipeline, NULL);
        phx_coalesced_pipeline = VK_NULL_HANDLE;
        phx_wide_columns = 0;
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
    for (unsigned iteration_index = 0; iteration_index < 14; ++iteration_index)
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
    phx_native_workgroup_compatibility = 0;
    phx_native_hardware_half = 0;
    phx_shape_matrices = 0;
    phx_wide_generic = 0;
    phx_native_matrix_rows = 0;
    phx_native_fp8_k = 0;
    phx_native_workgroup_rows = 0;
    phx_native_scalar_columns = 0;
    phx_native_skip = phx_native_gain = 0;
    phx_native_matrix2 = 0;
    phx_native_matrix2_shapes = 0;
    memset(phx_native_uuid, 0, sizeof(phx_native_uuid));
    memset(phx_native_view_offsets, 0, sizeof(phx_native_view_offsets));
}
