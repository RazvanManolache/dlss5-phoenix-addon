// Vulkan buffer/graph lifetime helpers included into the generated provider translation unit.
// Keep fence completion and reference ownership paired before releasing GPU-visible allocations.

/* Explicit C++ initialization avoids process environment and CRT snapshots. */
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
int phoenix_native_init(const char* uuid, const char* gemm, const char* unary, const char* row,
                        const char* history) {
    if (!uuid || strlen(uuid) != 32 || g.ready)
        return -1;
    memcpy(phx_native_uuid, uuid, 33);
    phx_native_configured = 1;
    int result = xmx_init(gemm);
    if (result < 0)
        return result;
    return xmx_res_init(gemm, unary, row, history, gemm, gemm);
}
void xmx_shutdown(void) {
    if (g.dev) {
        vkDeviceWaitIdle(g.dev);
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
    memset(phx_native_uuid, 0, sizeof(phx_native_uuid));
    memset(phx_native_view_offsets, 0, sizeof(phx_native_view_offsets));
}
