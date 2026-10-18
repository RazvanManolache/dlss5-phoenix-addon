"""Build DLSS5 Phoenix's private resident provider; leave imported source unmodified.

Requires locally unpacked Khronos headers/glslang, no global SDK installation.
The explicit replacements below are the auditable DLSS5 Phoenix patch to the pinned
Apache-2.0 runtime. Each anchor must match once or the build fails.
"""

from source_patches import code_count, replace_code

from pathlib import Path
import argparse, hashlib, json, os, re, shutil, subprocess

ROOT = Path(__file__).resolve().parents[1]
IMPORTED = ROOT / "third_party/intel_nr"


def imported_source_matches(path, expected, normalized_expected=None):
    """Preserve import pins across Git's LF/CRLF checkout conversion.

    These manifests describe text sources/notices, not executable/model bytes.
    Accept only byte-exact content or its line-ending equivalent; never bypass
    verification for edits to code, whitespace within lines or notice wording.
    """
    raw = path.read_bytes()
    candidates = [raw]
    if b"\0" not in raw:
        try:
            raw.decode("utf-8")
            normalized = raw.replace(b"\r\n", b"\n")
            if normalized_expected and hashlib.sha256(normalized).hexdigest() == normalized_expected:
                return True
            candidates += [normalized, normalized.replace(b"\n", b"\r\n")]
        except UnicodeDecodeError:
            pass
    return any(hashlib.sha256(candidate).hexdigest() == expected for candidate in candidates)


def patch_runtime(source):
    def replace(old, new):
        nonlocal source
        if code_count(source, old) != 1:
            raise RuntimeError(f"Patch anchor count: {old[:90]!r}")
        source = replace_code(source, old, new)

    replace(
        "#include <vulkan/vulkan.h>",
        '#include <vulkan/vulkan.h>\n#include "phoenix_devices.h"\nstatic int phx_native_configured;\nstatic char phx_native_uuid[33];\nstatic uint64_t phx_native_view_offsets[5];\nstatic unsigned phx_native_matrix_rows, phx_native_fp8_k, phx_native_workgroup_rows;\nstatic uint64_t phx_native_skip, phx_native_gain;\nstatic VkPipeline phx_wide_pipeline;\nstatic unsigned phx_wide_columns;',
    )
    replace("char* rpaths[12];", "char* rpaths[14];")
    replace("static VkPipeline phx_wide_pipeline;", "static VkPipeline phx_wide_pipeline, phx_coalesced_pipeline, phx_matrix2_pipeline, phx_matrix2_wide_pipeline;")
    replace("static int phx_native_configured;", "static unsigned phx_native_matrix2, phx_native_matrix2_shapes;\nstatic int phx_native_configured;")
    replace("static int phx_native_configured;", "static unsigned phx_native_workgroup_compatibility;\nstatic int phx_native_configured;")
    replace("static int phx_native_configured;", "static int phx_native_configured, phx_native_hardware_half, phx_shape_matrices, phx_wide_generic;")
    replace("static unsigned phx_native_matrix_rows,", "static unsigned phx_native_scalar_columns, phx_native_matrix_rows,")
    replace("specialization->mapEntryCount > sizeof entries / sizeof *entries - 2",
            "specialization->mapEntryCount > sizeof entries / sizeof *entries - 3")
    replace("specialization->dataSize > sizeof data - 2 * sizeof(VkBool32)",
            "specialization->dataSize > sizeof data - 3 * sizeof(VkBool32)")
    replace("VkSpecializationInfo constants = {", """VkBool32 native_half=phx_native_hardware_half?VK_TRUE:VK_FALSE;
    entries[count++]=(VkSpecializationMapEntry){.constantID=3,.offset=size,.size=sizeof native_half};
    memcpy(data+size,&native_half,sizeof native_half);
    size+=sizeof native_half;
    VkSpecializationInfo constants = {""")
    # Reading mapped NVIDIA VRAM through the CPU BAR was substantially slower
    # than a GPU copy into cached host storage. Keep the upstream environment
    # override and shared-memory AMD policy; change only our native NVIDIA route.
    replace(
        "if (!g.discrete) return 0;",
        """if (!g.discrete) return 0;
    if (phx_native_configured) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(g.pd,&properties);
        if(properties.vendorID==0x10de)return 1;
    }""",
    )
    # Unary and row shaders already use device addresses. Apply tensor byte
    # offsets directly to their push constants so one compact arena can back
    # every temporary, without copies or additional Vulkan buffer objects.
    views = """p.a+=phx_native_view_offsets[0];p.b+=phx_native_view_offsets[1];
    p.c+=phx_native_view_offsets[2];p.d+=phx_native_view_offsets[3];
    if(p.residual_cos)p.residual_cos+=phx_native_view_offsets[4];
    memset(phx_native_view_offsets,0,sizeof(phx_native_view_offsets));
    """
    replace(
        "VkPipeline pipeline;\n\tif (resident_pipeline(3, kind, g.runary, &pipeline))",
        views + "VkPipeline pipeline;\n\tif (resident_pipeline(3, kind, g.runary, &pipeline))",
    )
    replace(
        "unsigned row_stride = stride ? stride : width, groups = (rows + 31) / 32;",
        views + "unsigned row_stride = stride ? stride : width, groups = (rows + 31) / 32;",
    )
    # Streamed graphs have more bounded passes. Refuse timestamp truncation by
    # provisioning enough query slots for full-resolution per-layer attribution.
    replace("#define MAX_STAMPS 8192", "#define MAX_STAMPS 65536")
    # Mode 2 records only the start/end of a bounded submission. Detailed
    # operation queries remain explicitly available for offline profiling.
    replace("if (!g.prof || !g.qpool || g.prof_n >= MAX_STAMPS)",
            "if (!g.prof || !g.qpool || g.prof_n >= MAX_STAMPS || (g.prof == 2 && family != PK_START))")
    replace("g.prof = 1;", "g.prof = on == 2 ? 2 : 1;")
    replace("vkCmdResetQueryPool(g.rcb, g.qpool, 0, MAX_STAMPS);",
            "vkCmdResetQueryPool(g.rcb, g.qpool, 0, g.prof == 2 ? 2 : MAX_STAMPS);")
    replace("r = vkEndCommandBuffer(g.rcb);\n\tif (r) {", "if (g.prof == 2) stamp(PK_START, 0);\n    r = vkEndCommandBuffer(g.rcb);\n    if (r) {")

    replace(
        "VkDeviceSize size; int live, mapped;", "VkDeviceSize size, allocated; int live, mapped;"
    )
    replace(
        "static struct rbuf rbufs[MAX_RBUF];",
        """static struct rbuf rbufs[MAX_RBUF];
static uint64_t phx_resident_bytes, phx_resident_peak;
/* Resident buffer allocations only: excludes driver/pipeline and CPU storage. */
unsigned long long xmx_phoenix_memory(unsigned field) {
    if(field==1)return phx_resident_bytes;
    if(field==2)return phx_resident_peak;
    if(field==5)return g.stage.cap;
    uint64_t total=0;
    for(unsigned i=0;i<MAX_RBUF;++i)if(rbufs[i].live) {
        if(field==0)total+=rbufs[i].size;
        if(field==3)++total;
        if(field==4&&rbufs[i].mapped)total+=rbufs[i].allocated;
    }
    return total;
}""",
    )
    replace(
        "rb->live = 1;",
        """rb->allocated=mr.size;
    phx_resident_bytes+=mr.size;
    if(phx_resident_bytes>phx_resident_peak)phx_resident_peak=phx_resident_bytes;
    rb->live = 1;""",
    )
    replace(
        "rbufs[id] = (struct rbuf){ 0 };",
        """phx_resident_bytes-=rbufs[id].allocated;
    rbufs[id] = (struct rbuf){ 0 };""",
    )
    replace(
        "int ready, lost, discrete, unmapped;",
        "int ready, lost, discrete, unmapped, portable, matrix_rows, fp8_k;",
    )
    replace(
        "if (g.ready) return 0;",
        """if (g.ready) return 0;
    g.portable = phx_native_configured || (getenv("PHOENIX_VULKAN_GENERIC") && !strcmp(getenv("PHOENIX_VULKAN_GENERIC"),"1"));
    g.matrix_rows = phx_native_configured ? phx_native_matrix_rows : getenv("PHOENIX_VULKAN_MATRIX_ROWS") ? atoi(getenv("PHOENIX_VULKAN_MATRIX_ROWS")) : 0;
    g.fp8_k = phx_native_configured ? phx_native_fp8_k : getenv("PHOENIX_VULKAN_FP8_K") ? atoi(getenv("PHOENIX_VULKAN_FP8_K")) : 0;""",
    )
    replace(
        "g.pd = pds[0];\n\tfree(pds);",
        """g.pd = phx_select(pds,n,phx_native_configured?phx_native_uuid:getenv("PHOENIX_VULKAN_DEVICE"));
    free(pds);
    if (!g.pd) FAIL("Requested DLSS5 Phoenix Vulkan device UUID missing or ambiguous",0);""",
    )
    replace(
        "if (!(g.pin32 & 1))\n\t\tfprintf",
        'if (phx_native_workgroup_compatibility) g.pin32=0;\n'
        '    if (!(g.pin32 & 1) && !phx_native_workgroup_compatibility) FAIL("32-lane compute subgroup control required; opt in to vulkan_workgroup_compatibility for subgroup-independent shaders",0);\n'
        '    if (!(g.pin32 & 1) && !phx_native_workgroup_compatibility)\n\t\tfprintf',
    )
    replace(
        "float prio = 1.0f;",
        """VkPhysicalDeviceCooperativeMatrix2FeaturesNV matrix2={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV};
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    if (g.portable) {
        // Enable optional matrix/FP8 features only for the selected engine.
        v12.pNext=&v13; v13.pNext=NULL;
        if(g.matrix_rows) {
            unsigned shapes=phx_matrix_shapes(g.inst,g.pd);
            if(!((g.matrix_rows==8&&(shapes&1))||(g.matrix_rows==16&&(shapes&2)))) FAIL("Requested matrix configuration unsupported",0);
            v13.pNext=&cm;
        }
        v12.storageBuffer8BitAccess=VK_FALSE; v12.shaderInt8=VK_FALSE;
        if(g.fp8_k) {
            unsigned caps=phx_fp8_caps(g.inst,g.pd);
            unsigned bit=g.fp8_k==16?4:g.fp8_k==32?8:g.fp8_k==64?16:0;
            if(g.matrix_rows!=16||!bit||(caps&3)!=3||!(caps&bit))FAIL("Requested FP8 configuration unsupported",0);
            f8.shaderFloat8=VK_TRUE; f8.shaderFloat8CooperativeMatrix=VK_TRUE;
            cm.pNext=&f8; v12.storageBuffer8BitAccess=VK_TRUE; v12.shaderInt8=VK_TRUE;
        }
        if(phx_native_matrix2) {
            matrix2.cooperativeMatrixFlexibleDimensions=VK_TRUE;
            matrix2.cooperativeMatrixWorkgroupScope=VK_TRUE;
            matrix2.cooperativeMatrixTensorAddressing=VK_TRUE;
            matrix2.cooperativeMatrixConversions=VK_TRUE;
            matrix2.cooperativeMatrixPerElementOperations=VK_TRUE;
            cm.pNext=&f8;f8.pNext=&matrix2;
            f8.shaderFloat8=VK_TRUE;f8.shaderFloat8CooperativeMatrix=VK_TRUE;
            v12.storageBuffer8BitAccess=VK_TRUE;v12.shaderInt8=VK_TRUE;
        }
        VkPhysicalDeviceVulkan12Features have12={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features have11={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,.pNext=&have12};
        VkPhysicalDeviceFeatures2 have={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,.pNext=&have11};
        vkGetPhysicalDeviceFeatures2(g.pd,&have);
        if (!base.shaderInt64 || !have11.storageBuffer16BitAccess || !have12.shaderFloat16 ||
            !have12.bufferDeviceAddress || !have12.scalarBlockLayout || !have12.vulkanMemoryModel ||
            !have12.vulkanMemoryModelDeviceScope) FAIL("Missing portable Vulkan compute features",0);
    }
    float prio = 1.0f;""",
    )
    replace(
        'if (!g.preserve16 && !(denorm && !strcmp(denorm, "driver")))',
        'if (!phx_native_workgroup_compatibility && !g.preserve16 && !(denorm && !strcmp(denorm, "driver")))',
    )
    replace(
        "r = vkCreateDevice(g.pd, &dci, NULL, &g.dev);",
        """const char *portable_ext[4];
    if(g.portable) {
        unsigned count=0;
        if(g.matrix_rows)portable_ext[count++]=VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
        if(g.fp8_k||phx_native_matrix2)portable_ext[count++]=VK_EXT_SHADER_FLOAT8_EXTENSION_NAME;
        if(phx_native_matrix2)portable_ext[count++]=VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME;
        if(stats)portable_ext[count++]=VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME;
        dci.enabledExtensionCount=count; dci.ppEnabledExtensionNames=portable_ext;
    }
    r = vkCreateDevice(g.pd, &dci, NULL, &g.dev);""",
    )
    replace(
        "if (build_pipeline(spv_path, g.pl, &g.pipe))",
        "if (!g.portable && build_pipeline(spv_path, g.pl, &g.pipe))",
    )
    replace(
        "const char *paths[] = { gemm_spv, tiled_spv, staged_spv, unary_spv, row_spv };",
        """if(g.portable) {tiled_spv=gemm_spv; staged_spv=gemm_spv;}
    const char *paths[] = { gemm_spv, tiled_spv, staged_spv, unary_spv, row_spv };""",
    )
    replace(
        "g.staged32 = s32 ? (unsigned)atoi(s32) : 1;",
        """g.staged32 = s32 ? (unsigned)atoi(s32) : 1;
    if(g.portable) {g.tiling=0x7fffffffu;g.staging=0x7fffffffu;g.staged32=0;}""",
    )
    replace(
        "p.a += (uint64_t)oa * 2; p.b += (uint64_t)ob * 2;",
        "p.a += (uint64_t)oa * 2; p.b += (uint64_t)ob * ((bt&0x2000000u)?1:2);",
    )
    replace("p.a += (uint64_t)oa * 2;", """if (bt & 0x4000000u) {p.d=phx_native_skip; p.residual_cos=phx_native_gain;}
    phx_native_skip=phx_native_gain=0;
    p.a += (uint64_t)oa * 2;""")
    replace(
        "int small = g.staged32",
        """if(g.portable && (bt & ~0x3e011f01u)) FAIL("Unsupported fusion in portable GEMM",0);
    int small = g.staged32""",
    )
    replace("if (g.rstaged32[0]) return 0;", "if (g.portable || g.rstaged32[0]) return 0;")
    replace(
        "if (whole && g.rrows) {",
        """int parallel = g.portable && (((kind&255u)==0u && heads!=16u && !(kind&0x80000u)) ||
        ((kind&255u)==1u && width==64u && row_stride==64u)) || (kind&255u)==6u;
    if ((whole || parallel) && g.rrows) {""",
    )
    replace(
        "else       vkCmdDispatch(g.rcb, (N + 15) / 16, (M + 7) / 8, batch ? batch : 1);",
        """else {
            unsigned rows=coalesced?32u:wide?64u:phx_native_workgroup_rows?phx_native_workgroup_rows:g.matrix_rows?g.matrix_rows:8;
            uint64_t tiles=((uint64_t)M+rows-1)/rows*(batch?batch:1);
            unsigned y=tiles<65535u?(unsigned)tiles:65535u;
            if(!y||tiles>UINT32_MAX)FAIL("Portable GEMM grid exceeds indexing range",0);
            unsigned columns=coalesced?32u:wide?phx_wide_columns:
                !g.matrix_rows&&phx_native_scalar_columns?phx_native_scalar_columns:16u;
            vkCmdDispatch(g.rcb, (N+columns-1)/columns, y, (tiles+y-1)/y);
        }""",
    )
    replace(
        "VkPipeline pipeline;\n    if (small) {",
        """// The wider engine handles only packed model projections. Ordered
    // image boundaries and attention matrices retain their existing path.
    int wide=g.portable && phx_wide_pipeline && (bt&0x2000000u) && N>=phx_wide_columns;
    if(wide && phx_shape_matrices) {
        // Measured 1080p winners. Larger tiles lose on other projections,
        // notably the strided per-head contractions and wide expansions.
        unsigned publication=(bt>>8u)&15u;
        wide=(K==32u && N<=128u) ||
             (K==64u && (N==64u || N==192u || (N==256u && publication==6u))) ||
             (K>=2048u && N<=1024u);
    }
    // Coalesced weight loads help expanding packed projections and large
    // 128-to-32 contractions. Other shapes keep the lower-overhead tile.
    // This choice changes memory access only; both shaders retain ordered FMA.
    int coalesced=g.portable && !g.matrix_rows && phx_coalesced_pipeline &&
        (bt&0x2000000u) && ((K>=128u && N>K) || (K==128u && N==32u && M>=16384u));
    // The adapter proves native FP8 weight encoding (bit 29). Legacy E4M3
    // bytes 0x7f/0xff decode as +/-480 and must never become hardware NaNs.
    int matrix2_route=phx_matrix2_pipeline && K>=256u && N>=64u &&
        (bt&0x32000001u)==0x32000001u && K%16u==0u && lda%8u==0u && ldb%16u==0u &&
        (batch<=1u || (sa%8u==0u && sb%16u==0u)) &&
        (p.a&15u)==0u && (p.b&15u)==0u;
    VkPipeline pipeline;
    int matrix2_wide=matrix2_route && phx_matrix2_wide_pipeline && N>=128u && M>=512u;
    if(matrix2_route) {pipeline=matrix2_wide?phx_matrix2_wide_pipeline:phx_matrix2_pipeline;} else if(coalesced) {
        if(resident_pipeline(13,bt,phx_coalesced_pipeline,&pipeline))return -1;
    } else if(wide && phx_wide_generic) {
        pipeline=phx_wide_pipeline;
    } else if(wide) {
        if(resident_pipeline(12,bt,phx_wide_pipeline,&pipeline))return -1;
    } else if (small) {""",
    )
    for old, groups, planes in [
        ("vkCmdDispatch(g.rcb, (n + 255) / 256, 1, 1);", "((uint64_t)n+255)/256", "1"),
        ("vkCmdDispatch(g.rcb, groups, 1, 1);", "groups", "1"),
        ("vkCmdDispatch(g.rcb, (rows + 31) / 32, 3, 1);", "((uint64_t)rows+31)/32", "3"),
        ("vkCmdDispatch(g.rcb, (pixels + 63) / 64, 1, 1);", "((uint64_t)pixels+63)/64", "1"),
    ]:
        replace(
            old,
            f'{{ uint64_t groups_x={groups}; unsigned x=groups_x<65535u?(unsigned)groups_x:65535u; if(!x)FAIL("Empty dispatch",0); vkCmdDispatch(g.rcb,x,{planes},(groups_x+x-1)/x); }}',
        )
    # Validate every dispatch, including unused upstream fusion entry points.
    source, count = re.subn(
        r"vkCmdDispatch\(([^;]+)\);", r"{ if(phx_dispatch(\1))return -1; }", source
    )
    if count != 16:
        raise RuntimeError(f"Dispatch guard anchors: {count}")
    replace(
        "static uint64_t phx_resident_bytes, phx_resident_peak;",
        """static unsigned phx_limits[3], phx_max_dispatch[3], phx_rejected_dispatches;
static int phx_dispatch(VkCommandBuffer cmd,uint64_t x,uint64_t y,uint64_t z) {
    uint64_t dims[3]={x,y,z};
    for(unsigned i=0;i<3;++i)if(dims[i]>phx_limits[i]) {
        ++phx_rejected_dispatches;
        snprintf(g.err,sizeof g.err,"Dispatch axis %u exceeds device limit (%llu > %u)",i,(unsigned long long)dims[i],phx_limits[i]);
        return -1;
    }
    for(unsigned i=0;i<3;++i)if(dims[i]>phx_max_dispatch[i])phx_max_dispatch[i]=(unsigned)dims[i];
    vkCmdDispatch(cmd,(unsigned)x,(unsigned)y,(unsigned)z);return 0;
}
unsigned xmx_phoenix_dispatch_info(unsigned field) {
    return field<3?phx_limits[field]:field<6?phx_max_dispatch[field-3]:field==6?phx_rejected_dispatches:0;
}
static uint64_t phx_resident_bytes, phx_resident_peak;""",
    )
    replace(
        'snprintf(g.name, sizeof g.name, "%s", props.deviceName);',
        'memcpy(phx_limits,props.limits.maxComputeWorkGroupCount,sizeof phx_limits);\n\tsnprintf(g.name, sizeof g.name, "%s", props.deviceName);',
    )
    replace('if (staged) {\n        unsigned rows', '''if(matrix2_route) {
        unsigned tile_rows=matrix2_wide?64u:32u;
        uint64_t tiles=((uint64_t)M+tile_rows-1u)/tile_rows*(batch?batch:1u);
        unsigned y=tiles<65535u?(unsigned)tiles:65535u;
        if(!y || phx_dispatch(g.rcb,(N+63u)/64u,y,(tiles+y-1u)/y))return -1;
    } else if (staged) {
        unsigned rows''')
    return (
        "// DLSS5 Phoenix integration patch; see tools/build_vulkan.py and upstream NOTICE.\n"
        + source
    )


def prepare_sources(out):
    """Generate the same provider/shaders for Windows, Linux and Android builds."""
    out.mkdir(parents=True, exist_ok=True)
    gpu = IMPORTED / "src/gpu"
    own = ROOT / "src/backends/vulkan"
    manifest = json.loads((IMPORTED / "import_manifest.json").read_text())
    amd = ROOT / "third_party/amd_nr"
    amd_manifest = json.loads((amd / "import_manifest.json").read_text())
    for entry in amd_manifest["files"]:
        if not imported_source_matches(amd / entry["path"], entry["sha256"], entry.get("normalized_lf_sha256")):
            raise RuntimeError("Imported AMD file changed: " + entry["path"])
    for entry in manifest["files"]:
        if not imported_source_matches(IMPORTED / entry["path"], entry["sha256"], entry.get("normalized_lf_sha256")):
            raise RuntimeError("Imported file changed: " + entry["path"])
    runtime_source = (
        patch_runtime((gpu / "libxmx.c").read_text())
        + "\n"
        + (ROOT / "native_runtime/vulkan_lifetime.inc").read_text()
    )
    (out / "phoenix_vulkan.c").write_text(runtime_source)
    names = sorted(
        set(re.findall(r"(?m)^(?!static)[^\n]*?\b(xmx_\w+)\s*\([^;]*\)\s*\{", runtime_source))
    )
    names += [
        "phoenix_native_matrix2",
        "phoenix_device_count",
        "phoenix_device_info",
        "phoenix_device_info_options",
        "phoenix_matrix_shapes",
        "phoenix_fp8_capabilities",
        "phoenix_native_init",
        "phoenix_native_init_configured",
        "phoenix_native_matrix",
        "phoenix_native_device_id",
        "phoenix_native_features_init",
        "phoenix_control_map_abi",
        "phoenix_native_features",
        "phoenix_native_matrix_residual",
        "phoenix_native_window_attention",
        "phoenix_native_operand_offsets",
        "phoenix_native_vendor",
        "phoenix_native_wide_init",
        "phoenix_native_coalesced_init",
    ]
    (out / "provider.def").write_text("LIBRARY phoenix_vulkan\nEXPORTS\n" + "\n".join(names) + "\n")
    original_gpu = gpu
    gpu = out / "shader_sources"
    gpu.mkdir(exist_ok=True)
    for source_path in [
        *original_gpu.glob("*.comp"),
        *original_gpu.glob("*.glsl"),
        *own.glob("*.comp"),
    ]:
        shutil.copyfile(source_path, gpu / source_path.name)
    shutil.copyfile(own / "phoenix_fp8.glsl", gpu / "phoenix_fp8.glsl")
    shutil.copyfile(own / "packed_weight.glsl", gpu / "packed_weight.glsl")
    publish = (gpu / "publish.glsl").read_text()
    old = "float half_round(float x) {\n    return HALF_BY_CAST ? float(float16_t(x)) : unpackHalf2x16(packHalf2x16(vec2(x, 0.0))).x;\n}"
    if code_count(publish, old) != 1:
        raise RuntimeError("half_round patch anchor")
    publish = replace_code(publish, old, (own / "half_round.glsl").read_text())
    publish = replace_code(
        publish,
        "if (epilogue == 1u || epilogue == 3u) value = e4m3(value);",
        "if (epilogue == 5u) value = half_round(value);\n    if (epilogue == 1u || epilogue == 3u || epilogue == 5u) value = e4m3(value);",
    )
    (gpu / "publish.glsl").write_text(publish)
    resident = replace_code(
        (gpu / "resident.comp").read_text(),
        "if (epilogue == 1u || epilogue == 3u) value = e4m3(value);",
        "if (epilogue == 5u) value = half_round(value);\n    if (epilogue == 1u || epilogue == 3u || epilogue == 5u) value = e4m3(value);",
    )
    (gpu / "resident.comp").write_text(resident)
    # The independent reference rounds ALL ViT Q/K/V projections to half before
    # publishing; the upstream resident V path omitted that rounding point.
    python_source = (original_gpu / "nr_resident.py").read_text()
    pattern = r"(runtime\.split_heads\(\s*s\.proj,\s*s\.v16,[\s\S]*?epilogue=)xmxres\.EPI_E4M3"
    python_source, count = re.subn(pattern, r"\g<1>(5 if vit else xmxres.EPI_E4M3)", python_source)
    if count != 2:
        raise RuntimeError("ViT V rounding patch anchor")
    (out / "nr_resident.py").write_text(python_source)
    frame = (original_gpu / "nr_frame_resident.py").read_text()
    edits = {
        "self.split = (0.0, 0.0, 0.0)": "self.block_repeats = (1,)*8\n        self.split = (0.0, 0.0, 0.0)",
        "plan = (self.rt.scratch_key(), capture is not None)": "plan = (self.rt.scratch_key(), capture is not None, tuple(self.block_repeats))",
        "key = rt.graph_key()": "key = (rt.graph_key(), tuple(self.block_repeats))",
        'for index in range(31, 39):\n            block = self.block(index, 32, "global")': 'for index in (i for i in range(31, 39) for _ in range(self.block_repeats[i-31])):\n            block = self.block(index, 32, "global")',
        "    def run(self, features, submits=None, capture=None, timing=None, execution=None):\n        try:": '    def run(self, features, submits=None, capture=None, timing=None, execution=None):\n        if len(self.block_repeats)!=8 or any(type(n) is not int or not 0<=n<=4 for n in self.block_repeats):\n            raise ValueError("Eight block counts in 0..4 required")\n        try:',
    }
    for old, new in edits.items():
        if code_count(frame, old) != 1:
            raise RuntimeError("Resident frame patch anchor: " + old)
        frame = replace_code(frame, old, new)
    (out / "nr_frame_resident.py").write_text(frame)
    from vulkan_phoenix_math import patch

    patch(gpu, out, own)
    # Spread linear work over X/Z, leaving Y available for QKV's three planes.
    # Tail workgroups must return before shared-memory gathers subtract an
    # out-of-range base from an unsigned row count.
    for name in ("resident.comp", "attention.comp", "history.comp"):
        path = gpu / name
        text = path.read_text()
        text = replace_code(
            text,
            "gl_GlobalInvocationID.x",
            "(gl_GlobalInvocationID.x + gl_WorkGroupID.z*gl_NumWorkGroups.x*gl_WorkGroupSize.x)",
        )
        if name == "attention.comp":
            text = replace_code(
                text, "gl_WorkGroupID.x", "(gl_WorkGroupID.x + gl_WorkGroupID.z*gl_NumWorkGroups.x)"
            )
            anchor = "uint kind = operation_flags() & 0xFFu, local = gl_LocalInvocationID.x;"
            if code_count(text, anchor) != 1:
                raise RuntimeError("Attention tail guard anchor")
            text = replace_code(
                text,
                anchor,
                anchor
                + "\n    uint row_stride=pc.sa!=0u?pc.sa:pc.n;\n    uint rows_per_group=(kind==SOFTMAX&&(row_stride!=pc.n||row_stride>64u)&&row_stride+1u<=WHOLE)?min(32u,WHOLE/(row_stride+1u)):32u;\n    if((gl_WorkGroupID.x+gl_WorkGroupID.z*gl_NumWorkGroups.x)*rows_per_group>=pc.m)return;",
            )
            text = replace_code(
                text,
                "#extension GL_KHR_shader_subgroup_basic : require",
                "#ifndef PHX_WORKGROUP_COMPAT\n#extension GL_KHR_shader_subgroup_basic : require\n#extension GL_KHR_shader_subgroup_shuffle : require\n#endif",
            )
            text = replace_code(
                text, "void main() {", "#ifndef PHX_WORKGROUP_COMPAT\n" +
                (own / "phoenix_rows.glsl").read_text() + "\n#endif\nvoid main() {"
            )
            # Place after main's kind declaration, never inside gather_part.
            text = replace_code(
                text,
                anchor,
                anchor
                + "\n#if ROW_LANES == 256\n    if(kind==VIT_SOFTMAX) {\n        uint r=(gl_WorkGroupID.x+gl_WorkGroupID.z*gl_NumWorkGroups.x)*32u+local/8u;\n        if(r<pc.m)phx_vit_parallel(r,local%8u);\n        return;\n    }\n    if((kind==COSINE_PUBLISH&&pc.batch!=16u&&(operation_flags()&0x80000u)==0u)||\n       (kind==SOFTMAX&&pc.n==64u&&(pc.sa==0u||pc.sa==64u))) {\n        uint r=(gl_WorkGroupID.x+gl_WorkGroupID.z*gl_NumWorkGroups.x)*32u+local/8u;\n        if(r<pc.m){if(kind==COSINE_PUBLISH)phx_norm_parallel(r,local%8u);else phx_softmax_parallel(r,local%8u);}\n        return;\n    }\n#endif",
            )
        path.write_text(text)
    # Scalar row workgroups use local invocation IDs, never hardware subgroup IDs.
    # Workgroup barriers make their shared gathers/scatters safe even when the
    # hardware subgroup is larger than this 32-invocation workgroup. Keep the
    # existing ordered arithmetic; no subgroup shuffles or matrix collectives.
    attention = gpu / "attention.comp"
    text = attention.read_text()
    text = text.replace("subgroupMemoryBarrierShared();\n    subgroupBarrier();",
                        "#ifdef PHX_WORKGROUP_COMPAT\n    barrier();\n#else\n"
                        "    subgroupMemoryBarrierShared();\n    subgroupBarrier();\n#endif")
    attention.write_text(text)
    for source_path in [*gpu.glob("*.comp"), *gpu.glob("*.glsl")]:
        if source_path.name == "publish.glsl":
            continue
        text = replace_code(
            replace_code(source_path.read_text(), "unpackHalf2x16(", "phoenix_unpack_half("),
            "packHalf2x16(",
            "phoenix_pack_half(",
        )
        # Pin rounding before scalar half stores as well. Some AMD conversion
        # paths otherwise truncate an unquantized activation toward zero.
        if '#include "publish.glsl"' in text:
            text = re.sub(r"float16_t\(([^()]+)\)", r"float16_t(half_round(\1))", text)
        source_path.write_text(text)
    # Generated compatibility sources from an earlier incremental build are
    # re-created below, never discovered as additional standard entry points.
    shaders = {source_path.stem: (source_path, []) for source_path in gpu.glob("*.comp")
               if not source_path.stem.endswith('_workgroup') and source_path.stem != 'gemm_mobile'}
    shaders["phoenix_matrix2_wide"] = (gpu / "phoenix_matrix2.comp", ["-DBLOCK_ROWS=64"])
    shaders.pop("gemm_wide", None)  # Only the named 32/64-column variants are loaded.
    for reduction_size in (16, 32, 64):
        shaders[f"gemm_fp8{reduction_size}"] = (
            gpu / "gemm_matrix.comp",
            ["-DMATRIX_M=16", f"-DFP8_K={reduction_size}"],
        )
    shaders.update(
        gemm_portable_coalesced=(gpu / "gemm_portable_tiled.comp", ["-DCOALESCED_WEIGHTS=1"]),
        gemm_wide32=(gpu / "gemm_wide.comp", ["-DWIDE_COLUMNS=32"]),
        gemm_wide64=(gpu / "gemm_wide.comp", ["-DWIDE_COLUMNS=64"]),
        gemm_tiled=(gpu / "gemm_resident.comp", ["-DRM=2", "-DRN=2"]),
        phoenix_window_attention8=(gpu / "phoenix_window_attention.comp", ["-DPHX_ATTENTION_M=8"]),
        gemm_matrix8=(gpu / "gemm_matrix.comp", ["-DMATRIX_M=8"]),
        gemm_matrix16=(gpu / "gemm_matrix.comp", ["-DMATRIX_M=16"]),
        gemm_matrix16x4=(gpu / "gemm_matrix.comp", ["-DMATRIX_M=16", "-DMATRIX_WARPS=4"]),
        gemm_staged32=(gpu / "gemm_staged.comp", ["-DSTAGED_BM=32"]),
        gemm_staged32_deep=(gpu / "gemm_staged.comp", ["-DSTAGED_BM=32", "-DSTAGED_BK=64"]),
        attention_rows=(gpu / "attention.comp", ["-DROW_LANES=256"]),
    )
    from vulkan_compatibility import add_shaders
    add_shaders(gpu, shaders)
    return shaders, manifest, amd_manifest


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("output")
    argument_parser.add_argument("deps", type=Path)
    args = argument_parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_]+", args.output):
        raise ValueError("build name")
    out = ROOT / "build" / args.output
    out.mkdir(parents=True, exist_ok=False)
    deps = args.deps.resolve()
    shaders, manifest, amd_manifest = prepare_sources(out)
    own = ROOT / "src/backends/vulkan"
    gpu = out / "shader_sources"
    # Generate an import library for the installed Vulkan loader using its exports.
    vswhere = (
        Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    visual_studio_path = subprocess.check_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    loader = Path(os.environ["WINDIR"]) / "System32/vulkan-1.dll"
    cmd = f'''@echo off
call "{visual_studio_path}\\VC\\Auxiliary\\Build\\vcvars64.bat" > "{out}/toolchain.log" 2>&1
if errorlevel 1 exit /b 1
dumpbin /exports "{loader}" > "{out}/loader_exports.txt"
"{os.sys.executable}" "{ROOT}/tools/build_vulkan.py" --loader-def "{out}"
if errorlevel 1 exit /b 1
lib /nologo /def:"{out}/loader.def" /machine:x64 /out:"{out}/vulkan-1.lib"
if errorlevel 1 exit /b 1
cl /nologo /O2 /std:c11 /TC /MD /D_CRT_SECURE_NO_WARNINGS /I"{deps}/headers/include" /I"{own}" /Fo:"{out}/provider.obj" /Fe:"{out}/phoenix_vulkan.dll" /LD "{out}/phoenix_vulkan.c" /link /DEF:"{out}/provider.def" "{out}/vulkan-1.lib"
exit /b %ERRORLEVEL%
'''
    (out / "build.cmd").write_text(cmd)
    with (out / "build.log").open("w") as log:
        subprocess.run(
            ["cmd", "/c", str(out / "build.cmd")], stdout=log, stderr=subprocess.STDOUT, check=True
        )
    compiler = deps / "glslang/bin/glslang.exe"
    with (out / "shaders.log").open("w") as log:
        for name, (path, defs) in shaders.items():
            subprocess.run(
                [
                    str(compiler),
                    "--target-env",
                    "vulkan1.3",
                    "-I" + str(gpu),
                    *defs,
                    "-o",
                    str(out / (name + ".spv")),
                    str(path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    pins = {
        str(source_path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        for source_path in [
            Path(__file__),
            Path(__file__).with_name("vulkan_phoenix_math.py"),
            Path(__file__).with_name("vulkan_compatibility.py"),
            *own.glob("*"),
            *out.glob("*.dll"),
            *out.glob("*.spv"),
            out / "phoenix_vulkan.c",
            out / "nr_resident.py",
            out / "nr_frame_resident.py",
        ]
    }
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                pins=pins,
                import_commit=manifest["commit"],
                arithmetic="phoenix-reference-v1",
                matrix_engine="fp16-fp32-v1",
                matrix_default_vendors=[0x1002, 0x10DE],
                fp8_engine="lossless-tile-v1",
                packed_e4m3_weights=True,
                weight_decode_workspace=True,
                streamed_window=True,
                compact_head=True,
                nvidia_window_attention=True,
                matrix_window_attention8=True,
                amd_import_commit=amd_manifest["commit"],
            ),
            indent=2,
        )
        + "\n"
    )
    print(out)


if __name__ == "__main__":
    if len(os.sys.argv) == 3 and os.sys.argv[1] == "--loader-def":
        out = Path(os.sys.argv[2])
        text = (out / "loader_exports.txt").read_text()
        names = re.findall(r"^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(vk\w+)", text, re.M)
        if not names:
            raise RuntimeError("No Vulkan loader exports")
        (out / "loader.def").write_text("LIBRARY vulkan-1.dll\nEXPORTS\n" + "\n".join(names) + "\n")
    else:
        main()
