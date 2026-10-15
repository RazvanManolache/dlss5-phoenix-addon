"""Generate subgroup-independent shaders with bit-preserving half storage.

Some mobile drivers flush float16 subnormals even at a storage conversion.
Use 16-bit integer storage and the existing integer half converters instead.
The native/32-lane shaders retain their original types and scheduling.
"""
import re
from pathlib import Path


def parallel_rows(source):
    """Eight local invocations per row, exchanging through workgroup memory.

    This retains the scalar reduction tree without depending on subgroup size
    or subgroup-to-local invocation mapping. Tail rows participate in every
    barrier using a valid source row and suppress their writes.
    """
    start = source.index('#ifndef PHX_WORKGROUP_COMPAT\n// Eight lanes')
    end = source.index('\n#endif\nvoid main()', start)
    rows = source[start:end].replace('#ifndef PHX_WORKGROUP_COMPAT', '#if 1', 1)
    old = '''float phx_lane(float value, uint lane) {
    return subgroupShuffle(value, (gl_SubgroupInvocationID & ~7u) + lane);
}'''
    assert old in rows
    rows = rows.replace(old, '''shared float row_exchange[3 * 256];
void phx_exchange(uint slot, float value) {
    barrier(); // finish every reader before reusing this exchange plane
    row_exchange[slot * 256u + gl_LocalInvocationID.x] = value;
    barrier();
}
float phx_lane(uint slot, uint lane) {
    return row_exchange[slot * 256u + (gl_LocalInvocationID.x & ~7u) + lane];
}''')
    rows = rows.replace('precise float even_sum = half_round((phx_lane(partial_sum,',
                        'phx_exchange(0u, partial_sum);\n    precise float even_sum = half_round((phx_lane(partial_sum,')
    rows = rows.replace('precise float even_total =',
                        'phx_exchange(0u, even_sum); phx_exchange(1u, odd_sum);\n        precise float even_total =')
    rows = rows.replace('float parity_sum =',
                        'phx_exchange(0u, even_sum); phx_exchange(1u, odd_sum);\n        float parity_sum =')
    rows = rows.replace('total = hadd(hadd(phx_lane(parity_sum,',
                        'phx_exchange(2u, parity_sum);\n        total = hadd(hadd(phx_lane(parity_sum,')
    rows = rows.replace('float even = phx_denominator_add(phx_lane(column,',
                        'phx_exchange(0u, column);\n        float even = phx_denominator_add(phx_lane(column,')
    for value,slot in [('partial_sum',0),('even_sum',0),('odd_sum',1),('parity_sum',2),('column',0)]:
        rows = rows.replace('phx_lane('+value+',', 'phx_lane('+str(slot)+'u,')
    rows = rows.replace('if (lane == 0u)', 'if (lane == 0u && phx_row_active)')
    source = source[:start] + rows + source[end:]
    source = source.replace('void store(uint index, float value) {',
                            'bool phx_row_active = true;\nvoid store(uint index, float value) {\n    if(!phx_row_active)return;')
    source = source.replace('if(r<pc.m)phx_vit_parallel(r,local%8u);',
                            'phx_row_active=r<pc.m; phx_vit_parallel(min(r,pc.m-1u),local%8u);')
    source = source.replace('if(r<pc.m){if(kind==COSINE_PUBLISH)phx_norm_parallel(r,local%8u);else phx_softmax_parallel(r,local%8u);}',
                            'phx_row_active=r<pc.m; r=min(r,pc.m-1u); if(kind==COSINE_PUBLISH)phx_norm_parallel(r,local%8u);else phx_softmax_parallel(r,local%8u);')
    return source


def software_half_storage(source):
    source = source.replace(
        "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require",
        "#extension GL_EXT_shader_16bit_storage : require")
    source = source.replace("float16_t v[];", "uint16_t v[];")
    # Every half-buffer read in these scalar shaders explicitly converts to FP32.
    # The pattern allows nested brackets in the index and constructors in the base.
    read = r"float\(((?:HalfBuf\([^;\n]+?\)|pc\.[ab]|h)\.v\[[^;\n]+?\])\)"
    source = re.sub(read, r"phoenix_half_value(uint(\1))", source)
    if re.search(r"float\((?:HalfBuf\(|pc\.[ab]\.v\[|h\.v\[)", source):
        raise RuntimeError("Unconverted compatibility half-buffer read")
    # Parse balanced constructor arguments, rather than matching nested arithmetic
    # with a regex. These constructors are all half-buffer stores, not arithmetic.
    cursor = 0
    while (start := source.find("float16_t(", cursor)) >= 0:
        argument = start + len("float16_t(")
        depth, end = 1, argument
        while depth and end < len(source):
            depth += (source[end] == '(') - (source[end] == ')')
            end += 1
        if depth:
            raise RuntimeError("Unbalanced compatibility half store")
        value = source[argument:end-1]
        replacement = "uint16_t(phoenix_half_bits(" + value + "))"
        source = source[:start] + replacement + source[end:]
        cursor = start + len(replacement)
    if "float16_t" in source:
        raise RuntimeError("Compatibility shader still uses float16 storage/arithmetic")
    return source


def add_shaders(gpu, shaders):
    own = Path(__file__).resolve().parents[1] / 'src/backends/vulkan'
    (gpu / 'phoenix_ieee_math.glsl').write_text((own / 'phoenix_ieee_math.glsl').read_text())
    (gpu / 'publish_workgroup.glsl').write_text('#include "phoenix_ieee_math.glsl"\n' +
        (gpu / 'publish.glsl').read_text().replace('fma(', 'phoenix_ieee_fma('))
    cosine = (gpu / 'cosine_tree.glsl').read_text()
    cosine = cosine.replace('inversesqrt(norm)', 'phoenix_ieee_divide(1.0, phoenix_ieee_sqrt(norm))')
    cosine = cosine.replace('1.0 / sqrt(norm)', 'phoenix_ieee_divide(1.0, phoenix_ieee_sqrt(norm))')
    (gpu / 'cosine_tree_workgroup.glsl').write_text(cosine)
    for name in ("resident", "attention", "gemm_portable", "gemm_portable_tiled", "frame_features"):
        source = software_half_storage((gpu / (name + ".comp")).read_text())
        source = source.replace('#include "publish.glsl"', '#include "publish_workgroup.glsl"')
        if name in ('attention', 'resident'):
            source = source.replace('fma(', 'phoenix_ieee_fma(')
        if name == 'attention':
            source = source.replace('#include "cosine_tree.glsl"', '#include "cosine_tree_workgroup.glsl"')
            source = source.replace('1.0 / denominator', 'phoenix_ieee_divide(1.0, denominator)')
            source = source.replace('sqrt(norm)', 'phoenix_ieee_sqrt(norm)')
            source = source.replace('error / (2.0 * root)', 'phoenix_ieee_divide(error, 2.0 * root)')
            source = source.replace('float total = 0.0;', 'precise float total = 0.0;')
            source = source.replace('phoenix_ieee_fma(head_values[iteration_index], head_values[iteration_index], total)',
                                    'phoenix_square_add(head_values[iteration_index], total)')
            # For every positive finite half norm above the model's floor, the
            # CPU's Newton steps leave correctly rounded sqrt/division unchanged.
            # Native mobile fma residuals can instead move that correct answer.
            source = source.replace(
                'precise float inverse = phoenix_ieee_divide(1.0, denominator);\n'
                '    precise float error = phoenix_ieee_fma(-inverse, denominator, 1.0);\n'
                '    return phoenix_ieee_fma(error, inverse, inverse);',
                'return phoenix_ieee_divide(1.0, denominator);')
            source = source.replace(
                'precise float root = phoenix_ieee_sqrt(norm);\n'
                '    precise float error = phoenix_ieee_fma(-root, root, norm);\n'
                '    root += phoenix_ieee_divide(error, 2.0 * root);\n'
                '    return phoenix_reciprocal(root);',
                'return phoenix_ieee_divide(1.0, phoenix_ieee_sqrt(norm));')
            source = source.replace('inversesqrt(max(squared_sums[0], 0.00006198883056640625))',
                                    'phoenix_ieee_divide(1.0, phoenix_ieee_sqrt(max(squared_sums[0], 0.00006198883056640625)))')
            source = source.replace('1.0 / half_round(total)', 'phoenix_ieee_divide(1.0, half_round(total))')
            source = source.replace('1.0 / total', 'phoenix_ieee_divide(1.0, total)')
            source = source.replace('1. / total', 'phoenix_ieee_divide(1.0, total)')
        source = source.replace('#include "packed_weight.glsl"',
                                '#include "packed_weight_workgroup.glsl"')
        destination = gpu / (name + "_workgroup.comp")
        destination.write_text(source)
        shaders[name + "_workgroup"] = (destination, ["-DPHX_WORKGROUP_COMPAT=1"])
    (gpu / "packed_weight_workgroup.glsl").write_text(
        software_half_storage((gpu / "packed_weight.glsl").read_text()))
    # Both resident routes use the measured 32x32 mobile register tile. The
    # grid and packet ABI stay identical to the old scalar tiled implementation.
    mobile = gpu / 'gemm_mobile_workgroup.comp'
    mobile.write_text((own / 'gemm_mobile.glsl').read_text())
    for name in ('gemm_portable_tiled_workgroup', 'gemm_portable_coalesced_workgroup'):
        shaders[name] = (mobile, ['-DPHX_WORKGROUP_COMPAT=1', '-Os'])
    parallel = gpu / 'attention_parallel_workgroup.comp'
    parallel.write_text(parallel_rows((gpu / 'attention_workgroup.comp').read_text()))
    shaders['attention_parallel_workgroup'] = (parallel, ['-DPHX_WORKGROUP_COMPAT=1', '-DROW_LANES=256'])
