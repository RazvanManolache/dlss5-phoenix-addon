"""Explicit approximate mobile path, generated separately from reference shaders.

The graph and quantization grids remain unchanged. Native half conversions may
flush subnormals; FP32 normalization/publication uses driver arithmetic. This is
an opt-in quality/performance tradeoff, never a bit-equivalence claim.
"""
import re


def function_body(source, signature, replacement):
    start = source.index(signature)
    body = source.index('{', start)
    end, depth = body + 1, 1
    while depth:
        if end >= len(source):
            raise ValueError('Unbalanced GLSL function: ' + signature)
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[:body] + '{\n    ' + replacement + '\n}' + source[end:]


def approximate(source):
    source = re.sub(r'\bprecise\s+', '', source)
    source = source.replace('phoenix_ieee_fma(', 'fma(')
    source = source.replace('phoenix_square_add(', 'phx_fast_square_add(')
    source = source.replace('phoenix_ieee_divide(', 'phx_fast_divide(')
    source = source.replace('phoenix_ieee_sqrt(', 'sqrt(')
    return source


def add_fast_shaders(gpu, shaders):
    publish = (gpu/'publish_workgroup.glsl').read_text()
    publish = publish.replace('#include "phoenix_ieee_math.glsl"', '''// Approximate hardware arithmetic; no software IEEE normalization.
float phx_fast_divide(float a, float b) { return a / b; }
float phx_fast_square_add(float a, float sum) { return fma(a, a, sum); }''')
    publish = function_body(publish, 'float half_round(float value)',
                            'return unpackHalf2x16(packHalf2x16(vec2(value, 0.0))).x;')
    publish = function_body(publish, 'uint phoenix_half_bits(float value)',
                            'return packHalf2x16(vec2(value, 0.0)) & 65535u;')
    publish = function_body(publish, 'float phoenix_half_value(uint bits)',
                            'return unpackHalf2x16(bits & 65535u).x;')
    (gpu/'publish_fast_workgroup.glsl').write_text(approximate(publish))
    (gpu/'cosine_tree_fast_workgroup.glsl').write_text(
        approximate((gpu/'cosine_tree_workgroup.glsl').read_text()))
    names = [name for name in shaders if name.endswith('_workgroup')]
    for name in names:
        source, defines = shaders[name]
        text = approximate(source.read_text())
        text = text.replace('publish_workgroup.glsl', 'publish_fast_workgroup.glsl')
        text = text.replace('cosine_tree_workgroup.glsl', 'cosine_tree_fast_workgroup.glsl')
        if name in ('gemm_portable_tiled_workgroup', 'gemm_portable_coalesced_workgroup'):
            # 128 invocations / eight outputs each reduces register pressure;
            # 32x32 dispatch tiles keep the existing descriptor/grid ABI.
            text = text.replace('local_size_x = 64', 'local_size_x = 128')
            text = text.replace('e+=64u', 'e+=128u')
            text = text.replace('lane/8u', 'lane/16u').replace('lane%8u)*4u', 'lane%16u)*2u')
            text = text.replace('i<4u', 'i<2u').replace('vec4', 'vec2')
            text = text.replace('weight_tile[at+1u],weight_tile[at+2u],weight_tile[at+3u]', 'weight_tile[at+1u]')
            text = text.replace('#version 450', '#version 450\n#extension GL_EXT_control_flow_attributes : require')
            # Tile loads already zero-fill K tails. A fixed trip count lets the
            # driver schedule/unroll the matrix inner loop instead of repeatedly
            # evaluating dynamic limits and shared-memory address arithmetic.
            text = text.replace('for(uint k=0u;k<min(uint(MOBILE_REDUCTION),pc.k-kb);++k)',
                                '[[unroll]] for(uint k=0u;k<uint(MOBILE_REDUCTION);++k)')
            text = text.replace('#version 450', '#version 450\n#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require')
            text = text.replace('        [[unroll]] for(uint k=0u;',
                '        f16vec2 h0=f16vec2(0.0),h1=f16vec2(0.0),h2=f16vec2(0.0),h3=f16vec2(0.0);\n        [[unroll]] for(uint k=0u;')
            text = text.replace('vec2 weights=vec2(', 'f16vec2 weights=f16vec2(')
            for i in range(4):
                text = text.replace('sum'+str(i)+'=fma(vec2(', 'h'+str(i)+'=fma(f16vec2(')
                text = text.replace('weights,sum'+str(i)+');', 'weights,h'+str(i)+');')
            text = text.replace('        barrier();\n    }',
                '        sum0+=vec2(h0);sum1+=vec2(h1);sum2+=vec2(h2);sum3+=vec2(h3);\n        barrier();\n    }')
        fast_name = name.removesuffix('_workgroup') + '_fast_workgroup'
        output = gpu/(fast_name+'.comp')
        output.write_text(text)
        shaders[fast_name] = (output, list(defines))
