"""Generate Metal's reference kernels from the shared operator definition.

Only language/ABI adaptations occur here. Numerical operator bodies stay in
compute_ops.h so fixes to publication, indexing and reductions reach Metal too.
The generated text also supports host execution for arithmetic regression tests.
"""
import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def generate():
    source = (ROOT / "src/backends/hip_sycl/compute_ops.h").read_text()
    source = re.sub(r"^#(?:pragma|include).*\n", "", source, flags=re.M)
    # Metal uses C++14: select the portable implementations of optional C++20
    # Math hooks, which the Metal math adapter deliberately does not implement.
    source, hooks = re.subn(
        r"        if constexpr \(requires \{ Math::\w+\([^\n]+\); \}\)\n"
        r"            return Math::\w+\([^\n]+\);\n", "", source)
    if hooks != 5:
        raise ValueError("Update Metal adaptation for changed optional Math hooks")
    source = source.replace("namespace phx {", "namespace phoenix_metal {")
    source = source.replace("phx::", "phoenix_metal::")
    source = source.replace("void*", "device void*")
    source = re.sub(r"static_cast<((?:const )?(?:uint16_t|uint8_t|float))\*>",
                    r"static_cast<device \1*>", source)
    source = re.sub(r"\bhalf\(", "round_binary16(", source)
    # MSL has no standard offsetof macro; host-side descriptor layout is checked
    # independently. Device pointers are constructed from bound buffers below.
    source = re.sub(r"^static_assert.*\n", "", source, flags=re.M)
    # Avoid lambda support differences between Metal compiler versions.
    mirror = '''        auto mirror = [](u32 value, u32 extent) {
            int coordinate = value < extent ? int(value) : 2 * int(extent) - int(value) - 2;
            return coordinate < 0 ? 0u : coordinate >= int(extent) ? extent - 1 : u32(coordinate);
        };'''
    if mirror not in source:
        raise ValueError("Update Metal adaptation for changed mirror helper")
    source = source.replace(mirror, "")
    source = source.replace("    Operation operation;", '''    Operation operation;
    PHX_DEVICE u32 mirror(u32 value, u32 extent) const {
        int coordinate = value < extent ? int(value) : 2 * int(extent) - int(value) - 2;
        return coordinate < 0 ? 0u : coordinate >= int(extent) ? extent - 1 : u32(coordinate);
    }''')
    prefix = '''// Generated from compute_ops.h. Edit the shared source or its generator.
#ifdef PHOENIX_METAL_HOST_TEST
#include <cstdint>
#include "native_runtime/host_math.h"
#define device
using MetalMath = HostMath;
#else
#include <metal_stdlib>
using namespace metal;
using uint32_t = uint;
using uint64_t = ulong;
using uint16_t = ushort;
using uint8_t = uchar;
struct MetalMath {
    static uint bits(float value) { return as_type<uint>(value); }
    static float value(uint bits) { return as_type<float>(bits); }
    static float fma(float left, float right, float addend) { return metal::fma(left,right,addend); }
    static float sqrt(float value) { return metal::sqrt(value); }
    static float abs(float value) { return metal::abs(value); }
    static float min(float left,float right) { return metal::fmin(left,right); }
    static float max(float left,float right) { return metal::fmax(left,right); }
    static float even(float value) { return metal::rint(value); }
    static bool inf(float value) { return metal::isinf(value); }
};
#endif
#define PHX_DEVICE
'''
    # No CPU pointers cross into the GPU. Five bound buffers + scalar descriptor
    # reconstruct a device-local Operation. The host passes only bytes 40..95.
    suffix = '''
#ifndef PHOENIX_METAL_HOST_TEST
struct DispatchScalars {
    uint family, flags, row_count, column_count, reduction_count, batch_count;
    uint left_batch_stride, right_batch_stride, output_batch_stride;
    uint left_row_stride, right_row_stride, output_row_stride;
    float scale;
    uint padding;
    ulong count;
    ulong offsets[5];
};
kernel void phoenix_reference(
    device void* left [[buffer(0)]], device void* right [[buffer(1)]],
    device void* output [[buffer(2)]], device void* auxiliary [[buffer(3)]],
    device void* secondary [[buffer(4)]],
    constant DispatchScalars& parameters [[buffer(5)]], uint index [[thread_position_in_grid]]) {
    if (ulong(index) >= parameters.count) return;
    phoenix_metal::Operation operation;
    operation.left_input=static_cast<device uchar*>(left)+parameters.offsets[0];
    operation.right_input=static_cast<device uchar*>(right)+parameters.offsets[1];
    operation.output=static_cast<device uchar*>(output)+parameters.offsets[2];
    operation.auxiliary=static_cast<device uchar*>(auxiliary)+parameters.offsets[3];
    operation.secondary_output=static_cast<device uchar*>(secondary)+parameters.offsets[4];
    operation.family=parameters.family; operation.flags=parameters.flags;
    operation.row_count=parameters.row_count; operation.column_count=parameters.column_count;
    operation.reduction_count=parameters.reduction_count; operation.batch_count=parameters.batch_count;
    operation.left_batch_stride=parameters.left_batch_stride;
    operation.right_batch_stride=parameters.right_batch_stride;
    operation.output_batch_stride=parameters.output_batch_stride;
    operation.left_row_stride=parameters.left_row_stride;
    operation.right_row_stride=parameters.right_row_stride;
    operation.output_row_stride=parameters.output_row_stride; operation.scale=parameters.scale;
    phoenix_metal::Kernel<MetalMath> executor{operation};
    executor(index);
}
#else
#undef device
#endif
'''
    return prefix + source + suffix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(), encoding="utf-8")
