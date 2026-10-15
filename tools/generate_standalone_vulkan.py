"""Generate/build the production portable Vulkan shader set on any host."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import struct
from build_vulkan import prepare_sources


def check_workgroup_shader(path, allow_half=False):
    """Reject subgroup arithmetic; only approximate shaders may use native half."""
    words = struct.unpack('<%dI' % (path.stat().st_size // 4), path.read_bytes())
    cursor = 5
    while cursor < len(words):
        length, opcode = words[cursor] >> 16, words[cursor] & 65535
        if not length or cursor + length > len(words):
            raise RuntimeError('Malformed SPIR-V: ' + path.name)
        # OpCapability: Float16, Int16 arithmetic, GroupNonUniform and its operations.
        # StorageBuffer16BitAccess is intentional; half bits live in uint16 storage.
        rejected = (22, *range(61, 69)) if allow_half else (9, 22, *range(61, 69))
        if opcode == 17 and words[cursor + 1] in rejected:
            raise RuntimeError('Nonportable capability in ' + path.name + ': ' + str(words[cursor + 1]))
        cursor += length

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--compiler",type=Path,required=True)
    parser.add_argument("--android",action="store_true")
    args=parser.parse_args()
    shaders,_,_=prepare_sources(args.output)
    if args.android:
        # API 28's loader import library predates this Vulkan 1.2 symbol. Query
        # the enabled device function instead of raising the app's minimum API.
        # Existing capability checks still require bufferDeviceAddress support.
        path=args.output/"phoenix_vulkan.c"
        source=path.read_text().replace("vkGetBufferDeviceAddress(","phx_buffer_device_address(")
        adapter='''
static VkDeviceAddress phx_buffer_device_address(VkDevice device, const VkBufferDeviceAddressInfo* info) {
    PFN_vkGetBufferDeviceAddress query=(PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(device,"vkGetBufferDeviceAddress");
    if(!query)query=(PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(device,"vkGetBufferDeviceAddressKHR");
    return query ? query(device,info) : 0;
}
'''
        source=source.replace('#include "phoenix_devices.h"','#include "phoenix_devices.h"\n'+adapter,1)
        path.write_text(source)
    # Drop obsolete generated binaries so incremental packaging matches a clean
    # build. This directory contains generated shaders, never original sources.
    expected = {name + '.spv' for name in shaders}
    for path in args.output.glob('*.spv'):
        if path.name not in expected:
            path.unlink()
    with (args.output/"shaders.log").open("w") as log:
        for name,(source,defines) in shaders.items():
            subprocess.run([str(args.compiler),"--target-env","vulkan1.3",
                            "-I"+str(args.output/"shader_sources"),*defines,
                            "-o",str(args.output/(name+".spv")),str(source)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
            if name.endswith('_workgroup'):
                check_workgroup_shader(args.output/(name+".spv"), allow_half=name.endswith('_fast_workgroup'))
    (args.output/"standalone_manifest.json").write_text(json.dumps({
        "schema":"phoenix-vulkan-build/1",
        "sha256":{path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(args.output.glob("*.spv"))}},indent=2)+"\n")
