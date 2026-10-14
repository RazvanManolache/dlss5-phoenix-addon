"""Generate/build the production portable Vulkan shader set on any host."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from build_vulkan import prepare_sources

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
    with (args.output/"shaders.log").open("w") as log:
        for name,(source,defines) in shaders.items():
            subprocess.run([str(args.compiler),"--target-env","vulkan1.3",
                            "-I"+str(args.output/"shader_sources"),*defines,
                            "-o",str(args.output/(name+".spv")),str(source)],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    (args.output/"standalone_manifest.json").write_text(json.dumps({
        "schema":"phoenix-vulkan-build/1",
        "sha256":{path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(args.output.glob("*.spv"))}},indent=2)+"\n")
