"""Execute Metal operator tests when the CI host exposes a supported GPU."""
import argparse
import json
from pathlib import Path
import subprocess

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build",type=Path)
    args=parser.parse_args()
    devices=json.loads(subprocess.check_output(
        [str(args.build/"phoenix_frame"),"--devices",str(args.build.resolve())],text=True))
    print(json.dumps({"metal_devices":devices},indent=2),flush=True)
    if any(device["supported"] for device in devices):
        subprocess.run(["ctest","--test-dir",str(args.build),"--output-on-failure","-R","^metal_operators$"],check=True)
    else:
        print("Metal GPU tests NOT EXECUTED: this host exposes no supported Metal device.")
