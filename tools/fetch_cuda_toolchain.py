"""Fetch a pinned, private Linux x86_64 CUDA 13.1 build toolchain.

Writes only a fresh caller-selected directory. Does not install a driver, modify
PATH or run GPU work. NVIDIA's toolkit license remains with the SDK.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
PACKAGES = {
    "cuda_nvcc": {
        "relative_path": "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.1.80-archive.tar.xz",
        "sha256": "5ed3b7cfe7f12557199773e7769445357ee048958ff51e623e15f36d3393ca8b",
        "size": 30014972
    },
    "cuda_cudart": {
        "relative_path": "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.1.80-archive.tar.xz",
        "sha256": "b626f4790f46bc9324a1047f2fbcc9a42bc4a722b053056e61cc00da54ad6f32",
        "size": 1549648
    },
    "cuda_cccl": {
        "relative_path": "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-13.1.78-archive.tar.xz",
        "sha256": "e23c8847aa429431d7eb4ccb87ded5eecaa6ba92f7b4bbab90c75c6141e635a2",
        "size": 1096784
    },
    "cuda_crt": {
        "relative_path": "cuda_crt/linux-x86_64/cuda_crt-linux-x86_64-13.1.80-archive.tar.xz",
        "sha256": "0e7c365d3301a1b486dbee600b833f6bc771b1a7cc660abca0923269023355ed",
        "size": 79624
    },
    "libnvvm": {
        "relative_path": "libnvvm/linux-x86_64/libnvvm-linux-x86_64-13.1.80-archive.tar.xz",
        "sha256": "1a102f6658b6ecaa7a3aae1aa85a61a9aa6ba197be9f6b185d906deb2a6c5afd",
        "size": 44796540
    }
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a fresh SDK directory; existing dependencies are never overwritten")

    def download(item):
        name, entry = item
        with urllib.request.urlopen(BASE + entry["relative_path"], timeout=60) as response:
            data = response.read(entry["size"] + 1)
        if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError("CUDA archive size/checksum mismatch: " + name)
        print("Verified " + name, flush=True)
        return name, data

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        archives = list(pool.map(download, PACKAGES.items()))
    output.mkdir(parents=True)
    for _, data in archives:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as archive:
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if len(parts) > 1:
                    member.name = Path(*parts[1:]).as_posix()
                    archive.extract(member, output, filter="data")
    # NVIDIA's redistributions use lib/, while nvcc's default profile uses lib64/.
    if not (output / "lib64").exists():
        (output / "lib64").symlink_to("lib", target_is_directory=True)
    (output / "phoenix-source.json").write_text(json.dumps({
        "source": BASE + "redistrib_13.1.0.json", "packages": PACKAGES}, indent=2) + "\n")
    print("Private CUDA SDK ready: " + str(output))


if __name__ == "__main__":
    main()
