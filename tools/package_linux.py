#!/usr/bin/env python3
"""Build a curated, model-free, relocatable Linux preview archive."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_digest():
    # Identify dirty and newly added source too; a commit ID alone cannot
    # reproduce a development package. Respect Git's ignored local assets.
    paths = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]).split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(set(paths)):
        if not name:
            continue
        path = ROOT / os.fsdecode(name)
        if path.is_file():
            digest.update(name + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def elf_requirements(paths):
    """Read ELF metadata without executing libraries or probing graphics devices."""
    versions = {}
    needed = set()
    for path in paths:
        if path.read_bytes()[:4] != b"\x7fELF":
            continue
        text = subprocess.check_output(["readelf", "--version-info", "--dynamic", str(path)], text=True)
        needed.update(re.findall(r"Shared library: \[([^\]]+)\]", text))
        for family, number in re.findall(r"\b(GLIBCXX|GLIBC|CXXABI)_([0-9.]+)", text):
            current = tuple(map(int, number.split(".")))
            if current > tuple(map(int, versions.get(family, "0").split("."))):
                versions[family] = number
    return {"symbol_versions": versions, "needed_libraries": sorted(needed)}


def package(prefix, output, version, reference_only=False):
    if platform.system() != "Linux":
        raise ValueError("Build Linux archives on Linux.")
    if not version or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_" for c in version):
        raise ValueError("Version must contain only letters, digits, dots, dashes and underscores.")
    prefix, output = prefix.resolve(), output.resolve()
    required = ["bin/phoenix", "bin/phoenix_frame", "lib/libphoenix_layer.so",
                "lib/libphoenix_runtime_cpp.so", "lib/libphoenix_cpu_cpp.so",
                "share/phoenix/VkLayer_phoenix.json"]
    if not reference_only:
        required += ["lib/libphoenix_vulkan_cpp.so", "lib/vulkan/libphoenix_vulkan.so", "lib/vulkan/standalone_manifest.json"]
        if not list((prefix / "lib/vulkan").glob("*.spv")):
            raise ValueError("Vulkan shader bundle is missing; use --reference-only only for a CPU reference package.")
    backends = ["cpu"] + ([] if reference_only else ["vulkan"])
    if not reference_only and (prefix / "lib/libphoenix_cuda_cpp.so").is_file():
        required += ["lib/libphoenix_cuda_cpp.so", "share/phoenix/cuda_backend.json"]
        backends.append("cuda")
    native_files = []
    if not reference_only and (prefix / "lib/libphoenix_native_cuda.so").is_file():
        bundle = prefix / "lib/native-cuda"
        native = json.loads((bundle / "native_manifest.json").read_text())
        if (native.get("schema") != "phoenix-native-generated-build/1" or
                native.get("boundary_abi") != "cuda-texture-objects/1" or
                native.get("architectures") != ["sm_120"] or not native.get("passed") or
                native.get("model_weights_included") is not False or
                native.get("captured_plans_included") is not False or len(native.get("kernels", [])) != 41):
            raise ValueError("Unsupported native CUDA bundle")
        if hashlib.sha256((bundle / "native_graph.json").read_bytes()).hexdigest() != native["native_graph_sha256"]:
            raise ValueError("Native CUDA graph checksum mismatch")
        for kernel in native["kernels"]:
            path = (bundle / kernel["binary"]).resolve()
            if not path.is_relative_to(bundle) or path.suffix != ".cubin" or hashlib.sha256(path.read_bytes()).hexdigest() != kernel["binary_sha256"]:
                raise ValueError("Native CUDA kernel checksum/path mismatch")
            native_files.append(path.relative_to(prefix).as_posix())
        if len(set(native_files)) != 41 or set(native_files) != {path.relative_to(prefix).as_posix() for path in bundle.rglob("*.cubin")}:
            raise ValueError("Native CUDA bundle contains unlisted or duplicate modules")
        required += ["lib/libphoenix_native_cuda.so", "lib/native-cuda/native_manifest.json",
                     "lib/native-cuda/native_graph.json", "include/phoenix.h", "include/phoenix_linux.h", *native_files]
        backends.append("native-cuda")
    if any(backend in backends for backend in ("cuda", "native-cuda")) and not (prefix / "share/licenses/phoenix/CUDA-LICENSE.txt").is_file():
        raise ValueError("CUDA license notice is missing")
    required = list(dict.fromkeys(required))
    for name in required:
        if not (prefix / name).is_file():
            raise ValueError(f"Missing installed package input: {name}")
    if not reference_only:
        shaders = json.loads((prefix / "lib/vulkan/standalone_manifest.json").read_text())
        if shaders.get("schema") != "phoenix-vulkan-build/1":
            raise ValueError("Unsupported Vulkan shader manifest.")
        actual = {p.name for p in (prefix / "lib/vulkan").glob("*.spv")}
        if actual != set(shaders["sha256"]):
            raise ValueError("Vulkan shader files differ from the generated manifest.")
        for name, expected in shaders["sha256"].items():
            path = (prefix / "lib/vulkan" / name).resolve()
            if path.parent != prefix / "lib/vulkan" or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Vulkan shader checksum mismatch: {name}")
    # Host libraries and CUDA cubins are both ELF, with different machines.
    # Verify host files against the package architecture, not just ELF magic.
    machine = platform.machine()
    elf_machine = {"x86_64": 62, "aarch64": 183, "i386": 3, "i686": 3,
                   "armv7l": 40, "riscv64": 243, "ppc64le": 21, "s390x": 22}.get(machine)
    if elf_machine is None:
        raise ValueError(f"Unsupported Linux package architecture: {machine}")
    for name in required:
        if name.endswith(".so") or name == "bin/phoenix_frame":
            header = (prefix / name).read_bytes()[:20]
            if (len(header) < 20 or header[:4] != b"\x7fELF" or
                    header[4] != (2 if elf_machine in (62, 183, 243, 21, 22) else 1) or header[5] not in (1, 2) or
                    int.from_bytes(header[18:20], "little" if header[5] == 1 else "big") != elf_machine or
                    header[5] != (1 if sys.byteorder == "little" else 2)):
                raise ValueError(f"ELF architecture mismatch for {machine}: {name}")
    name = f"phoenix-linux-{machine}" + ("-reference" if reference_only else "")
    revision = git("rev-parse", "HEAD")
    epoch = int(git("show", "-s", "--format=%ct", "HEAD"))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phoenix-package-") as temporary:
        stage = Path(temporary) / name
        for relative in required:
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prefix / relative, target)
        if not reference_only:
            for path in (prefix / "lib/vulkan").glob("*.spv"):
                shutil.copy2(path, stage / "lib/vulkan" / path.name)
        shutil.copytree(prefix / "share/doc/phoenix", stage / "share/doc/phoenix")
        shutil.copytree(prefix / "share/licenses/phoenix", stage / "share/licenses/phoenix")
        # Keep complete third-party license and provenance text, including the
        # JSON header's additional bundled notices. Never copy weights/binaries.
        for dependency in ("nlohmann", "intel_nr", "amd_nr", "portable_nr"):
            source = ROOT / "third_party" / dependency
            for path in source.rglob("*"):
                if path.is_file() and (path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE", "PROVENANCE")) or path.name == "json.hpp"):
                    target = stage / "share/licenses/phoenix/third_party" / dependency / path.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
        for document in ("LIBRARY.md", "MULTI_MODEL.md", "SMALL_FP8.md", "SUPPORT.md", "LINUX_0AD_VALIDATION.md", "LINUX_NATIVE_CUDA.md"):
            shutil.copy2(ROOT / "docs" / document, stage / "share/doc/phoenix" / document)
        if "native-cuda" in backends:
            guide = stage / "share/doc/phoenix/LINUX_NATIVE_CUDA.md"
            guide.write_text(guide.read_text().replace("../native_runtime/phoenix_linux.h", "../../../include/phoenix_linux.h")
                             .replace("../tests/linux/cuda_launch_probe.cpp", "examples/cuda_launch_probe.cpp"))
            examples = guide.parent / "examples"
            examples.mkdir(exist_ok=True)
            shutil.copy2(ROOT / "tests/linux/cuda_launch_probe.cpp", examples / "cuda_launch_probe.cpp")
        (stage / "README.md").write_text(
            "# Phoenix for Linux — experimental preview\n\n"
            "Run `./bin/phoenix --doctor` to verify this package.\n\n"
            "Read [the Linux guide](share/doc/phoenix/LINUX.md) for setup, limits and uninstall instructions.\n\n"
            "Model weights are external. See the Linux guide for the scope of runtime validation; packaging checks alone do not certify GPUs or games.\n")
        manifest = {
            "schema": "phoenix-linux/1", "version": version, "stage": "experimental-host-transport",
            "machine": machine, "backends": backends,
            "game_device_selection": "per-instance Vulkan 1.1+ UUID filter",
            "source_revision": revision, "source_dirty": bool(git("status", "--porcelain")),
            "source_tree_sha256": source_digest(),
            "build_platform": platform.platform(), "libc": platform.libc_ver(),
            "requirements": {"python": "3.10+", "vulkan_loader": "1.3.234+", "models": "external", "system_libraries": ["libstdc++", "libgcc_s", "libc", "libm", "libvulkan (Vulkan backend)"]},
            "validation": {"gpu_inference": "not certified by packaging", "live_games": "not certified by packaging", "proton": "unvalidated", "opengl": "not implemented"},
            "sha256": {},
        }
        if "cuda" in backends or "native-cuda" in backends:
            manifest["optional_backend_requirements"] = {
                "cuda_driver": "NVIDIA driver libcuda.so.1; toolkit is not required to run",
                "native_cuda": "SM120, bounded reset-only SDR RGB/RGBA8" if "native-cuda" in backends else "not included",
            }
        manifest["elf_requirements"] = elf_requirements(path for path in stage.rglob("*") if path.is_file() and (path.suffix == ".so" or path.parent == stage / "bin"))
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                manifest["sha256"][str(path.relative_to(stage))] = hashlib.sha256(path.read_bytes()).hexdigest()
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Stable member ordering, timestamps and ownership. The manifest records
        # build inputs; repeated packaging of identical inputs is deterministic.
        archive = Path(temporary) / (name + ".tar.gz")
        with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed, tarfile.open(fileobj=compressed, mode="w") as tar:
            for path in [stage, *sorted(stage.rglob("*"))]:
                info = tar.gettarinfo(str(path), arcname=str(Path(name) / path.relative_to(stage)))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = epoch
                info.mode = 0o755 if path.is_dir() or path.parent == stage / "bin" else 0o644
                if path.is_file():
                    with path.open("rb") as data:
                        tar.addfile(info, data)
                else:
                    tar.addfile(info)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        destination = output / archive.name
        shutil.copy2(archive, destination)
        (output / (archive.name + ".sha256")).write_text(f"{digest}  {archive.name}\n")
        return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    try:
        print(package(args.prefix, args.output, args.version, args.reference_only))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"package_linux: {exc}\n")


if __name__ == "__main__":
    main()
