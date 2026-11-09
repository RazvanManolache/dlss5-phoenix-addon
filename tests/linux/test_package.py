#!/usr/bin/env python3
"""Exercise the installed distribution without loading a driver or model."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("packager", ROOT / "tools/package_linux.py")
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--report", type=Path, help="write a compact software-validation receipt")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="phoenix-package-test-") as temporary:
        root = Path(temporary)
        archive = packager.package(args.prefix, root / "first", "test-preview", args.reference_only)
        second = packager.package(args.prefix, root / "second", "test-preview", args.reference_only)
        check(archive.read_bytes() == second.read_bytes(), "Archive is not reproducible for identical inputs")
        with tarfile.open(archive) as tar:
            tar.extractall(root / "extract", filter="data")
        package = next((root / "extract").iterdir())
        relocated = root / "relocated package with spaces"
        shutil.move(package, relocated)
        launcher = relocated / "bin/phoenix"
        report = subprocess.check_output([str(launcher), "--doctor"], text=True)
        check(json.loads(report)["checksums"] == "passed", "Relocated package doctor failed")
        manifest = json.loads((relocated / "manifest.json").read_text())
        layer = json.loads((relocated / "share/phoenix/VkLayer_phoenix.json").read_text())
        library = (relocated / "share/phoenix" / layer["layer"]["library_path"]).resolve()
        check(library.is_file(), "Relocated manifest cannot find the layer")
        # Loading this frontend and negotiating the loader ABI do not create a
        # Vulkan instance/device, load model weights or perform inference.
        module = ctypes.CDLL(str(library))
        for symbol in ("vkGetInstanceProcAddr", "vkGetDeviceProcAddr", "vkCreateInstance", "vkCreateDevice", "vkQueuePresentKHR", "vkEnumeratePhysicalDevices", "vkEnumeratePhysicalDeviceGroups", "vkEnumeratePhysicalDeviceGroupsKHR", "vkNegotiateLoaderLayerInterfaceVersion"):
            getattr(module, symbol)
        runtime = ctypes.CDLL(str(relocated / "lib/libphoenix_runtime_cpp.so"))
        for symbol in ("phoenix_runtime_create", "phoenix_runtime_destroy", "phoenix_runtime_frame_v2", "phoenix_runtime_error"):
            getattr(runtime, symbol)
        class Negotiation(ctypes.Structure):
            _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                        ("version", ctypes.c_uint32), ("gpa", ctypes.c_void_p),
                        ("gdpa", ctypes.c_void_p), ("physical", ctypes.c_void_p)]
        negotiate = module.vkNegotiateLoaderLayerInterfaceVersion
        negotiate.argtypes = [ctypes.POINTER(Negotiation)]
        negotiate.restype = ctypes.c_int
        request = Negotiation(1, None, 2, None, None, None)
        check(negotiate(ctypes.byref(request)) == 0 and request.gpa and request.gdpa, "Packaged loader ABI negotiation failed")
        expected_backends = ["cpu"] if args.reference_only else ["cpu", "vulkan"]
        if not args.reference_only:
            if (args.prefix / "lib/libphoenix_cuda_cpp.so").is_file(): expected_backends.append("cuda")
            if (args.prefix / "lib/libphoenix_native_cuda.so").is_file(): expected_backends.append("native-cuda")
        check(manifest["backends"] == expected_backends, "Backend declaration differs from package")
        if "native-cuda" in expected_backends:
            check((relocated / "include/phoenix_linux.h").is_file(), "Native byte API header is missing")
            check((relocated / "share/doc/phoenix/examples/cuda_launch_probe.cpp").is_file(), "Documented probe example is missing")
            symbols = subprocess.check_output(["readelf", "--dyn-syms", "--wide", str(relocated / "lib/libphoenix_native_cuda.so")], text=True)
            for symbol in ("phoenix_runtime_create", "phoenix_runtime_destroy", "phoenix_runtime_frame_v2", "phoenix_runtime_status", "phoenix_runtime_error", "phoenix_linux_frame_rgba8"):
                check(symbol in symbols, "Native CUDA export missing: " + symbol)
        model = root / "external model"
        model.mkdir()
        config = root / "configuration.json"
        text = json.dumps({"backend": "cpu", "device": "cpu", "weight_source": {"kind": "model_folder", "path": str(model)}})
        config.write_text(text)
        child = root / "inspect_child.py"
        result_file = root / "child.json"
        child.write_text(
            "import json, os, pathlib, sys\n"
            "config=pathlib.Path(os.environ['PHOENIX_CONFIG'])\n"
            "payload={'args':sys.argv[2:], 'env':dict(os.environ), 'config':json.loads(config.read_text())}\n"
            "pathlib.Path(sys.argv[1]).write_text(json.dumps(payload))\n"
            "sys.exit(7)\n")
        env = dict(os.environ, VK_INSTANCE_LAYERS="existing-overlay", VK_ADD_LAYER_PATH="/existing/layers")
        # This child does not load Vulkan: no physical device or model is used.
        game_device = "vulkan:" + "a"*32
        result = subprocess.run([str(launcher), "--game-device", game_device, "--config", str(config), "--", sys.executable, str(child), str(result_file), "argument with spaces", "$(literal)"], env=env, check=False)
        check(result.returncode == 7, "Game exit status not preserved")
        payload = json.loads(result_file.read_text())
        check(payload["args"] == ["argument with spaces", "$(literal)"], "Game arguments changed or were interpreted by a shell")
        check(payload["env"]["VK_INSTANCE_LAYERS"].endswith(":existing-overlay"), "Existing layers lost")
        check(payload["env"]["VK_ADD_LAYER_PATH"].endswith(":/existing/layers"), "Existing search paths lost")
        check(payload["config"]["runtime_root"] == str(relocated / "lib"), "Runtime root did not relocate")
        check(payload["env"]["PHOENIX_GAME_DEVICE"] == game_device and payload["config"]["device"] == "cpu", "Game GPU option changed the independent inference device")
        check(not Path(payload["env"]["PHOENIX_CONFIG"]).exists(), "Private runtime config was not cleaned up")
        check(config.read_text() == text, "User configuration was modified")
        with library.open("ab") as output:
            output.write(b"tampered")
        result = subprocess.run([str(launcher), "--doctor"], capture_output=True, text=True)
        check(result.returncode == 2 and "checksum mismatch" in result.stderr, "Tampered package was accepted")
        cpu = relocated / "lib/libphoenix_cpu_cpp.so"
        data = bytearray(cpu.read_bytes())
        data[18:20] = b"\0\0"
        cpu.write_bytes(data)
        try:
            packager.package(relocated, root / "wrong-architecture", "test-preview", args.reference_only)
        except ValueError as error:
            check("architecture mismatch" in str(error), "Wrong architecture failed for an unrelated reason")
        else:
            raise RuntimeError("Wrong-architecture library was packaged")
        print("Linux package: reproducibility, relocation, layer paths, launcher isolation, exit status and tamper detection passed; no GPU work run.")
    if args.report:
        report = {
            "schema": "phoenix-linux-software-validation/1",
            "source_revision": packager.git("rev-parse", "HEAD"),
            "source_tree_sha256": packager.source_digest(),
            "package_contracts": "passed",
            "checks": ["deterministic archive", "relocation with spaces", "packaged layer/runtime exports", "loader ABI negotiation", "launch arguments", "environment preservation", "exit status", "temporary config cleanup", "user config preservation", "tamper detection"],
            "gpu_inference": "not run", "live_games": "not run",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
