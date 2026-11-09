#!/usr/bin/env python3
"""Relocatable, per-process Linux launcher. Never registers a global layer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

LAYER = "VK_LAYER_PHOENIX_neural_rendering"


def package_root():
    return Path(__file__).resolve().parent.parent


def read_manifest(root):
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError("Package manifest is missing; run the launcher from a complete Linux package.")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "phoenix-linux/1":
        raise ValueError("Unsupported package manifest schema.")
    return manifest


def verify(root, manifest):
    for relative, expected in manifest["sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe package file: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Package checksum mismatch: {relative}")


def validate_device(device):
    if not isinstance(device, str) or not device.startswith("vulkan:") or len(device) != 39:
        raise ValueError("A Vulkan device must be vulkan:<32 hexadecimal UUID digits>.")
    if any(c not in "0123456789abcdefABCDEF" for c in device[7:]):
        raise ValueError("The Vulkan device UUID must be hexadecimal.")
    return device.lower()


def make_environment(root, config_path, original, game_device=None, backend=None):
    env = dict(original)
    env["PHOENIX_ENABLE"] = "1"
    env["PHOENIX_CONFIG"] = str(config_path)
    library = "libphoenix_native_cuda.so" if backend == "native-cuda" else "libphoenix_runtime_cpp.so"
    env["PHOENIX_RUNTIME_LIBRARY"] = str(root / "lib" / library)
    if game_device is not None:
        env["PHOENIX_GAME_DEVICE"] = validate_device(game_device)
    # ADD preserves the loader's normal system layer search paths. Replacing
    # VK_LAYER_PATH would hide overlays and validation layers installed there.
    previous = env.get("VK_ADD_LAYER_PATH", "")
    env["VK_ADD_LAYER_PATH"] = str(root / "share/phoenix") + (os.pathsep + previous if previous else "")
    # VK_LAYER_PATH overrides additive discovery when already set. Retain its
    # custom entries while making our manifest visible in that case too.
    if env.get("VK_LAYER_PATH"):
        env["VK_LAYER_PATH"] = str(root / "share/phoenix") + os.pathsep + env["VK_LAYER_PATH"]
    layers = [x for x in env.get("VK_INSTANCE_LAYERS", "").split(":") if x and x != LAYER]
    env["VK_INSTANCE_LAYERS"] = ":".join([LAYER, *layers])
    return env


def runtime_config(root, filename, manifest):
    config = json.loads(Path(filename).expanduser().read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    backend = config.get("backend")
    if backend not in manifest["backends"]:
        raise ValueError(f"Backend {backend!r} is not included; available: {manifest['backends']}")
    if backend not in ("cpu", "vulkan", "cuda", "native-cuda"):
        raise ValueError("Unsupported Linux inference backend.")
    device = config.get("device", "")
    if backend == "vulkan":
        config["device"] = validate_device(device)
    if backend in ("cuda", "native-cuda"):
        if not isinstance(device, str) or len(device) != 37 or not device.startswith("cuda:"):
            raise ValueError("CUDA requires cuda:<32 hexadecimal UUID digits>.")
        if any(c not in "0123456789abcdefABCDEF" for c in device[5:]):
            raise ValueError("The CUDA device UUID must be hexadecimal.")
        config["device"] = device.lower()
    if backend == "cpu" and device != "cpu":
        raise ValueError("The CPU backend requires device: cpu.")
    source = config.get("weight_source", {})
    model_path = source.get("path") if isinstance(source, dict) else None
    if not isinstance(model_path, str) or not Path(model_path).is_absolute():
        raise ValueError("weight_source.path must be an absolute path to your external model.")
    if not Path(model_path).exists():
        raise ValueError("The external model path does not exist.")
    config["runtime_root"] = str(root / "lib")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phoenix for Linux — experimental Vulkan presentation layer")
    parser.add_argument("--doctor", action="store_true", help="verify package and report requirements; does not run inference or probe the GPU")
    parser.add_argument("--config", type=Path, help="runtime JSON configuration with an external model and explicit device")
    parser.add_argument("--game-device", type=validate_device, help="expose only this Vulkan GPU UUID to the game; Phoenix inference retains the available GPU list")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- game [arguments...] (Steam: -- %%command%%)")
    args = parser.parse_args(argv)
    root = package_root()
    manifest = read_manifest(root)
    if platform.system() != "Linux":
        raise ValueError("This package targets native Linux.")
    if platform.machine() != manifest["machine"]:
        raise ValueError(f"Package targets {manifest['machine']}; this machine is {platform.machine()}.")
    verify(root, manifest)
    if args.doctor:
        report = {
            "package": manifest["version"], "stage": manifest["stage"],
            "machine": manifest["machine"], "checksums": "passed",
            "backends": manifest["backends"], "python": platform.python_version(),
            "elf_requirements": manifest.get("elf_requirements", {}),
            "game_device": args.game_device or os.environ.get("PHOENIX_GAME_DEVICE"),
            "game_device_selection": manifest.get("game_device_selection", "unspecified"),
            "gpu_validation": "not run", "game_validation": "not run",
            "transport": "synchronous CPU-buffer readback/inference/upload",
            "capture": "SDR RGBA8/BGRA8 Vulkan, one swapchain per presentation, no extension chains",
            "temporal_reuse": False, "hud_included": True,
            "vulkan_requirement": "Vulkan loader 1.3.234+ for VK_ADD_LAYER_PATH; compute capabilities also required",
        }
        if args.config:
            runtime_config(root, args.config, manifest)
            report["configuration"] = "paths and backend checked; model contents and GPU execution not validated"
        print(json.dumps(report, indent=2))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not args.config:
        parser.error("launch with --config /absolute/config.json -- game [arguments...]")
    config = runtime_config(root, args.config, manifest)
    print("[Phoenix Linux] Experimental host transport; HUD included; temporal reuse disabled.", file=sys.stderr)
    if args.game_device:
        print(f"[Phoenix Linux] Game GPU: {args.game_device}; inference device: {config['device']}", file=sys.stderr)
    if config["backend"] == "cpu":
        print("[Phoenix Linux] CPU reference backend selected; not a real-time performance route.", file=sys.stderr)
    # The private directory survives while the child process runs and is removed
    # on normal exit, launch failure and Python exceptions. No user config writes.
    with tempfile.TemporaryDirectory(prefix="phoenix-linux-") as temporary:
        path = Path(temporary) / "runtime.json"
        path.write_text(json.dumps(config))
        path.chmod(0o600)
        result = subprocess.run(command, env=make_environment(root, path, os.environ, args.game_device, config["backend"]), check=False)
        return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (ValueError, OSError, KeyError) as exc:
        print(f"phoenix: {exc}", file=sys.stderr)
        sys.exit(2)
