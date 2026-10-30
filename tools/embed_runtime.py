"""Package model-free runtime files as compressed Windows addon resources."""

from pathlib import Path
import ctypes
from ctypes import wintypes
import hashlib
import json


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require_architecture_coverage(manifest):
    architectures = manifest.get("architectures", [])
    if len(architectures) > 1:
        evidence = manifest.get("architecture_validation", {})
        if set(evidence) != set(architectures) or any(evidence[arch] != "validated" for arch in architectures):
            raise ValueError("Every native architecture needs device inference validation before packaging")


def require_native_packaging(manifest, experimental=False):
    if not manifest.get("passed"):
        raise ValueError("Native kernel compilation must pass before packaging")
    if manifest.get('kernel_style', 'auto') != 'auto':
        raise ValueError('Standalone style kernels cannot replace the native default; use build_kernel_styles.py')
    for style, variant in manifest.get('kernel_styles', {}).items():
        minimum = {'rtx20': 75, 'rtx30': 86, 'rtx40': 89}.get(style)
        targets = variant.get('architectures', [])
        if (not minimum or not targets or len(set(targets)) != len(targets) or
                any(target not in ('sm_86', 'sm_89', 'sm_120') or int(target[3:]) <= minimum for target in targets) or
                len(variant.get('kernels', [])) != 41):
            raise ValueError('Invalid native kernel-style catalogue')
        if not experimental and variant.get('validation_pending', True):
            raise ValueError('Kernel-style device inference validation is pending')
    if experimental:
        if (manifest.get("selection_status") not in ("compiled", "validated") or
                manifest.get("model_weights_included") is not False):
            raise ValueError("Experimental native bundle must be successfully compiled and model-free")
        return
    if manifest.get("selection_status") != "validated":
        raise ValueError("Native kernels must be validated before packaging")
    require_architecture_coverage(manifest)


def native_graph_files(native, manifest):
    """Keep every selected graph and its pin together in the embedded payload."""
    native = Path(native)
    result = {}
    for filename, pin in (("native_graph.json", "native_graph_sha256"),
                          ("native_graph_v2.json", "native_graph_v2_sha256")):
        if pin not in manifest:
            continue
        raw = (native / filename).read_bytes()
        if digest(raw) != manifest[pin]:
            raise ValueError("Native graph changed since validation: " + filename)
        result["native/" + filename] = raw
    return result


def compress(data):
    cabinet = ctypes.WinDLL("cabinet", use_last_error=True)
    cabinet.CreateCompressor.argtypes = [
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    cabinet.Compress.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    cabinet.CloseCompressor.argtypes = [ctypes.c_void_p]
    handle = ctypes.c_void_p()
    if not cabinet.CreateCompressor(4, None, ctypes.byref(handle)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        source = ctypes.create_string_buffer(data)
        needed = ctypes.c_size_t()
        cabinet.Compress(handle, source, len(data), None, 0, ctypes.byref(needed))
        if not needed.value:
            raise ctypes.WinError(ctypes.get_last_error())
        target = ctypes.create_string_buffer(needed.value)
        if not cabinet.Compress(
            handle, source, len(data), target, needed.value, ctypes.byref(needed)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return target.raw[: needed.value]
    finally:
        cabinet.CloseCompressor(handle)


def prepare(runtime, native, output, nr_forwarder=None, *, experimental_native=False):
    runtime, native, output = map(
        lambda source_path: Path(source_path).resolve(), (runtime, native, output)
    )
    validation = json.loads((runtime / "validation_manifest.json").read_text(encoding="utf-8"))
    native_manifest = json.loads((native / "native_manifest.json").read_text(encoding="utf-8"))
    if not validation.get("passed") or not validation.get("inference_validated"):
        raise ValueError("Runtime must pass inference validation before packaging")
    require_native_packaging(native_manifest, experimental_native)
    if not experimental_native and any("large_input_entry" in kernel for kernel in native_manifest["kernels"]):
        from validate_native import kernel_selection_digest, require_variant_coverage

        evidence = native_manifest.get("validation", {})
        if evidence.get("kernel_selection_sha256") != kernel_selection_digest(native_manifest):
            raise ValueError("Native resolution selection changed since validation")
        coverage = {(item["name"], item["mode"])
                    for item in evidence.get("resolution_variant_coverage", [])}
        require_variant_coverage(native_manifest, coverage)
    files = {}
    excluded = {
        "phoenix_cpp_backend_contracts.exe",
        "phoenix_cpp_graph_profile.exe",
        "phoenix_cpp_matrix_benchmark.exe",
        "phoenix_cpp_cuda_vendor_matrix_benchmark.exe",
        "phoenix_cpp_cuda_math_contracts.exe",
        "phoenix_cpp_client_contracts.exe",
        "phoenix_cpp_offload_probe.exe",
        "phoenix_cpp_runtime_contracts.exe",
        "phoenix_cpp_probe.dll",
        "phoenix_cpp_runner.exe",
    }
    allowed = {".dll", ".exe", ".spv", ".rtl", ".o", ".cfg"}
    for path in runtime.rglob("*"):
        if not path.is_file() or path.name in excluded or path.suffix.lower() not in allowed:
            continue
        relative = path.relative_to(runtime).as_posix()
        if "nvngx" in relative.lower() or "weights" in relative.lower():
            raise ValueError("Model/NVIDIA DLL cannot be embedded: " + relative)
        data = path.read_bytes()
        expected = validation["binaries"].get(relative)
        if expected and digest(data) != expected:
            raise ValueError("Runtime binary changed since validation: " + relative)
        files["runtime/" + relative] = data
    # Source is always supplied by the addon; never bake local model paths in.
    config = json.loads((runtime / "phoenix_native_cpp.json").read_text(encoding="utf-8"))
    config.pop("weight_source", None)
    files["runtime/phoenix_native_cpp.json"] = (json.dumps(config, indent=2) + "\n").encode()
    files["runtime/validation_manifest.json"] = (runtime / "validation_manifest.json").read_bytes()
    files.update(native_graph_files(native, native_manifest))
    files["native/native_manifest.json"] = (native / "native_manifest.json").read_bytes()
    selected_kernels = list(native_manifest["kernels"])
    for variant in native_manifest.get('kernel_styles', {}).values():
        selected_kernels.extend(variant['kernels'])
    for kernel in selected_kernels:
        relative = Path(kernel["binary"])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError("Native kernel must use a relative bundle path")
        path = (native / relative).resolve()
        if not path.is_relative_to(native):
            raise ValueError("Native kernel escapes bundle")
        data = path.read_bytes()
        if digest(data) != kernel["binary_sha256"]:
            raise ValueError("Native kernel changed since validation")
        files["native/" + relative.as_posix()] = data
    if nr_forwarder is not None:
        files['runtime/phoenix-nvngx.dll'] = Path(nr_forwarder).read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    entries, resources = [], []
    for index, (name, data) in enumerate(sorted(files.items()), 1100):
        compressed = compress(data)
        algorithm = "xpress_huff" if len(compressed) < len(data) else "raw"
        payload = compressed if algorithm == "xpress_huff" else data
        path = output / (str(index) + ".bin")
        path.write_bytes(payload)
        entries.append(
            {
                "path": name,
                "resource": index,
                "bytes": len(data),
                "sha256": digest(data),
                "algorithm": algorithm,
                "packed_bytes": len(payload),
            }
        )
        resources.append(f'{index} RCDATA "{path.as_posix()}"')
    manifest = {
        "schema": "dlss5-phoenix-embedded-runtime/1",
        "model_data_included": False,
        "nvidia_dll_included": False,
        "release_channel": "experimental" if experimental_native else "stable",
        "native_architecture_validation": native_manifest.get("architecture_validation", {}),
        "native_validation_pending": native_manifest.get("validation_pending", False),
        "files": entries,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    resources.insert(0, f'1001 RCDATA "{manifest_path.as_posix()}"')
    recipe = output / "runtime.rc"
    recipe.write_text("\n".join(resources) + "\n", encoding="utf-8")
    return recipe, {
        "files": len(entries),
        "unpacked_bytes": sum(resource_entry["bytes"] for resource_entry in entries),
        "packed_bytes": sum(resource_entry["packed_bytes"] for resource_entry in entries),
        "manifest_sha256": digest(manifest_path.read_bytes()),
        "model_data_included": False,
        "release_channel": manifest["release_channel"],
    }
