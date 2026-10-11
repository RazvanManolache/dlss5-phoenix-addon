"""Audit the production source tree and optionally make a source-only ZIP."""

from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import zipfile

from production_build_native_kernels import production_inputs

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {
    "build",
    "providers",
    "settings",
    "output",
    "dist",
    "validation",
    "models",
    ".git",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    "captures",
    "traces",
    "backups",
}
FORBIDDEN = {
    ".dll",
    ".exe",
    ".addon64",
    ".asi",
    ".lib",
    ".obj",
    ".pdb",
    ".exp",
    ".cubin",
    ".ptx",
    ".spv",
    ".bin",
    ".safetensors",
    ".onnx",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".zip",
    ".7z",
}


def candidates():
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        # Local runtime DLLs and source annotations never enter publication.
        if rel.parts[0] in {"runtime", "source_maps"}:
            continue
        # Only the selected production kernels belong in this source package.
        if rel.parts[:3] == ("src", "native", "kernels") and rel.parts[3:4] != ("production",):
            continue
        if rel.parts[:5] == ("src", "native", "kernels", "production", "reference"):
            continue
        if rel.parts[:3] == ("src", "native", "model") and path.name.startswith("kernel_tuning"):
            continue
        if (
            rel.parts[0] in IGNORED
            or any(part in {".git", "__pycache__"} for part in rel.parts)
            or path.suffix == ".pyc"
        ):
            continue
        if path.is_symlink():
            raise ValueError("Publication symlink is not allowed: " + rel.as_posix())
        if path.is_file():
            yield path


def kernel_closure(catalogue, found):
    """Follow local includes so missing common headers fail before packaging."""
    kernel_root = (ROOT / "src/native/kernels/production").resolve()
    pending = [row["source"] for row in catalogue["kernels"]]
    required = set()
    while pending:
        name = pending.pop()
        if name in required:
            continue
        if name not in found:
            raise ValueError("Production kernel dependency missing: " + name)
        path = found[name]
        required.add(name)
        for include in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', path.read_text(), re.M):
            dependency = (path.parent / include).resolve()
            if not dependency.is_relative_to(kernel_root):
                raise ValueError("Kernel include escapes production sources: " + include)
            pending.append(dependency.relative_to(ROOT.resolve()).as_posix())
    return required


def validate_payload(name, path):
    """Publish maintained sources; refuse generated GPU code and model payloads."""
    if path.suffix.lower() in FORBIDDEN:
        raise ValueError("Forbidden publication payload: " + name)
    if path.name in {"extract.py", "cli.py", "__main__.py"}:
        raise ValueError("Excluded extraction entry point: " + name)
    raw = path.read_bytes()
    if raw[:2] == b"MZ" or raw[:4] == b"\x7fELF":
        raise ValueError("Binary executable payload: " + name)
    return hashlib.sha256(raw).hexdigest()


def audit():
    # The production catalogue replaces the private tuning/source manifests.
    # Generate hashes from this audit instead of requiring a stale copy in Git.
    catalogue, _ = production_inputs(ROOT)
    found = {source_path.relative_to(ROOT).as_posix(): source_path for source_path in candidates()}
    required = kernel_closure(catalogue, found)
    kernel_files = {name for name in found if name.startswith("src/native/kernels/production/")}
    if kernel_files != required:
        raise ValueError(
            "Unselected kernel files in package: " + str(sorted(kernel_files - required))
        )
    manifest = {"schema": "phoenix-production-source-audit/1", "files": {}}
    for name, path in found.items():
        if Path(name).parts[0] == "runtime" or any(
            part in {"models", "captures", "traces", "backups", "archive"}
            for part in Path(name).parts
        ):
            raise ValueError("Private payload directory: " + name)
        manifest["files"][name] = {"sha256": validate_payload(name, path)}
    return manifest, found


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--archive", action="store_true")
    argument_parser.add_argument(
        "--git-visible",
        action="store_true",
        help="Also require every source-package file to be present in Git's index",
    )
    args = argument_parser.parse_args()
    manifest, files = audit()
    report = {
        "passed": True,
        "files": len(files),
        "bytes": sum(source_path.stat().st_size for source_path in files.values()),
        "model_assets": 0,
        "extraction_entry_points": 0,
    }
    if args.git_visible:
        tracked = set(
            subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=ROOT)
            .decode("utf-8")
            .split("\0")
        )
        missing = sorted(set(files) - tracked)
        if missing:
            raise ValueError("Source files are not Git-visible: " + ", ".join(missing))
        report["git_visible"] = True
    if args.archive:
        dest = ROOT / "dist/dlss5_phoenix-source.zip"
        dest.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(
            dest, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, path in sorted(files.items()):
                archive.write(path, "dlss5_phoenix/" + name)
        report["archive"] = str(dest)
        report["archive_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
