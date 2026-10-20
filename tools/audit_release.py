"""Audit the production source tree and optionally make a source-only ZIP."""

from pathlib import Path, PurePosixPath
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
    "artifacts",
    "validation",
    ".git",
    ".vscode",
    ".idea",
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
    ".addon32",
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
    ".log",
    ".dmp",
}
MODEL_SOURCE_ROOTS = (
    ("native_runtime", "models"),
    ("src", "native", "model", "models"),
)
MODEL_SOURCE_SUFFIXES = {".c", ".cpp", ".h", ".hpp", ".cuh", ".inc", ".md"}


def candidates():
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    for name in sorted(set(filter(None, names))):
        path = ROOT / name
        rel = path.relative_to(ROOT)
        folded = tuple(part.casefold() for part in rel.parts)
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
            folded[0] in IGNORED
            or any(part in {".git", ".vscode", ".idea", "__pycache__"} for part in folded)
            or path.suffix.lower() == ".pyc"
        ):
            continue
        for component in (path, *path.parents):
            if component == ROOT.parent:
                break
            if component.is_symlink() or getattr(component, "is_junction", lambda: False)():
                raise ValueError("Publication redirect is not allowed: " + rel.as_posix())
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
    validate_public_content(name, raw)
    return hashlib.sha256(raw).hexdigest()


def validate_public_content(name, raw):
    """Reject host-specific paths and credential formats without echoing matches."""
    # Device IDs are machine identifiers too. Only explicit parser-test examples
    # may use UUID-shaped strings; measurement exports use descriptive labels.
    synthetic = {b'cuda:GPU-01234567-89ab-cdef-0123-456789abcdef',
                 b'vulkan:0123456789abcdef0123456789abcdef',
                 b'vulkan:0123456789abcdef0123456789abcdef0'}
    for match in re.finditer(rb'(?:cuda:GPU-[0-9a-fA-F-]{36}|vulkan:[0-9a-fA-F]{32,33})', raw):
        if match.group() not in synthetic:
            raise ValueError('Private device identifier in publication file: ' + name)
    patterns = (
        rb"(?i)[a-z]:[/\\\\]Users[/\\\\](?!Public(?:[/\\\\]|$))[^/\\\\\s]+",
        rb"(?i)[ed]:[/\\\\](?:Games|Kit|AI)(?:[/\\\\]|$)",
        rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})",
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    )
    for pattern in patterns:
        if re.search(pattern, raw):
            raise ValueError("Private path or credential pattern in publication file: " + name)


def validate_source_path(name):
    """Distinguish architecture source from downloaded/extracted model data."""
    path = PurePosixPath(name)
    parts = path.parts
    folded = tuple(part.casefold() for part in parts)
    if (not parts or path.is_absolute() or ".." in parts or "\\" in name or ":" in name or
            folded[0] == "runtime" or any(
                part in {".git", ".vscode", ".idea", "captures", "traces", "backups", "archive"}
                for part in folded)):
        raise ValueError("Private or unsafe source-package path: " + name)
    if "models" not in folded:
        return
    # These directories contain maintained C++ graphs/layout declarations, not
    # model payloads. File-type and executable-magic checks still apply below.
    model_source = folded.count("models") == 1 and path.suffix.lower() in MODEL_SOURCE_SUFFIXES and any(
        folded[:len(root)] == root for root in MODEL_SOURCE_ROOTS)
    if name.casefold() != "models/readme.md" and not model_source:
        raise ValueError("Private model payload directory: " + name)


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
        validate_source_path(name)
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
