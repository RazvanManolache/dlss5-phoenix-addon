"""Assemble local C++ runtime binaries; never includes a model or compiler toolchain."""

from pathlib import Path
import argparse, hashlib, json, shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("build")
    argument_parser.add_argument("package")
    argument_parser.add_argument("--config", type=Path)
    argument_parser.add_argument(
        "--dependency",
        action="append",
        type=Path,
        default=[],
        help="Vendor runtime DLL to place beside the compiled modules",
    )
    argument_parser.add_argument(
        "--opencl-cpu",
        type=Path,
        help="Optional private CPU ICD directory; never registers it globally",
    )
    args = argument_parser.parse_args()
    for value in (args.build, args.package):
        if not value.replace("_", "").isalnum():
            argument_parser.error("Simple fresh build/package directory names required")
    source = ROOT / "build" / args.build
    report = json.loads((source / "build_report.json").read_text(encoding="utf-8"))
    if not report["all_requested_compiled"]:
        raise RuntimeError("Build contains failed targets")
    for target in ("runtime", "cpu", "broker"):
        if target not in report["targets"]:
            raise RuntimeError("Package requires " + target)
    out = ROOT / "build" / args.package
    out.mkdir(exist_ok=False)
    origins = {}

    def copy(path, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        origins[dest.relative_to(out).as_posix()] = str(path)

    for row in report["targets"].values():
        if "artifact" in row:
            path = ROOT / row["artifact"]
            if path.suffix in (".dll", ".exe", ".addon64"):
                copy(
                    path,
                    out / "opencl_cpu" / path.name
                    if path.parent.name == "opencl_cpu"
                    else out / path.name,
                )
    for pattern in ("*.dll", "*.spv"):
        for path in (source / "vulkan").glob(pattern):
            copy(path, out / "vulkan" / path.name)
    for path in args.dependency:
        if path.suffix.lower() != ".dll":
            raise ValueError("Dependencies must be DLLs")
        copy(path, out / path.name)
    if args.opencl_cpu:
        if not (out / "opencl_cpu/OpenCL.dll").is_file():
            raise RuntimeError("Private CPU ICD requires the opencl_cpu_proxy build target")
        for path in args.opencl_cpu.iterdir():
            if (
                path.is_file()
                and path.suffix.lower() not in (".bat", ".exe")
                and path.name != "OpenCL.dll"
            ):
                copy(path, out / "opencl_cpu" / path.name)
    copy(
        args.config or ROOT / "native_runtime/phoenix_native_cpp.example.json",
        out / "phoenix_native_cpp.json",
    )
    copy(source / "build_report.json", out / "build_report.json")
    files = {
        package_path.relative_to(out).as_posix(): hashlib.sha256(
            package_path.read_bytes()
        ).hexdigest()
        for package_path in out.rglob("*")
        if package_path.is_file()
    }
    manifest = {
        "schema": "phoenix-portable-cpp-package/1",
        "compiled": True,
        "inference_validated": False,
        "python_required": False,
        "torch_required": False,
        "model_assets": 0,
        "vendor_runtime_notice": "Local assembly only; preserve and review vendor redistribution terms separately",
        "files": files,
        "origins": origins,
    }
    (out / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(out)


if __name__ == "__main__":
    main()
