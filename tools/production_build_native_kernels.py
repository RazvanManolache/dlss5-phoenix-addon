"""Build architecture-specific production kernels, keeping their launch ABI shared."""

from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("src/native/model")
CATALOGUE = MODEL / "production_kernel_contracts.json"
GRAPH = MODEL / "production_native_graph.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def production_inputs(root, *, check_sources=True):
    """Validate the fixed graph/kernel interface before creating build outputs."""
    catalogue = json.loads((root / CATALOGUE).read_text(encoding="utf-8"))
    graph = json.loads((root / GRAPH).read_text(encoding="utf-8"))
    if catalogue.get("schema") != "phoenix-native-production-kernels/1":
        raise ValueError("Unsupported production kernel catalogue")
    if not re.fullmatch(r"sm_\d+", catalogue.get("architecture", "")):
        raise ValueError("Invalid production GPU architecture")
    rows = catalogue["kernels"]
    if len(rows) != 41:
        raise ValueError("Production catalogue must contain 41 kernels")
    names, sources = set(), set()
    kernel_root = (root / "src/native/kernels/production").resolve()
    for row in rows:
        source = (root / row["source"]).resolve()
        if source in sources or row["name"] in names:
            raise ValueError("Duplicate production source/kernel")
        sources.add(source)
        names.add(row["name"])
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", row["entry"]):
            raise ValueError("Invalid production entry point")
        large_entry = row.get("large_input_entry")
        large_minimum = row.get("large_input_min_pixels")
        if large_entry is not None or large_minimum is not None:
            if (
                not isinstance(large_entry, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", large_entry)
                or large_entry == row["entry"]
                or type(large_minimum) is not int
                or not 0 < large_minimum <= 16384 * 16384
                or row.get("ptx_options") is not None
            ):
                raise ValueError("Invalid large-input kernel entry/threshold")
        block = row["block"]
        if (
            len(block) != 3
            or any(type(axis) is not int or axis < 1 for axis in block)
            or math.prod(block) > 1024
        ):
            raise ValueError("Invalid production thread block")
        size = row["parameter_bytes"]
        if type(size) is not int or not 0 < size <= 264 or size % 4:
            raise ValueError("Invalid production argument size")
        if not source.is_relative_to(kernel_root) or source.suffix != ".cu":
            raise ValueError("Production source must be inside the production kernel folder")
        options = row.get("ptx_options")
        if options is not None and options != {"texture_mode": "independent", "max_registers": 128}:
            raise ValueError("Unsupported frame-boundary PTX options")
        optimization = row.get("assembler_optimization", 3)
        if type(optimization) is not int or optimization not in (1, 2, 3):
            raise ValueError("Invalid production assembler optimization level")
        if check_sources and not source.is_file():
            raise ValueError("Missing production source: " + row["source"])
    if graph.get("schema") != "phoenix-native-graph/1" or len(graph["layers"]) != 154:
        raise ValueError("Unsupported production graph")
    if {layer["kernel"] for layer in graph["layers"]} != names:
        raise ValueError("Production graph and kernel catalogue differ")
    return catalogue, graph


def configure_boundary_ptx(source, architecture, entry, options):
    """Apply the D3D12 sampler mode and the selected register budget to generated PTX.

    nvcc emits unified texture mode even when inline PTX supplies separate texture
    and sampler handles. These two stages require independent mode. The register
    budget is applied after CUDA compilation to reproduce the selected production
    kernels; passing a different nvcc register flag can change instruction order.
    """
    text = source.read_text(encoding="utf-8")
    target = re.compile(r"(?m)^\s*\.target " + re.escape(architecture) + r"\s*$")
    text, targets = target.subn(".target " + architecture + ", texmode_independent", text)
    declaration = re.compile(
        r"(\.visible\s+\.entry\s+" + re.escape(entry)
        + r"\s*\([^)]*\)\s*\.maxntid[^\n]*\n)"
    )
    text, entries = declaration.subn(
        lambda match: match[1] + ".maxnreg " + str(options["max_registers"]) + "\n", text
    )
    if targets != 1 or entries != 1 or len(re.findall(r"\.visible\s+\.entry\b", text)) != 1:
        raise ValueError("Unexpected generated frame-boundary PTX layout")
    source.write_text(text, encoding="utf-8")


def preserved_kernels(folder, catalogue):
    """Authenticate the shared ABI and exact cubins before preserving a fast path."""
    folder = folder.resolve()
    manifest = json.loads((folder / "native_manifest.json").read_text(encoding="utf-8"))
    if (not manifest.get("passed") or manifest.get("model_weights_included") is not False or
            digest(folder / "native_graph.json") != manifest["native_graph_sha256"]):
        raise ValueError("Preserved native bundle has a different or unverified graph")
    graph = json.loads((folder / "native_graph.json").read_text(encoding="utf-8"))
    current_graph = json.loads((ROOT / GRAPH).read_text(encoding="utf-8"))
    # These additive identity fields migrated the same graph to resource-based
    # authentication. Tensor layouts, launch plans and existing identities must match.
    for field in ("model_architecture", "reference_model_resource"):
        if field not in graph and field in current_graph:
            graph[field] = current_graph[field]
    if graph != current_graph:
        raise ValueError("Preserved native graph execution contract differs")
    kernels = {row["name"]: row for row in manifest["kernels"]}
    if len(kernels) != 41 or len(manifest["kernels"]) != 41:
        raise ValueError("Preserved native bundle must contain 41 unique kernels")
    result = {}
    for row in catalogue["kernels"]:
        base = kernels[row["name"]]
        for field in ("entry", "parameter_bytes", "block", "large_input_entry", "large_input_min_pixels"):
            if base.get(field) != row.get(field):
                raise ValueError("Preserved native kernel ABI differs: " + row["name"])
        binary = (folder / base["binary"]).resolve()
        if not binary.is_relative_to(folder) or digest(binary) != base["binary_sha256"]:
            raise ValueError("Preserved native kernel digest/path mismatch")
        data = binary.read_bytes()
        if (len(data) < 52 or data[:6] != b"\x7fELF\x02\x01" or
                int.from_bytes(data[18:20], "little") != 190 or
                (int.from_bytes(data[48:52], "little") >> 8) & 255 != 120):
            raise ValueError("Preserved native binary must be an sm_120 cubin")
        result[row["name"]] = binary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Directory name under build/; must be fresh unless --resume is used")
    parser.add_argument("--compiler", type=Path, required=True, help="CUDA nvcc.exe path")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted build after checking its source snapshot and cubins")
    parser.add_argument("--compile-timeout", type=int, default=3600,
                        help="Maximum seconds per compiler/assembler invocation")
    parser.add_argument("--architectures", nargs="+", choices=("sm_75", "sm_86", "sm_89", "sm_120"),
                        default=["sm_75", "sm_86", "sm_89", "sm_120"])
    parser.add_argument("--preserve-sm120", type=Path,
                        help="Reuse authenticated sm_120 cubins from an existing bundle unchanged")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name) or not 1 <= args.jobs <= 8:
        parser.error("Use a simple build name and between 1 and 8 compiler jobs")
    if len(set(args.architectures)) != len(args.architectures):
        parser.error("GPU architectures must be unique")
    if args.compile_timeout < 60:
        parser.error("Compiler timeout must be at least 60 seconds")
    try:
        catalogue, _ = production_inputs(ROOT)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    compiler = args.compiler.resolve()
    assembler = compiler.with_name("ptxas.exe")
    if not compiler.is_file() or not assembler.is_file():
        parser.error("CUDA nvcc.exe and its sibling ptxas.exe are required")
    packer = compiler.with_name("fatbinary.exe")
    if len(args.architectures) > 1 and not packer.is_file():
        parser.error("CUDA fatbinary.exe is required for multi-architecture modules")
    preserved = preserved_kernels(args.preserve_sm120, catalogue) if args.preserve_sm120 else {}
    vswhere = (
        Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    installation = subprocess.check_output(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    if not installation:
        parser.error("MSVC x64 build tools and Windows SDK are required")
    out = ROOT / "build" / args.name
    source_hashes = {path.relative_to(ROOT).as_posix(): digest(path)
                     for path in (ROOT / "src/native/kernels/production").rglob("*")
                     if path.suffix in (".cu", ".cuh", ".h")}
    if args.resume:
        previous = json.loads((out / "native_manifest.json").read_text(encoding="utf-8"))
        if (previous.get("source_hashes") != source_hashes or
                previous.get("architectures") != args.architectures or
                previous.get("native_graph_sha256") != digest(ROOT / GRAPH) or
                digest(out / "native_graph.json") != digest(ROOT / GRAPH)):
            parser.error("Cannot resume a build with different sources, graph or architectures")
        if previous.get("catalogue_sha256", digest(ROOT / CATALOGUE)) != digest(ROOT / CATALOGUE):
            parser.error("Cannot resume a build with a different kernel catalogue")
    else:
        out.mkdir(parents=True, exist_ok=False)
        # Runtime bundle filenames stay compatible with the existing addon loader.
        shutil.copyfile(ROOT / GRAPH, out / "native_graph.json")
    def compile_kernel(job):
        row, architecture = job
        source = ROOT / row["source"]
        # Source paths already identify the operation; avoid a second identifier
        # in the production catalogue just to name build-output directories.
        relative = source.resolve().relative_to((ROOT / "src/native/kernels/production").resolve())
        folder = out / relative.with_suffix("")
        if len(args.architectures) > 1:
            folder /= architecture
        folder.mkdir(parents=True, exist_ok=args.resume)
        binary = folder / "kernel.cubin"
        if architecture == "sm_120" and preserved:
            shutil.copyfile(preserved[row["name"]], binary)
            return row, architecture, binary
        # These flags reproduce the selected production build. There are no
        # alternative profiles or register-limit switches in this entry point.
        options = row.get("ptx_options")
        optimization = row.get("assembler_optimization", 3)
        generated_ptx = folder / "kernel.ptx"
        command = [
            str(compiler),
            "--entries",
            ",".join([row["entry"]] + ([row["large_input_entry"]] if "large_input_entry" in row else [])),
            "-allow-unsupported-compiler",
            "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
            "-diag-suppress",
            "177",
            "-diag-suppress",
            "550",
            "-ptx" if options else "-cubin",
            "-arch=" + architecture,
            "-O3",
            "-std=c++17",
            *(["--split-compile=2"] if architecture != "sm_120" else []),
            "-lineinfo",
            # Device line tables are offline build artifacts. Keeping them in
            # every shipping module increases CPU storage without changing
            # executable instructions or constant data.
            *([] if options else [f"-Xptxas=-v,-O{optimization},--suppress-debug-info"]),
            # Relative paths keep generated source labels within the checkout.
            source.relative_to(ROOT).as_posix(),
            "-o",
            str(generated_ptx if options else binary),
        ]
        recipe = folder / "build.cmd"
        recipe_bytes = (
            '@echo off\r\ncall "'
            + installation
            + '/VC/Auxiliary/Build/vcvars64.bat"\r\n'
            + "if errorlevel 1 exit /b 5\r\n"
            + subprocess.list2cmdline(command)
            + "\r\n"
        ).encode("utf-8")
        if args.resume and binary.is_file() and recipe.is_file() and recipe.read_bytes() == recipe_bytes:
            data = binary.read_bytes()
            entries = [row["entry"]] + ([row["large_input_entry"]] if "large_input_entry" in row else [])
            if (len(data) >= 64 and data[:6] == b"\x7fELF\x02\x01" and
                    int.from_bytes(data[18:20], "little") == 190 and
                    (int.from_bytes(data[48:52], "little") >> 8) & 255 == int(architecture[3:]) and
                    all(entry.encode() + b"\0" in data for entry in entries)):
                # A timeout can occur after ptxas finished writing its output.
                # Parse the entire ELF before retaining it; header checks alone
                # could accept a truncated compiler output.
                parsed = subprocess.run([str(compiler.with_name("cuobjdump.exe")), "--dump-elf-symbols", str(binary)],
                                        capture_output=True, timeout=60)
                if parsed.returncode == 0 and all(re.search(
                        rb"(?m)^STT_FUNC\s+STB_GLOBAL\s+STO_ENTRY\s+" + re.escape(entry.encode()) + rb"\s*$",
                        parsed.stdout) for entry in entries):
                    print("Reused " + architecture + ": " + row["name"], flush=True)
                    return row, architecture, binary
        recipe.write_bytes(recipe_bytes)
        invocation = ["cmd", "/d", "/c", str(recipe)]
        with (folder / "build.log").open("wb") as log:
            subprocess.run(
                invocation,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.compile_timeout,
            )
            if options:
                configure_boundary_ptx(generated_ptx, architecture, row["entry"], options)
                subprocess.run(
                    [str(assembler), "-arch=" + architecture, "--entry", row["entry"],
                     f"-O{optimization}", "-v", "--suppress-debug-info", str(generated_ptx), "-o", str(binary)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.compile_timeout,
                )
        print("Compiled " + architecture + ": " + row["name"], flush=True)
        return row, architecture, binary

    report = {
        "schema": "phoenix-native-generated-build/1",
        "passed": False,
        "model_weights_included": False,
        "captured_plans_included": False,
        "native_graph_sha256": digest(out / "native_graph.json"),
        # Compilation alone must not satisfy the embedding tool's requirement
        # for a separately validated native bundle.
        "selection_status": "compiled",
        "validation_pending": True,
        "architectures": args.architectures,
        "architecture_validation": {arch: "pending_device_inference" for arch in args.architectures},
        "source_hashes": source_hashes,
        "catalogue_sha256": digest(ROOT / CATALOGUE),
        "resumed": args.resume,
    }
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(compile_kernel, (row, arch))
                       for row in catalogue["kernels"] for arch in args.architectures]
            try:
                compiled = [future.result() for future in concurrent.futures.as_completed(futures)]
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        if any(digest(ROOT / path) != pin for path, pin in source_hashes.items()):
            raise ValueError("Native kernel sources changed during compilation")
        report["kernels"] = []
        for row in catalogue["kernels"]:
            images = [(arch, binary) for item, arch, binary in compiled if item["name"] == row["name"]]
            images.sort(key=lambda image: args.architectures.index(image[0]))
            binary = images[0][1]
            if len(images) > 1:
                binary = binary.parent.parent / "kernel.fatbin"
                subprocess.run([str(packer), "--64", "--create", str(binary), "--compress-all",
                    *["--image3=file=" + str(path) + ",kind=elf,sm=" + arch[3:] for arch, path in images]],
                    check=True, capture_output=True, timeout=120)
            report["kernels"].append({
                **{key: row[key] for key in ("name", "entry", "parameter_bytes", "block")},
                **{key: row[key] for key in ("large_input_entry", "large_input_min_pixels") if key in row},
                "binary": binary.relative_to(out).as_posix(), "binary_sha256": digest(binary),
                "architecture_images": [{"architecture": arch, "sha256": digest(path), "bytes": path.stat().st_size}
                                        for arch, path in images],
            })
        report["sm120_preserved_byte_exact"] = bool(preserved) and "sm_120" in args.architectures
        report["passed"] = True
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        (out / "native_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"passed": True, "kernels": 41, "architectures": args.architectures, "build": str(out)}))


if __name__ == "__main__":
    main()
