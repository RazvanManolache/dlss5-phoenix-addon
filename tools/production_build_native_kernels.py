"""Build the single production kernel set; no tuning catalogues are required."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Fresh directory name under build/")
    parser.add_argument("--compiler", type=Path, required=True, help="CUDA nvcc.exe path")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name) or not 1 <= args.jobs <= 8:
        parser.error("Use a simple build name and between 1 and 8 compiler jobs")
    try:
        catalogue, _ = production_inputs(ROOT)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    compiler = args.compiler.resolve()
    assembler = compiler.with_name("ptxas.exe")
    if not compiler.is_file() or not assembler.is_file():
        parser.error("CUDA nvcc.exe and its sibling ptxas.exe are required")
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
    out.mkdir(parents=True, exist_ok=False)
    # Runtime bundle filenames stay compatible with the existing addon loader.
    shutil.copyfile(ROOT / GRAPH, out / "native_graph.json")
    architecture = catalogue["architecture"]

    def compile_kernel(row):
        source = ROOT / row["source"]
        # Source paths already identify the operation; avoid a second identifier
        # in the production catalogue just to name build-output directories.
        relative = source.resolve().relative_to((ROOT / "src/native/kernels/production").resolve())
        folder = out / relative.with_suffix("")
        folder.mkdir(parents=True)
        binary = folder / "kernel.cubin"
        # These flags reproduce the selected production build. There are no
        # alternative profiles or register-limit switches in this entry point.
        options = row.get("ptx_options")
        optimization = row.get("assembler_optimization", 3)
        generated_ptx = folder / "kernel.ptx"
        command = [
            str(compiler),
            "--entries",
            row["entry"],
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
        recipe.write_bytes(
            (
                '@echo off\r\ncall "'
                + installation
                + '/VC/Auxiliary/Build/vcvars64.bat"\r\n'
                + "if errorlevel 1 exit /b 5\r\n"
                + subprocess.list2cmdline(command)
                + "\r\n"
            ).encode("utf-8")
        )
        invocation = ["cmd", "/d", "/c", str(recipe)]
        with (folder / "build.log").open("wb") as log:
            subprocess.run(
                invocation,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
            if options:
                configure_boundary_ptx(generated_ptx, architecture, row["entry"], options)
                subprocess.run(
                    [str(assembler), "-arch=" + architecture, "--entry", row["entry"],
                     f"-O{optimization}", "-v", "--suppress-debug-info", str(generated_ptx), "-o", str(binary)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600,
                )
        print("Compiled: " + row["name"], flush=True)
        return {
            **{key: row[key] for key in ("name", "entry", "parameter_bytes", "block")},
            "binary": binary.relative_to(out).as_posix(),
            "binary_sha256": digest(binary),
        }

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
    }
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            report["kernels"] = list(pool.map(compile_kernel, catalogue["kernels"]))
        report["passed"] = True
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        (out / "native_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"passed": True, "kernels": 41, "build": str(out)}))


if __name__ == "__main__":
    main()
