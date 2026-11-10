"""Build model-free SM120 modules for Linux CUDA texture objects.

Compilation is not runtime validation. This tool never runs GPU work.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from production_build_native_kernels import production_inputs

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("Use 1..8 compiler jobs")
    compiler = args.compiler.resolve()
    assembler = compiler.with_name("ptxas.exe" if compiler.suffix.lower() == ".exe" else "ptxas")
    if not compiler.is_file() or not assembler.is_file():
        parser.error("Provide nvcc with its sibling ptxas")
    catalogue, _ = production_inputs(ROOT)
    source_hashes = {path.relative_to(ROOT).as_posix(): digest(path)
                     for path in (ROOT / "src/native/kernels/production").rglob("*") if path.is_file()}
    catalogue_sha = digest(ROOT / "src/native/model/production_kernel_contracts.json")
    graph_sha = digest(ROOT / "src/native/model/production_native_graph.json")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "native_manifest.json"
    # A failed rebuild must never leave a previous success receipt in place.
    manifest.write_text(json.dumps({"schema": "phoenix-native-generated-build/1", "passed": False}) + "\n")
    version = subprocess.check_output([str(compiler), "--version"], text=True)
    shutil.copyfile(ROOT / "src/native/model/production_native_graph.json", out / "native_graph.json")

    def build(row):
        folder = out / row["name"]
        folder.mkdir(exist_ok=True)
        binary, ptx = folder / "kernel.cubin", folder / "kernel.ptx"
        boundary = row.get("ptx_options")
        entries = [row["entry"]] + ([row["large_input_entry"]] if "large_input_entry" in row else [])
        optimization = row.get("assembler_optimization", 3)
        command = [str(compiler), "--entries", ",".join(entries), "-diag-suppress", "177",
                   "-diag-suppress", "550", "-DPHOENIX_CUDA_TEXTURE_OBJECTS", "-arch=sm_120",
                   "-O3", "-std=c++17", "-ptx" if boundary else "-cubin"]
        if compiler.suffix.lower() == ".exe":
            command += ["-allow-unsupported-compiler", "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"]
        if not boundary:
            command += [f"-Xptxas=-O{optimization},--suppress-debug-info"]
        command += [row["source"], "-o", str(ptx if boundary else binary)]
        (folder / "recipe.json").write_text(json.dumps(command, indent=2) + "\n")
        with (folder / "build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=600, cwd=ROOT)
            if boundary:
                # Keep the production 128-register boundary recipe, but use
                # unified CUDA texture objects instead of D3D12 sampler mode.
                pattern = (r"(\.visible\s+\.entry\s+" + re.escape(row["entry"]) +
                           r"\s*\([^)]*\)\s*\.maxntid[^\n]*\n)")
                text, count = re.subn(pattern, r"\1.maxnreg 128\n", ptx.read_text())
                if count != 1:
                    raise RuntimeError("Native CUDA boundary PTX declaration changed")
                ptx.write_text(text)
                subprocess.run([str(assembler), "-arch=sm_120", "--entry",
                                row["entry"], f"-O{optimization}", "--suppress-debug-info",
                                str(ptx), "-o", str(binary)], stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=600, cwd=ROOT)
        data = binary.read_bytes()
        if (len(data) < 64 or data[:6] != b"\x7fELF\x02\x01" or
                int.from_bytes(data[18:20], "little") != 190 or
                (int.from_bytes(data[48:52], "little") >> 8) & 255 != 120 or
                any(entry.encode() + b"\0" not in data for entry in entries)):
            raise RuntimeError("Invalid native CUDA module: " + row["name"])
        print("Compiled " + row["name"], flush=True)
        return {**{key: row[key] for key in ("name", "entry", "parameter_bytes", "block",
                   "large_input_entry", "large_input_min_pixels") if key in row},
                "binary": binary.relative_to(out).as_posix(), "binary_sha256": digest(binary)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        kernels = list(pool.map(build, catalogue["kernels"]))
    if (any(digest(ROOT / name) != expected for name, expected in source_hashes.items()) or
            digest(ROOT / "src/native/model/production_kernel_contracts.json") != catalogue_sha or
            digest(ROOT / "src/native/model/production_native_graph.json") != graph_sha):
        raise RuntimeError("Native CUDA sources changed during compilation; rebuild from a stable tree")
    report = {
        "schema": "phoenix-native-generated-build/1", "passed": True,
        "model_weights_included": False, "captured_plans_included": False,
        "boundary_abi": "cuda-texture-objects/1", "architectures": ["sm_120"],
        "kernel_style": "auto", "native_graph_sha256": digest(out / "native_graph.json"),
        "kernels": kernels, "selection_status": "compiled", "validation_pending": True,
        "compiler_version": version,
        "source_sha256": source_hashes,
        "catalogue_sha256": catalogue_sha,
    }
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(manifest)


if __name__ == "__main__":
    main()
