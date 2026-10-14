"""Check legacy FP16 CPU iteration rounding with a matching embedded helper.

Execution uses the GPU. --compile-only builds both client bitnesses without
starting any helper, game, graphics device or inference.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--weight-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    source = args.weight_source.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "build"):
        parser.error("Test output must be inside build")
    output.mkdir(parents=True, exist_ok=False)
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    studio = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    cases = []
    for bits in (32, 64):
        case = output / str(bits)
        case.mkdir()
        recipe = case / "build.cmd"
        recipe.write_text(f'@echo off\ncall "{studio}/VC/Auxiliary/Build/vcvars{bits}.bat" > "{case}/toolchain.log" 2>&1\n'
            f'cl /nologo /std:c++20 /Zc:__cplusplus /O2 /EHsc /MT /DNOMINMAX '
            f'"{ROOT}/tests/native/compatibility_cpu_iterations.cpp" /Fo:"{case}/iterations.obj" /Fe:"{case}/iterations.exe" '
            f'/link "{package}/embedded32/runtime.res" d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib '
            f'bcrypt.lib cabinet.lib shell32.lib ole32.lib uuid.lib > "{case}/build.log" 2>&1\nexit /b %errorlevel%\n')
        subprocess.run(["cmd", "/d", "/c", str(recipe)], cwd=case, check=True)
        item = {"client_bits": bits, "compiled": True, "gpu_executed": False}
        if not args.compile_only:
            result = subprocess.run([str(case / "iterations.exe"), str(source)], cwd=case,
                capture_output=True, text=True, timeout=300)
            (case / "stdout.log").write_text(result.stdout)
            (case / "stderr.log").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f"{bits}-bit iteration test failed ({result.returncode}): {result.stderr}")
            item.update(json.loads(result.stdout))
            print(result.stdout, flush=True)
        cases.append(item)
    (output / "report.json").write_text(json.dumps({"passed": True,
        "compile_only": args.compile_only, "gpu_executed": not args.compile_only,
        "package": str(package), "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
