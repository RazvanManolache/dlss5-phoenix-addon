"""Build and run CPU-only IPC ownership/cancellation tests in x86 and x64."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "build"):
        parser.error("Test output must be inside build")
    output.mkdir(parents=True, exist_ok=False)
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    studio = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    results = []
    for bits in (32, 64):
        case = output / str(bits)
        case.mkdir()
        recipe = case / "build.cmd"
        recipe.write_text(f'@echo off\ncall "{studio}/VC/Auxiliary/Build/vcvars{bits}.bat" > "{case}/toolchain.log" 2>&1\n'
            f'cl /nologo /std:c++20 /Zc:__cplusplus /O2 /EHsc /MT /DNOMINMAX '
            f'"{ROOT}/tests/native/compatibility_ipc_tests.cpp" /Fo:"{case}/ipc.obj" /Fe:"{case}/ipc.exe" '
            f'/link d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib bcrypt.lib cabinet.lib shell32.lib ole32.lib uuid.lib '
            f'> "{case}/build.log" 2>&1\nexit /b %errorlevel%\n')
        subprocess.run(["cmd", "/d", "/c", str(recipe)], cwd=case, check=True)
        result = subprocess.run([str(case / "ipc.exe")], cwd=case, capture_output=True, text=True, timeout=20)
        (case / "stdout.log").write_text(result.stdout)
        (case / "stderr.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{bits}-bit IPC tests failed ({result.returncode}): {result.stderr}")
        results.append(json.loads(result.stdout))
        print(result.stdout, flush=True)
    (output / "report.json").write_text(json.dumps({"passed": all(case["passed"] for case in results),
        "gpu_executed": False, "cases": results}, indent=2))


if __name__ == "__main__":
    main()
