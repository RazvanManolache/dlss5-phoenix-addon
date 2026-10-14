"""Run x86/x64 transport, inference and helper-death probes with a supplied model.

These tests execute the GPU and are separate from model-free source builds.
No NVIDIA DLL or model data is copied into the source repository.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--weight-source", type=Path, required=True)
    parser.add_argument("--full-hd", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    source = args.weight_source.resolve()
    output = package / "validation"
    output.mkdir(exist_ok=False)
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    studio = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                                     "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    libraries = "d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib bcrypt.lib cabinet.lib shell32.lib ole32.lib uuid.lib"
    results = []
    report = dict(schema="phoenix-graphics-validation/1", passed=False, tests=results,
                  nvidia_helper="Not tested by this suite; see documented 2026-10-14 NGX initialization failure",
                  other_graphics_apis="Not tested by this suite")
    report_path = output / "report.json"
    try:
        for bits in (32, 64):
            recipe = output / f"build_probe{bits}.cmd"
            recipe.write_text(f'@echo off\ncall "{studio}/VC/Auxiliary/Build/vcvars{bits}.bat" > "{output}/toolchain{bits}.log" 2>&1\n'
                f'cl /nologo /std:c++20 /Zc:__cplusplus /O2 /EHsc /MT /DNOMINMAX "{ROOT}/tests/native/compatibility_bridge_probe.cpp" '
                f'/Fo:"{output}/probe{bits}.obj" /Fe:"{output}/probe{bits}.exe" /link "{package}/embedded32/runtime.res" '
                f'{libraries} > "{output}/probe{bits}.build.log" 2>&1\nexit /b %errorlevel%\n')
            subprocess.run(["cmd", "/d", "/c", str(recipe)], cwd=output, check=True)
        cases = [(32, "transport", 257, 255), (64, "transport", 257, 255),
                 (32, "phoenix", 257, 255), (64, "phoenix", 257, 255),
                 (32, "split", 257, 255), (32, "cpu_transport", 257, 255),
                 (32, "cpu_phoenix", 257, 255)]
        if args.full_hd:
            cases.append((32, "phoenix", 1920, 1080))
        for bits, mode, width, height in cases:
            folder = output / f"{bits}_{mode}_{width}x{height}"
            folder.mkdir()
            with (folder / "stderr.log").open("w") as errors:
                result = subprocess.run([str(output / f"probe{bits}.exe"), str(source), mode, str(width), str(height)],
                                        cwd=folder, text=True, stdout=subprocess.PIPE, stderr=errors, timeout=110)
            (folder / "stdout.log").write_text(result.stdout)
            if result.returncode:
                raise RuntimeError(f"{bits}-bit {mode} failed: {(folder / 'stderr.log').read_text()}")
            evidence = json.loads(result.stdout)
            if not evidence["passed"]:
                raise RuntimeError("Probe reported failure")
            results.append(evidence)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(evidence), flush=True)
        matching = [item["output_sha256"] for item in results
                    if item["mode"] in ("phoenix", "cpu_phoenix") and item["width"] == 257]
        if len(set(matching)) != 1:
            raise RuntimeError("x86, x64 and CPU-transport Phoenix outputs differ")
        report.update(passed=True, cross_bitness_and_transport_output_identical=True)
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
