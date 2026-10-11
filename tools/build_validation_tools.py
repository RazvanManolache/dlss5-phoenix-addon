"""Build offline native inference and embedded-resource tests from tracked source."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Fresh output directory under build/")
    parser.add_argument("--ngx-sdk", required=True, type=Path)
    parser.add_argument("--ngx-library", type=Path, help="Optional NVIDIA nvsdk_ngx_d.lib for offline DLSS SR probe initialization")
    args = parser.parse_args()
    if args.ngx_library and not args.ngx_library.is_file():
        parser.error("--ngx-library must name nvsdk_ngx_d.lib")
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name):
        parser.error("Simple output name required")
    sdk = args.ngx_sdk.resolve()
    if not (sdk / "nvsdk_ngx.h").is_file():
        parser.error("--ngx-sdk must contain nvsdk_ngx.h")
    output = ROOT / "build" / args.name
    output.mkdir(parents=True, exist_ok=False)
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
        raise RuntimeError("MSVC x64 tools and Windows SDK required")
    setup = Path(installation) / "VC/Auxiliary/Build/vcvars64.bat"
    targets = {
        "temporal_upscaler_probe": [ROOT / "tests/native/temporal_upscaler_probe.cpp", ROOT / "src/native/runtime/device_recovery.cpp"],
        "direct_memory_probe": [ROOT / "tests/native/direct_memory_probe.cpp",
                                ROOT / "src/native/runtime/device_recovery.cpp"],
        "native_inference_probe": [
            ROOT / "tests/native/native_inference_probe.cpp",
            ROOT / "src/native/runtime/device_recovery.cpp",
        ],
        "embedded_runtime_tests": [ROOT / "tests/native/embedded_runtime_tests.cpp"],
    }
    report = {"passed": False, "tests_run": False, "targets": {}}
    for name, sources in targets.items():
        dependencies = output / (name + "_dependencies")
        dependencies.mkdir()
        executable = output / (name + ".exe")
        command = [
            "cl",
            "/nologo",
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/MD",
            "/DNOMINMAX",
            "/I" + str(sdk),
            "/I" + str(ROOT / "third_party/nvapi"),
            "/sourceDependencies",
            str(dependencies) + os.sep,
            *map(str, sources),
            "/Fo" + str(output) + os.sep,
            "/Fe" + str(executable),
            "/link",
            "d3d12.lib",
            "dxgi.lib",
            "d3dcompiler.lib",
            "user32.lib",
            "bcrypt.lib",
            "psapi.lib",
            "cabinet.lib",
            "shell32.lib",
            "ole32.lib",
            "uuid.lib",
        ]
        if name in ("temporal_upscaler_probe", "direct_memory_probe") and args.ngx_library:
            command.insert(1,"/DPHOENIX_PROBE_NGX_SDK")
            command.extend([str(args.ngx_library.resolve()), "advapi32.lib", "version.lib", "wintrust.lib"])
        recipe = output / (name + ".cmd")
        recipe.write_text(
            '@echo off\ncall "' + str(setup) + '" >nul\n'
            "if errorlevel 1 exit /b 5\n"
            + subprocess.list2cmdline(command)
            + "\nexit /b %ERRORLEVEL%\n",
            encoding="utf-8",
        )
        with (output / (name + ".log")).open("wb") as log:
            subprocess.run(
                ["cmd.exe", "/d", "/c", str(recipe)],
                cwd=output,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        closure = set()
        for dependency_file in dependencies.glob("*.json"):
            data = json.loads(dependency_file.read_text(encoding="utf-8"))["Data"]
            for included in [data["Source"], *data["Includes"]]:
                path = Path(included).resolve()
                if path.is_relative_to(ROOT):
                    closure.add(path.relative_to(ROOT).as_posix())
                elif path.is_relative_to(sdk):
                    continue
                elif path.is_relative_to(ROOT.parent):
                    raise RuntimeError("Unexpected sibling project include: " + str(path))
        report["targets"][name] = {
            "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "sources": [path.relative_to(ROOT).as_posix() for path in sources],
            "project_dependencies": sorted(closure),
        }
        print("Compiled: " + name, flush=True)
    report["passed"] = True
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
