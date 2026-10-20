"""Focused Windows x86/x64 CPU checks. No game, GPU inference or installation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def toolchain(bits):
    finder = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    studio = subprocess.check_output([str(finder), "-latest", "-products", "*", "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()
    if not studio:
        raise RuntimeError("MSVC tools required")
    setup = Path(studio) / f"VC/Auxiliary/Build/vcvars{bits}.bat"
    lines = subprocess.check_output(f'cmd.exe /d /s /c ""{setup}" >nul && set"', text=True)
    env = {key.upper(): value for key, value in os.environ.items()}
    for line in lines.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            env[key.upper()] = value
    compiler = shutil.which("cl.exe", path=env["PATH"])
    if not compiler:
        raise RuntimeError("Initialized compiler not found")
    return compiler, env


def run(command, *, cwd, env, timeout=240):
    result = subprocess.run([str(value) for value in command], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(str(command[0])).name} failed ({result.returncode})\n{result.stdout}\n{result.stderr}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, nargs="+", choices=(32, 64), default=[32, 64])
    parser.add_argument("--model", type=Path, action="append", default=[], help="Optional read-only original-model DLL")
    parser.add_argument("--small-model", type=Path, help="Optional DLL containing WEIGHTS_CC_SMALL_FP8; decoder/graph checks only")
    parser.add_argument("--output", type=Path, help="Compact JSON receipt only; temporary binaries are removed")
    parser.add_argument("--ngx-include", type=Path, help="Also compile/check the x64 native library; no GPU creation")
    parser.add_argument("--bundle", type=Path, help="Optional read-only native bundle for resource-equivalence checks")
    args = parser.parse_args()
    build_root = ROOT / "build"
    build_root.mkdir(exist_ok=True)
    results = []
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"passed": False, "state": "incomplete", "gpu_used": False}) + "\n", encoding="utf-8")
    for bits in args.bits:
        compiler, env = toolchain(bits)
        with tempfile.TemporaryDirectory(prefix=f"model-selection-{bits}-", dir=build_root) as temporary:
            work = Path(temporary)
            common = [compiler, "/nologo", "/std:c++20", "/Zc:__cplusplus", "/EHsc", "/bigobj", "/O2", "/fp:strict", "/MD", "/DNOMINMAX"]
            for name, source in [
                ("selection", ROOT / "tests/portable_cpp/model_selection_tests.cpp"),
                ("legacy_loader", ROOT / "tests/native/dll_loader_tests.cpp"),
                ("resource_identity", ROOT / "tests/portable_cpp/model_resource_identity_tests.cpp"),
                ("picker_selection", ROOT / "tests/native/model_picker_selection_tests.cpp"),
            ]:
                executable = work / f"{name}.exe"
                run(common + [source, f"/Fe:{executable}", f"/Fo:{work / (name + '.obj')}", "bcrypt.lib",
                    "cabinet.lib", "shell32.lib", "ole32.lib", "uuid.lib", "user32.lib"], cwd=work, env=env)
                arguments = [path.resolve() for path in args.model] if name in ("selection", "resource_identity", "picker_selection") else []
                if name == "resource_identity" and args.bundle:
                    arguments = ["--bundle", args.bundle.resolve(), *arguments]
                result = json.loads(run([executable, *arguments], cwd=work, env=env))
                results.append({"bits": bits, "test": name, "result": result})
                print(json.dumps(results[-1]), flush=True)
            dll = work / "phoenix_runtime_cpp.dll"
            if args.small_model:
                executable = work / "small_model.exe"
                run(common + [ROOT / "tests/portable_cpp/small_model_tests.cpp",
                    f"/Fe:{executable}", f"/Fo:{work / 'small_model.obj'}", "bcrypt.lib"], cwd=work, env=env)
                result = json.loads(run([executable, args.small_model.resolve()], cwd=work, env=env))
                results.append({"bits": bits, "test": "small_model", "result": result})
                print(json.dumps(results[-1]), flush=True)
                executable = work / "small_model_refactor_probe.exe"
                run(common + [ROOT / "tests/portable_cpp/small_model_refactor_probe.cpp",
                    f"/Fe:{executable}", f"/Fo:{work / 'small_model_refactor_probe.obj'}", "bcrypt.lib"], cwd=work, env=env)
                result = json.loads(run([executable, ROOT, args.small_model.resolve()], cwd=work, env=env))
                results.append({"bits": bits, "test": "small_model_refactor", "result": result})
                print(json.dumps({"bits": bits, "test": "small_model_refactor", "passed": result["passed"],
                    "portable_configurations": len(result["portable"]), "native_graph_sha256": result["native_graph_sha256"]}), flush=True)
            link = ["/link", f"/IMPLIB:{work / 'phoenix_runtime_cpp.lib'}"]
            if bits == 32:
                link.append(f"/DEF:{ROOT / 'native_runtime/phoenix_runtime.def'}")
            run(common + ["/LD", ROOT / "native_runtime/runtime.cpp", f"/Fe:{dll}",
                f"/Fo:{work / 'runtime.obj'}", "bcrypt.lib", *link], cwd=work, env=env)
            executable = work / "runtime_selection.exe"
            run(common + [ROOT / "tests/portable_cpp/model_runtime_selection_tests.cpp",
                f"/Fe:{executable}", f"/Fo:{work / 'runtime_selection.obj'}",
                work / "phoenix_runtime_cpp.lib", "bcrypt.lib"], cwd=work, env=env)
            result = json.loads(run([executable], cwd=work, env=env))
            results.append({"bits": bits, "test": "runtime_selection", "result": result})
            print(json.dumps(results[-1]), flush=True)
            if bits == 64 and args.ngx_include:
                native = work / "phoenix_native.dll"
                run(common + ["/LD", ROOT / "native_runtime/native_api.cpp",
                    "/I" + str(args.ngx_include.resolve()), "/I" + str(ROOT / "third_party/nvapi"),
                    f"/Fe:{native}", f"/Fo:{work / 'native_api.obj'}", "d3d12.lib", "dxgi.lib",
                    "d3dcompiler.lib", "bcrypt.lib", "/link", f"/IMPLIB:{work / 'phoenix_native.lib'}"], cwd=work, env=env)
                executable = work / "native_model_selection.exe"
                run(common + [ROOT / "tests/native/native_model_selection_tests.cpp", f"/Fe:{executable}",
                    f"/Fo:{work / 'native_selection.obj'}", work / "phoenix_native.lib"], cwd=work, env=env)
                result = json.loads(run([executable], cwd=work, env=env))
                results.append({"bits": bits, "test": "native_model_selection", "result": result})
                print(json.dumps(results[-1]), flush=True)
    report = {"passed": all(case["result"].get("passed", False) for case in results),
        "gpu_used": False, "installed_files_modified": False,
        "temporary_builds_removed": True, "cases": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
