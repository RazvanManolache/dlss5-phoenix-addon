"""Focused native RTX CPU contracts and build. Does not create GPU devices."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from test_model_selection import ROOT, toolchain, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ngx-include", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontend", action="store_true", help="Also link the x64 addon and refresh its runtime catalogue")
    args = parser.parse_args()
    work = args.build.resolve()
    work.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed":False,"state":"incomplete","gpu_executed":False}) + "\n", encoding="utf-8")
    compiler, env = toolchain(64)
    common = [compiler, "/nologo", "/std:c++20", "/Zc:__cplusplus", "/EHsc", "/bigobj", "/O2", "/fp:strict", "/MD", "/DNOMINMAX",
              "/I" + str(args.ngx_include.resolve()), "/I" + str(ROOT / "third_party/nvapi")]
    libraries = ["d3d12.lib", "dxgi.lib", "d3dcompiler.lib", "bcrypt.lib"]
    results = []
    for name, arguments in [
        ("native_small_model_tests", [ROOT, args.bundle.resolve(), args.model.resolve()]),
        ("generated_plan_tests", [ROOT]),
    ]:
        executable = work / (name + ".exe")
        run(common + [ROOT / ("tests/native/" + name + ".cpp"), f"/Fe:{executable}", f"/Fo:{work / (name + '.obj')}", *libraries], cwd=work, env=env)
        result = json.loads(run([executable, *arguments], cwd=work, env=env))
        results.append({"test": name, "result": result})
        print(json.dumps(results[-1]), flush=True)
    native = work / "phoenix_native.dll"
    run(common + ["/LD", ROOT / "native_runtime/native_api.cpp", f"/Fe:{native}", f"/Fo:{work / 'native_api.obj'}", *libraries,
                  "/link", f"/IMPLIB:{work / 'phoenix_native.lib'}"], cwd=work, env=env)
    executable = work / "native_model_selection_tests.exe"
    run(common + [ROOT / "tests/native/native_model_selection_tests.cpp", f"/Fe:{executable}", f"/Fo:{work / 'native_selection.obj'}",
                  work / "phoenix_native.lib"], cwd=work, env=env)
    result = json.loads(run([executable], cwd=work, env=env))
    results.append({"test": "native_model_selection_tests", "result": result})
    if args.frontend:
        runtime = work / "phoenix_runtime_cpp.dll"
        run(common + ["/LD", ROOT / "native_runtime/runtime.cpp", f"/Fe:{runtime}", f"/Fo:{work / 'runtime.obj'}", "bcrypt.lib",
                      "/link", f"/IMPLIB:{work / 'phoenix_runtime_cpp.lib'}"], cwd=work, env=env)
        includes = ["/I" + str(ROOT / ("third_party/" + name)) for name in ("minhook/include", "imgui_reshade", "reshade")]
        sources = [ROOT / ("third_party/minhook/src/" + name) for name in ("hook.c", "buffer.c", "trampoline.c", "hde/hde64.c")]
        addon = work / "phoenix.addon64"
        run(common + ["/LD", "/DImTextureID=ImU64", "/DPHOENIX_BUILD_TAG=native_small_model", *includes,
                      ROOT / "src/integration/reshade/addon.cpp", *sources, f"/Fe:{addon}", "/link",
                      *libraries, "d3d11.lib", "user32.lib", "psapi.lib", "cabinet.lib", "shell32.lib", "ole32.lib", "uuid.lib", "windowscodecs.lib"],
            cwd=work, env=env)
        for artifact in (runtime,addon):
            shutil.copy2(artifact, args.output.parent / artifact.name)
    report = {"passed": all(item["result"]["passed"] for item in results), "gpu_executed": False,
              "vendor_reference_executed": False, "native_rtx_small_model_implemented": True, "cases": results,
              "native_dll_sha256": hashlib.sha256(native.read_bytes()).hexdigest()}
    report["x64_addon_linked"] = args.frontend
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(native, args.output.parent / native.name)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
