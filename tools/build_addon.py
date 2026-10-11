"""Build the addon and test CPU contracts and WARP capture without model files."""

from pathlib import Path
import argparse, hashlib, json, os, re, subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="New directory name under build/")
    parser.add_argument(
        "--ngx-sdk",
        required=True,
        type=Path,
        help="Local SDK header directory containing nvsdk_ngx.h; not redistributed",
    )
    parser.add_argument(
        "--embed-runtime",
        type=Path,
        help="Validated C++ runtime to embed for a single-addon install",
    )
    parser.add_argument(
        "--embed-native", type=Path, help="Validated model-free native kernel bundle to embed"
    )
    parser.add_argument(
        "--production-only", action="store_true", help="Omit the separate diagnostic addon"
    )
    args = parser.parse_args()
    if bool(args.embed_runtime) != bool(args.embed_native):
        parser.error("Supply both embedded runtime and native bundle")
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name):
        parser.error("Use a simple build directory name")
    sdk = args.ngx_sdk.resolve()
    for name in ("nvsdk_ngx.h", "nvsdk_ngx_defs.h", "nvsdk_ngx_params.h"):
        if not (sdk / name).is_file():
            parser.error("Missing local NGX SDK header: " + name)
    out = ROOT / "build" / args.name
    out.mkdir(parents=True, exist_ok=False)
    vswhere = (
        Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    visual_studio_path = subprocess.check_output(
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
    if not visual_studio_path:
        raise RuntimeError("Install the MSVC x64 C++ build tools and Windows SDK")
    lines = [
        "@echo off",
        "setlocal",
        f'call "{visual_studio_path}/VC/Auxiliary/Build/vcvars64.bat" > "{out}/toolchain.log" 2>&1',
        "if errorlevel 1 exit /b 5",
    ]
    common = f'/nologo /std:c++17 /O2 /W4 /EHsc /MD /DNOMINMAX /I"{sdk}" /I"{ROOT}/third_party/nvapi" /I"{ROOT}/third_party/minhook/include"'
    libraries = "d3d12.lib dxgi.lib d3dcompiler.lib user32.lib bcrypt.lib psapi.lib cabinet.lib shell32.lib ole32.lib uuid.lib"
    minhook = " ".join(
        f'"{ROOT}/third_party/minhook/src/{source_path}"'
        for source_path in ("hook.c", "buffer.c", "trampoline.c", "hde/hde64.c")
    )
    tests = [
        ("session_tests", "tests.cpp"),
        ("memory_tests", "memory_tests.cpp"),
        ("observer_tests", "observer_tests.cpp"),
        ("colour_bridge_tests", "colour_bridge_tests.cpp"),
        ("diagnostic_tests", "diagnostic_tests.cpp"),
        ("dispatch_tests", "dispatch_cpu_tests.cpp"),
        ("direct_tests", "direct_tests.cpp"),
        ("dll_loader_tests", "dll_loader_tests.cpp"),
        ("generated_plan_tests", "generated_plan_tests.cpp"),
        ("model_status_tests", "model_status_tests.cpp"),
        ("model_tools_tests", "model_tools_tests.cpp"),
    ]
    software_tests = [("production_capture_tests", "production_capture_tests.cpp"),
                      ("resolution_bridge_tests", "resolution_bridge_tests.cpp")]
    embedded_report = None
    resource = ""
    if args.embed_runtime:
        from embed_runtime import prepare

        recipe, embedded_report = prepare(args.embed_runtime, args.embed_native, out / "embedded")
        resource = f'"{out}/embedded/runtime.res"'
        lines.extend(
            [
                f'rc /nologo /fo {resource} "{recipe}" > "{out}/embedded.build.log" 2>&1',
                "if errorlevel 1 exit /b 12",
            ]
        )
    outputs = []

    def target(name, source, extra="", addon=False):
        extension = ".addon64" if addon else ".exe"
        target_common = common.replace("/MD", "/MT") if addon else common
        payload = resource if addon else ""
        lines.extend(
            [
                f'cl {target_common} {extra} /sourceDependencies "{out}/" "{ROOT}/{source}" /Fo:"{out}\\\\" /Fe:"{out}/{name}{extension}" /link /INCREMENTAL:NO {libraries} {payload} > "{out}/{name}.build.log" 2>&1',
                "if errorlevel 1 exit /b 10",
            ]
        )
        outputs.append(name + extension)

    for name, source in tests + software_tests:
        target(name, "tests/native/" + source)
        test_args = f' "{ROOT}"' if name == "generated_plan_tests" else ""
        lines.extend(
            [
                f'"{out}/{name}.exe"{test_args} > "{out}/{name}.result.json" 2>&1',
                "if errorlevel 1 exit /b 11",
            ]
        )
    flags = f'/DImTextureID=ImU64 /I"{ROOT}/third_party/imgui_reshade" /I"{ROOT}/third_party/reshade" /DPHOENIX_BUILD_TAG={args.name} /LD {minhook}'
    target("phoenix", "src/integration/reshade/addon.cpp", flags, True)
    diagnostic_source = ROOT / "src/integration/reshade/diagnostic_addon.cpp"
    if not args.production_only and diagnostic_source.is_file():
        target("phoenix_diagnostic", "src/integration/reshade/diagnostic_addon.cpp", flags, True)
    lines.append("exit /b 0")
    recipe = out / "build.cmd"
    recipe.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    report = {
        "schema": "phoenix-source-build/1",
        "passed": False,
        "outputs": outputs,
        "model_files_required": False,
        "gpu_executed": False,
        "compilation_completed": False,
        "contract_tests_passed": False,
        "artifact_verification_completed": False,
    }
    if embedded_report:
        report["embedded_runtime"] = embedded_report
    stage = "compile_and_test"
    try:
        subprocess.run(["cmd", "/d", "/c", str(recipe)], cwd=ROOT, check=True)
        report["compilation_completed"] = True
        stage = "read_contract_results"
        report["cpu_tests"] = {
            name: json.loads((out / (name + ".result.json")).read_text()) for name, _ in tests
        }
        report["software_d3d12_tests"] = {
            name: json.loads((out / (name + ".result.json")).read_text())
            for name, _ in software_tests
        }
        if not all(
            result["passed"]
            for section in ("cpu_tests", "software_d3d12_tests")
            for result in report[section].values()
        ):
            raise RuntimeError("Source build test failed")
        report["contract_tests_passed"] = True
        stage = "audit_source_dependencies"
        closure = set()
        for path in out.glob("*.json"):
            content = json.loads(path.read_text())
            if "Data" not in content:
                continue
            for name in [content["Data"]["Source"], *content["Data"]["Includes"]]:
                dependency = Path(name).resolve()
                if dependency.is_relative_to(ROOT):
                    closure.add(dependency.relative_to(ROOT).as_posix())
                elif dependency.is_relative_to(sdk):
                    continue
                elif dependency.is_relative_to(ROOT.parent):
                    raise RuntimeError("Unexpected sibling-checkout dependency: " + name)
        report["project_dependencies"] = sorted(closure)
        report["external_ngx_headers"] = [source_path.name for source_path in sdk.glob("*.h")]
        stage = "verify_artifacts"
        report["binaries"] = {}
        for name in outputs:
            try:
                report["binaries"][name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
            except OSError as error:
                raise RuntimeError(
                    f"Cannot verify final artifact {out / name}: {error}. "
                    "Inspect build logs and security history if the file disappeared or is locked; "
                    "compiler completion alone does not establish a usable output."
                ) from error
        report["artifact_verification_completed"] = True
        report["passed"] = True
    except BaseException as error:
        report["failed_stage"] = stage
        report["error"] = str(error)
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "cpu_suites": len(tests),
                "software_d3d12_suites": len(software_tests),
                "build": str(out),
                "models_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()
