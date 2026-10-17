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
    common = f'/nologo /std:c++20 /Zc:__cplusplus /O2 /W4 /EHsc /MD /DNOMINMAX /I"{sdk}" /I"{ROOT}/third_party/nvapi" /I"{ROOT}/third_party/minhook/include"'
    libraries = "d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib bcrypt.lib psapi.lib cabinet.lib shell32.lib ole32.lib uuid.lib windowscodecs.lib"
    minhook = " ".join(
        f'"{ROOT}/third_party/minhook/src/{source_path}"'
        for source_path in ("hook.c", "buffer.c", "trampoline.c", "hde/hde64.c")
    )
    tests = [
        ("session_tests", "tests.cpp"),
        ("feeder_api_tests", "feeder_api_tests.cpp"),
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
        ("module_capability_tests", "module_capability_tests.cpp"),
        ("nr_forwarder_tests", "nr_forwarder_tests.cpp"),
        ("compatibility_policy_tests", "compatibility_policy_tests.cpp"),
        ("graphics_transport_tests", "graphics_transport_tests.cpp"),
        ("backbuffer_route_tests", "backbuffer_route_tests.cpp"),
        ("backbuffer_shader_tests", "backbuffer_shader_tests.cpp"),
        ("compatibility_ipc_tests", "compatibility_ipc_tests.cpp"),
        ("screenshot_tests", "screenshot_tests.cpp"),
    ]
    software_tests = [("production_capture_tests", "production_capture_tests.cpp"),
                      ("resolution_bridge_tests", "resolution_bridge_tests.cpp")]
    embedded_report = None
    resource = ""
    if args.embed_runtime:
        from embed_runtime import prepare
        forwarder_source = ROOT / 'src/integration/reshade/nr_forwarder.cpp'
        forwarder_digest = hashlib.sha256(forwarder_source.read_bytes()).hexdigest()
        forwarder = out / 'phoenix-nvngx.dll'
        wrapper_recipe = out / 'nr_forwarder.cmd'
        wrapper_recipe.write_text('\n'.join([
            '@echo off',
            f'call "{visual_studio_path}/VC/Auxiliary/Build/vcvars64.bat" > "{out}/nr_forwarder.toolchain.log" 2>&1',
            'if errorlevel 1 exit /b 5',
            f'cl {common.replace("/MD", "/MT")} /LD "{forwarder_source}" /Fo:"{out}/nr_forwarder.obj" /Fe:"{forwarder}" /link /INCREMENTAL:NO > "{out}/nr_forwarder.build.log" 2>&1',
            'exit /b %errorlevel%',
        ]))
        subprocess.run(['cmd', '/d', '/c', str(wrapper_recipe)], cwd=out, check=True)
        if hashlib.sha256(forwarder_source.read_bytes()).hexdigest() != forwarder_digest:
            raise RuntimeError('NR forwarder source changed during compilation')
        recipe, embedded_report = prepare(args.embed_runtime, args.embed_native, out / "embedded", forwarder)
        resource = f'"{out}/embedded/runtime.res"'
        lines.extend(
            [
                f'rc /nologo /fo {resource} "{recipe}" > "{out}/embedded.build.log" 2>&1',
                "if errorlevel 1 exit /b 12",
            ]
        )
    # Capture source identity before compilation; receipts must not silently
    # bless files changed while a build/test is running.
    source_suffixes = {'.h', '.hpp', '.c', '.cpp', '.cu', '.cuh', '.inc', '.py', '.json'}
    source_inputs = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for folder in ('src', 'native_runtime', 'third_party', 'tools', 'tests')
                     for path in (ROOT / folder).rglob('*')
                     if path.is_file() and path.suffix in source_suffixes}
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
        if name == 'nr_forwarder_tests' and args.embed_runtime:
            test_args = f' "{forwarder}"'
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
        for name, digest in source_inputs.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Source changed during compilation/tests: " + name)
        if not set(closure).issubset(source_inputs):
            raise RuntimeError("Compiler used source outside the pre-build identity snapshot")
        report["source_inputs"] = source_inputs
        if args.embed_runtime:
            manifest = json.loads((out / 'embedded/manifest.json').read_text())
            linked = [name + '.obj' for name in ('addon', 'hook', 'buffer', 'trampoline', 'hde64')]
            linked += ['embedded/manifest.json']
            linked += [f"embedded/{entry['resource']}.bin" for entry in manifest['files']]
            report['link_inputs'] = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                     for name in linked}
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
