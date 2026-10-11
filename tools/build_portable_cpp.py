"""Build Python-free portable inference runners. Python is build tooling only."""

from pathlib import Path
import argparse, json, os, subprocess, hashlib, shutil, time

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native_runtime"


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("name")
    argument_parser.add_argument(
        "--only", nargs="+", default=["cpu", "runtime", "broker", "runner", "device_query"]
    )
    argument_parser.add_argument("--hip-sdk", type=Path)
    argument_parser.add_argument("--sycl-compiler", type=Path)
    argument_parser.add_argument(
        "--cuda-compiler",
        type=Path,
        default=Path(
            os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
        )
        / "bin/nvcc.exe",
    )
    argument_parser.add_argument("--cuda-arch", default="sm_120")
    argument_parser.add_argument("--hip-arch", default="gfx1036")
    argument_parser.add_argument("--vulkan-deps", type=Path)
    argument_parser.add_argument("--ngx-sdk", type=Path)
    args = argument_parser.parse_args()
    for option in ("hip_sdk", "sycl_compiler", "cuda_compiler", "vulkan_deps", "ngx_sdk"):
        value = getattr(args, option)
        if value is not None:
            setattr(args, option, value.resolve())
    known = {
        "cpu",
        "runtime",
        "broker",
        "runner",
        "device_query",
        "probe",
        "offload_probe",
        "client_contracts",
        "runtime_contracts",
        "opencl_cpu_proxy",
        "vulkan_adapter",
        "d3d12_transport",
        "cuda",
        "hip",
        "sycl",
        "addon",
        "vulkan_shaders",
    }
    if set(args.only) - known:
        argument_parser.error("Unknown target(s): " + str(set(args.only) - known))
    for target, option in [
        ("hip", "hip_sdk"),
        ("sycl", "sycl_compiler"),
        ("vulkan_shaders", "vulkan_deps"),
        ("addon", "ngx_sdk"),
        ("offload_probe", "ngx_sdk"),
    ]:
        if target in args.only and getattr(args, option) is None:
            argument_parser.error(target + " requires --" + option.replace("_", "-"))
    if not args.name.replace("_", "").isalnum():
        raise ValueError("Simple fresh output name required")
    out = ROOT / "build" / args.name
    out.mkdir(parents=True, exist_ok=False)
    generated = out / "generated"
    generated.mkdir()
    vswhere = (
        Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    install = subprocess.check_output(
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
    if not install:
        raise RuntimeError("MSVC x64 build tools and Windows SDK required")
    visual_studio_setup = Path(install) / "VC/Auxiliary/Build/vcvars64.bat"
    hip = args.hip_sdk or Path("UNCONFIGURED_HIP")
    sycl = args.sycl_compiler or Path("UNCONFIGURED_SYCL")
    cuda = args.cuda_compiler
    result = {
        "schema": "phoenix-native-migration-build/1",
        "tests_run": False,
        "gpu_used": False,
        "inference_validated": False,
        "targets": {},
        "source_inputs": {},
    }

    def sha(file_path):
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    def build(name, command, output=None):
        if args.only and name not in args.only and not (name == "cpu_avx2" and "cpu" in args.only):
            return
        script = out / (name + ".cmd")
        script.write_bytes(
            (
                '@echo off\r\ncall "'
                + str(visual_studio_setup)
                + '" >nul\r\nif errorlevel 1 exit /b 5\r\n'
                + subprocess.list2cmdline([str(command_argument) for command_argument in command])
                + "\r\nexit /b %ERRORLEVEL%\r\n"
            ).encode()
        )
        start = time.monotonic()
        with (out / (name + ".log")).open("wb") as log_file:
            process_result = subprocess.run(
                ["cmd.exe", "/d", "/c", str(script)],
                cwd=out,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        row = {
            "compiled": process_result.returncode == 0,
            "exit_code": process_result.returncode,
            "seconds": time.monotonic() - start,
            "command": [str(command_argument) for command_argument in command],
        }
        if output and output.exists():
            row.update(artifact=str(output.relative_to(ROOT)), sha256=sha(output))
        result["targets"][name] = row
        (out / "build_report.json").write_text(json.dumps(result, indent=2) + "\n", newline="\n")
        print(
            name + ": " + ("compiled" if process_result.returncode == 0 else "FAILED (see log)"),
            flush=True,
        )

    common = [
        "cl",
        "/nologo",
        "/std:c++20",
        "/O2",
        "/EHsc",
        "/MD",
        "/DNOMINMAX",
        "/D_CRT_SECURE_NO_WARNINGS",
        "/I" + str(NATIVE),
        "/LD",
    ]
    if not args.only or "cpu" in args.only:
        build(
            "cpu_avx2",
            [
                "cl",
                "/nologo",
                "/std:c++20",
                "/O2",
                "/EHsc",
                "/MD",
                "/DNOMINMAX",
                "/arch:AVX2",
                "/c",
                NATIVE / "cpu_avx2.cpp",
                "/Fo" + str(out / "cpu_avx2.obj"),
            ],
            out / "cpu_avx2.obj",
        )
    build(
        "cpu",
        common
        + [
            NATIVE / "backend_cpu.cpp",
            out / "cpu_avx2.obj",
            "/Fo" + str(out / "cpu.obj"),
            "/Fe" + str(out / "phoenix_cpu_cpp.dll"),
        ],
        out / "phoenix_cpu_cpp.dll",
    )
    build(
        "runtime",
        common
        + [
            NATIVE / "runtime.cpp",
            "/Fo" + str(out / "runtime.obj"),
            "/Fe" + str(out / "phoenix_runtime_cpp.dll"),
            "/link",
            "bcrypt.lib",
        ],
        out / "phoenix_runtime_cpp.dll",
    )
    if "opencl_cpu_proxy" in args.only:
        (out / "opencl_cpu").mkdir()
        build(
            "opencl_cpu_proxy",
            [
                "link",
                "/nologo",
                "/DLL",
                "/NOENTRY",
                "/DEF:" + str(NATIVE / "opencl_cpu_proxy.def"),
                "/OUT:" + str(out / "opencl_cpu/OpenCL.dll"),
            ],
            out / "opencl_cpu/OpenCL.dll",
        )
    build(
        "probe",
        common
        + [
            ROOT / "tests/portable_cpp/probe.cpp",
            "/Fo" + str(out / "probe.obj"),
            "/Fe" + str(out / "phoenix_cpp_probe.dll"),
            "/link",
            "bcrypt.lib",
        ],
        out / "phoenix_cpp_probe.dll",
    )
    build(
        "runner",
        [
            "cl",
            "/nologo",
            "/std:c++20",
            "/O2",
            "/EHsc",
            "/MD",
            "/DNOMINMAX",
            ROOT / "tests/portable_cpp/runner.cpp",
            "/Fo" + str(out / "runner.obj"),
            "/Fe" + str(out / "phoenix_cpp_runner.exe"),
            "/link",
            "bcrypt.lib",
            "psapi.lib",
        ],
        out / "phoenix_cpp_runner.exe",
    )
    build(
        "offload_probe",
        [
            "cl",
            "/nologo",
            "/std:c++17",
            "/O2",
            "/EHsc",
            "/MD",
            "/DNOMINMAX",
            "/I" + str(args.ngx_sdk),
            ROOT / "tests/portable_cpp/offload_probe.cpp",
            "/Fo" + str(out / "offload_probe.obj"),
            "/Fe" + str(out / "phoenix_cpp_offload_probe.exe"),
            "/link",
            "bcrypt.lib",
            "psapi.lib",
            "d3d12.lib",
            "dxgi.lib",
            "d3dcompiler.lib",
            "user32.lib",
        ],
        out / "phoenix_cpp_offload_probe.exe",
    )
    build(
        "client_contracts",
        [
            "cl",
            "/nologo",
            "/std:c++20",
            "/O2",
            "/EHsc",
            "/MD",
            "/DNOMINMAX",
            ROOT / "tests/portable_cpp/client_contracts.cpp",
            "/Fo" + str(out / "client_contracts.obj"),
            "/Fe" + str(out / "phoenix_cpp_client_contracts.exe"),
            "/link",
            "bcrypt.lib",
        ],
        out / "phoenix_cpp_client_contracts.exe",
    )
    build(
        "runtime_contracts",
        [
            "cl",
            "/nologo",
            "/std:c++20",
            "/O2",
            "/EHsc",
            "/MD",
            "/DNOMINMAX",
            ROOT / "tests/portable_cpp/runtime_contracts.cpp",
            "/Fo" + str(out / "runtime_contracts.obj"),
            "/Fe" + str(out / "phoenix_cpp_runtime_contracts.exe"),
        ],
        out / "phoenix_cpp_runtime_contracts.exe",
    )
    build(
        "device_query",
        [
            "cl",
            "/nologo",
            "/std:c++20",
            "/O2",
            "/EHsc",
            "/MD",
            "/DNOMINMAX",
            NATIVE / "device_query.cpp",
            "/Fo" + str(out / "device_query.obj"),
            "/Fe" + str(out / "phoenix_cpp_devices.exe"),
        ],
        out / "phoenix_cpp_devices.exe",
    )
    build(
        "vulkan_adapter",
        common
        + [
            NATIVE / "backend_vulkan.cpp",
            "/Fo" + str(out / "vulkan.obj"),
            "/Fe" + str(out / "phoenix_vulkan_cpp.dll"),
        ],
        out / "phoenix_vulkan_cpp.dll",
    )
    build(
        "d3d12_transport",
        common
        + [
            NATIVE / "d3d12_transport.cpp",
            "/Fo" + str(out / "transport.obj"),
            "/Fe" + str(out / "phoenix_d3d12_transport.dll"),
            "/link",
            "d3d12.lib",
            "dxgi.lib",
        ],
        out / "phoenix_d3d12_transport.dll",
    )
    source = (NATIVE / "backend_gpu.cpp").read_text(encoding="utf-8")
    for kind in ("cuda", "hip", "sycl"):
        text = (
            f'#define PHX_{kind.upper()}\n#define DEVICE_NAME "{kind}-cpp"\n#define DEVICE_PREFIX "{kind}:"\n'
            + source
        )
        if kind == "cuda":
            text = text.replace("hip", "cuda").replace("cudaUUID uuid", "cudaUUID_t uuid")
        file = generated / ("backend_" + kind + (".cu" if kind == "cuda" else ".cpp"))
        file.write_text(text, encoding="utf-8", newline="\n")
        dll = out / ("phoenix_" + kind + "_cpp.dll")
        if kind == "cuda":
            command = [
                cuda,
                "-allow-unsupported-compiler",
                "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
                "-std=c++20",
                "-O2",
                "--fmad=false",
                "-arch=" + args.cuda_arch,
                "-shared",
                "-Xcompiler=/MD,/EHsc,/DNOMINMAX",
                "-I" + str(NATIVE),
                file,
                "-o",
                dll,
            ]
        else:
            compiler = sycl if kind == "sycl" else hip / "lib/llvm/bin/amdclang++.exe"
            command = [
                compiler,
                "-std=c++20",
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-DNOMINMAX",
                "-I" + str(NATIVE),
                "-shared",
                file,
                "-o",
                dll,
            ]
            if kind == "sycl":
                command += ["-fsycl", "-fsycl-targets=spir64"]
            else:
                command[1:1] = ["-x", "hip"]
                command += [
                    "--offload-arch=" + args.hip_arch,
                    "--hip-path=" + str(hip),
                    "-I" + str(hip / "include"),
                    "-L" + str(hip / "lib"),
                    "-lamdhip64",
                ]
        build(kind, command, dll)
    sdk = args.ngx_sdk or Path("UNCONFIGURED_NGX")
    minhook = ROOT / "third_party/minhook"
    addon = common + [
        "/std:c++17",
        "/DImTextureID=ImU64",
        "/DPHOENIX_BUILD_TAG=" + args.name,
        "/I" + str(sdk),
        "/I" + str(ROOT / "third_party/nvapi"),
        "/I" + str(minhook / "include"),
        "/I" + str(ROOT / "third_party/imgui_reshade"),
        "/I" + str(ROOT / "third_party/reshade"),
        ROOT / "src/integration/reshade/addon.cpp",
    ]
    addon += [
        minhook / "src" / source_path
        for source_path in ("hook.c", "buffer.c", "trampoline.c", "hde/hde64.c")
    ]
    addon += [
        "/Fe" + str(out / "phoenix.addon64"),
        "/link",
        "d3d12.lib",
        "dxgi.lib",
        "d3dcompiler.lib",
        "user32.lib",
        "bcrypt.lib",
        "psapi.lib",
    ]
    build("addon", addon, out / "phoenix.addon64")
    if (NATIVE / "broker.cpp").exists():
        build(
            "broker",
            [
                "cl",
                "/nologo",
                "/std:c++20",
                "/O2",
                "/EHsc",
                "/MD",
                "/DNOMINMAX",
                NATIVE / "broker.cpp",
                "/Fe" + str(out / "phoenix_native_broker.exe"),
                "/link",
                "bcrypt.lib",
            ],
            out / "phoenix_native_broker.exe",
        )
    # Existing Vulkan source compiler only invokes MSVC/lib/dumpbin/glslang.
    if not args.only or "vulkan_shaders" in args.only:
        build(
            "vulkan_shaders",
            [
                Path(os.sys.executable),
                ROOT / "tools/build_vulkan.py",
                args.name + "_vulkan",
                args.vulkan_deps,
            ],
        )
        if (ROOT / "build" / (args.name + "_vulkan")).is_dir():
            shutil.copytree(ROOT / "build" / (args.name + "_vulkan"), out / "vulkan")
    for source_path in NATIVE.rglob("*"):
        if source_path.is_file() and "__pycache__" not in source_path.parts:
            result["source_inputs"][source_path.relative_to(ROOT).as_posix()] = sha(source_path)
    result["all_requested_compiled"] = all(
        target_result["compiled"] for target_result in result["targets"].values()
    )
    (out / "build_report.json").write_text(json.dumps(result, indent=2) + "\n", newline="\n")
    print(out, flush=True)
    if not result["all_requested_compiled"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
