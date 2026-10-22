"""Targeted model-switch, allocation retirement and output-writeback checks.

CPU-only by default. --gpu explicitly opts into bounded RTX inference; external
model DLLs are read-only inputs and are never copied into test outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from test_model_selection import ROOT, run, toolchain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--lightweight", type=Path)
    args = parser.parse_args()
    if args.gpu and not all((args.bundle, args.original, args.lightweight)):
        parser.error("--gpu requires --bundle, --original and --lightweight")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "gpu_used": args.gpu, "cases": []}
    receipt = output / "validation.json"
    receipt.write_text(json.dumps(report), encoding="utf-8")
    try:
        (ROOT / "build").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="model_switch_", dir=ROOT / "build") as temporary:
            work = Path(temporary)
            for bits in (32, 64):
                compiler, env = toolchain(bits)
                flags = [compiler, "/nologo", "/std:c++20", "/Zc:__cplusplus", "/EHsc", "/bigobj",
                         "/O2", "/MT", "/DNOMINMAX"]
                exe = work / f"lifetime{bits}.exe"
                build = run(flags + [ROOT / "tests/native/model_switch_lifetime_tests.cpp",
                    f"/Fo:{work / 'lifetime.obj'}", f"/Fe:{exe}"], cwd=work, env=env)
                (output / f"cpu{bits}-compile.log").write_text(build, encoding="utf-8")
                result = json.loads(run([exe], cwd=work, env=env))
                report["cases"].append({"suite": f"lifetime{bits}", **result})
                print(json.dumps(report["cases"][-1]), flush=True)
            if args.gpu:
                compiler, env = toolchain(64)
                exe = work / "switch_gpu.exe"
                flags = [compiler, "/nologo", "/std:c++20", "/Zc:__cplusplus", "/EHsc", "/bigobj",
                    "/O2", "/MT", "/DNOMINMAX", "/I" + str(ROOT.parent / "external/DLSS/include"),
                    "/I" + str(ROOT / "third_party/nvapi")]
                build = run(flags + [ROOT / "tests/native/model_switch_gpu_probe.cpp",
                    f"/Fo:{work / 'switch_gpu.obj'}", f"/Fe:{exe}", "d3d12.lib", "dxgi.lib",
                    "d3dcompiler.lib", "bcrypt.lib", "cabinet.lib", "shell32.lib", "ole32.lib",
                    "uuid.lib", "user32.lib"], cwd=work, env=env)
                (output / "gpu-compile.log").write_text(build, encoding="utf-8")
                result = subprocess.run([str(exe), str(args.bundle.resolve()), str(args.original.resolve()),
                    str(args.lightweight.resolve())], cwd=work, env=env, capture_output=True,
                    text=True, timeout=180)
                (output / "gpu.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
                if result.returncode:
                    raise RuntimeError(f"GPU probe failed ({result.returncode}): {result.stderr}")
                gpu = json.loads(result.stdout.strip().splitlines()[-1])
                (output / "gpu.json").write_text(json.dumps(gpu, indent=2), encoding="utf-8")
                report["cases"].append({"suite": "production-runner-gpu", "passed": gpu["passed"],
                    "rows": gpu["rows"], "retirements": gpu["retirements"]})
                print(json.dumps(report["cases"][-1]), flush=True)
            report["passed"] = all(case["passed"] for case in report["cases"])
            report["retail_game_presentation_validated"] = False
            report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [ROOT / "src/integration/reshade/addon.cpp",
                             ROOT / "src/native/support/production_direct.h",
                             ROOT / "src/native/network/production_network.h",
                             ROOT / "tests/native/model_switch_gpu_probe.cpp",
                             ROOT / "src/native/support/model_selection_state.h",
                             ROOT / "src/native/support/optional_feature_lifetime.h"]}
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
