"""Build the optional Windows x64 ONNX Runtime provider for legacy depth."""
import argparse
from pathlib import Path
import shutil
from test_model_selection import run, toolchain

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ort-sdk", type=Path, required=True,
                        help="Extracted ONNX Runtime x64 SDK with include and lib directories")
    parser.add_argument("--runtime-dir", type=Path, help="Matching CPU, CUDA or DirectML x64 runtime DLL directory")
    parser.add_argument("--directml-dll", type=Path, help="Microsoft.AI.DirectML x64 runtime DLL, when using DirectML")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sdk = args.ort_sdk.resolve()
    output = args.output.resolve()
    library = sdk / "lib"
    if not (sdk / "include/onnxruntime_cxx_api.h").is_file() or not (library / "onnxruntime.lib").is_file():
        parser.error("Use an ONNX Runtime Windows x64 SDK")
    runtime = args.runtime_dir.resolve() if args.runtime_dir else library
    if not (runtime / "onnxruntime.dll").is_file():
        parser.error("The runtime directory must contain onnxruntime.dll")
    output.mkdir(parents=True, exist_ok=True)
    cl, env = toolchain(64)
    print(run([cl, "/nologo", "/std:c++20", "/EHsc", "/O2", "/MT", "/LD",
               "/I" + str(sdk / "include"),
               ROOT / "src/integration/compatibility/host/legacy_depth_ort.cpp",
               "/link", "/OUT:" + str(output / "phoenix_legacy_depth.dll"),
               library / "onnxruntime.lib", "dxgi.lib"], cwd=output, env=env), end="")
    for name in ("onnxruntime.dll", "onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll"):
        source = runtime / name
        if source.is_file() and source.resolve() != (output / name).resolve():
            shutil.copy2(source, output / name)
    if args.directml_dll:
        shutil.copy2(args.directml_dll, output / "DirectML.dll")
    for name in ("LICENSE", "ThirdPartyNotices.txt"):
        if (sdk / name).is_file():
            shutil.copy2(sdk / name, output / ("ONNXRuntime-" + name))
    print(output / "phoenix_legacy_depth.dll")


if __name__ == "__main__":
    main()
