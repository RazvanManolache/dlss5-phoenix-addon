"""Explicit matched Phoenix/NVIDIA original/small native GPU comparison."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from test_model_selection import ROOT, toolchain, run


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def telemetry():
    command = ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,temperature.gpu,clocks.sm,power.draw,utilization.gpu", "--format=csv,noheader"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    return result.stdout.strip()


def build(work, source_root=ROOT):
    compiler, env = toolchain(64)
    sdk = ROOT.parent / "external/DLSS/include"
    base = [compiler, "/nologo", "/std:c++20", "/Zc:__cplusplus", "/EHsc", "/bigobj", "/O2", "/fp:strict", "/MD", "/DNOMINMAX", "/I" + str(sdk), "/I" + str(source_root / "third_party/nvapi"), "/I" + str(source_root / "third_party/minhook/include")]
    forwarder = work / "phoenix-nvngx.dll"
    run(base + ["/LD", source_root / "src/integration/reshade/nr_forwarder.cpp", f"/Fo:{work / 'forwarder.obj'}", f"/Fe:{forwarder}"], cwd=work, env=env)
    sources = [source_root / "tests/native/four_model_probe.cpp"] + [source_root / ("third_party/minhook/src/" + name) for name in ("hook.c", "buffer.c", "trampoline.c", "hde/hde64.c")]
    executable = work / "four_model_probe.exe"
    run(base + sources + [f"/Fe:{executable}", "d3d12.lib", "dxgi.lib", "d3dcompiler.lib", "bcrypt.lib", "psapi.lib"], cwd=work, env=env)
    return executable, forwarder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--resolutions", nargs="+", default=["1280x720", "1920x1080", "3840x2160"])
    args = parser.parse_args()
    work = ROOT / "build/four_model_comparison"
    destination = ROOT / "artifacts/current/four-model-comparison"
    work.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    executable, forwarder = (work / "four_model_probe.exe", work / "phoenix-nvngx.dll") if args.skip_build else build(work)
    model = ROOT.parent / "research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll"
    image = ROOT.parent / "dlss5-phoenix-gallery/docs/readme001/images/room0__beauty__off.jpg"
    core_files = list(Path("C:/Windows/System32/DriverStore/FileRepository").glob("nv_*.inf_amd64_*/_nvngx.dll"))
    if len(core_files) != 1:
        raise RuntimeError("Need an unambiguous installed NGX core")
    metadata = dict(schema="phoenix-nvidia-four-model-comparison/1", passed=False,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), gpu_before=telemetry(),
        source_image=str(image), source_image_sha256=sha(image), model_dll=str(model), model_dll_sha256=sha(model),
        native_bundle=str(ROOT / "artifacts/current/native-kernels"), ngx_core=str(core_files[0]), ngx_core_sha256=sha(core_files[0]),
        executable_sha256=sha(executable), forwarder_sha256=sha(forwarder), full_graph=True, passes=1, repeats=[1]*8,
        fixture="Shared sRGB SDR rendered museum scene proxy resized with Lanczos to RGBA16F; zero RG16F motion; no depth/UI/mask",
        games_launched=False, installed_files_modified=False, cases=[])
    sizes = ["320x180"] if args.smoke else args.resolutions
    report_path = destination / ("smoke.json" if args.smoke else "run.json")
    for resolution in sizes:
        w, h = map(int, resolution.split("x"))
        rgba = np.ones((h,w,4),np.float16)
        rgba[:,:,:3] = np.asarray(Image.open(image).convert("RGB").resize((w,h),Image.Resampling.LANCZOS),np.float32)/255
        color, motion = work / "input.rgba16f", work / "motion.rg16f"
        color.write_bytes(rgba.tobytes()); motion.write_bytes(np.zeros((h,w,2),np.float16).tobytes())
        if resolution == "1920x1080":
            Image.fromarray(np.uint8(np.clip(rgba[:,:,:3].astype(np.float32),0,1)*255+.5)).save(destination / "input.png")
        for engine, model_id in [("phoenix","original"),("nvidia","original"),("phoenix","small-fp8"),("nvidia","small-fp8")]:
            name = f"{engine}-{model_id}-{resolution}" + ("-smoke" if args.smoke else "")
            config = dict(width=w,height=h,engine=engine,weight_source=dict(kind="nvidia_dll",path=str(model),sha256=metadata["model_dll_sha256"],model=model_id),
                bundle_path=metadata["native_bundle"],ngx_core=str(core_files[0]),forwarder=str(forwarder),cache=str(work / "ngx-cache"),
                color_file=str(color),motion_file=str(motion),capture=not args.smoke and resolution=="1920x1080",
                debug=args.smoke,warmup_seconds=.1 if args.smoke else .75,measure_seconds=.1 if args.smoke else 2.)
            job, output = work / "job.json", destination / (name + ".json")
            write(job,config);write(report_path,metadata)
            started=datetime.datetime.now(datetime.timezone.utc).isoformat()
            process=subprocess.run([str(executable),str(job),str(output)],cwd=work,capture_output=True,text=True,timeout=180)
            row=json.loads(output.read_text(encoding="utf-8")) if output.exists() else dict(passed=False,error="Probe did not write a result")
            row.update(process_exit_code=process.returncode,started_utc=started,gpu_after=telemetry())
            if process.returncode:
                row["passed"]=False;row["stderr"]=process.stderr[-5000:]
            write(output,row);metadata["cases"].append(dict(engine=engine,model_id=model_id,resolution=resolution,file=output.name,passed=row["passed"]))
            print(name,"PASS" if row["passed"] else "FAIL",row.get("gpu",{}).get("median_ms"),row.get("error",""),flush=True)
            write(report_path,metadata)
            if not row["passed"]:
                raise RuntimeError(f"{name}: {row.get('error',process.stderr)}")
    metadata.update(passed=True,gpu_after=telemetry(),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(report_path,metadata)


if __name__ == "__main__":
    main()
