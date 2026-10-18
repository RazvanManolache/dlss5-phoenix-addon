"""Promote the one current model-free candidate; preserve the released SDK."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,indent=2)+"\n")

def validate_native_graph(root):
    bundle=root/"artifacts/current/native-kernels"
    manifest=json.loads((bundle/"native_manifest.json").read_text())
    if sha(bundle/"native_graph.json")!=manifest["native_graph_sha256"]:
        raise ValueError("Legacy native graph pin differs from its manifest")
    name="native_graph_v2.json" if "native_graph_v2_sha256" in manifest else "native_graph.json"
    pin=manifest.get("native_graph_v2_sha256",manifest["native_graph_sha256"])
    actual=sha(bundle/name)
    if actual!=pin or actual!=sha(root/"src/native/model/production_native_graph.json"):
        raise ValueError("Native bundle graph is stale or unpinned; refresh the tile-aligned graph before packaging")
    return dict(path=str(bundle/name),sha256=actual,legacy_graph_sha256=manifest["native_graph_sha256"])

def main():
    graph=validate_native_graph(ROOT) # Fail before copying any candidate files.
    destination=ROOT/"artifacts/current/neural-candidate"
    root64=destination/"windows-x64/bin"
    source64=ROOT/"build/neural_controls_candidate/Release"
    source32=ROOT/"build/neural_controls_x86_candidate/Release"
    root32=destination/"windows-x86/bin";root32.mkdir(parents=True,exist_ok=True)
    for name in ("phoenix_runtime_cpp.dll","phoenix_native.dll","phoenix_cpu_cpp.dll",
                 "phoenix_vulkan_cpp.dll","phoenix_cuda_cpp.dll","phoenix_native_broker.exe",
                 "phoenix_frame.exe","phoenix_addon_candidate.addon64"):
        shutil.copy2(source64/name,root64/name)
    for name in ("phoenix_runtime_cpp.dll","phoenix_cpu_cpp.dll","phoenix_frame.exe"):
        shutil.copy2(source32/name,root32/name)
    headers=destination/"windows-x86/include";headers.mkdir(exist_ok=True)
    shutil.copy2(ROOT/"native_runtime/phoenix.h",headers/"phoenix.h")
    comparison=ROOT/"artifacts/current/neural-comparison"
    config=json.loads((comparison/"candidate-job.json").read_text())["config"]
    config["runtime_root"]=str(root64)
    config["devices"]=[{"id":config["device"],"name":"RTX 5080 / Vulkan"},
        {"id":config["device"].replace("vulkan:","cuda:"),"name":"RTX 5080 / CUDA"},
        {"id":"cpu","name":"CPU"}]
    config["validated_devices"]=[]
    # This shared addon-worker configuration is usable by either host bitness.
    write(root64/"phoenix_native_cpp.json",dict(config,execution="broker",broker_protocol=2))
    write(destination/"portable-vulkan.json",config)
    write(destination/"portable-cuda.json",dict(config,backend="cuda",device=config["device"].replace("vulkan:","cuda:")))
    write(destination/"x86-helper.json",dict(config,execution="broker",broker_protocol=2))
    native=json.loads((comparison/"native-candidate-job.json").read_text())["config"]
    write(destination/"native-rtx.json",native)
    # Candidate jobs now use promoted binaries, leaving baseline paths intact.
    for name in ("candidate-job","cuda-job","legacy-job","native-candidate-job"):
        path=comparison/(name+".json");job=json.loads(path.read_text())
        job["runtime"]=str(root64/Path(job["runtime"]).name)
        if "runtime_root" in job["config"]:job["config"]["runtime_root"]=str(root64)
        write(path,job)
    (destination/"README.txt").write_text("""Phoenix neural controls experimental candidate — 2026-10-18

This is one current model-free candidate alongside the unchanged released SDK.
Actual GPU results and receipts: ../neural-comparison/summary.json
Human-readable checklist: ../../../docs/NEURAL_RENDERING_DEVELOPMENT_TODO.md

windows-x64/bin: native RTX library, portable Vulkan/CUDA/CPU libraries,
standalone frame tool, 64-bit broker, and compiled addon candidate.
windows-x86/bin: portable runtime/CPU and frame client. Use x86-helper.json
for GPU work through windows-x64/bin. Direct Win32 Vulkan is not validated.

Local JSON configurations pin the existing external model and kernel bundle.
No NVIDIA model DLL, extracted weights or duplicate native kernel bundle is
included. Paths are intentionally machine-local; update them after moving.

V1 ABI is retained. The corrected native graph reserves full 32-token tiles
and persistent spatial padding; it fixes the 1024x768 hang and 4K reset corruption.
Current correctness evidence: ../native-correctness/REPORT.md
Earlier timing/image receipts describe their recorded pre-fix runtime hashes.
Portable legacy replay needs BOTH
history_gain_mode=legacy_unit and output_profile=legacy. New APIs and feeder
planes are documented in docs/LIBRARY.md and docs/NATIVE_LIBRARY.md.

The addon is compile-checked and its shared display/capture wrapper was tested
on software D3D12. No live game/engine producer or complete Streamline NR provider
was exercised. It has not been installed into a game. Retain the existing game
configuration and stable addon until a separate live integration test.

This candidate declares an experimental Phoenix RGB/1 mask/UI/field contract.
It does not certify NVIDIA output parity, arbitrary HDR UI, scaled advanced
planes, curated appearance-model editions, or real-time game performance.
""",encoding="utf-8")
    files={str(path.relative_to(destination)).replace("\\","/"):{"bytes":path.stat().st_size,"sha256":sha(path)}
        for path in sorted(destination.rglob("*")) if path.is_file() and path.name!="manifest.json"}
    source_matches={}
    for name in ("phoenix_runtime_cpp.dll","phoenix_native.dll","phoenix_vulkan_cpp.dll","phoenix_cuda_cpp.dll","phoenix_native_broker.exe","phoenix_addon_candidate.addon64"):
        source_matches[name]=sha(source64/name)==sha(root64/name)
    assert all(source_matches.values())
    write(destination/"manifest.json",dict(schema="phoenix-neural-candidate/1",model_free=True,
        deployed_to_games=False,vendor_parity=False,source_copy_hashes_match=source_matches,
        native_graph=graph,files=files))
    print(json.dumps(dict(files=len(files),bytes=sum(v["bytes"] for v in files.values()),copy_hashes_match=True)))

if __name__=="__main__":main()
