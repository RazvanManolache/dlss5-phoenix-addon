"""Refresh compact evidence from actual results and current source hashes."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    folder=ROOT/"artifacts/current/neural-comparison"
    summary=json.loads((folder/"summary.json").read_text())
    contracts={}
    for bitness in ("x64","x86"):
        xml=ET.parse(folder/("contracts-"+bitness+".xml")).getroot()
        contracts[bitness]={"tests":int(xml.attrib["tests"]),"failures":int(xml.attrib["failures"]),
            "cases":[case.attrib["name"] for case in xml.findall("testcase")]}
        assert contracts[bitness]["failures"]==0
    paths=subprocess.check_output(["git","ls-files","-m","-o","--exclude-standard"],cwd=ROOT,text=True).splitlines()
    paths+=['src/native/kernels/production/optimized/input/preparation.cu','tools/build_portable_cpp.py']
    snapshot={name:sha(ROOT/name) for name in sorted(set(paths)) if
        (name.startswith(("native_runtime/","src/","tests/","tools/","docs/","examples/"))) and
        (ROOT/name).is_file() and (ROOT/name).suffix in (".h",".hpp",".c",".cpp",".mm",".cu",".cuh",".comp",".hlsl",".py",".md",".txt",".def")}
    path=ROOT/"artifacts/current/neural-rendering-development.json"
    previous=json.loads(path.read_text());initial=previous.get("initial_slice",previous)
    receipt=dict(schema="phoenix-neural-rendering-development/2",
        recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        scope="Experimental candidate implementation and bounded GPU comparison; no vendor/game parity certification",
        gpu_used=True,full_model_inference_run=True,games_launched=False,vendor_code_executed=False,
        frozen_reference_modified=False,model=initial["model"],
        candidate_manifest="neural-candidate/manifest.json",comparison_summary="neural-comparison/summary.json",
        contracts=contracts,control_mask_builder_python_tests=3,
        model_data_preflight_checks=130,model_data_preflight_inference_run=False,
        comparison=summary,
        source_snapshot_scope="Current working tree hashes include pre-existing unrelated edits; not change attribution",
        source_sha256=snapshot,
        temporary_builds_removed=False,
        cleanup={"automatic_approval_review":"rejected","reason":"blocked by policy",
            "retained_project_builds":["build/neural_controls_candidate","build/neural_controls_x86_candidate"]},
        remaining_acceptance=["NVIDIA numerical/color/noise/map-alpha reference",
            "Live engine object/material producer, motion/occlusion/stereo and game rendering",
            "HDR/scaled advanced-plane and standalone native depth-validity contracts",
            "HIP/SYCL/Metal/mobile mapped-feature execution",
            "Complete NR Streamline provider and curated appearance-model data",
            "Older direct binary replay probe consumers; direct Win32 Vulkan imports",
            "Long motion/curved fields, allocation peaks and game-performance matrix"],
        initial_slice=initial)
    path.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(source_files=len(snapshot),contract_tests=contracts,
        candidate_manifest_sha256=sha(ROOT/"artifacts/current/neural-candidate/manifest.json")),indent=2))

if __name__=="__main__":main()
