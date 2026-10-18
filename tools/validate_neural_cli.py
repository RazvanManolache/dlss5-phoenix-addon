"""Exercise shipped standalone file replay, including the 32-bit broker client."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/"build/neural_controls_candidate/cli-fixture";folder.mkdir(exist_ok=True)
    dest=ROOT/"artifacts/current/neural-candidate"
    w,h=64,48
    image=np.linspace(.1,.8,w*h*3,dtype=np.float32).reshape(h,w,3)
    mask=np.ones((h,w,4),np.float32);mask[:,:,0]=0
    ui=np.zeros((h,w,4),np.float32);ui[:8,:24]=1
    alpha=ui[:,:,3].copy();backbuffer=ui[:,:,:3]+image*(1-alpha[:,:,None])
    planes=dict(color=image,motion=np.zeros((h,w,2),np.float32),depth=np.full((h,w),.5,np.float32),
        validity=np.ones((h,w),np.float32),control_mask=mask,ui=ui,ui_alpha=alpha,backbuffer=backbuffer,
        distortion=np.zeros((h,w,4),np.float32))
    job=dict(width=w,height=h,control_profile="phoenix-experimental-rgb/1",metadata={"validation_stage_hashes":True})
    for name,values in planes.items():
        file=folder/(name+".f32");values.tofile(file);job[name]=str(file)
    report=dict(passed=False,gpu_used=True,full_graph=True,vendor_parity=False,routes=[])
    reference=None
    for name,bitness,broker in (("local64","x64",False),("broker64","x64",True),("broker32","x86",True)):
        config=json.loads((dest/("x86-helper.json" if broker else "portable-vulkan.json")).read_text())
        config["validation_stage_hashes"]=True
        job["config"]=config;path=folder/(name+".json");path.write_text(json.dumps(job))
        output=folder/(name+".output.rgb32f")
        # This tool owns exactly these regular output files inside its fixture.
        if output.exists():output.unlink()
        executable=dest/("windows-"+bitness)/"bin/phoenix_frame.exe"
        process=subprocess.run([str(executable),str(path),str(output)],text=True,capture_output=True,check=True)
        status=json.loads(process.stdout)
        expected_hashes={key:hashlib.sha256(planes[key].tobytes()).hexdigest() for key in
            ("control_mask","ui","ui_alpha","backbuffer","distortion")}
        assert status.get("optional_input_hashes")==expected_hashes,"Optional-plane capture hash mismatch"
        values=np.fromfile(output,dtype=np.float32).reshape(h,w,3)
        assert np.array_equal(values,backbuffer),"CLI map/UI composition changed original protected pixels"
        if reference is None:reference=values
        else:assert np.array_equal(values,reference),"CLI local/helper/bitness mismatch"
        report["routes"].append(dict(route=name,all_optional_planes=True,protected_pixels_exact=True,
            binary_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            input_hashes=status.get("optional_input_hashes",{})))
    report["passed"]=True
    (ROOT/"artifacts/current/neural-comparison/cli-checks.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()
