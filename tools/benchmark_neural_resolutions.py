"""Explicit, sequential all-preset performance sweep; compact current receipts."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CURRENT=ROOT/"artifacts/current"
DEST=CURRENT/"neural-performance"
WORK=ROOT/"build/neural_controls_candidate/performance"
RESOLUTIONS=[(320,180),(384,256),(640,360),(640,480),(800,600),(854,480),
    (960,540),(1024,768),(1280,720),(1280,800),(1280,960),(1280,1024),
    (1600,900),(1920,1080),(1920,1200),(2560,1080),(2560,1440),(2560,1600),
    (3440,1440),(3840,1600),(3840,2160),(5120,1440),(5120,2160),(5120,2880),(7680,4320)]

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+"\n")
def telemetry():
    p=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),csv=p.stdout.strip(),error=p.stderr.strip())

def run_native(width,height,kind,label):
    output=DEST/f"native-{label}-{width}x{height}.json"
    job=read(CURRENT/"neural-candidate/native-rtx.json");job.update(width=width,height=height)
    config=WORK/"native-job.json";write(config,job)
    dll=CURRENT/("sdk/windows-x64/bin/phoenix_native.dll" if kind=="legacy" else "neural-candidate/windows-x64/bin/phoenix_native.dll")
    start=time.perf_counter()
    try:
        p=subprocess.run([str(ROOT/"build/neural_controls_candidate/Release/neural_performance_probe.exe"),
            str(dll),str(config),str(output),kind],capture_output=True,text=True,timeout=300)
        result=read(output) if output.exists() else dict(passed=False,profiles=[])
        result.update(runtime_sha256=sha(dll),process_exit_code=p.returncode,elapsed_seconds=time.perf_counter()-start)
        if p.returncode:result.update(passed=False,error=p.stderr[-2000:])
        print(p.stdout.strip(),flush=True)
    except subprocess.TimeoutExpired:
        result=read(output) if output.exists() else dict(profiles=[])
        result.update(passed=False,error="Isolated native benchmark timed out after 300 seconds")
    write(output,result)
    return result

def run_portable(width,height,backend,advanced,legacy=False):
    route=("legacy_" if legacy else "candidate_")+backend+("_combined" if advanced else "_default")
    output=DEST/f"portable-{route}-{width}x{height}.json"
    config=read(CURRENT/"neural-candidate"/("portable-"+backend+".json"))
    runtime=CURRENT/("sdk/windows-x64/bin" if legacy else "neural-candidate/windows-x64/bin")
    config["runtime_root"]=str(runtime)
    job=dict(extent=[width,height],runtime=str(runtime/"phoenix_runtime_cpp.dll"),config=config,advanced=advanced,route=route)
    path=WORK/"portable-job.json";write(path,job)
    start=time.perf_counter()
    try:
        p=subprocess.run([sys.executable,str(ROOT/"tools/benchmark_neural_portable.py"),str(path),str(output)],
            capture_output=True,text=True,timeout=180)
        result=read(output) if output.exists() else dict(passed=False)
        result.update(runtime_sha256=sha(runtime/"phoenix_runtime_cpp.dll"),backend_sha256=sha(runtime/("phoenix_"+backend+"_cpp.dll")),
            process_exit_code=p.returncode,elapsed_seconds=time.perf_counter()-start)
        if p.returncode:result.update(passed=False,error=p.stderr[-2000:])
        print(p.stdout.strip() or (route+" "+str(width)+"x"+str(height)+": "+result.get("error","failed")),flush=True)
    except subprocess.TimeoutExpired:result=dict(passed=False,error="Isolated portable case timed out after 180 seconds")
    result.update(extent=[width,height],route=route);write(output,result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['native','portable','all'],default='all')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--only',help='Single WIDTHxHEIGHT checkpoint')
    args=parser.parse_args();DEST.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    sizes=[tuple(map(int,args.only.split('x')))] if args.only else RESOLUTIONS
    manifest=read(DEST/"run.json") if args.resume and (DEST/"run.json").exists() else dict(
        schema="phoenix-neural-resolution-performance/1",started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        gpu=telemetry(),resolutions=RESOLUTIONS,games_launched=False,vendor_code_executed=False,vendor_parity=False,
        full_graph=True,passes=1,repeats=[1]*8,fixture="Synthetic structured SDR proxy, zero motion, accepted temporal continuation",
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/"tools/benchmark_neural_portable.py",ROOT/"tests/native/neural_performance_probe.cpp"]},
        native={},portable={})
    for w,h in sizes:
        key=f"{w}x{h}";manifest['current_resolution']=key;write(DEST/"run.json",manifest)
        if args.stage in ('native','all'):
            print('NATIVE '+key,flush=True);rows=manifest['native'].setdefault(key,{})
            rows['gpu_before']=telemetry()
            for kind,label in [('legacy','before'),('candidate','candidate'),('legacy','after')]:
                if args.resume and rows.get(label,{}).get('passed'):continue
                result=run_native(w,h,kind,label)
                rows[label]=dict(passed=result.get('passed',False),error=result.get('error'),file=f"native-{label}-{key}.json")
                write(DEST/"run.json",manifest)
            rows['gpu_after']=telemetry()
            before=read(DEST/rows['before']['file']);after=read(DEST/rows['after']['file']);candidate=read(DEST/rows['candidate']['file'])
            if before.get('passed') and after.get('passed') and candidate.get('passed'):
                profiles={r['profile']:r for r in candidate['profiles']}
                hashes=[before['profiles'][0]['reset_output_sha256'],after['profiles'][0]['reset_output_sha256'],
                    profiles['default']['reset_output_sha256'],profiles['default_after_features']['reset_output_sha256']]
                rows['legacy_reset_output_exact']=len(set(hashes))==1
                print(key+' native compatibility exact='+str(rows['legacy_reset_output_exact']),flush=True)
            write(DEST/"run.json",manifest)
        if args.stage in ('portable','all'):
            print('PORTABLE '+key,flush=True);rows=manifest['portable'].setdefault(key,{})
            rows['gpu_before']=telemetry()
            for backend,advanced,legacy in [('vulkan',False,True),('vulkan',False,False),('vulkan',True,False),('cuda',False,False),('cuda',True,False)]:
                route=('legacy_' if legacy else 'candidate_')+backend+('_combined' if advanced else '_default')
                if args.resume and rows.get(route,{}).get('passed'):continue
                result=run_portable(w,h,backend,advanced,legacy)
                rows[route]=dict(passed=result.get('passed',False),error=result.get('error'),file=f"portable-{route}-{key}.json")
                write(DEST/"run.json",manifest)
            rows['gpu_after']=telemetry();write(DEST/"run.json",manifest)
    manifest['stage_completed']=args.stage;manifest['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write(DEST/"run.json",manifest)

if __name__=="__main__":main()
