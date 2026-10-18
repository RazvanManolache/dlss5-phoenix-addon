"""Longer-warmup repeats of flagged timings; preserve the original sweep."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from benchmark_neural_resolutions import ROOT,CURRENT,DEST,WORK,read,write,sha,telemetry

def main():
    report=dict(schema='phoenix-neural-timing-rechecks/1',warmup_seconds=5,measured_seconds_min=5,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tests/native/neural_performance_probe.cpp',ROOT/'tools/benchmark_neural_portable.py']},
        cases={},portable={})
    for w,h in [(384,256),(5120,1440),(5120,2160),(5120,2880)]:
        key=f'{w}x{h}';rows=report['cases'].setdefault(key,{})
        rows['gpu_before']=telemetry()
        for kind,label in [('legacy','before'),('candidate','candidate'),('legacy','after')]:
            config=read(CURRENT/'neural-candidate/native-rtx.json');config.update(width=w,height=h,
                benchmark_warmup_seconds=5,benchmark_measured_seconds=5,benchmark_cap_seconds=30)
            if kind=='candidate':config['benchmark_profiles']=['default','ui_overlay','combined','default_after_features']
            job=WORK/'recheck-native.json';write(job,config)
            dll=CURRENT/('sdk/windows-x64/bin/phoenix_native.dll' if kind=='legacy' else 'neural-candidate/windows-x64/bin/phoenix_native.dll')
            out=DEST/f'native-recheck-{label}-{key}.json'
            p=subprocess.run([str(ROOT/'build/neural_controls_candidate/Release/neural_performance_probe.exe'),str(dll),str(job),str(out),kind],
                text=True,capture_output=True,timeout=180)
            result=read(out) if out.exists() else dict(passed=False)
            result.update(runtime_sha256=sha(dll),process_exit_code=p.returncode)
            if p.returncode:result.update(passed=False,error=p.stderr[-2000:])
            write(out,result);rows[label]=dict(passed=result.get('passed',False),file=out.name,error=result.get('error'))
            write(DEST/'rechecks.json',report);print(key,label,p.stdout.strip() or p.stderr.strip(),flush=True)
        rows['gpu_after']=telemetry()
    for legacy in [True,False]:
        key='legacy_vulkan_default' if legacy else 'candidate_vulkan_default'
        runtime=CURRENT/('sdk/windows-x64/bin' if legacy else 'neural-candidate/windows-x64/bin')
        config=read(CURRENT/'neural-candidate/portable-vulkan.json');config['runtime_root']=str(runtime)
        job=dict(extent=[5120,2160],runtime=str(runtime/'phoenix_runtime_cpp.dll'),config=config,advanced=False,route=key)
        path=WORK/'recheck-portable.json';write(path,job);out=DEST/f'portable-recheck-{key}-5120x2160.json'
        p=subprocess.run([sys.executable,str(ROOT/'tools/benchmark_neural_portable.py'),str(path),str(out),
            '--warmup','5','--measure','5','--cap','30'],text=True,capture_output=True,timeout=180)
        result=read(out) if out.exists() else dict(passed=False)
        result.update(runtime_sha256=sha(runtime/'phoenix_runtime_cpp.dll'),process_exit_code=p.returncode)
        if p.returncode:result.update(passed=False,error=p.stderr[-2000:])
        write(out,result);report['portable'][key]=dict(passed=result.get('passed',False),file=out.name,error=result.get('error'))
        write(DEST/'rechecks.json',report);print(p.stdout.strip() or p.stderr.strip(),flush=True)
    report['completed']=True;write(DEST/'rechecks.json',report)
if __name__=='__main__':main()
