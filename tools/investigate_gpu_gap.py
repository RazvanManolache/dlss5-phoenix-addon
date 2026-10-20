"""Matched offline engine/copy timing diagnostics; no games or installed files."""
import datetime
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import tempfile

from compare_four_models import build, sha, telemetry, write
from test_model_selection import ROOT

OUT=ROOT/'artifacts/current/gpu-gap'
STAGES=('engine','to_copy_transition','output_copy','restore_transition')
DEEP_STAGES=('engine_prepare','dispatch_region','engine_output_transition','to_copy_transition','output_copy','restore_transition')


def stage_names(case):
    return DEEP_STAGES if 'engine_prepare' in case.get('gpu_stages',{}) else STAGES


def validate_segments(case):
    """Reject incomplete, negative or nonconserving GPU timestamp partitions."""
    if not case.get('passed') or case.get('process_exit_code',0):
        raise ValueError('Probe failed')
    if not case.get('gpu_stages'):
        return
    samples=case['gpu_samples_ms']
    stages=case['gpu_stage_samples_ms']
    names=stage_names(case)
    if set(stages)!=set(names) or not samples or any(len(stages[k])!=len(samples) for k in names):
        raise ValueError('Incomplete GPU stage samples')
    for index,total in enumerate(samples):
        parts=[stages[k][index] for k in names]
        if not all(math.isfinite(p) and p>=0 for p in [total,*parts]):
            raise ValueError('Invalid GPU interval')
        if not math.isclose(sum(parts),total,rel_tol=0,abs_tol=1e-9):
            raise ValueError('GPU stage accounting differs')


def summarize(cases,extents=((1280,720),(1920,1080),(3840,2160))):
    rows=[]
    for model in ('original','small-fp8'):
        for extent in extents:
            same_fixture=[c for c in cases if c['model_id']==model and (c['width'],c['height'])==extent]
            if not same_fixture or any(c['captures']!=same_fixture[0]['captures'] for c in same_fixture):
                raise ValueError('Instrumentation or renderer changed output')
            for profiled in ([False,True] if extent==(1920,1080) else [True]):
                pair={}
                for engine in ('phoenix','nvidia'):
                    matches=[c for c in cases if c['engine']==engine and c['model_id']==model
                             and (c['width'],c['height'])==extent and c['profiled']==profiled]
                    if len(matches)!=2:
                        raise ValueError('Two independent processes required per cell')
                    for case in matches:validate_segments(case)
                    if matches[0]['captures']!=matches[1]['captures']:
                        raise ValueError('Across-process output differs')
                    names=stage_names(matches[0])
                    pair[engine]=dict(gpu_ms=statistics.mean(c['gpu']['median_ms'] for c in matches),
                        gpu_mean_ms=statistics.mean(c['gpu']['mean_ms'] for c in matches),
                        gpu_process_medians_ms=[c['gpu']['median_ms'] for c in matches],
                        stages_ms={stage:statistics.mean(c['gpu_stages'][stage]['median_ms'] for c in matches)
                                   for stage in names} if profiled else {},
                        stage_means_ms={stage:statistics.mean(c['gpu_stages'][stage]['mean_ms'] for c in matches)
                                        for stage in names} if profiled else {},
                        calibration_us=[c['gpu_timestamp_calibration']['gpu']['median_ms']*1000 for c in matches] if profiled else [],
                        captures=matches[0]['captures'])
                if pair['phoenix']['captures']!=pair['nvidia']['captures']:
                    raise ValueError('Native NVIDIA DLSS5 parity differs')
                for value in pair.values():del value['captures']
                a,b=pair['phoenix'],pair['nvidia']
                rows.append(dict(model=model,extent=list(extent),profiled=profiled,renderers=pair,
                    gpu_delta_us=1000*(a['gpu_ms']-b['gpu_ms']),
                    stage_median_deltas_us={stage:1000*(a['stages_ms'][stage]-b['stages_ms'][stage])
                                           for stage in names} if profiled else {},
                    stage_mean_deltas_us={stage:1000*(a['stage_means_ms'][stage]-b['stage_means_ms'][stage])
                                         for stage in names} if profiled else {}))
    for row in rows:
        if row['extent']!=[1920,1080] or not row['profiled']:continue
        bare=next(r for r in rows if r['model']==row['model'] and r['extent']==row['extent'] and not r['profiled'])
        row['instrumentation_change_us']={engine:1000*(row['renderers'][engine]['gpu_ms']-
            bare['renderers'][engine]['gpu_ms']) for engine in ('phoenix','nvidia')}
    return rows


def main():
    global OUT
    # The receipt validator/report and model-free tests do not need capture or
    # portable-runtime dependencies. Load them only when running GPU cases.
    import numpy as np
    from benchmark_readme_case import fixture
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deep',action='store_true',help='Subdivide engine preparation, dispatch and output transition at 1080p')
    parser.add_argument('--output',type=Path,default=OUT,help='Receipt folder; choose a separate folder to retain an earlier study')
    args=parser.parse_args()
    OUT=args.output.resolve()
    if args.deep:OUT=OUT/'deep'
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    receipt=json.loads((ROOT/'artifacts/current/readme-refresh/build.json').read_text())
    bundle=Path(receipt['native_bundle'])
    if sha(bundle/'native_manifest.json')!=receipt['native_bundle_manifest_sha256']:
        raise RuntimeError('Retained README kernel identity changed')
    manifest=json.loads((bundle/'native_manifest.json').read_text())
    for kernel in manifest['kernels']:
        if sha(bundle/kernel['binary'])!=kernel['binary_sha256']:
            raise RuntimeError('Kernel binary changed: '+kernel['name'])
    model=ROOT.parent/'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
    cores=list(Path('C:/Windows/System32/DriverStore/FileRepository').glob('nv_*.inf_amd64_*/_nvngx.dll'))
    if len(cores)!=1:raise RuntimeError('Unambiguous installed NGX core required')
    source=dict(kind='nvidia_dll',path=str(model),sha256=sha(model))
    report=dict(schema='phoenix-gpu-gap-diagnostic/1',passed=False,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_before=telemetry(),
        model_dll_sha256=source['sha256'],ngx_core_sha256=sha(cores[0]),
        bundle_manifest_sha256=sha(bundle/'native_manifest.json'),
        probe_source_sha256=sha(ROOT/'tests/native/four_model_probe.cpp'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty_checkout=True,games_launched=False,installed_files_modified=False,cases=[])
    write(OUT/'run.json',report)
    with tempfile.TemporaryDirectory(prefix='gpu-gap-',dir=ROOT/'build') as temporary:
        work=Path(temporary);executable,forwarder=build(work)
        report.update(probe_sha256=sha(executable),forwarder_sha256=sha(forwarder))
        cases=[]
        extents=((1920,1080),) if args.deep else ((1280,720),(1920,1080),(3840,2160))
        for w,h in extents:
            color,motion=work/'color.rgba16f',work/'motion.rg16f'
            pixels=np.ones((h,w,4),np.float16);pixels[...,:3]=fixture(w,h)
            color.write_bytes(pixels.tobytes());motion.write_bytes(np.zeros((h,w,2),np.float16).tobytes())
            order=[('phoenix','original'),('nvidia','original'),('phoenix','small-fp8'),('nvidia','small-fp8')]
            for process in range(2):
                for engine,model_id in (order if process==0 else reversed(order)):
                    profiles=[False,True] if (w,h)==(1920,1080) else [True]
                    for profiled in (profiles if process==0 else reversed(profiles)):
                        name=f'{engine}-{model_id}-{w}x{h}-{"stages" if profiled else "bare"}-p{process}'
                        job=dict(width=w,height=h,engine=engine,weight_source={**source,'model':model_id},
                            bundle_path=str(bundle),ngx_core=str(cores[0]),forwarder=str(forwarder),
                            cache=str(work/'ngx-cache'),color_file=str(color),motion_file=str(motion),
                            capture=False,debug=False,warmup_seconds=.75,measure_seconds=2.,profile_gpu_stages=profiled,
                            deep_profile_gpu_stages=args.deep and profiled)
                        write(work/'job.json',job)
                        result=subprocess.run([str(executable),str(work/'job.json'),str(OUT/'cases'/(name+'.json'))],
                            cwd=work,text=True,capture_output=True,timeout=180)
                        case=json.loads((OUT/'cases'/(name+'.json')).read_text())
                        case.update(profiled=profiled,process_index=process,process_exit_code=result.returncode,
                                    telemetry_after=telemetry())
                        write(OUT/'cases'/(name+'.json'),case)
                        validate_segments(case);cases.append(case)
                        report['cases'].append(dict(file=name+'.json',sha256=sha(OUT/'cases'/(name+'.json'))))
                        write(OUT/'run.json',report)
                        print(name,f"{case['gpu']['median_ms']:.6f} ms",flush=True)
        report.update(passed=True,rows=summarize(cases,extents),gpu_after=telemetry())
    report['temporary_build_removed']=True
    report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write(OUT/'run.json',report)
    print(json.dumps(dict(passed=True,cases=len(cases),rows=report['rows']),indent=2),flush=True)


if __name__=='__main__':main()
