"""Fresh paired Phoenix DLSS5 / NVIDIA DLSS5 native measurements for the README.

Use the existing structured fixture, current native kernels and the same model
DLL for both engines. Only compact receipts survive; no games or deployments.
"""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import subprocess
import tempfile

import numpy as np
from benchmark_readme_case import fixture
from compare_four_models import ROOT, build, sha, telemetry, write
from experiment_native_tuning import MODEL

EXTENTS=('320x180','640x360','640x480','960x540','1280x720','1920x1080',
         '2560x1080','2560x1440','3840x2160','5760x3240','6144x3456','7680x4320')


def aggregate(rows):
    """Keep process medians and observed ranges, without pooling unequal runs."""
    result={}
    for kind in ('gpu','wall'):
        values=[row[kind]['median_ms'] for row in rows]
        result[kind]=dict(median_process_median_ms=statistics.median(values),
                          mean_process_median_ms=statistics.mean(values),process_medians_ms=values,
                          min_process_median_ms=min(values),max_process_median_ms=max(values))
    result['memory']={key:statistics.median(row['memory'][key] for row in rows)
                      for key in rows[0]['memory'] if key.endswith('_bytes')}
    result['create_ms']=[row['create_ms'] for row in rows]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT/'artifacts/current/native-kernel-styles')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/current/native-vendor-refresh/run.json')
    parser.add_argument('--extents',nargs='+',default=EXTENTS)
    parser.add_argument('--rounds',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.rounds<=5:
        parser.error('Use one to five matched process pairs')
    core=list(Path('C:/Windows/System32/DriverStore/FileRepository').rglob('_nvngx.dll'))
    if len(core)!=1:
        raise ValueError('Need an unambiguous installed x64 NGX core')
    bundle=args.bundle.resolve()
    manifest=json.loads((bundle/'native_manifest.json').read_text())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(schema='phoenix-native-vendor-refresh/1',passed=False,state='building',
                measured_date=datetime.date.today().isoformat(),games_launched=False,installed_files_modified=False,
                older_device_validated=False,published=False,rounds=args.rounds,extents=list(args.extents),
                fixture='Existing structured SDR RGB/checker/central-color fixture, FP16 RGBA, zero RG16F motion; full eight blocks, one pass, automatic mask, default controls',
                aggregate='Median of fresh-process medians; matched-pair changes retained separately',
                timing_scope='Complete engine GPU recording/barriers and matching output copy; wall includes CPU recording/submission/wait; neither includes upload/readback',
                warmup_seconds=3.,measure_seconds=3.,minimum_measured_frames=30,
                model_dll_sha256=sha(MODEL),model_dll_bytes=MODEL.stat().st_size,ngx_core_sha256=sha(core[0]),
                native_manifest=manifest,runner_sha256=sha(Path(__file__)),
                probe_source_sha256=sha(ROOT/'tests/native/four_model_probe.cpp'),
                gpu_before=telemetry(),cases=[],comparisons=[],records=[])
    def save():write(args.output,report)
    save()
    try:
        with tempfile.TemporaryDirectory(prefix='native-vendor-refresh-',dir=ROOT/'build') as temporary:
            work=Path(temporary)
            executable,forwarder=build(work)
            report.update(probe_sha256=sha(executable),forwarder_sha256=sha(forwarder),state='running')
            for extent in args.extents:
                width,height=map(int,extent.split('x'))
                rgba=np.ones((height,width,4),np.float16)
                rgba[:,:,:3]=fixture(width,height)
                (work/'color.rgba16f').write_bytes(rgba.tobytes())
                (work/'motion.rg16f').write_bytes(np.zeros((height,width,2),np.float16).tobytes())
                del rgba
                for model in ('original','small-fp8'):
                    rows=[]
                    for repeat in range(args.rounds):
                        for engine in (('phoenix','nvidia') if repeat%2==0 else ('nvidia','phoenix')):
                            report['current_case']=dict(extent=extent,model=model,engine=engine,round=repeat)
                            save()
                            config=dict(width=width,height=height,engine=engine,
                                weight_source=dict(kind='nvidia_dll',path=str(MODEL),sha256=report['model_dll_sha256'],model=model),
                                bundle_path=str(bundle),native_kernel_style='auto',ngx_core=str(core[0]),
                                forwarder=str(forwarder),cache=str(work/'ngx-cache'),
                                color_file=str(work/'color.rgba16f'),motion_file=str(work/'motion.rg16f'),
                                capture=False,debug=False,warmup_seconds=3.,measure_seconds=3.,
                                minimum_measured_frames=30,maximum_measure_seconds=120.)
                            write(work/'job.json',config)
                            output=work/'result.json';output.unlink(missing_ok=True)
                            process=subprocess.run([str(executable),str(work/'job.json'),str(output)],cwd=work,
                                                   capture_output=True,text=True,timeout=240)
                            result=json.loads(output.read_text()) if output.exists() else {}
                            if process.returncode or not result.get('passed'):
                                raise RuntimeError(result.get('error',process.stderr or 'No successful probe result'))
                            if not all(c['finite'] for c in result['captures'].values()) or result['gpu']['samples']<30:
                                raise RuntimeError('Nonfinite output or incomplete measured sample count')
                            if engine=='phoenix' and (result['native_kernel_style']!='auto' or result['physical_cuda_architecture']!=120):
                                raise RuntimeError('Wrong Phoenix instruction style or physical GPU')
                            row=dict(extent=extent,model=model,engine=engine,round=repeat,passed=True,
                                     **{k:result[k] for k in ('gpu','wall','memory','create_ms','captures','reset_repeatable')})
                            rows.append(row);report['cases'].append(row);save()
                    groups={engine:[row for row in rows if row['engine']==engine] for engine in ('phoenix','nvidia')}
                    exact=all(row['captures']==rows[0]['captures'] and row['reset_repeatable'] for row in rows)
                    paired=[(groups['phoenix'][i]['gpu']['median_ms']/groups['nvidia'][i]['gpu']['median_ms']-1)*100
                            for i in range(args.rounds)]
                    comparison=dict(extent=extent,model=model,exact=exact,paired_phoenix_extra_gpu_percent=paired,
                                    median_phoenix_extra_gpu_percent=statistics.median(paired),gpu_after=telemetry())
                    report['comparisons'].append(comparison)
                    for engine,group in groups.items():
                        report['records'].append(dict(route=engine,model=model,extent=[width,height],profile='default',
                                                     processes=args.rounds,measured_date=report['measured_date'],**aggregate(group)))
                    save();print(json.dumps(comparison),flush=True)
                    if not exact:
                        raise RuntimeError('Phoenix/NVIDIA reset/history outputs differ')
            if sha(bundle/'native_manifest.json')!=sha_from_json(report['native_manifest']):
                # Compare parsed metadata below: its on-disk formatting is not
                # part of the logical source-identity equality check.
                if json.loads((bundle/'native_manifest.json').read_text())!=report['native_manifest']:
                    raise RuntimeError('Native bundle changed during measurement')
            report.update(passed=True,state='completed',gpu_after=telemetry())
            report.pop('current_case',None)
    except Exception as error:
        report.update(state='failed',error=str(error));raise
    finally:
        report['temporary_builds_removed']=True
        save()


def sha_from_json(value):
    import hashlib
    return hashlib.sha256((json.dumps(value,indent=2)+'\n').encode()).hexdigest()


if __name__=='__main__':
    main()
