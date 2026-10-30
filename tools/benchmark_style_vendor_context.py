"""1080p instruction-style context for the matched native/NVIDIA README study.

All modes run on RTX 5080. Reuse the structured fixture, current kernels and
sampling protocol; wait for the vendor sweep to avoid GPU contention.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from benchmark_native_vendor import aggregate
from benchmark_readme_case import fixture
from compare_four_models import ROOT, build, sha, write
from experiment_native_tuning import MODEL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=ROOT/'artifacts/current/native-vendor-refresh/run.json')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/current/native-vendor-refresh/styles-1080p.json')
    args=parser.parse_args()
    bundle=ROOT/'artifacts/current/native-kernel-styles'
    report=dict(passed=False,state='preparing',gpu='RTX 5080',older_device_validated=False,
                games_launched=False,installed_files_modified=False,cases=[],records=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():write(args.output,report)
    save()
    try:
        with tempfile.TemporaryDirectory(prefix='style-vendor-context-',dir=ROOT/'build') as temporary:
            work=Path(temporary)
            executable,forwarder=build(work)
            report.update(probe_sha256=sha(executable),native_manifest_sha256=sha(bundle/'native_manifest.json'),
                          state='waiting-for-native-sweep')
            save()
            while True:
                try:reference=json.loads(args.reference.read_text())
                except json.JSONDecodeError:
                    time.sleep(1);continue
                if reference['state']=='completed':break
                if reference['state']=='failed':raise RuntimeError('Native/NVIDIA reference failed')
                time.sleep(1)
            if not reference['passed']:raise RuntimeError('Invalid native/NVIDIA reference')
            rgba=np.ones((1080,1920,4),np.float16);rgba[:,:,:3]=fixture(1920,1080)
            (work/'color.rgba16f').write_bytes(rgba.tobytes())
            (work/'motion.rg16f').write_bytes(np.zeros((1080,1920,2),np.float16).tobytes())
            report.update(state='running',reference_receipt_sha256=sha(args.reference))
            save()
            for model in ('original','small-fp8'):
                groups={style:[] for style in ('rtx40','rtx30','rtx20')}
                for repeat in range(3):
                    for style in (('rtx40','rtx30','rtx20') if repeat%2==0 else ('rtx20','rtx30','rtx40')):
                        report['current_case']=dict(model=model,style=style,round=repeat);save()
                        config=dict(width=1920,height=1080,engine='phoenix',native_kernel_style=style,
                            weight_source=dict(kind='nvidia_dll',path=str(MODEL),sha256=reference['model_dll_sha256'],model=model),
                            bundle_path=str(bundle),forwarder=str(forwarder),color_file=str(work/'color.rgba16f'),
                            motion_file=str(work/'motion.rg16f'),capture=False,debug=False,
                            warmup_seconds=3.,measure_seconds=3.,minimum_measured_frames=30,maximum_measure_seconds=120.)
                        write(work/'job.json',config);output=work/'result.json';output.unlink(missing_ok=True)
                        process=subprocess.run([str(executable),str(work/'job.json'),str(output)],cwd=work,
                                               capture_output=True,text=True,timeout=180)
                        result=json.loads(output.read_text()) if output.exists() else {}
                        if process.returncode or not result.get('passed'):
                            raise RuntimeError(result.get('error',process.stderr or 'Style probe failed'))
                        if result['native_kernel_style']!=style or result['physical_cuda_architecture']!=120:
                            raise RuntimeError('Wrong style/device')
                        if not result['reset_repeatable'] or not all(c['finite'] for c in result['captures'].values()):
                            raise RuntimeError('Invalid style output')
                        row=dict(model=model,style=style,round=repeat,
                                 **{key:result[key] for key in ('gpu','wall','memory','create_ms','captures','reset_repeatable')})
                        groups[style].append(row);report['cases'].append(row);save()
                for style,rows in groups.items():
                    if not all(row['captures']==rows[0]['captures'] for row in rows):
                        raise RuntimeError('Style output varies across processes')
                    report['records'].append(dict(model=model,style=style,extent=[1920,1080],
                        processes=3,repeatable=True,**aggregate(rows)))
                save()
                print(model,'style context complete',flush=True)
            report.update(passed=True,state='completed');report.pop('current_case',None)
    except Exception as error:
        report.update(state='failed',error=str(error));raise
    finally:
        report['temporary_builds_removed']=True;save()


if __name__=='__main__':
    main()
