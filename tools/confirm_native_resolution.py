"""Recheck a small native-50 resolution effect, with longer warmup and five pairs.

Preparation is CPU-only. GPU work starts only after the main serial sweep exits.
The native control is the exact SDK image set already authenticated by that sweep.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from benchmark_tuning_resolutions import fixture, sm120_pin
from compare_four_models import ROOT, build, sha, telemetry, write
from experiment_native_tuning import MODEL
from verify_native_tuning import summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep',type=Path,default=ROOT/'artifacts/current/kernel-tuning/resolutions.json')
    parser.add_argument('--previous',type=Path,default=ROOT/'artifacts/current/kernel-tuning/final-comparison.json')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/current/kernel-tuning/native-8k-confirmation.json')
    args=parser.parse_args()
    candidate=ROOT/'artifacts/current/native-kernel-styles'
    sdk=ROOT/'artifacts/current/sdk/windows-x64/native'
    previous=json.loads(args.previous.read_text())
    report=dict(passed=False,state='preparing',extent='7680x4320',rounds=5,warmup_seconds=3.,measure_seconds=3.,
                games_launched=False,installed_files_modified=False,older_device_validated=False,
                model_sha256=sha(MODEL),runner_sha256=sha(Path(__file__)),cases=[],decisions=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():write(args.output,report)
    save()
    try:
        with tempfile.TemporaryDirectory(prefix='confirm-native-resolution-',dir=ROOT/'build') as temporary:
            work=Path(temporary);control=work/'control';control.mkdir()
            manifest=copy.deepcopy(previous['manifests']['control'])
            manifest.pop('kernel_styles',None)
            manifest['architectures']=['sm_120']
            sdk_rows={r['name']:r for r in json.loads((sdk/'native_manifest.json').read_text())['kernels']}
            for row in manifest['kernels']:
                source=sdk/sdk_rows[row['name']]['binary']
                if sha(source)!=sm120_pin(row):
                    raise ValueError('Native control image differs from saved benchmark')
                target=control/row['binary'];target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
                row.update(binary_sha256=sha(target),architecture_images=[dict(
                    architecture='sm_120',sha256=sha(target),bytes=target.stat().st_size)])
            shutil.copy2(candidate/'native_graph.json',control/'native_graph.json')
            if sha(control/'native_graph.json')!=manifest['native_graph_sha256']:
                raise ValueError('Native control graph differs')
            write(control/'native_manifest.json',manifest)
            executable,forwarder=build(work)
            report.update(control_manifest=manifest,candidate_manifest_sha256=sha(candidate/'native_manifest.json'),
                          probe_sha256=sha(executable),probe_source_sha256=sha(ROOT/'tests/native/four_model_probe.cpp'),
                          state='waiting-for-sweep')
            save()
            while True:
                try:sweep=json.loads(args.sweep.read_text())
                except json.JSONDecodeError:
                    time.sleep(1);continue
                if sweep['state']=='completed':break
                if sweep['state']=='failed':raise RuntimeError('Main sweep failed; do not start another GPU run')
                time.sleep(1)
            if not sweep['passed']:
                raise RuntimeError('Main sweep has invalid cells; investigate those first')
            fixture(work,7680,4320)
            report.update(state='running',gpu_before=telemetry());save()
            for model in ('original','small-fp8'):
                expected=next(r['captures'] for r in sweep['cases'] if r['extent']=='7680x4320' and
                              r['style']=='auto' and r['model']==model and r['label']=='control')
                rows=[]
                for repeat in range(5):
                    for label in (('control','candidate') if repeat%2==0 else ('candidate','control')):
                        config=dict(width=7680,height=4320,engine='phoenix',
                            weight_source=dict(kind='nvidia_dll',path=str(MODEL),sha256=report['model_sha256'],model=model),
                            bundle_path=str(control if label=='control' else candidate),native_kernel_style='auto',
                            forwarder=str(forwarder),color_file=str(work/'color.rgba16f'),motion_file=str(work/'motion.rg16f'),
                            capture=False,debug=False,warmup_seconds=3.,measure_seconds=3.,
                            minimum_measured_frames=30,maximum_measure_seconds=120.)
                        write(work/'job.json',config)
                        output=work/'result.json';output.unlink(missing_ok=True)
                        process=subprocess.run([str(executable),str(work/'job.json'),str(output)],cwd=work,
                                               capture_output=True,text=True,timeout=240)
                        result=json.loads(output.read_text()) if output.exists() else {}
                        if (process.returncode or not result.get('passed') or
                                result.get('captures')!=expected or not all(r['finite'] for r in result['captures'].values())):
                            raise RuntimeError(result.get('error','Native confirmation failed output checks'))
                        row=dict(label=label,round=repeat,model=model,
                                 **{k:result[k] for k in ('gpu','captures','memory','reset_repeatable')})
                        rows.append(row);report['cases'].append(row);save()
                decision=dict(model=model,**summarize(rows,5))
                report['decisions'].append(decision);save()
                print(json.dumps(decision),flush=True)
            report.update(passed=True,state='completed',gpu_after=telemetry())
    except Exception as error:
        report.update(state='failed',error=str(error));raise
    finally:
        report['temporary_builds_removed']=True
        save()


if __name__=='__main__':
    main()
