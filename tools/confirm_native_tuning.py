"""Confirm combined, full-kernel candidates before changing production sources."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

from compare_four_models import ROOT, build, sha, write, telemetry
from experiment_native_tuning import BASE, MODEL, PRODUCTION, source_variant,compile_case,fixture
from test_model_selection import toolchain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/current/kernel-tuning/confirmation.json')
    parser.add_argument('--weight-cache',action='store_true',help='Confirm the global decoded-weight pair cache across every arithmetic module')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(passed=False,state='compiling',games_launched=False,installed_files_modified=False,
                older_device_validated=False,extent=[1920,1080],rounds=5,cases=[],decisions=[],candidate_manifests={})
    write(args.output,report);_,env=toolchain(64)
    contracts=json.loads((ROOT/'src/native/model/production_kernel_contracts.json').read_text())['kernels']
    try:
        with tempfile.TemporaryDirectory(prefix='native-confirmation-',dir=ROOT/'build') as temporary:
            work=Path(temporary);exe,fwd=build(work);fixture(work);prepared={}
            vector=work/'vector';source_variant(vector,'vector_memory')
            late=work/'late';source_variant(late,'late_fragments')
            overlap=work/'overlap';source_variant(overlap,'deferred_copy_wait')
            # This confirmation expands the Turing lifetime change to every
            # arithmetic module; it is no longer merely a hot-module screen.
            shutil.copy2(vector/'common/device_memory_scalar.cuh',late/'common/device_memory_scalar.cuh')
            shutil.copy2(vector/'common/device_memory_scalar.cuh',overlap/'common/device_memory_scalar.cuh')
            plan=[('rtx20',late,'all_math'),('rtx40',overlap,'deferred_copy_wait'),('auto',vector,'vector_memory')]
            if args.weight_cache:
                turing=work/'turing-cache';ampere=work/'ampere-cache'
                source_variant(turing,'global_weight_pairs');source_variant(ampere,'global_weight_pairs')
                shutil.copy2(vector/'common/device_memory_scalar.cuh',turing/'common/device_memory_scalar.cuh')
                plan=[('rtx20',turing,'all_math'),('rtx30',ampere,'all_math')]
            for style,source,selection in plan:
                folder=work/('bundle-'+style)
                changed=compile_case(source,folder,selection,style,contracts,env)
                prepared[style]=folder
                manifest=json.loads((folder/'native_manifest.json').read_text())
                report['candidate_manifests'][style]=manifest;write(args.output,report)
                print('Prepared full confirmation',style,len(changed),'modules',flush=True)
            if args.wait_for:
                while json.loads(args.wait_for.read_text()).get('state') in ('running','compiling','waiting'):
                    time.sleep(1)
            report.update(state='running',gpu_before=telemetry());write(args.output,report)
            for style in prepared:
                for model in ('original','small-fp8'):
                    groups={'control':[],'candidate':[]}
                    for repeat in range(5):
                        for label in (('control','candidate') if repeat%2==0 else ('candidate','control')):
                            config=dict(width=1920,height=1080,engine='phoenix',
                                weight_source=dict(kind='nvidia_dll',path=str(MODEL),sha256=sha(MODEL),model=model),
                                bundle_path=str(BASE if label=='control' else prepared[style]),native_kernel_style=style,
                                forwarder=str(fwd),color_file=str(work/'color.rgba16f'),motion_file=str(work/'motion.rg16f'),
                                capture=False,debug=False,warmup_seconds=1.,measure_seconds=2.,minimum_measured_frames=30)
                            write(work/'job.json',config);(work/'result.json').unlink(missing_ok=True)
                            p=subprocess.run([str(exe),str(work/'job.json'),str(work/'result.json')],cwd=work,
                                             capture_output=True,text=True,timeout=90)
                            r=json.loads((work/'result.json').read_text()) if (work/'result.json').exists() else {}
                            if p.returncode or not r.get('passed'):raise RuntimeError(r.get('error',p.stderr))
                            row=dict(style=style,model=model,label=label,round=repeat,gpu=r['gpu'],captures=r['captures'],memory=r['memory'])
                            groups[label].append(row);report['cases'].append(row);write(args.output,report)
                    exact=all(r['captures']==groups['control'][0]['captures'] for g in groups.values() for r in g)
                    a,b=[statistics.median(r['gpu']['median_ms'] for r in groups[k]) for k in ('control','candidate')]
                    ratios=[groups['candidate'][i]['gpu']['median_ms']/groups['control'][i]['gpu']['median_ms'] for i in range(5)]
                    decision=dict(style=style,model=model,exact=exact,control_ms=a,candidate_ms=b,
                                  change_percent=(b/a-1)*100,paired_changes_percent=[(v-1)*100 for v in ratios],
                                  control_rounds_ms=[r['gpu']['median_ms'] for r in groups['control']],
                                  candidate_rounds_ms=[r['gpu']['median_ms'] for r in groups['candidate']])
                    report['decisions'].append(decision);write(args.output,report)
                    print('Confirmation',style,model,decision,flush=True)
                    if not exact:raise RuntimeError('Combined candidate changed output')
            # Retain only this current, model-free candidate until integration.
            staged=args.output.parent/(args.output.stem+'-candidates');staged.mkdir(exist_ok=True)
            for style,folder in prepared.items():
                target=staged/style
                if target.exists():raise RuntimeError('Do not overwrite a previous confirmation candidate')
                target.mkdir();shutil.copy2(folder/'native_manifest.json',target/'native_manifest.json')
                for binary in (folder/'overrides').rglob('*.cubin'):
                    path=target/binary.relative_to(folder);path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(binary,path)
            report.update(passed=True,state='completed',gpu_after=telemetry())
    except Exception as error:report.update(state='failed',error=str(error));raise
    finally:report['temporary_builds_removed']=True;write(args.output,report)


if __name__=='__main__':main()
