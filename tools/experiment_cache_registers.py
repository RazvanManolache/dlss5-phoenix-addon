"""Target the new cache's measured spill hotspot; never change global budgets."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
import time

from compare_four_models import ROOT,build,sha,write
from experiment_native_tuning import CUDA,MODEL,STYLES,PRODUCTION,fixture
from test_model_selection import toolchain,run
from benchmark_register_limits import ResourceQuery

NAME='cc_tinlayout_fused_swin_2h_64_2_chained_fp8'
RELATIVE=Path('optimized/window_attention/channels_64/attention.cu')


def with_register_budget(source, budget):
    """Change the compatibility annotation, never the native FP8 branch."""
    conditional = '#if PHOENIX_STYLE_ARCH < 890\n__maxnreg__(160)'
    if source.count(conditional) == 1:
        return source.replace(conditional, '#if PHOENIX_STYLE_ARCH < 890\n__maxnreg__('+str(budget)+')')
    result, count = re.subn(r'__maxnreg__\(\s*128\s*\)',f'__maxnreg__({budget})',source)
    if count != 1:
        raise ValueError('Unexpected register annotation; refuse to change the wrong path')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(passed=False,state='waiting',scope='Only the 64-channel attention register budget changes',
                games_launched=False,installed_files_modified=False,older_device_validated=False,cases=[],decisions=[])
    write(args.output,report)
    while not (args.bundle/'native_manifest.json').exists():time.sleep(1)
    base=args.bundle.resolve();manifest=json.loads((base/'native_manifest.json').read_text())
    _,env=toolchain(64)
    try:
        with tempfile.TemporaryDirectory(prefix='cache-registers-',dir=ROOT/'build') as temporary:
            work=Path(temporary);source=work/'production';shutil.copytree(PRODUCTION,source)
            original=(source/RELATIVE).read_text()
            executable,forwarder=build(work);fixture(work)
            for style in ('rtx20','rtx30'):
                row=next(r for r in manifest['kernel_styles'][style]['kernels'] if r['name']==NAME)
                query=ResourceQuery()
                try:control_resources=query.query(base/row['binary'],row['entry'],64)
                finally:query.close()
                for budget in (160,192,224,255):
                    folder=work/f'{style}-{budget}';folder.mkdir()
                    (source/RELATIVE).write_text(with_register_budget(original,budget))
                    binary=folder/'attention.cubin'
                    run([CUDA/'nvcc.exe','--entries',row['entry'],'-allow-unsupported-compiler',
                         '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH','-cubin','-arch=sm_120','-O3','-std=c++17',
                         f'-DPHOENIX_KERNEL_STYLE={STYLES[style]}','-Xptxas=-O3,--suppress-debug-info',
                         source/RELATIVE,'-o',binary],cwd=work,env=env,timeout=600)
                    query=ResourceQuery()
                    try:resources=query.query(binary,row['entry'],64)
                    finally:query.close()
                    candidate=copy.deepcopy(manifest)
                    paths={r['binary'] for r in candidate['kernels']}
                    paths|={r['binary'] for v in candidate['kernel_styles'].values() for r in v['kernels']}
                    paths.add('native_graph.json')
                    for name in paths:
                        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);os.link(base/name,path)
                    changed=next(r for r in candidate['kernel_styles'][style]['kernels'] if r['name']==NAME)
                    changed.update(binary='attention.cubin',binary_sha256=sha(binary),architecture_images=[dict(architecture='sm_120',sha256=sha(binary),bytes=binary.stat().st_size)])
                    candidate['kernel_styles'][style]['architectures']=['sm_120'];write(folder/'native_manifest.json',candidate)
                    groups={'control':[],'candidate':[]}
                    report['state']='running';write(args.output,report)
                    for repeat in range(3):
                        for label in (('control','candidate') if repeat%2==0 else ('candidate','control')):
                            config=dict(width=1920,height=1080,engine='phoenix',
                                weight_source=dict(kind='nvidia_dll',path=str(MODEL),sha256=sha(MODEL),model='original'),
                                bundle_path=str(base if label=='control' else folder),native_kernel_style=style,
                                forwarder=str(forwarder),color_file=str(work/'color.rgba16f'),motion_file=str(work/'motion.rg16f'),
                                capture=False,debug=False,warmup_seconds=.5,measure_seconds=1.,minimum_measured_frames=30)
                            write(work/'job.json',config);(work/'result.json').unlink(missing_ok=True)
                            p=subprocess.run([str(executable),str(work/'job.json'),str(work/'result.json')],cwd=work,
                                             capture_output=True,text=True,timeout=90)
                            r=json.loads((work/'result.json').read_text()) if (work/'result.json').exists() else {}
                            if p.returncode or not r.get('passed'):raise RuntimeError(r.get('error',p.stderr))
                            result=dict(style=style,budget=budget,label=label,round=repeat,gpu=r['gpu'],captures=r['captures'],memory=r['memory'])
                            groups[label].append(result);report['cases'].append(result);write(args.output,report)
                    exact=all(r['captures']==groups['control'][0]['captures'] for group in groups.values() for r in group)
                    pairs=[(groups['candidate'][i]['gpu']['median_ms']/groups['control'][i]['gpu']['median_ms']-1)*100 for i in range(3)]
                    result=dict(style=style,budget=budget,exact=exact,paired_changes_percent=pairs,
                                paired_median_change_percent=statistics.median(pairs),before=control_resources,after=resources)
                    report['decisions'].append(result);write(args.output,report)
                    print(style,budget,result,flush=True)
                    if not exact:raise RuntimeError('Register candidate changed output')
            report.update(passed=True,state='completed')
    except Exception as error:report.update(state='failed',error=str(error));raise
    finally:report['temporary_builds_removed']=True;write(args.output,report)


if __name__=='__main__':main()
