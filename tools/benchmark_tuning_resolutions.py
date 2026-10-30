"""Explicit multi-resolution before/after GPU experiment, without game deployment.

Reconstruct the earlier control in temporary storage, authenticate its native
images and verify all styles against the saved 1080p reset/history outputs.
Run alternating paired processes at each requested extent. Failed cells retain
their reason, never a successful FPS/timing summary. Only compact receipts remain.
"""
import argparse
import concurrent.futures
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
from compare_four_models import ROOT, build, sha, telemetry, write
from experiment_native_tuning import CUDA, MODEL, PRODUCTION, STYLES
from production_build_native_kernels import configure_boundary_ptx
from test_model_selection import toolchain, run
from verify_native_tuning import summarize

DEFAULT_EXTENTS = ('1280x720','2560x1080','2560x1440','3840x2160','6144x3456','7680x4320')


def replace_once(path, before, after):
    text=path.read_text()
    if text.count(before)!=1:
        raise ValueError('Control reconstruction anchor changed: '+path.name)
    path.write_text(text.replace(before,after))


def control_source(destination):
    """Undo only the winning second-wave changes in an isolated source copy."""
    shutil.copytree(PRODUCTION,destination)
    math=destination/'common/architecture_math.cuh'
    replace_once(math,'#include "fp8_weight_values.cuh"\n','')
    replace_once(math,'decode_fp8_weight_fragment(b[i], right_low[i], right_high[i]);',
                 'decode_fp8_fragment(b[i], right_low[i], right_high[i]);')
    (destination/'common/fp8_weight_values.cuh').unlink()
    memory=destination/'common/device_memory_scalar.cuh'
    text=memory.read_text()
    # Keep each old scalar branch, removing its new vector-read alternative.
    pattern=r'#if PHOENIX_STYLE_ARCH >= 890\n.*?#else\n(.*?)#endif\n'
    text,count=re.subn(pattern,lambda match:match[1],text,flags=re.S)
    if count!=2:
        raise ValueError('Expected two vector-read alternatives')
    memory.write_text(text)
    copy_helper=destination/'common/compatibility_copy.cuh'
    text=copy_helper.read_text()
    wait='''#if PHOENIX_STYLE_ARCH == 890
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
'''
    if text.count(wait)!=1:
        raise ValueError('Consumer wait anchor changed')
    text=text.replace(wait,'')
    commit='''    asm volatile("cp.async.commit_group;" ::: "memory");
#if PHOENIX_STYLE_ARCH != 890
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif'''
    if text.count(commit)!=1:
        raise ValueError('Producer wait anchor changed')
    text=text.replace(commit,'    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");')
    copy_helper.write_text(text)
    attention=destination/'optimized/window_attention/channels_64/attention.cu'
    replace_once(attention,'''#if PHOENIX_STYLE_ARCH < 890
__maxnreg__(160)
#else
__maxnreg__(128)
#endif''','__maxnreg__(128)')
    return {p.relative_to(destination).as_posix():sha(p) for p in destination.rglob('*') if p.is_file()}


def sm120_pin(row):
    return next(image['sha256'] for image in row['architecture_images'] if image['architecture']=='sm_120')


def reconstruct_control(work, candidate, previous, report, save, jobs):
    """Retain exact older bytes when available; compile other active-target images."""
    source=work/'control-source'
    report['reconstructed_source_hashes']=control_source(source)
    control=work/'control';control.mkdir()
    manifest=copy.deepcopy(previous['manifests']['control'])
    tuned=previous['manifests']['candidate']
    if json.loads((candidate/'native_manifest.json').read_text())!=tuned:
        raise ValueError('Current winning bundle differs from the validated previous candidate')
    shutil.copy2(candidate/'native_graph.json',control/'native_graph.json')
    if sha(control/'native_graph.json')!=manifest['native_graph_sha256']:
        raise ValueError('Control execution graph changed')
    sdk=ROOT/'artifacts/current/sdk/windows-x64/native'
    sdk_rows={r['name']:r for r in json.loads((sdk/'native_manifest.json').read_text())['kernels']}
    _,env=toolchain(64)
    for row in manifest['kernels']:
        original=sdk_rows[row['name']]
        binary=sdk/original['binary']
        if sha(binary)!=sm120_pin(row):
            raise ValueError('Native control is not byte-identical to saved RTX 50 image')
        relative='native/'+row['name']+'.cubin'
        target=control/relative;target.parent.mkdir(exist_ok=True)
        shutil.copy2(binary,target)
        row.update(binary=relative,binary_sha256=sha(target),architecture_images=[
            dict(architecture='sm_120',sha256=sha(target),bytes=target.stat().st_size)])
    contracts={r['name']:r for r in json.loads((ROOT/'src/native/model/production_kernel_contracts.json').read_text())['kernels']}
    tasks=[]
    report['control_images']=[]
    for style in ('rtx40','rtx30','rtx20'):
        new_rows={r['name']:r for r in tuned['kernel_styles'][style]['kernels']}
        group=manifest['kernel_styles'][style]
        for row in group['kernels']:
            target=control/'styles'/style/(row['name']+'.cubin')
            target.parent.mkdir(parents=True,exist_ok=True)
            tasks.append((style,row,contracts[row['name']],new_rows[row['name']],target))
        group['architectures']=['sm_120']

    def compile_or_extract(task):
        style,row,contract,current,target=task
        expected=sm120_pin(row)
        if sm120_pin(current)==expected:
            directory=work/'extract'/style/row['name'];directory.mkdir(parents=True)
            run([CUDA/'cuobjdump.exe','--extract-elf','all',candidate/current['binary']],cwd=directory,env=env)
            images=[p for p in directory.iterdir() if p.read_bytes()[:4]==b'\x7fELF' and
                    (int.from_bytes(p.read_bytes()[48:52],'little')>>8)&255==120]
            if len(images)!=1 or sha(images[0])!=expected:
                raise ValueError('Retained control image mismatch')
            shutil.copy2(images[0],target)
            retained=True
        else:
            relative=Path(contract['source']).relative_to('src/native/kernels/production')
            options=contract.get('ptx_options')
            ptx=target.with_suffix('.ptx')
            command=[CUDA/'nvcc.exe','--entries',','.join([row['entry']]+([row['large_input_entry']] if 'large_input_entry' in row else [])),
                     '-allow-unsupported-compiler','-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
                     '-diag-suppress','177','-diag-suppress','550',
                     '-ptx' if options else '-cubin','-arch=sm_120','-O3','-std=c++17','-lineinfo',
                     '-DPHOENIX_KERNEL_STYLE='+str(STYLES[style]),
                     *([] if options else [f'-Xptxas=-O{contract.get("assembler_optimization",3)},--suppress-debug-info']),
                     source/relative,'-o',ptx if options else target]
            run(command,cwd=work,env=env,timeout=900)
            if options:
                configure_boundary_ptx(ptx,'sm_120',row['entry'],options)
                run([CUDA/'ptxas.exe','-arch=sm_120','--entry',row['entry'],
                     f'-O{contract.get("assembler_optimization",3)}','--suppress-debug-info',ptx,'-o',target],
                    cwd=work,env=env,timeout=900)
                ptx.unlink()
            retained=False
        return style,row,target,dict(style=style,kernel=row['name'],retained=retained,
                                    earlier_image_sha256=expected,actual_sha256=sha(target),
                                    byte_identical=expected==sha(target))

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures=[pool.submit(compile_or_extract,task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            style,row,target,evidence=future.result()
            row.update(binary=target.relative_to(control).as_posix(),binary_sha256=sha(target),architecture_images=[
                dict(architecture='sm_120',sha256=sha(target),bytes=target.stat().st_size)])
            report['control_images'].append(evidence)
            save()
            if len(report['control_images'])%12==0:
                print('Prepared control images',len(report['control_images']),'/ 123',flush=True)
    manifest.update(architectures=['sm_120'],source_hashes=report['reconstructed_source_hashes'],
                    reconstruction='Second-wave changes removed; validated against saved 1080p outputs')
    write(control/'native_manifest.json',manifest)
    report['control_manifest']=manifest
    save()
    return control


def fixture(work, width, height):
    """Write one frame in bounded row blocks, including 8K without a full grid."""
    pixels=np.memmap(work/'color.rgba16f',mode='w+',dtype=np.float16,shape=(height,width,4))
    x=np.arange(width,dtype=np.float32)[None,:]
    for first in range(0,height,128):
        last=min(first+128,height)
        y=np.arange(first,last,dtype=np.float32)[:,None]
        pixels[first:last,:,0]=x/width
        pixels[first:last,:,1]=y/height
        pixels[first:last,:,2]=.5+.25*np.sin(x*.17)*np.cos(y*.11)
        pixels[first:last,:,3]=1
    pixels.flush();pixels._mmap.close()
    remaining=width*height*4
    with (work/'motion.rg16f').open('wb') as stream:
        zeros=bytes(1024*1024)
        while remaining:
            count=min(remaining,len(zeros));stream.write(zeros[:count]);remaining-=count


def execute(executable,forwarder,work,bundle,style,model,width,height,model_sha,validation=False):
    config=dict(width=width,height=height,engine='phoenix',
                weight_source=dict(kind='nvidia_dll',path=str(MODEL),sha256=model_sha,model=model),
                bundle_path=str(bundle),native_kernel_style=style,forwarder=str(forwarder),
                color_file=str(work/'color.rgba16f'),motion_file=str(work/'motion.rg16f'),
                capture=False,debug=False,warmup_seconds=.1 if validation else 1.,
                measure_seconds=.1 if validation else 2.,minimum_measured_frames=3 if validation else 30,
                maximum_measure_seconds=120.)
    write(work/'job.json',config)
    output=work/'result.json';output.unlink(missing_ok=True)
    started=time.monotonic()
    try:
        process=subprocess.run([str(executable),str(work/'job.json'),str(output)],cwd=work,
                               capture_output=True,text=True,timeout=240)
        result=json.loads(output.read_text()) if output.exists() else {}
        if process.returncode or not result.get('passed'):
            return dict(passed=False,error=result.get('error',process.stderr or 'No successful probe result'),exit_code=process.returncode)
        if result.get('native_kernel_style')!=style or result.get('physical_cuda_architecture')!=120:
            return dict(passed=False,error='Wrong style or physical GPU selected')
        if not result['reset_repeatable'] or not all(r['finite'] for r in result['captures'].values()):
            return dict(passed=False,error='Nonfinite or nonrepeatable output')
        if result['gpu']['samples']<config['minimum_measured_frames']:
            return dict(passed=False,error='Measurement cap reached before requested sample count')
        return dict(passed=True,seconds=time.monotonic()-started,
                    **{k:result[k] for k in ('gpu','wall','memory','captures','reset_repeatable','owned_memory')})
    except subprocess.TimeoutExpired:
        return dict(passed=False,error='Isolated probe exceeded 240 seconds',timeout=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,default=ROOT/'artifacts/current/native-kernel-styles')
    parser.add_argument('--previous',type=Path,default=ROOT/'artifacts/current/kernel-tuning/final-comparison.json')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/current/kernel-tuning/resolutions.json')
    parser.add_argument('--extents',nargs='+',default=DEFAULT_EXTENTS)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--jobs',type=int,default=8)
    parser.add_argument('--resume-work',type=Path,
                        help='Resume a saved sweep from its verified temporary build folder')
    args=parser.parse_args()
    if not 1<=args.jobs<=8 or not 1<=args.rounds<=5:
        parser.error('Use 1-8 build jobs and 1-5 paired rounds')
    sizes=[]
    for extent in args.extents:
        width,height=map(int,extent.split('x'))
        if not 1<=width<=16384 or not 1<=height<=16384:
            parser.error('Extent exceeds supported D3D texture dimensions')
        sizes.append((extent,width,height))
    previous=json.loads(args.previous.read_text())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(schema='phoenix-tuning-resolutions/1',passed=False,state='preparing-control',
                gpu_before=telemetry(),games_launched=False,installed_files_modified=False,published=False,
                older_device_validated=False,rounds=args.rounds,extents=args.extents,
                measure_seconds=2.,warmup_seconds=1.,minimum_measured_frames=30,
                timing_scope='Complete model and matching output copy; excludes upload/readback/CPU gaps',
                model_sha256=sha(MODEL),runner_sha256=sha(Path(__file__)),
                probe_source_sha256=sha(ROOT/'tests/native/four_model_probe.cpp'),
                previous_receipt_sha256=sha(args.previous),candidate_manifest=previous['manifests']['candidate'],
                baseline_validation=[],cases=[],decisions=[])
    completed=set()
    if args.resume_work:
        report=json.loads(args.output.read_text())
        if report['previous_receipt_sha256']!=sha(args.previous) or report['candidate_manifest']!=previous['manifests']['candidate']:
            raise ValueError('Cannot resume with different control or candidate identities')
        if report['extents']!=list(args.extents) or report['rounds']!=args.rounds:
            raise ValueError('Cannot change the resolution matrix while resuming')
        completed={(r['extent'],r['style'],r['model']) for r in report['decisions']}
        retained=[r for r in report['cases'] if (r['extent'],r['style'],r['model']) in completed]
        for row in retained:
            row.setdefault('probe_sha256',report['probe_sha256'])
        report.setdefault('harness_revisions',[]).append(dict(
            probe_sha256=report['probe_sha256'],source_sha256=report['probe_source_sha256'],
            runner_sha256=report['runner_sha256'],
            retained_cases=len(retained),interrupted_cases_restarted=len(report['cases'])-len(retained),
            reason='Avoid a temporary error-message allocation per FP16 output component; GPU recording unchanged'))
        report.update(cases=retained,state='resuming',passed=False,runner_sha256=sha(Path(__file__)))
        report.pop('current_case',None)
    def save():write(args.output,report)
    save()
    try:
        with ExitStack() as cleanup:
            if args.resume_work:
                work=args.resume_work.resolve()
                if (work.parent!=(ROOT/'build').resolve() or not work.name.startswith('tuning-resolutions-')
                        or work.is_symlink() or work.is_junction()):
                    raise ValueError('Unsafe resumed build folder')
                if any(p.is_symlink() or p.is_junction() for p in work.rglob('*')):
                    raise ValueError('Resumed build contains redirected paths')
                cleanup.callback(shutil.rmtree,work)
                control=work/'control'
                manifest=json.loads((control/'native_manifest.json').read_text())
                if manifest!=report['control_manifest']:
                    raise ValueError('Resumed control manifest changed')
                for row in manifest['kernels']+[r for group in manifest['kernel_styles'].values() for r in group['kernels']]:
                    if sha(control/row['binary'])!=row['binary_sha256']:
                        raise ValueError('Resumed control binary changed')
                if json.loads((args.candidate/'native_manifest.json').read_text())!=report['candidate_manifest']:
                    raise ValueError('Resumed candidate changed')
            else:
                work=Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix='tuning-resolutions-',dir=ROOT/'build')))
                control=reconstruct_control(work,args.candidate.resolve(),previous,report,save,args.jobs)
            executable,forwarder=build(work)
            report.update(probe_sha256=sha(executable),forwarder_sha256=sha(forwarder),state='validating-control',
                          probe_source_sha256=sha(ROOT/'tests/native/four_model_probe.cpp'))
            if args.resume_work:
                report['earlier_baseline_validation']=report['baseline_validation']
                report['baseline_validation']=[]
            fixture(work,1920,1080)
            for style in ('auto','rtx40','rtx30','rtx20'):
                for model in ('original','small-fp8'):
                    result=execute(executable,forwarder,work,control,style,model,1920,1080,report['model_sha256'],True)
                    expected=next(r for r in previous['cases'] if r['extent']=='1920x1080' and
                                  r['style']==style and r['model']==model and r['label']=='control')['captures']
                    good=result['passed'] and result['captures']==expected
                    report['baseline_validation'].append(dict(style=style,model=model,exact=good,result=result))
                    save()
                    if not good:raise RuntimeError('Reconstructed control differs from saved outputs: '+style+' '+model)
            report['state']='running';save()
            for extent,width,height in sizes:
                if all((extent,style,model) in completed for style in STYLES for model in ('original','small-fp8')):
                    continue
                fixture(work,width,height)
                for style in ('auto','rtx40','rtx30','rtx20'):
                    for model in ('original','small-fp8'):
                        if (extent,style,model) in completed:
                            continue
                        cases=[]
                        for repeat in range(args.rounds):
                            for label in (('control','candidate') if repeat%2==0 else ('candidate','control')):
                                report['current_case']=dict(extent=extent,style=style,model=model,round=repeat,label=label)
                                save()
                                result=execute(executable,forwarder,work,control if label=='control' else args.candidate.resolve(),
                                               style,model,width,height,report['model_sha256'])
                                row=dict(extent=extent,style=style,model=model,round=repeat,label=label,
                                         probe_sha256=report['probe_sha256'],**result)
                                cases.append(row);report['cases'].append(row);save()
                        decision=dict(extent=extent,style=style,model=model,rounds=args.rounds,gpu_after=telemetry())
                        if all(row['passed'] for row in cases):
                            decision.update(**summarize(cases,args.rounds))
                            decision['passed']=decision['exact']
                        else:
                            decision.update(passed=False,errors=[dict(label=r['label'],round=r['round'],error=r['error'])
                                                                 for r in cases if not r['passed']])
                        report['decisions'].append(decision);save()
                        print(json.dumps(decision),flush=True)
            report.update(passed=all(row['passed'] for row in report['decisions']),state='completed',gpu_after=telemetry())
            report.pop('current_case',None)
    except Exception as error:
        report.update(state='failed',error=str(error))
        raise
    finally:
        report['temporary_builds_removed']=True
        save()


if __name__=='__main__':
    main()
