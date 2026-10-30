"""Explicit GPU primitive/operator tests, optionally queued after model screens."""
import argparse
import json
from pathlib import Path
import tempfile
import time

from compare_four_models import ROOT, sha, write
from experiment_native_tuning import source_variant, CUDA, PRODUCTION
from test_model_selection import toolchain, run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true',help='Required to execute these isolated GPU checks')
    parser.add_argument('--wait-for',nargs='*',type=Path,default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.gpu:parser.error('GPU execution requires explicit --gpu')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(passed=False,gpu_used=True,older_device_validated=False,full_model_weight_cache_tested=False,
                primitive_cases=[],preexpanded_weight_operator=[],state='compiling')
    write(args.output,report);_,env=toolchain(64)
    flags=[CUDA/'nvcc.exe','-allow-unsupported-compiler','-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
           '-arch=sm_120','-O3','-std=c++17']
    try:
        with tempfile.TemporaryDirectory(prefix='tuning-primitives-',dir=ROOT/'build') as temporary:
            work=Path(temporary);jobs=[]
            for profile in ('control','packed_decode','late_fragments','deferred_copy_wait'):
                source=PRODUCTION
                if profile!='control':source=work/profile;source_variant(source,profile)
                text=(ROOT/'tests/native/compatibility_copy_tests.cu').read_text()
                text=text.replace('../../src/native/kernels/production/common/pipeline_memory.cuh',
                                  (source/'common/pipeline_memory.cuh').as_posix())
                if profile=='deferred_copy_wait':
                    text=text.replace('__syncthreads();','phoenix_architecture::finish_compatibility_copies();')
                unit=work/(profile+'.cu');unit.write_text(text)
                for style in (750,860,890):
                    exe=work/(profile+str(style)+'.exe')
                    run(flags+[f'-DPHOENIX_KERNEL_STYLE={style}',unit,'-o',exe],cwd=work,env=env,timeout=600)
                    jobs.append((profile,style,exe))
            operators=[]
            for style in (750,860):
                exe=work/f'predecoded{style}.exe'
                run(flags+[f'-DPHOENIX_KERNEL_STYLE={style}',ROOT/'tests/native/predecoded_weights_probe.cu','-o',exe],
                    cwd=work,env=env,timeout=600)
                operators.append((style,exe))
            report['state']='waiting';write(args.output,report)
            for path in args.wait_for:
                while True:
                    try:value=json.loads(path.read_text())
                    except json.JSONDecodeError:time.sleep(1);continue
                    if value.get('state') not in ('running','compiling','waiting'):break
                    time.sleep(1)
            report['state']='running';write(args.output,report)
            for profile,style,exe in jobs:
                result=json.loads(run([exe],cwd=work,env=env,timeout=60))
                report['primitive_cases'].append(dict(profile=profile,style=style,result=result,binary_sha256=sha(exe)))
                write(args.output,report)
                print(profile,style,'passed',result['passed'],flush=True)
            for style,exe in operators:
                result=json.loads(run([exe],cwd=work,env=env,timeout=60))
                report['preexpanded_weight_operator'].append(dict(style=style,result=result,binary_sha256=sha(exe)))
                write(args.output,report)
                print('Pre-expanded operator',style,result,flush=True)
            report.update(passed=True,state='completed')
    except Exception as error:report.update(state='failed',error=str(error));raise
    finally:report['temporary_builds_removed']=True;write(args.output,report)


if __name__=='__main__':main()
