"""Compile once and run explicit Windows portable GPU transport contracts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import hashlib
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from test_model_selection import toolchain


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('package','image','report','cuda'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config',type=Path,nargs='+',required=True,help='One config, or a matrix using --report as an output directory')
    parser.add_argument("--trials",type=int,default=1)
    parser.add_argument('--producer-device',help='Explicit CUDA UUID of the D3D12 producer; independent of the neural device')
    parser.add_argument('--size',type=int,nargs=2,metavar=('WIDTH','HEIGHT'))
    parser.add_argument('--baseline-package',type=Path,help='Also require identical temporal/control frame hashes from this pre-change runtime')
    args=parser.parse_args();assert 1<=args.trials<=5
    compiler,env=toolchain(64)
    source=Image.open(args.image).convert('RGB')
    if args.size:source=source.resize(tuple(args.size),Image.Resampling.LANCZOS)
    image=np.asarray(source,dtype=np.float32)/np.float32(255)
    height,width,_=image.shape
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    build_file=args.package/'BUILD.json'
    build=json.loads(build_file.read_text()) if build_file.exists() else None
    if build and 'source_inputs' in build:
        assert build['passed'] and all(sha(ROOT/n)==h for n,h in build['source_inputs'].items()),'Rebuild source-stale runtime before testing'
        assert all(sha(args.package/n)==h for n,h in build['files'].items()),'Runtime inventory changed'
    inputs={str(p.resolve()):sha(p) for p in args.config}
    assert len({p.stem for p in args.config})==len(args.config),'Matrix config names must be unique'
    with tempfile.TemporaryDirectory(prefix='cuda-d3d12-contract-',dir=ROOT/'build') as directory:
        temporary=Path(directory);exe=temporary/'contracts.exe';raw=temporary/'input.rgb32f';image.tofile(raw)
        result=subprocess.run([compiler,'/nologo','/std:c++20','/EHsc','/O2','/fp:strict','/DNOMINMAX',
            '/I'+str(args.cuda/'include'),str(ROOT/'tests/native/portable_gpu_contracts.cpp'),'/Fe:'+str(exe),
            '/link','/LIBPATH:'+str(args.cuda/'lib/x64'),'cuda.lib','d3d11.lib','d3d12.lib','dxgi.lib','dxguid.lib','d3dcompiler.lib','bcrypt.lib','windowscodecs.lib','ole32.lib','uuid.lib'],cwd=temporary,env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        for source_config in args.config:
            print('CONTRACTS '+source_config.stem,flush=True)
            job=dict(runtime_root=str(args.package.resolve()),config=json.loads(source_config.read_text()),width=width,height=height,input=str(raw))
            if args.producer_device:job['producer_device']=args.producer_device
            config=temporary/'job.json';config.write_text(json.dumps(job))
            trials=[]
            for trial in range(args.trials):
                result=subprocess.run([str(exe),str(config),str(temporary/'report.json')],cwd=temporary,capture_output=True,text=True)
                print(result.stdout+result.stderr,flush=True)
                assert result.returncode==0,'GPU transport contracts failed'
                trials.append(json.loads((temporary/'report.json').read_text()))
            report=dict(trials[-1],trials=trials,pair_files_removed=True,config_sha256=inputs[str(source_config.resolve())])
            runtime_name='phoenix_native_cuda.dll' if job['config']['backend']=='native-cuda' else 'phoenix_runtime_cpp.dll'
            report['runtime_sha256']=sha(args.package/runtime_name)
            report['sdk_sha256']=sha(args.package/'phoenix_native.dll')
            report['source_sha256']={name:sha(ROOT/name) for name in ['tests/native/portable_gpu_contracts.cpp','src/integration/compatibility/portable_gpu.h','native_runtime/pipeline_client.h', 'tests/native/portable_gpu_contracts.py']}
            if build:
                report['compiled_source_sha256']=build.get('source_inputs',{})
                report['runtime_files_sha256']=build.get('files',{})
            assert all(t['frame_hashes']==trials[0]['frame_hashes'] for t in trials),'Cross-process output changed'
            if args.baseline_package:
                job['runtime_root']=str(args.baseline_package.resolve());config.write_text(json.dumps(job))
                result=subprocess.run([str(exe),str(config),str(temporary/'baseline.json')],cwd=temporary,capture_output=True,text=True)
                print(result.stdout+result.stderr,flush=True);assert result.returncode==0,'Baseline transport contracts failed'
                baseline=json.loads((temporary/'baseline.json').read_text());assert baseline['frame_hashes']==report['frame_hashes'],'Temporal/control output differs from baseline'
                report['baseline_exact_frames']=len(report['frame_hashes']);report['baseline_sdk_sha256']=sha(args.baseline_package/'phoenix_native.dll')
            target=args.report/(source_config.stem+'.json') if len(args.config)>1 else args.report
            target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
        assert all(sha(Path(n))==h for n,h in inputs.items()),'Configs changed during matrix'
        if build and 'source_inputs' in build:
            assert all(sha(ROOT/n)==h for n,h in build['source_inputs'].items()),'Compiled sources changed during matrix'
            assert all(sha(args.package/n)==h for n,h in build['files'].items()),'Runtime changed during matrix'



if __name__=='__main__':main()
