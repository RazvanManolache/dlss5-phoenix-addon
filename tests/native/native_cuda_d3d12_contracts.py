"""Compile/run the explicit Windows D3D12-CUDA GPU transport contracts."""
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
    for name in ('package','config','image','report','cuda'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();compiler,env=toolchain(64)
    build_file=args.package/'BUILD.json'
    build=json.loads(build_file.read_text()) if build_file.exists() else None
    if build and 'source_inputs' in build:
        assert build['passed'] and all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in build['source_inputs'].items()),'Rebuild source-stale runtime before testing'
    image=np.asarray(Image.open(args.image).convert('RGB'),dtype=np.float32)/np.float32(255)
    height,width,_=image.shape
    with tempfile.TemporaryDirectory(prefix='cuda-d3d12-contract-',dir=ROOT/'build') as directory:
        temporary=Path(directory);exe=temporary/'contracts.exe';raw=temporary/'input.rgb32f';image.tofile(raw)
        job=dict(runtime_root=str(args.package.resolve()),config=json.loads(args.config.read_text()),width=width,height=height,input=str(raw))
        config=temporary/'job.json';config.write_text(json.dumps(job))
        result=subprocess.run([compiler,'/nologo','/std:c++20','/EHsc','/O2','/fp:strict','/DNOMINMAX',
            '/I'+str(args.cuda/'include'),str(ROOT/'tests/native/native_cuda_d3d12_tests.cpp'),'/Fe:'+str(exe),
            '/link','/LIBPATH:'+str(args.cuda/'lib/x64'),'cuda.lib','d3d12.lib','dxgi.lib','dxguid.lib'],cwd=temporary,env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        result=subprocess.run([str(exe),str(config),str(args.report.resolve())],cwd=temporary,capture_output=True,text=True)
        print(result.stdout+result.stderr)
        assert result.returncode==0,'GPU transport contracts failed'
        report=json.loads(args.report.read_text())
        report['runtime_sha256']=hashlib.sha256((args.package/'phoenix_native_cuda.dll').read_bytes()).hexdigest()
        if build:report['compiled_source_sha256']=build.get('source_inputs',{})
        report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['native_runtime/native_cuda.cpp','native_runtime/native_cuda_executor.h','native_runtime/native_cuda_pixels.cu','tests/native/native_cuda_d3d12_tests.cpp']}
    report['temporary_build_removed']=True
    args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
