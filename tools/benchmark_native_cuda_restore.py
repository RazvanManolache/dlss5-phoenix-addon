"""Explicit current/pre-change GPU-texture timing of full-workload restoration."""
import argparse,hashlib,json,statistics,subprocess,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from test_model_selection import ROOT,toolchain

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['runtime','before','image','cuda','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();compiler,env=toolchain(64);a.output.parent.mkdir(parents=True,exist_ok=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report={'passed':False,'scope':'Synchronous resident D3D12 textures -> Native CUDA frame -> completed caller GPU output. Upload, readback, setup and hash excluded; different from the common host-boundary chart.','cases':[],'rounds':2,'samples':30,'warmup':20}
    builds={name:json.loads((path/'BUILD.json').read_text()) for name,path in [('current',a.runtime),('before',a.before)]}
    for name,path in [('current',a.runtime),('before',a.before)]:
        assert builds[name]['passed'] and all(sha(path/n)==h for n,h in builds[name]['files'].items())
    report['builds']=builds
    assert all(sha(ROOT/n)==h for n,h in builds['current']['source_inputs'].items())
    report['source_sha256']={n:sha(ROOT/n) for n in ['tools/benchmark_native_cuda_restore.py','tests/native/native_cuda_restore_probe.cpp','tests/native/native_cuda_d3d12_tests.cpp']}
    with tempfile.TemporaryDirectory(prefix='cuda-restore-',dir=ROOT/'build') as folder:
        work=Path(folder);exe=work/'probe.exe';pixels=np.asarray(Image.open(a.image).convert('RGB').resize((1280,720),Image.Resampling.LANCZOS),np.float32)/np.float32(255);pixels.tofile(work/'input.rgb32f')
        result=subprocess.run([compiler,'/nologo','/std:c++20','/O2','/EHsc','/fp:strict','/DNOMINMAX','/I'+str(a.cuda/'include'),ROOT/'tests/native/native_cuda_restore_probe.cpp','/Fe:'+str(exe),'/link','/LIBPATH:'+str(a.cuda/'lib/x64'),'cuda.lib','d3d12.lib','dxgi.lib','dxguid.lib','bcrypt.lib'],cwd=work,env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        report['probe_sha256']=sha(exe)
        for round in range(2):
            cells=[(name,path,model) for model in ['original','small-fp8'] for name,path in [('current',a.runtime),('before',a.before)]]
            if round:cells.reverse()
            for name,path,model in cells:
                print('RESTORE '+name+' '+model+' '+str(round),flush=True)
                config=json.loads((ROOT.parent/f'dlss5-everything-testing/runs/windows-linux-after/outer-transport/native-cuda-{model}-config.json').read_text())
                job={'runtime_root':str(path.resolve()),'config':config,'width':1280,'height':720,'input':str(work/'input.rgb32f')}
                (work/'job.json').write_text(json.dumps(job));(work/'result.json').unlink(missing_ok=True)
                result=subprocess.run([exe,work/'job.json',work/'result.json'],cwd=work,env=env,capture_output=True,text=True,timeout=180)
                assert result.returncode==0,result.stdout+result.stderr
                value=json.loads((work/'result.json').read_text());assert value['passed']
                report['cases'].append(dict(value,variant=name,model=model,round=round));print(name+' '+model+' fresh/restored '+str(value['fresh']['wall']['median_ms'])+'/'+str(value['restored']['wall']['median_ms']),flush=True)
        for model in ['original','small-fp8']:
            cases=[c for c in report['cases'] if c['model']==model]
            assert len({c[phase]['output_sha256'] for c in cases for phase in ['fresh','restored']})==1
        assert all(sha(ROOT/n)==h for n,h in builds['current']['source_inputs'].items())
    report.update(passed=True,temporary_build_removed=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
