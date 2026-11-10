"""Exhaustive portable pixel codec contracts; --gpu explicitly runs D3D12 compute."""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
from test_model_selection import ROOT,toolchain

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    compiler,env=toolchain(64);report={'passed':False,'gpu_executed':a.gpu,'checks':[]}
    sources=['native_runtime/half_rgba.h','native_runtime/portable_pixels.h','native_runtime/portable_d3d12_pixels.h','native_runtime/frame_resolve.h','native_runtime/frame_contract.h','tests/native/portable_pixels_tests.cpp']
    if a.gpu:sources.append('tests/native/portable_d3d12_pixels_tests.cpp')
    report['source_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='portable-pixels-',dir=ROOT/'build') as folder:
            work=Path(folder)
            for name in ['portable_pixels']+(['portable_d3d12_pixels'] if a.gpu else []):
                exe=work/(name+'.exe');cmd=[compiler,'/nologo','/std:c++20','/O2','/EHsc','/bigobj','/MD','/fp:strict','/DNOMINMAX',ROOT/'tests/native'/(name+'_tests.cpp'),'/Fe:'+str(exe)]
                if name=='portable_d3d12_pixels':
                    cmd+=['/I'+str(ROOT.parent/'external/DLSS/include'),'/I'+str(ROOT/'third_party/nvapi'),'/I'+str(ROOT/'third_party/minhook/include')]
                    cmd += [ROOT/'third_party/minhook/src'/n for n in ['hook.c','buffer.c','trampoline.c','hde/hde64.c']]
                    cmd += ['d3d12.lib','dxgi.lib','d3dcompiler.lib','bcrypt.lib','psapi.lib']
                built=subprocess.run([str(c) for c in cmd],cwd=work,env=env,capture_output=True,text=True)
                if built.returncode:raise RuntimeError(built.stdout+built.stderr)
                result=subprocess.run([exe],cwd=work,env=env,capture_output=True,text=True,timeout=120)
                row={'name':name,'passed':result.returncode==0,'stdout':result.stdout,'stderr':result.stderr};report['checks'].append(row)
                print(result.stdout+result.stderr,flush=True)
                if result.returncode:raise RuntimeError(name+' failed')
        report['passed']=True;report['temporary_build_removed']=True
    except Exception as error:report['error']=str(error);raise
    finally:a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
