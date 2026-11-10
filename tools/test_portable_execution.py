"""Explicit CPU/CUDA operator and portable broker V1/V3 regression checks."""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
from test_model_selection import ROOT,toolchain

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--configs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.runtime=a.runtime.resolve()
    compiler,env=toolchain(64);report=dict(passed=False,gpu_executed=True,operators=[],transports=[])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report['runtime_sha256']={f.relative_to(a.runtime).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in a.runtime.rglob('*') if f.suffix in ['.dll','.exe','.spv']}
    report['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'tests/portable_cpp/backend_operator_tests.cpp',ROOT/'tests/portable_cpp/advanced_transport_tests.cpp',ROOT/'native_runtime/backend_cpu.cpp',ROOT/'native_runtime/cpu_avx512.cpp',ROOT/'native_runtime/broker_client.h']}
    def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
    try:
        with tempfile.TemporaryDirectory(prefix='portable-execution-',dir=ROOT/'build') as folder:
            work=Path(folder);executables={}
            for name in ['backend_operator_tests','advanced_transport_tests']:
                exe=work/(name+'.exe');cmd=[compiler,'/nologo','/std:c++20','/O2','/EHsc','/bigobj','/MD','/fp:strict','/DNOMINMAX',ROOT/'tests/portable_cpp'/(name+'.cpp'),'/Fe:'+str(exe),'bcrypt.lib','d3d12.lib','dxgi.lib','d3dcompiler.lib','windowscodecs.lib','ole32.lib','uuid.lib']
                built=subprocess.run([str(c) for c in cmd],cwd=work,env=env,capture_output=True,text=True)
                if built.returncode:raise RuntimeError(built.stdout+built.stderr)
                executables[name]=exe
            configs=[json.loads(f.read_text()) for f in a.configs]
            for backend,device,options in [('cpu','cpu',0),('cuda',next(c['device'] for c in configs if c['backend']=='cuda'),128)]:
                result=subprocess.run([executables['backend_operator_tests'],a.runtime,backend,device,str(options)],cwd=work,env=env,capture_output=True,text=True,timeout=900)
                report['operators'].append(dict(backend=backend,options=options,passed=result.returncode==0,stdout=result.stdout,stderr=result.stderr));save()
                print(backend+' operators: '+str(result.returncode==0),flush=True)
                if result.returncode:raise RuntimeError(result.stdout+result.stderr)
            for source,config in zip(a.configs,configs):
                target=work/'transport.json';target.unlink(missing_ok=True)
                effective=dict(config,runtime_root=str(a.runtime.resolve()));effective_path=work/'config.json';effective_path.write_text(json.dumps(effective))
                result=subprocess.run([executables['advanced_transport_tests'],a.runtime,effective_path,target],cwd=work,env=env,capture_output=True,text=True,timeout=900)
                row=dict(config=source.name,effective_config=effective,passed=result.returncode==0,stdout=result.stdout,stderr=result.stderr)
                if target.exists():row['result']=json.loads(target.read_text())
                report['transports'].append(row);save();print(source.name+': '+str(row['passed']),flush=True)
                if not row['passed']:raise RuntimeError(result.stdout+result.stderr)
        report.update(passed=True,temporary_build_removed=True)
    except Exception as error:report['error']=str(error);raise
    finally:save()

if __name__=='__main__':main()
