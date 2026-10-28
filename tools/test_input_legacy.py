"""Build a real DirectInput 7 A/W polling probe without a GPU or desktop input."""
import json,subprocess,tempfile,argparse
from pathlib import Path
from test_model_selection import ROOT,toolchain,run
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    results=[]
    with tempfile.TemporaryDirectory(prefix='legacy-input-',dir=ROOT/'build') as tmp:
        for bits in (32,64):
            folder=Path(tmp)/str(bits);folder.mkdir();compiler,env=toolchain(bits);target=folder/'probe.exe'
            mh=ROOT/'third_party/minhook'
            run([compiler,'/nologo','/std:c++20','/Zc:__cplusplus','/O2','/EHsc','/MT','/DNOMINMAX','/I'+str(ROOT/'third_party'),'/I'+str(ROOT/'third_party/reshade'),'/I'+str(mh/'include'),ROOT/'tests/native/input_legacy_tests.cpp',*[mh/'src'/x for x in ('hook.c','buffer.c','trampoline.c',f'hde/hde{bits}.c')],'/Fo:'+str(folder)+'\\','/Fe:'+str(target),'user32.lib','dxguid.lib','dinput8.lib'],cwd=folder,env=env)
            r=subprocess.run([str(target)],cwd=folder,env=env,capture_output=True,text=True,timeout=30)
            results.append(dict(bits=bits,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,passed=r.returncode==0))
    result=dict(passed=all(x['passed'] for x in results),cases=results,gpu_used=False,desktop_input_sent=False)
    (a.output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
