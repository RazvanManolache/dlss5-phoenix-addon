"""Measure actual Input companion unloading without rebuilding its DLLs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from test_model_selection import ROOT, toolchain, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = dict(passed=False, pending=True, gpu_used=False, cases={})
    try:
        with tempfile.TemporaryDirectory(prefix='input-lifetime-', dir=ROOT/'build') as temporary:
            for bits in (32,64):
                work=Path(temporary)/str(bits);work.mkdir()
                compiler,env=toolchain(bits);target=work/'probe.exe'
                run([compiler,'/nologo','/std:c++20','/EHsc','/O2','/MT',ROOT/'tests/native/input_addon_lifetime_probe.cpp',
                     '/Fo:'+str(work/'probe.obj'),'/Fe:'+str(target),'psapi.lib'],cwd=work,env=env)
                source=args.input_directory/f'PhoenixInput.addon{bits}'
                addon=work/source.name;shutil.copy2(source,addon)
                log=args.output/f'{bits}.jsonl'
                with log.open('w',encoding='utf-8') as stream:
                    process=subprocess.Popen([str(target),str(addon),str(args.cycles)],cwd=work,env=env,stdout=stream,stderr=stream)
                    try:code=process.wait(max(240,args.cycles*15))
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait();code='timeout'
                samples=[json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
                result=dict(exit=code,passed=False,samples=samples,
                    addon_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                if len(samples)>5:
                    result.update(heap_growth=samples[-1]['heaps']-samples[4]['heaps'],
                                  handle_growth=samples[-1]['handles']-samples[4]['handles'],
                                  private_growth_mib=(samples[-1]['private']-samples[4]['private'])/1048576)
                    result['passed']=code==0 and len(samples)==args.cycles and result['heap_growth']==0 and result['handle_growth']<=4 and result['private_growth_mib']<8
                results['cases'][str(bits)]=result
                (args.output/'report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
                print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)
        results['passed']=len(results['cases'])==2 and all(v['passed'] for v in results['cases'].values())
    finally:
        results['pending']=False
        (args.output/'report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return 0 if results['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
