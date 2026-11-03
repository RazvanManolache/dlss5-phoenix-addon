"""Opt-in 1080p x86/internal-x64 worker profile; no games, no model redistribution."""
import argparse,json,pathlib,subprocess,tempfile,statistics
from test_model_selection import ROOT,toolchain,run

def records(text):
    decoder=json.JSONDecoder();cursor=0
    while cursor<len(text):
        start=text.find('{',cursor)
        if start<0:return
        try:value,cursor=decoder.raw_decode(text,start);yield value
        except ValueError:cursor=start+1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=pathlib.Path,required=True)
    p.add_argument('--source',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    args=p.parse_args();compiler,env=toolchain(32)
    sdk=ROOT.parent/'external/Streamline/external/ngx-sdk/include'
    with tempfile.TemporaryDirectory(prefix='worker_profile_',dir=ROOT/'build') as folder:
        work=pathlib.Path(folder);exe=work/'profile.exe'
        run([compiler,'/nologo','/std:c++20','/Zc:__cplusplus','/O2','/EHsc','/MT','/bigobj','/DNOMINMAX',
             '/I'+str(sdk),'/I'+str(ROOT/'third_party/nvapi'),ROOT/'tests/native/wow64_worker_profile.cpp',
             '/Fo:'+str(work)+'\\','/Fe:'+str(exe),'d3d11.lib','d3d12.lib','dxgi.lib','d3dcompiler.lib',
             'bcrypt.lib','user32.lib','gdi32.lib','advapi32.lib','opengl32.lib','ole32.lib','cabinet.lib',
             'shell32.lib','uuid.lib','windowscodecs.lib','/link','/LARGEADDRESSAWARE',
             '/SUBSYSTEM:WINDOWS','/ENTRY:wmainCRTStartup'],cwd=work,env=env,timeout=300)
        result=subprocess.run([str(exe),str(args.runtime.resolve()),str(args.source.resolve())],
            cwd=ROOT,capture_output=True,text=True,timeout=180)
        report=next((r for r in records(result.stdout) if r.get('passed')),None)
        if result.returncode or not report:raise RuntimeError(result.stderr or result.stdout)
        expected=json.loads(args.source.read_text()).get('model','original')
        assert report['model']==expected
        for row in report['runs']:
            samples=row['samples'];assert len(samples)==32 and all(s['motion_valid'] for s in samples)
            metrics={'wall_ms':[s['wall_ms'] for s in samples]}
            fresh_gpu=[s['gpu_ms'] for s in samples if s.get('gpu_sampled')]
            if fresh_gpu:metrics['gpu_ms']=fresh_gpu
            for group in ('stages','transfer','sdk','worker','motion_stages'):
                if not isinstance(samples[0].get(group),dict):continue
                for key in samples[0][group]:metrics[group+'.'+key]=[s[group][key] for s in samples]
            row['summary']={k:{'median':statistics.median(v),'mean':statistics.mean(v),
                'p95':sorted(v)[int(.95*(len(v)-1))]} for k,v in metrics.items()}
        report['games_run']=False
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'model':report['model'],'runs':[{k:v for k,v in row.items() if k!='samples'} for row in report['runs']]}))

if __name__=='__main__':main()
