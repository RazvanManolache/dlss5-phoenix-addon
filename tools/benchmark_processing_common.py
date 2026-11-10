"""Fresh common-boundary backend benchmark. Explicit GPU/CPU work; no games/installations."""
import argparse,datetime,hashlib,json,os,shutil,statistics,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from test_model_selection import ROOT,toolchain


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--samples',type=int,default=15);p.add_argument('--slow-samples',type=int,default=5)
    p.add_argument('--routes',nargs='+',help='Restrict measured engines for targeted optimization')
    p.add_argument('--models',nargs='+',choices=['original','small-fp8'],default=['original','small-fp8'])
    p.add_argument('--ablation',type=Path,help='JSON mapping route IDs to additional named option variants')
    p.add_argument('--baseline-report',type=Path,help='Require exact per-route/model output hashes from this common-boundary receipt')
    p.add_argument('--profile-commands',action='store_true',help='Separate diagnostic command timings after the first FP8 measurement of each portable engine')
    p.add_argument('--packaged-providers',action='store_true',help='Measure the source-verified current runtime providers without rebuilding alternate providers')
    p.add_argument('--compare-runtime',type=Path,help='Model-free pre-change runtime snapshot; interleave matching CUDA/RTX Vulkan control processes')
    a=p.parse_args()
    assert 1<=a.rounds<=5 and a.samples>=3 and a.slow_samples>=3
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists():raise RuntimeError('Choose a new explicitly named run; do not mix snapshots')
    compiler,env=toolchain(64);cuda=Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1');sdk=ROOT.parent/'external/DLSS/include'
    raw=ROOT/'artifacts/current/windows-cuda';raw_build=json.loads((raw/'BUILD.json').read_text())
    assert raw_build['passed'] and all(sha(ROOT/k)==v for k,v in raw_build['source_inputs'].items()),'Rebuild source-stale Windows runtime'
    model=ROOT.parent/'rtx-test-cache/main/Nvidia/nodix-compat/1.3.7-ac6ddca163ea/nvngx_dlssnr.dll'
    core=list(Path(r'C:\Windows\System32\DriverStore\FileRepository').rglob('_nvngx.dll'));assert len(core)==1
    report=dict(schema='phoenix-common-processing-benchmark/1',passed=False,state='building',date=datetime.datetime.now().astimezone().isoformat(),width=1280,height=720,rounds=a.rounds,
        timing_scope='Synchronous host RGBA16F input -> GPU upload -> one full reset neural pass -> GPU readback -> completed host RGBA16F output. Model setup/allocation, output validation and hashing excluded. No optical flow, scene detection, scaling or UI.',
        model_sha256=sha(model),source_image_sha256=sha(a.image),cases=[],unavailable={'hip':'No usable HIP provider/SDK on this host','sycl':'No usable SYCL provider/SDK on this host','metal':'Windows host; Apple Metal unavailable'},
        notes=['Backend processing benchmark; not live game FPS','Transport choices (helper/x86 library/driver) are not separately ranked as neural backends','All timed calls use the same outer upload/readback boundary, on the NVIDIA producer/consumer device','Vulkan AMD inference crosses to the integrated GPU through the portable host-plane interface','CPU uses the current optimized provider with 32 threads'],games_launched=False,installed_games_modified=False,published=False)
    report['compiled_source_sha256']=raw_build['source_inputs']
    comparison=None
    if a.compare_runtime:
        assert a.packaged_providers,'A/B control requires packaged providers'
        a.compare_runtime=a.compare_runtime.resolve()
        comparison=json.loads((a.compare_runtime/'BUILD.json').read_text())
        assert comparison['passed'] and all(sha(a.compare_runtime/k)==v for k,v in comparison['files'].items()),'Invalid pre-change build'
        report['comparison_runtime']={'path':str(a.compare_runtime),'files':comparison['files'],'compiled_source_sha256':comparison['source_inputs']}
    save(out/'run.json',report)
    def run(stage,cmd,cwd,timeout=900):
        print(stage,flush=True);report['current']=stage;save(out/'run.json',report)
        with (out/(stage+'.log')).open('w') as log:r=subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if r.returncode:raise RuntimeError(stage+' failed; see '+str(out/(stage+'.log')))
    with tempfile.TemporaryDirectory(prefix='common_processing_',dir=ROOT/'build') as temp,tempfile.TemporaryDirectory(prefix='common_native_',dir=ROOT/'build') as native_name:
        work=Path(temp);bundle=Path(native_name);bundle.rmdir() # Empty, newly owned path; production builder requires a fresh name.
        if not a.routes or 'native_rtx' in a.routes:
            run('native-build',[sys.executable,ROOT/'tools/production_build_native_kernels.py',bundle.name,'--compiler',cuda/'bin/nvcc.exe','--architectures','sm_120','--jobs','4'],ROOT,1800)
        else:
            bundle.mkdir();shutil.copy2(raw/'native-cuda/native_manifest.json',bundle/'native_manifest.json')
        stage=work/'runtime';stage.mkdir()
        for name,digest in raw_build['files'].items():
            assert sha(raw/name)==digest
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(raw/name,target)
        common=[compiler,'/nologo','/std:c++20','/Zc:__cplusplus','/O2','/EHsc','/MD','/DNOMINMAX','/fp:strict']
        if not a.packaged_providers:
            for name,flag in [('cpu_avx2','/arch:AVX2'),('cpu_avx512','/arch:AVX512')]:run(name,common+[flag,'/c',ROOT/'native_runtime'/(name+'.cpp'),'/Fo:'+str(work/(name+'.obj'))],work)
            run('cpu-provider',common+['/LD',ROOT/'native_runtime/backend_cpu.cpp',work/'cpu_avx2.obj',work/'cpu_avx512.obj','/Fo:'+str(work/'cpu.obj'),'/Fe:'+str(stage/'phoenix_cpu_cpp.dll')],work)
            glslang=ROOT.parent/'external/renodx/bin/glslang.exe';headers=ROOT.parent/'dlss5-everything-testing/vendor';vk=stage/'vulkan'
            run('vulkan-shaders',[sys.executable,ROOT/'tools/generate_standalone_vulkan.py','--output',vk,'--compiler',glslang],ROOT)
            lib=shutil.which('lib.exe',path=env['PATH']);run('vulkan-import',[lib,'/nologo','/def:'+str(ROOT/'third_party/vulkan-loader.def'),'/machine:x64','/out:'+str(work/'vulkan-1.lib')],work)
            run('vulkan-provider',[compiler,'/nologo','/O2','/std:c11','/TC','/MD','/D_CRT_SECURE_NO_WARNINGS','/I'+str(headers),'/I'+str(ROOT/'src/backends/vulkan'),'/LD',vk/'phoenix_vulkan.c','/Fo:'+str(work/'vulkan.obj'),'/Fe:'+str(vk/'phoenix_vulkan.dll'),'/link','/DEF:'+str(vk/'provider.def'),work/'vulkan-1.lib'],work)
            run('vulkan-adapter',common+['/LD',ROOT/'native_runtime/backend_vulkan.cpp','/Fo:'+str(work/'vulkan-adapter.obj'),'/Fe:'+str(stage/'phoenix_vulkan_cpp.dll')],work)
        else:
            assert all((stage/n).is_file() for n in ('phoenix_cpu_cpp.dll','phoenix_vulkan_cpp.dll','vulkan/phoenix_vulkan.dll')),'Build the current runtime with --vulkan first'
            report['providers']='source-verified current Windows runtime binaries'
        query=work/'devices.exe';run('device-query',common+[ROOT/'native_runtime/device_query.cpp','/Fo:'+str(work/'devices.obj'),'/Fe:'+str(query)],work)
        inventory=json.loads(subprocess.check_output([query,stage],cwd=work,env=env,text=True));report['inventory']=inventory
        options=common+['/bigobj','/I'+str(sdk),'/I'+str(ROOT/'third_party/nvapi'),'/I'+str(ROOT/'third_party/minhook/include')]
        forwarder=work/'phoenix-nvngx.dll';run('forwarder',options+['/LD',ROOT/'src/integration/reshade/nr_forwarder.cpp','/Fo:'+str(work/'forwarder.obj'),'/Fe:'+str(forwarder)],work)
        exe=work/'unified.exe';hooks=[ROOT/'third_party/minhook/src'/name for name in ['hook.c','buffer.c','trampoline.c','hde/hde64.c']]
        run('unified-probe',options+[ROOT/'tests/native/unified_processing_probe.cpp',*hooks,'/Fe:'+str(exe),'d3d12.lib','dxgi.lib','d3dcompiler.lib','bcrypt.lib','psapi.lib'],work)
        if a.profile_commands:
            run('command-probe',common+[ROOT/'tests/portable_cpp/graph_profile.cpp','/Fe:'+str(work/'commands.exe'),'bcrypt.lib'],work)
        report['binaries']={k.relative_to(stage).as_posix():sha(k) for k in stage.rglob('*') if k.is_file() and k.suffix in ('.dll','.exe','.cubin','.spv')};report['probe_sha256']=sha(exe);report['native_manifest']=json.loads((bundle/'native_manifest.json').read_text())
        source_paths=[p for folder in ['native_runtime','src/native','src/pipeline','src/backends','third_party/intel_nr/src/gpu'] for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.cpp','.c','.h','.hpp','.cu','.cuh','.inc','.comp','.glsl','.json')]
        source_paths += [ROOT/'tests/native/unified_processing_probe.cpp',ROOT/'tests/native/four_model_probe.cpp',Path(__file__)]
        report['source_sha256']={k.relative_to(ROOT).as_posix():sha(k) for k in source_paths}
        rgba=np.asarray(Image.open(a.image).convert('RGBA').resize((1280,720),Image.Resampling.LANCZOS),np.float32)/255
        rgba=rgba.astype(np.float16);input_file=work/'input.rgba16f';rgba.tofile(input_file);report['input_rgba16f_sha256']=sha(input_file)
        if a.profile_commands:rgba[...,:3].astype(np.float32).copy().tofile(work/'input.rgb32f')
        routes=[('native_rtx','Native RTX / D3D12',None,False),('nvidia','NVIDIA DLSS5',None,False)]
        for backend,label in [('native-cuda','Native CUDA'),('cuda','Portable CUDA')]:
            entry=next(d for d in inventory['devices'] if d['id'].startswith('cuda:'));device=entry['id'];device='cuda:'+device[9:].replace('-','') if backend=='native-cuda' else device
            routes.append((backend,label,dict(backend=backend,device=device),False))
        for d in inventory['devices']:
            if d['id'].startswith('vulkan:') and d.get('capable'):
                name='vulkan_nvidia' if d['vendor']==0x10de else 'vulkan_amd' if d['vendor']==0x1002 else 'vulkan_other'
                routes.append((name,'Vulkan / '+('RTX 5080' if d['vendor']==0x10de else d['label']),dict(backend='vulkan',device=d['id']),d['vendor']!=0x10de))
        routes.append(('cpu32','CPU / 32 threads',dict(backend='cpu',device='cpu',cpu_threads=32),True))
        if a.routes:
            assert set(a.routes)<=set(r[0] for r in routes),'Unknown route filter'
            routes=[r for r in routes if r[0] in a.routes]
        base_route={r[0]:r[0] for r in routes}
        if a.ablation:
            variants=json.loads(a.ablation.read_text());report['ablation']=variants
            for name,label,portable,slow in list(routes):
                for variant in variants.get(name,[]):
                    key=name+'/'+variant['id'];assert key not in base_route
                    routes.append((key,label+' / '+variant['id'],dict(portable,**variant['options']),slow));base_route[key]=name
        if comparison:
            paired=[]
            for route in routes:
                paired.append(route)
                name,label,portable,slow=route
                if name in ['native-cuda','cuda','vulkan_nvidia']:
                    key=name+'/before';paired.append((key,label+' / before',dict(portable,_benchmark_runtime=str(a.compare_runtime)),slow));base_route[key]=name
            routes=paired
        golden={}
        if a.baseline_report:
            baseline=json.loads(a.baseline_report.read_text());assert baseline['passed']
            assert baseline['input_rgba16f_sha256']==report['input_rgba16f_sha256']
            golden={(c['route'],c['model']):c['output_sha256'] for c in baseline['cases']}
            report['baseline_report']=str(a.baseline_report.resolve());report['baseline_report_sha256']=sha(a.baseline_report)
        report['routes']=[{'id':r[0],'label':r[1]} for r in routes];report['state']='running';save(out/'run.json',report)
        for round in range(a.rounds):
            cells=[(r,m) for r in routes for m in a.models]
            if round%2:cells.reverse()
            for (name,label,portable,slow),model_id in cells:
                key=name+'-'+model_id+'-'+str(round);row=dict(route=name,label=label,model=model_id,round=round,passed=False)
                job=dict(engine='nvidia' if name=='nvidia' else 'sdk',width=1280,height=720,runtime_root=str(stage),bundle_path=str(bundle),weight_source=dict(kind='nvidia_dll',path=str(model),sha256=report['model_sha256'],model=model_id),
                    ngx_core=str(core[0]),forwarder=str(forwarder),cache=str(work/'ngx-cache'),color_file=str(input_file),warmup=3,samples=a.slow_samples if slow else a.samples,output_file=str(work/'last.rgba16f'))
                if portable:
                    runner=dict(portable)
                    effective_runtime=Path(runner.pop('_benchmark_runtime',str(stage)))
                    runner.update(runtime_root=str(effective_runtime),weight_source=job['weight_source'],allow_unvalidated=True)
                    runner['runtime_sha256']=sha(effective_runtime/('phoenix_native_cuda.dll' if portable['backend']=='native-cuda' else 'phoenix_runtime_cpp.dll'));job['portable_runtime']=runner
                    # The SDK itself must also come from the selected snapshot.
                    job['runtime_root']=str(effective_runtime)
                jobfile=work/'job.json';result=work/'result.json';save(jobfile,job);result.unlink(missing_ok=True)
                print('MEASURING '+key,flush=True);report['current']=key;save(out/'run.json',report)
                try:
                    process=subprocess.run([exe,jobfile,result],cwd=work,env=env,capture_output=True,text=True,timeout=1200)
                    value=json.loads(result.read_text()) if result.exists() else {}
                    if process.returncode or not value.get('passed'):raise RuntimeError(value.get('error',process.stderr or process.stdout or 'No probe result'))
                    row.update(value);row['route']=name;row['label']=label;row['round']=round
                    pixels=np.fromfile(work/'last.rgba16f',np.float16).reshape((720,1280,4));row['output_rgb_mean']=pixels[...,:3].astype(np.float32).mean(axis=(0,1)).tolist();row['output_rgb_std']=float(pixels[...,:3].astype(np.float32).std())
                    assert row['input_sha256']==report['input_rgba16f_sha256']
                    if golden:
                        assert row['output_sha256']==golden[base_route[name],model_id],'Output differs from baseline'
                        row['baseline_exact']=True
                    print(key+': '+str(row['wall']['median_ms'])+' ms',flush=True)
                except Exception as error:
                    row['passed']=False;row['error']=str(error);print(key+': FAILED '+str(error),flush=True)
                report['cases'].append(row);save(out/'run.json',report)
                if a.profile_commands and row['passed'] and round==0 and model_id=='small-fp8' and portable and '/' not in name:
                    diagnostic=dict(runner,window_batch_limit=row['status']['portable_neural_stages'][0]['window_batch_limit'])
                    if name=='cuda':diagnostic['cuda_matrix_columns']=row['status']['portable_neural_stages'][0].get('cuda_matrix_columns',16)
                    schedule=row['status']['portable_neural_stages'][0]
                    jobfile=work/'commands.json';save(jobfile,dict(config=diagnostic,width=1280,height=720,padded_width=schedule['working_width'],padded_height=schedule['working_height'],cases=[dict(color=str(work/'input.rgb32f'))]))
                    run('profile-'+name,[work/'commands.exe',jobfile,out/('profile-'+name+'.json')],work)
        report['source_changed_during_run']=[k for k,v in report['source_sha256'].items() if sha(ROOT/k)!=v]
        assert not report['source_changed_during_run'],'Compiled sources changed during run'
        report['state']='complete';report['passed']=all(c['passed'] for c in report['cases']);report['temporary_build_removed']=True
    save(out/'run.json',report)
    print(json.dumps({'passed':report['passed'],'cases':len(report['cases']),'report':str(out/'run.json')}),flush=True)

if __name__=='__main__':main()
