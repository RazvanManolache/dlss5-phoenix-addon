"""Sequential current-checkout model/backend benchmarks, with incremental receipts."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import winreg
import numpy as np
import psutil
from test_model_selection import ROOT
from benchmark_readme_case import fixture

DEST = ROOT / 'artifacts/current/readme-benchmarks'
WORK = ROOT / 'build/four_model_comparison'
MODELS = ['original', 'small-fp8']
NATIVE_SIZES = [(320,180),(640,360),(640,480),(960,540),(1280,720),(1920,1080),(2560,1440),(3840,2160),(5760,3240),(7680,4320)]
FAST_SIZES = [(320,180),(1280,720),(1920,1080),(2560,1440),(3840,2160)]
SLOW_SIZES = [(320,180),(1920,1080)]

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path): return json.loads(path.read_text())
def write(path, value): path.write_text(json.dumps(value, indent=2) + '\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def telemetry():
    p = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True,text=True,timeout=10)
    return {'utc':now(),'csv':p.stdout.strip(),'error':p.stderr.strip()}


def native_order(process):
    cells=[(model,engine) for model in MODELS for engine in ['phoenix','nvidia']]
    return cells if process%2==0 else list(reversed(cells))


def main():
    global DEST, WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['inventory','smoke','native','portable','features','model_sources'], required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--routes', nargs='+')
    parser.add_argument('--only-size', help='Optional WIDTHxHEIGHT checkpoint')
    parser.add_argument('--output', type=Path, default=DEST)
    parser.add_argument('--model-library', type=Path, help='Read-only shared model library for matched alternate-source measurements')
    parser.add_argument('--keep-going', action='store_true', help='Record failed cases and continue independent cells without publishing them as measurements')
    args = parser.parse_args()
    DEST = args.output.resolve()
    DEST.mkdir(parents=True,exist_ok=True); (DEST/'cases').mkdir(exist_ok=True)
    build = read(DEST/'build.json'); stage = Path(build['runtime_root'])
    WORK = Path(build.get('work_root', WORK))
    source = ROOT.parent/'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
    core = list(Path('C:/Windows/System32/DriverStore/FileRepository').glob('nv_*.inf_amd64_*/_nvngx.dll'))
    if len(core)!=1: raise RuntimeError('Need an unambiguous installed NGX core')
    manifest_path = DEST/'run.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest['model_dll_sha256']!=sha(source): raise RuntimeError('Model DLL changed; do not mix benchmark inputs')
        if manifest.get('build_receipt_sha256')!=sha(DEST/'build.json'):
            raise RuntimeError('Build identity changed; use a fresh result folder')
    else:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
        manifest = {'schema':'phoenix-readme-model-backend-benchmarks/1','started_utc':now(),'state':'running',
            'build_receipt_sha256':sha(DEST/'build.json'),
            'model_dll_sha256':sha(source),'models':MODELS,'hardware':{'cpu':cpu,'physical_cores':psutil.cpu_count(logical=False),
                'logical_processors':psutil.cpu_count(),'ram_bytes':psutil.virtual_memory().total,'nvidia':telemetry()},
            'fixture':'Deterministic structured SDR RGB, quantized through float16, zero motion, no depth, fixed seed, automatic mask, default tone/structure',
            'full_model_default':{'passes':1,'transformer_blocks':8,'repeats':[1]*8},
            'games_launched':False,'installed_files_modified':False,'cases':{},'stages':{},
            'unavailable':{'metal':'No Apple silicon host attached; no current Metal measurements',
                           'hip':'Provider/SDK unavailable in the local benchmark runtime',
                           'sycl':'Provider/private CPU ICD or GPU runtime unavailable locally'}}
    manifest['hardware']['display_adapters'] = json.loads(subprocess.check_output(
        ['powershell.exe','-NoProfile','-Command',
         'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json'],
        text=True,timeout=30))
    inventory = json.loads(subprocess.check_output([build['device_query'],str(stage)],text=True))
    manifest['inventory']=inventory
    routes = {'cpu32':{'backend':'cpu','device':'cpu','threads':32,'vendor':0,'slow':True},
              'cpu1':{'backend':'cpu','device':'cpu','threads':1,'vendor':0,'slow':True}}
    for device in inventory['devices']:
        key=device['id']
        if key.startswith('cuda:'): routes['cuda']={'backend':'cuda','device':key,'vendor':0x10de,'slow':False}
        if key.startswith('vulkan:') and device.get('capable'):
            name='vulkan_nvidia' if device['vendor']==0x10de else 'vulkan_amd' if device['vendor']==0x1002 else 'vulkan_other'
            routes[name]={'backend':'vulkan','device':key,'vendor':device['vendor'],'slow':device['vendor']!=0x10de}
        if key.startswith('hip:'): routes['hip']={'backend':'hip','device':key,'vendor':0x1002,'slow':True};manifest['unavailable'].pop('hip',None)
        if key.startswith('sycl:'): routes['sycl']={'backend':'sycl','device':key,'vendor':0,'slow':True};manifest['unavailable'].pop('sycl',None)
    manifest['routes']=routes
    manifest['model_dll_bytes']=source.stat().st_size
    manifest['portable_backend_configuration']={
        'vulkan_fused_feed_forward': {'effective':False,'selection':'current runtime default','caller_override':False}}
    manifest['stages'][args.stage]={'state':'running','started_utc':now()}
    def save(): write(manifest_path,manifest)
    save()
    if args.stage=='inventory':
        manifest['stages']['inventory']['state']='completed';save();print(json.dumps(inventory,indent=2));return
    stage_keys=[]
    def execute(key,command,job,output,timeout):
        stage_keys.append(key)
        if args.resume and manifest['cases'].get(key,{}).get('passed'):
            if read(output).get('benchmark_job_sha256')!=sha(job): raise RuntimeError('Resume job differs: '+key)
            print('SKIP '+key,flush=True);return
        manifest['current_case']=key
        manifest['cases'][key]={'state':'running','file':str(output.relative_to(DEST)),'started_utc':now()};save()
        before=telemetry();started=time.perf_counter()
        try:
            process=subprocess.run([str(c) for c in command],cwd=WORK,capture_output=True,text=True,timeout=timeout)
            result=read(output) if output.exists() else {'passed':False,'error':'No child result'}
            result.update(process_exit_code=process.returncode,stderr=process.stderr[-5000:] if process.returncode else '',
                          stdout=process.stdout[-1000:],process_elapsed_seconds=time.perf_counter()-started)
            if process.returncode:result['passed']=False
        except subprocess.TimeoutExpired:
            result=read(output) if output.exists() else {}
            result.update(passed=False,error=f'Isolated process timed out after {timeout}s')
        result.update(telemetry_before=before,telemetry_after=telemetry(),benchmark_job_sha256=sha(job),measured_utc=now())
        write(output,result)
        manifest['cases'][key].update(state='completed' if result.get('passed') else 'failed',passed=result.get('passed',False),
                                     finished_utc=now(),sha256=sha(output));save()
        median=result.get('wall',{}).get('median_ms')
        print(key,'PASS' if result.get('passed') else 'FAIL',f'{median:.3f} ms call' if median else result.get('error',''),flush=True)
        if not result.get('passed') and not args.keep_going: raise RuntimeError(key+': '+str(result.get('error',result.get('stderr'))))
    def native(model,size,process,profile='default',patch=None,engine='phoenix',debug=False,weight_source=None):
        w,h=size
        image=fixture(w,h);rgba=np.ones((h,w,4),np.float16);rgba[...,:3]=image
        color=WORK/'readme-input.rgba16f';motion=WORK/'readme-motion.rg16f'
        color.write_bytes(rgba.tobytes());motion.write_bytes(np.zeros((h,w,2),np.float16).tobytes())
        key=f'{"smoke-" if debug else ""}{engine}-{model}-{w}x{h}-{profile}-p{process}'
        config={'width':w,'height':h,'engine':engine,'weight_source':{'kind':'nvidia_dll','path':str(source),
            'sha256':manifest['model_dll_sha256'],'model':model},'bundle_path':build['native_bundle'],
            'ngx_core':str(core[0]),'forwarder':build['forwarder'],'cache':str(WORK/'readme-ngx-cache'),
            'color_file':str(color),'motion_file':str(motion),'capture':False,'debug':debug,
            'warmup_seconds':.1 if debug else .75,'measure_seconds':.1 if debug else 2.}
        config.update(patch or {})
        if weight_source is not None:config['weight_source']=weight_source
        job=WORK/'readme-native-job.json';write(job,config)
        execute(key,[build['native_probe'],job,DEST/'cases'/(key+'.json')],job,DEST/'cases'/(key+'.json'),180)
    def portable(route,model,size,process,profile='default',patch=None,advanced=False,weight_source=None):
        info=routes[route];w,h=size;key=f'{route}-{model}-{w}x{h}-{profile}-p{process}'
        config={'runtime_root':str(stage),'backend':info['backend'],'device':info['device'],'allow_unvalidated':True,
            'cpu_threads':info.get('threads',0),'weight_source':{'kind':'nvidia_dll','path':str(source),
                'sha256':manifest['model_dll_sha256'],'model':model}}
        # Exercise today's safe default, not the historical explicit workaround.
        if weight_source is not None:config['weight_source']=weight_source
        # The register-only FFN remains an experimental opt-in, excluded here.
        job={'extent':[w,h],'model':model,'route':route,'profile':profile,'process_index':process,'config':config,
             'runtime':str(stage/'phoenix_runtime_cpp.dll'),'slow':info['slow'],'vendor_id':info['vendor'],
             'metadata':patch or {},'advanced':advanced}
        if size==(320,180) and profile=='default':job['golden']=str(WORK/('readme-golden-'+model+'.npz'))
        path=WORK/'readme-portable-job.json';write(path,job)
        execute(key,[sys.executable,ROOT/'tools/benchmark_readme_case.py',path,DEST/'cases'/(key+'.json')],path,DEST/'cases'/(key+'.json'),600 if info['slow'] else 180)
    chosen=lambda route:args.routes is None or route in args.routes
    selected_size=lambda sizes:[tuple(map(int,args.only_size.split('x')))] if args.only_size else sizes
    profiles=[('depth4',{'transformer_blocks':4}),('skip4',{'repeats':[1,1,1,0,1,1,1,1]}),
              ('repeat4x2',{'repeats':[1,1,1,2,1,1,1,1]}),('passes2',{'passes':2}),('passes4',{'passes':4}),('passes10',{'passes':10}),
              ('adaptive_gpu',{'adaptive_budget':{'enabled':True,'mode':'gpu_time','target_ms':.1,'minimum_blocks':2}})]
    if args.stage=='smoke':
        for model in MODELS:
            for engine in ['phoenix','nvidia']:native(model,(320,180),0,engine=engine,debug=True)
            for profile,patch in profiles[:4]:native(model,(320,180),0,profile,patch,debug=True)
            for route in routes:
                if route!='cpu1' and chosen(route):portable(route,model,(320,180),0)
    if args.stage=='native':
        for size in selected_size(NATIVE_SIZES):
            for process in range(2):
                # Reverse the second process block so one renderer is not always
                # sampled earlier in the temperature/clock trajectory.
                for model,engine in native_order(process):
                    native(model,size,process,engine=engine)
    if args.stage=='portable':
        for route,info in routes.items():
            if not chosen(route):continue
            sizes=[(320,180)] if route=='cpu1' else SLOW_SIZES if info['slow'] else FAST_SIZES
            for size in selected_size(sizes):
                for process in range(2):
                    for model in MODELS:portable(route,model,size,process)
    if args.stage=='features':
        for model in MODELS:
            for profile,patch in profiles:native(model,(320,180),0,profile,patch)
            for route in ['cuda','vulkan_nvidia']:
                if route not in routes or not chosen(route):continue
                for profile,patch in profiles:portable(route,model,(320,180),0,profile,patch)
                for profile,patch in [('v3_default',{}),('natural',{'controls':{'style':1}}),('cinematic',{'controls':{'style':2}}),
                                      ('control_map',{}),('ui_overlay',{}),('distortion',{}),('combined',{'controls':{'style':1}})]:
                    portable(route,model,(320,180),0,profile,patch,advanced=True)
    if args.stage=='model_sources':
        if not args.model_library:raise RuntimeError('Alternate-source benchmarks need --model-library')
        library=args.model_library.resolve()
        index=read(library/'model-library.json')
        if index.get('schema')!='phoenix-model-library/1':raise RuntimeError('Unknown model library schema')
        sources={}
        for entry in index['models']:
            path=(library/entry['manifest']).resolve()
            if not path.is_relative_to(library) or sha(path)!=entry['manifest_sha256']:
                raise RuntimeError('Shared model manifest changed or escapes the library')
            description=read(path)
            if description.get('model')!=entry['model']:raise RuntimeError('Shared model architecture selection differs')
            resource=(path.parent/description['resource']['path']).resolve()
            if not resource.is_relative_to(path.parent) or sha(resource)!=description['resource']['sha256']:
                raise RuntimeError('Shared model resource changed or escapes its package')
            if entry['model'] in sources:raise RuntimeError('Ambiguous alternate source per model')
            sources[entry['model']]={'kind':'model_folder','path':str(path.parent),'model':entry['model']}
        if set(sources)!=set(MODELS):raise RuntimeError('Both model sources required')
        manifest['alternate_source_scope']='Canonical native exports: alternate storage for the same two networks, not new trained models'
        for model,source_config in sources.items():
            for process in range(2):
                for size in ((320,180),(1920,1080),(3840,2160)):
                    native(model,size,process,'shared_export',weight_source=source_config)
                for route in routes:
                    if not chosen(route):continue
                    sizes=[(320,180)] if route=='cpu1' else [(320,180),(1920,1080)]
                    for size in sizes:portable(route,model,size,process,'shared_export',weight_source=source_config)
    failed=[key for key in stage_keys if not manifest['cases'][key].get('passed')]
    manifest['stages'][args.stage].update(state='failed' if failed else 'completed',finished_utc=now(),failed_cases=failed)
    manifest['current_case']=None;save()


if __name__=='__main__':main()
