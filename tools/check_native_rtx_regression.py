"""Explicit isolated before/after Native RTX SDK regression check.

Reconstructs the pre-portable-optimization SDK files and requires their exact
recorded source hashes before compiling. Never edits or promotes working files.
"""
import argparse,datetime,hashlib,json,os,re,shutil,statistics,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from test_model_selection import ROOT,toolchain

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def text_sha(value):return hashlib.sha256(value.replace('\n','\r\n').encode()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def replace_once(value,old,new):
    assert value.count(old)==1,old[:100]
    return value.replace(old,new)

def reconstruct(before):
    changes={}
    name='native_runtime/native_api.cpp';s=(ROOT/name).read_text()
    for line in ['#include "portable_pixels.h"\n','#include "portable_d3d12_pixels.h"\n','    std::unique_ptr<phoenix_portable_pixels::D3D12Codec> portable_codec;\n']:
        s=replace_once(s,line,'')
    changes[name]=s
    name='native_runtime/native_portable.inc';s=(ROOT/name).read_text()
    begin=s.index('    const auto& runner_configuration=');end=s.index('    auto prepare=',begin);s=s[:begin]+s[end:]
    s=replace_once(s,'    auto& rgb=portable_rgb;auto& vectors=portable_vectors;auto& result=portable_result;auto& rgba=portable_rgba;\n',
        '    auto& rgb=portable_rgb;auto& vectors=portable_vectors;auto& result=portable_result;auto& rgba=portable_rgba;\n    phx::Kernel<HostMath> math{};\n')
    s=replace_once(s,'auto read=[&](Plane& plane,auto consume){','auto read=[&](Plane& plane,unsigned channels,auto consume){')
    s=replace_once(s,'                consume(y,row);','                for(unsigned x=0;x<width;++x)for(unsigned c=0;c<channels;++c)consume((size_t(y)*width+x)*channels+c,row[x*channels+c]);')
    begin=s.index('        read(portable_color_readback,[&]');end=s.index('        require(portable_session->frame',begin)
    s=s[:begin]+'''        read(portable_color_readback,4,[&](size_t i,uint16_t value){rgba[i]=value;if(i%4<3)rgb[i/4*3+i%4]=std::clamp(math.half_value(value),0.f,1.f);});
        if(motion)read(portable_motion_readback,2,[&](size_t i,uint16_t value){vectors[i]=math.half_value(value)*(i%2?controls.motion_scale_y:controls.motion_scale_x);});
'''+s[end:]
    s=replace_once(s,'        portable_status["pixel_conversion"]="host_simd";\n        portable_status["texture_boundary_host_bytes"]=pixels*(motion?20:16);\n','')
    s=replace_once(s,'        phoenix_portable_pixels::output(result.data(),rgba.data(),pixels);','        for(size_t p=0;p<pixels;++p)for(unsigned c=0;c<3;++c)rgba[p*4+c]=uint16_t(math.half_bits(result[p*3+c]));')
    changes[name]=s
    name='native_runtime/half_rgba.h';s=(ROOT/name).read_text()
    s=replace_once(s,'inline bool host_f16c_available() {\n#ifdef PHOENIX_HAS_F16C_DISPATCH','inline void half_rgba_to_rgb(const uint16_t* source, float* target, size_t pixels) {\n#ifdef PHOENIX_HAS_F16C_DISPATCH')
    s=replace_once(s,'''    return supported;
#else
    return false;
#endif
}
inline void half_rgba_to_rgb(const uint16_t* source, float* target, size_t pixels) {
#ifdef PHOENIX_HAS_F16C_DISPATCH
    if (host_f16c_available()) {''','''    if (supported) {''')
    changes[name]=s
    name='native_runtime/broker_client.h';s=(ROOT/name).read_text()
    begin=s.index('            // V3 views are absent');end=s.index('\n        }',begin)
    s=s[:begin]+'''            copy(10*pixels, 4*pixels, mask);
            copy(14*pixels, 4*pixels, ui);
            copy(18*pixels, pixels, alpha);
            copy(19*pixels, 3*pixels, backbuffer);
            copy(22*pixels, 4*pixels, distortion);'''+s[end:]
    changes[name]=s
    name='src/native/network/production_network.h'
    # HEAD's only difference is the later NVAPI error diagnostic, verified by hash.
    s=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode().replace('\r\n','\n')
    changes[name]=s
    verification={};encoded={}
    for name,s in changes.items():
        expected=before['source_sha256'][name]
        candidates=[s.encode(),s.replace('\n','\r\n').encode()]
        raw=next((v for v in candidates if hashlib.sha256(v).hexdigest()==expected),candidates[0])
        encoded[name]=raw;verification[name]={'expected':expected,'reconstructed':hashlib.sha256(raw).hexdigest()}
    return encoded,verification

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-check-only',action='store_true');p.add_argument('--rounds',type=int,default=6);p.add_argument('--samples',type=int,default=30)
    a=p.parse_args();assert 3<=a.rounds<=10 and a.samples>=15
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    before=json.loads((ROOT.parent/'dlss5-everything-testing/runs/latest-processing-comparison/run.json').read_text())
    changes,verification=reconstruct(before)
    report=dict(schema='phoenix-native-rtx-regression/1',passed=False,state='source-check',reconstructed_sources=verification,cases=[],games_launched=False,installed_games_modified=False)
    save(out/'run.json',report)
    assert all(v['expected']==v['reconstructed'] for v in verification.values()),'Baseline reconstruction differs; see run.json'
    print('Five earlier SDK files reconstructed byte-for-byte',flush=True)
    if a.source_check_only:return
    compiler,env=toolchain(64);cuda=Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1')
    raw=ROOT/'artifacts/current/windows-cuda';manifest=json.loads((raw/'BUILD.json').read_text())
    assert manifest['passed'] and all(sha(ROOT/k)==v for k,v in manifest['source_inputs'].items()),'Current package source binding is stale'
    report.update(state='building',date=datetime.datetime.now(datetime.timezone.utc).isoformat(),rounds=a.rounds,samples=a.samples,
        baseline_receipt_sha256=sha(ROOT.parent/'dlss5-everything-testing/runs/latest-processing-comparison/run.json'),source_changes_applied_to_workspace=False)
    def run(stage,cmd,cwd,timeout=900):
        print(stage,flush=True);report['current']=stage;save(out/'run.json',report)
        with (out/(stage+'.log')).open('w') as log:
            result=subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:raise RuntimeError(stage+' failed: '+str(out/(stage+'.log')))
    def telemetry():
        result=subprocess.run(['nvidia-smi','--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,pstate','--format=csv,noheader,nounits'],capture_output=True,text=True)
        return result.stdout.strip()
    source_snapshot={k:sha(ROOT/k) for k in before['source_sha256'] if (ROOT/k).exists()}
    with tempfile.TemporaryDirectory(prefix='rtx-regression-',dir=ROOT/'build') as temporary,tempfile.TemporaryDirectory(prefix='rtx_regression_bundle_',dir=ROOT/'build') as kernel_name:
        work=Path(temporary);bundle=Path(kernel_name);bundle.rmdir()
        run('native-kernels',[sys.executable,ROOT/'tools/production_build_native_kernels.py',bundle.name,'--compiler',cuda/'bin/nvcc.exe','--architectures','sm_120','--jobs','4'],ROOT,1800)
        kernels=json.loads((bundle/'native_manifest.json').read_text());assert kernels['kernels']==before['native_manifest']['kernels'],'Kernel binaries changed'
        report['kernel_bundle_identical']=True;report['native_manifest']=kernels
        suffixes={'.h','.hpp','.cpp','.c','.cu','.cuh','.inc','.json','.hlsl','.glsl','.comp','.def'}
        current=work/'current';baseline=work/'baseline'
        for name in ['native_runtime','src','third_party']:
            for source in (ROOT/name).rglob('*'):
                if source.is_file() and source.suffix in suffixes:
                    target=current/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        shutil.copytree(current,baseline)
        for name,value in changes.items():(baseline/name).write_bytes(value)
        dependency_candidates=[k for k in before['source_sha256'] if (baseline/k).exists()]
        report['baseline_source_differences']=[k for k in dependency_candidates if sha(baseline/k)!=before['source_sha256'][k]]
        # The other changed files implement separate provider DLLs, not this SDK.
        allowed={'native_runtime/backend_cpu.cpp','native_runtime/backend_gpu.cpp','native_runtime/backend_vulkan.cpp','native_runtime/cpu_avx512.cpp','native_runtime/runtime.cpp'}
        assert set(report['baseline_source_differences'])<=allowed,report['baseline_source_differences']
        common=[compiler,'/nologo','/std:c++20','/O2','/Ob2','/MD','/DNDEBUG','/DWIN32','/D_WINDOWS','/DNOMINMAX','/EHsc','/bigobj','/fp:strict']
        runtime_dirs={}
        for name,source in [('before',baseline),('after',current)]:
            destination=work/name;destination.mkdir();runtime_dirs[name]=destination
            run('sdk-'+name,common+['/experimental:deterministic','/pathmap:'+str(source)+'=.',
                '/I'+str(ROOT.parent/'external/DLSS/include'),'/I'+str(source/'third_party/nvapi'),'/showIncludes','/LD',
                source/'native_runtime/native_api.cpp','/Fo:'+str(destination/'native.obj'),'/Fe:'+str(destination/'phoenix_native.dll'),
                'd3d12.lib','dxgi.lib','d3dcompiler.lib','bcrypt.lib'],destination)
            log=(out/('sdk-'+name+'.log')).read_text(errors='replace');used=[]
            for line in log.splitlines():
                if 'including file:' in line:
                    path=Path(line.split('including file:',1)[1].strip())
                    if path.is_relative_to(source):used.append(path.relative_to(source).as_posix())
            assert used,'No compiler include dependency trace'
            report.setdefault('sdk_dependencies',{})[name]={n:sha(source/n) for n in sorted(set(used))}
            if name=='before':assert not (set(used)&set(report['baseline_source_differences'])),'Non-baseline dependency used'
        packaged=work/'packaged';packaged.mkdir();shutil.copy2(raw/'phoenix_native.dll',packaged/'phoenix_native.dll');runtime_dirs['packaged']=packaged
        report['sdk_sha256']={n:sha(p/'phoenix_native.dll') for n,p in runtime_dirs.items()}
        native_tests=current/'tests/native';native_tests.mkdir(parents=True)
        for name in ['four_model_probe.cpp','kernel_timing.h']:shutil.copy2(ROOT/'tests/native'/name,native_tests/name)
        probe=(ROOT/'tests/native/unified_processing_probe.cpp').read_text()
        probe=replace_once(probe,'{"PipelineMonitoring",0}','{"PipelineMonitoring",job.value("monitoring",0)},{"MonitoringSampleInterval",1}')
        probe=replace_once(probe,'    auto frame=[&]{','    std::vector<double> diagnostic_gpu;Json diagnostic_stages=Json::array();\n    auto frame=[&]{')
        probe=replace_once(probe,'if(vendor){gpu.begin();nvidia->record(gpu,color,motion,output,true,w,h);gpu.submit();}',
            'if(vendor){gpu.begin();if(job.value("monitoring",0))gpu.mark(0);nvidia->record(gpu,color,motion,output,true,w,h);if(job.value("monitoring",0)){gpu.mark(1);gpu.command->ResolveQueryData(gpu.queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,gpu.stamps.Get(),0);}gpu.submit();}')
        probe=replace_once(probe,'        return elapsed;};',
            '        if(job.value("monitoring",0)){if(vendor)diagnostic_gpu.push_back(gpu.timing());else{const auto status=sdk->snapshot();require(status.at("gpu_sampled_last_frame")==true,"Missing SDK GPU sample");diagnostic_gpu.push_back(status.at("gpu_processing_ms"));diagnostic_stages.push_back(status.at("frame_stages_ms"));}}\n        return elapsed;};')
        probe=replace_once(probe,'for(unsigned i=0;i<warmup;++i)frame();std::vector<double> samples;',
            'for(unsigned i=0;i<warmup;++i)frame();diagnostic_gpu.clear();diagnostic_stages.clear();std::vector<double> samples;')
        probe=replace_once(probe,'    if(sdk)report["status"]=',
            '    if(!diagnostic_gpu.empty()){report["diagnostic_gpu_ms"]=diagnostic_gpu;report["diagnostic_gpu"]=stats(diagnostic_gpu);report["diagnostic_cpu_stages"]=diagnostic_stages;}\n    if(sdk)report["status"]=')
        (native_tests/'unified_processing_probe.cpp').write_text(probe)
        report['probe_source_sha256']=sha(native_tests/'unified_processing_probe.cpp')
        options=common+['/Zc:__cplusplus','/I'+str(ROOT.parent/'external/DLSS/include'),'/I'+str(current/'third_party/nvapi'),'/I'+str(current/'third_party/minhook/include')]
        forwarder=work/'phoenix-nvngx.dll'
        run('forwarder',options+['/LD',current/'src/integration/reshade/nr_forwarder.cpp','/Fo:'+str(work/'forwarder.obj'),'/Fe:'+str(forwarder)],work)
        exe=work/'probe.exe';hooks=[current/'third_party/minhook/src'/n for n in ['hook.c','buffer.c','trampoline.c','hde/hde64.c']]
        run('probe',options+[native_tests/'unified_processing_probe.cpp',*hooks,'/Fe:'+str(exe),'d3d12.lib','dxgi.lib','d3dcompiler.lib','bcrypt.lib','psapi.lib'],work)
        report['probe_sha256']=sha(exe)
        image=ROOT.parent/'dlss5-everything-testing/runs/windows-linux-before/windows/dx12-x64/capture.png'
        rgba=(np.asarray(Image.open(image).convert('RGBA').resize((1280,720),Image.Resampling.LANCZOS),np.float32)/255).astype(np.float16)
        source=work/'input.rgba16f';rgba.tofile(source);assert sha(source)==before['input_rgba16f_sha256']
        model=ROOT.parent/'rtx-test-cache/main/Nvidia/nodix-compat/1.3.7-ac6ddca163ea/nvngx_dlssnr.dll'
        core=list(Path(r'C:\Windows\System32\DriverStore\FileRepository').rglob('_nvngx.dll'));assert len(core)==1
        report.update(state='running',input_sha256=sha(source),model_sha256=sha(model),gpu_before=telemetry(),scope='Same 1280x720 host RGBA16F upload -> full reset model -> completed host readback, with 100 warmup calls; balanced fresh-process ordering. Diagnostic GPU intervals measured separately.',
            diagnostic_scope='Phoenix SDK timestamps cover recorded network work and final output copy, excluding its initial private input copies. NVIDIA timestamps bracket its evaluation. These intervals are valid within-route before/after controls, not isolated equivalent neural-kernel timing.')
        golden={m:next(c['output_sha256'] for c in before['cases'] if c['route']=='native_rtx' and c['model']==m) for m in ['original','small-fp8']}
        for diagnostics,rounds in [(False,a.rounds),(True,2)]:
            for iteration in range(rounds):
                cells=[(m,n) for m in ['original','small-fp8'] for n in ['before','after','packaged','nvidia']]
                if iteration%2:cells.reverse()
                # Rotate the start to avoid always charging startup/clock drift to before.
                shift=(iteration//2)%len(cells);cells=cells[shift:]+cells[:shift]
                for model_id,variant in cells:
                    key=('diagnostic-' if diagnostics else 'plain-')+model_id+'-'+variant+'-'+str(iteration)
                    print(key,flush=True);report['current']=key;save(out/'run.json',report)
                    job=dict(engine='nvidia' if variant=='nvidia' else 'sdk',runtime_root=str(runtime_dirs.get(variant,packaged)),
                        width=1280,height=720,bundle_path=str(bundle),weight_source=dict(kind='nvidia_dll',path=str(model),sha256=sha(model),model=model_id),
                        ngx_core=str(core[0]),forwarder=str(forwarder),cache=str(work/'ngx-cache'),color_file=str(source),warmup=100 if not diagnostics else 30,
                        samples=a.samples if not diagnostics else 15,monitoring=2 if diagnostics else 0)
                    jobfile=work/'job.json';result=work/'result.json';save(jobfile,job);result.unlink(missing_ok=True)
                    telemetry_before=telemetry()
                    process=subprocess.run([exe,jobfile,result],cwd=work,env=env,capture_output=True,text=True,timeout=180)
                    value=json.loads(result.read_text()) if result.exists() else {}
                    assert process.returncode==0 and value.get('passed'),value.get('error',process.stdout+process.stderr)
                    assert value['output_sha256']==golden[model_id],'Output regression'
                    value.update(variant=variant,round=iteration,diagnostic=diagnostics,gpu_before=telemetry_before,gpu_after=telemetry())
                    report['cases'].append(value);save(out/'run.json',report)
                    print(str(value['wall']['median_ms'])+' ms',flush=True)
        report['source_changed_during_run']=[k for k,v in source_snapshot.items() if sha(ROOT/k)!=v];assert not report['source_changed_during_run']
        report.update(passed=True,state='complete',gpu_after=telemetry(),temporary_builds_removed=True)
    save(out/'run.json',report)
    print(json.dumps({'passed':True,'cases':len(report['cases']),'report':str(out/'run.json')}),flush=True)

if __name__=='__main__':main()
