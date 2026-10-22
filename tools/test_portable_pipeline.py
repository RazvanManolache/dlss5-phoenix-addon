"""Validate native portable-chain stages without game or desktop automation."""
import argparse
import ctypes as C
import _ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from compare_neural_controls import configure,status
from test_model_selection import ROOT,toolchain,run


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def sample(image,width,height):
    sh,sw,_=image.shape
    yy,xx=np.mgrid[:height,:width];px=(xx+.5)*sw/width-.5;py=(yy+.5)*sh/height-.5
    def bilinear(px,py):
        x=np.floor(px).astype(int);y=np.floor(py).astype(int);fx=(px-x)[...,None];fy=(py-y)[...,None]
        def read(x,y):return image[np.clip(y,0,sh-1),np.clip(x,0,sw-1)].astype(np.float64)
        return ((read(x,y)*(1-fx)+read(x+1,y)*fx)*(1-fy)+(read(x,y+1)*(1-fx)+read(x+1,y+1)*fx)*fy)
    dx=max(sw/width-1,0)*.25;dy=max(sh/height-1,0)*.25
    return np.ascontiguousarray((.25*(bilinear(px+dx,py+dy)+bilinear(px-dx,py-dy)+bilinear(px+dx,py-dy)+bilinear(px-dx,py+dy))).astype(np.float32))


def case(library,root,source,model,backend,device,output_stage='presented'):
    api=configure(library)
    config={'runtime_root':str(root),'backend':backend,'device':device,'allow_unvalidated':True,
        'cpu_threads':2,'weight_source':{'kind':'nvidia_dll','path':str(source),'sha256':sha(source),'model':model}}
    handles=[]
    def create(w,h):
        session=api.phoenix_runtime_create(json.dumps(config).encode(),w,h)
        if not session:raise RuntimeError(api.phoenix_runtime_error().decode())
        handles.append(session);return session
    metadata={'reset':True,'has_motion':True,'passes':2,'repeats':[1]*8,'output_stage':output_stage,
        'controls':{'auto_mask':False,'seed':0,'sampling_mode':'fixed'}}
    if backend=='cpu':metadata.update(transformer_blocks=1,repeats=[1,0,0,0,0,0,0,0])
    def evaluate(session,image,meta):
        h,w,_=image.shape;motion=np.zeros((h,w,2),np.float32);output=np.empty_like(image)
        code=api.phoenix_runtime_frame(session,image.ctypes.data,motion.ctypes.data,None,None,json.dumps(meta).encode(),output.ctypes.data)
        if code:raise RuntimeError(api.phoenix_runtime_error().decode())
        if not np.isfinite(output).all():raise RuntimeError('Nonfinite chain result')
        return output
    try:
        w,h=32,24
        yy,xx=np.mgrid[:h,:w];image=np.stack([.2+(xx%9)/32,.1+(yy%7)/16,.25+((xx+yy)%5)/16],axis=-1).astype(np.float16).astype(np.float32)
        frames=[np.ascontiguousarray(np.roll(image,i,axis=1)) for i in range(3)]
        references=[]
        # The public Vulkan API still owns one context at a time. Produce the
        # independent oracles in sequential sessions, just like two separate jobs.
        for scale in (1,2):
            handle=create(w//scale,h//scale);predictions=[]
            for i,pixels in enumerate(frames):
                source_pixels=pixels if scale==1 else sample(pixels,w//2,h//2)
                prediction=evaluate(handle,source_pixels,{**metadata,'reset':i==0})
                predictions.append(prediction if scale==1 else sample(prediction,w,h))
            references.append(predictions);api.phoenix_runtime_destroy(handle);handles.remove(handle)
        pipeline={'ProcessingChain':True,'ChainFusion':True,'MultiscaleFusion':True,'FusionLayers':2,'PipelineMonitoring':0}
        errors=[{'frame':i} for i in range(3)];report=None
        for standalone in (False,True):
            fused=create(w,h)
            selected={**pipeline,'ChainNeural':not standalone}
            for frame,pixels in enumerate(frames):
                meta={**metadata,'reset':frame==0,'pipeline':selected}
                if frame==1:
                    # A rejected reset must not erase working branch recurrence.
                    broken={**meta,'reset':True,'passes':0}
                    dummy=np.empty_like(pixels);mv=np.zeros((h,w,2),np.float32)
                    if api.phoenix_runtime_frame(fused,pixels.ctypes.data,mv.ctypes.data,None,None,json.dumps(broken).encode(),dummy.ctypes.data)==0:
                        raise RuntimeError('Invalid chain pass count accepted')
                expected=(references[0][frame]+references[1][frame])*.5
                actual=evaluate(fused,pixels,meta);error=float(np.max(np.abs(actual-expected)))
                if error>.0002:raise RuntimeError(f'Independent fusion oracle failed: {error}')
                errors[frame]['standalone_max_error' if standalone else 'max_error']=error
            report=status(api,fused)
            if report['independent_temporal_stages']!=4:raise RuntimeError('Whole-network passes or branches share histories')
            if report['network_schedule']['adaptive']:raise RuntimeError('Unexpected independent adaptation')
            if report['pipeline_histories_reset']:raise RuntimeError('Rejected request erased valid histories')
            neutral={'ProcessingChain':True,'ChainNeural':False,'ChainScale':False,'ChainColor':False,'PipelineMonitoring':0}
            if not np.array_equal(evaluate(fused,image,{**metadata,'reset':False,'pipeline':neutral}),image):
                raise RuntimeError('Changing an active chain to bypass did not preserve pixels')
            bypass_status=status(api,fused)
            if bypass_status['independent_temporal_stages'] or bypass_status['history_used']:
                raise RuntimeError('Bypassed chain reports stale neural execution')
            api.phoenix_runtime_destroy(fused);handles.remove(fused)
        bypass=create(w,h)
        neutral={'ProcessingChain':True,'ChainNeural':False,'ChainScale':False,'ChainColor':False,'PipelineMonitoring':0}
        if not np.array_equal(evaluate(bypass,image,{**metadata,'pipeline':neutral}),image):raise RuntimeError('Disabled chain was not exact bypass')
        return {'passed':True,'model':model,'backend':backend,'device':device,'output_stage':output_stage,'independent_fusion_oracle':errors,
            'passes':2,'blocks':metadata.get('transformer_blocks',8),'independent_temporal_stages':4,'rejected_reset_preserves_history':True,'disabled_exact_bypass':True,
            'network_schedule':report['network_schedule']}
    finally:
        for handle in reversed(handles):api.phoenix_runtime_destroy(handle)
        _ctypes.FreeLibrary(api._handle)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--runtime-root',type=Path,help='Optional backend folder; default extracts the actual packaged x64 payload')
    args=parser.parse_args()
    output=ROOT/'artifacts/current/portable-pipeline';output.mkdir(parents=True,exist_ok=True)
    report={'passed':False,'cases':[]}
    source=ROOT.parent/'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
    inputs={p.relative_to(ROOT).as_posix():sha(p) for directory in ('src','native_runtime','third_party','tests/native')
        for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in ('.h','.hpp','.cpp','.inc','.inl')}
    try:
        compiler,env=toolchain(64)
        with tempfile.TemporaryDirectory(prefix='portable-pipeline-',dir=ROOT/'build') as temporary:
            work=Path(temporary);library=work/'phoenix_runtime_cpp.dll'
            if args.runtime_root:root=args.runtime_root.resolve()
            else:
                from build_experimental_rtx_addon import extract_runtime
                addon=ROOT/'artifacts/current/addon/Phoenix.addon64'
                payload=work/'payload';payload.mkdir();scratch=work/'resources';scratch.mkdir()
                extract_runtime(addon,payload,scratch)
                def x64(path):
                    data=path.read_bytes();pe=int.from_bytes(data[60:64],'little')
                    return data[pe:pe+4]==b'PE\x00\x00' and data[pe+4:pe+6]==b'\x64\x86'
                roots=[p.parent for p in payload.rglob('phoenix_runtime_cpp.dll') if x64(p)]
                if len(roots)!=1:raise RuntimeError('Need one packaged x64 backend root')
                root=roots[0];report['source_addon_sha256']=sha(addon)
                report['packaged_backend_root']=root.relative_to(payload).as_posix()
            report['model_dll_sha256']=sha(source)
            report['backend_files']={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*')
                if p.is_file() and p.suffix.lower() in ('.dll','.spv') and p.name!='phoenix_runtime_cpp.dll'}
            flags=[compiler,'/nologo','/std:c++20','/EHsc','/O2','/MD','/bigobj','/DNOMINMAX','/D_CRT_SECURE_NO_WARNINGS',
                '/experimental:deterministic','/pathmap:'+str(ROOT.parent)+'=.']
            contracts=work/'host-contracts.exe'
            run([*flags,ROOT/'tests/native/pipeline_host_chain_tests.cpp','/Fo:'+str(work/'host.obj'),'/Fe:'+str(contracts)],cwd=work,env=env)
            report['host_contracts']=json.loads(run([contracts],cwd=work,env=env));print('Host chain contracts passed.',flush=True)
            run([*flags,'/LD',ROOT/'native_runtime/runtime.cpp','/Fo:'+str(work/'runtime.obj'),'/Fe:'+str(library),'/link','bcrypt.lib'],cwd=work,env=env)
            print('Portable pipeline DLL compiled.',flush=True)
            devices=[('cpu','cpu')]
            if args.gpu:
                for name in ('cuda','vulkan_nvidia','vulkan_amd'):
                    receipt=json.loads((ROOT/f'artifacts/current/readme-benchmarks/cases/{name}-small-fp8-320x180-default-p0.json').read_text())
                    devices.append(('cuda' if name=='cuda' else 'vulkan',receipt['runtime_status']['device']))
            for backend,device in devices:
                for model in ('original','small-fp8'):
                    result=case(library,root,source,model,backend,device)
                    report['cases'].append(result);print(model+' '+device+' passed.',flush=True)
                    if backend!='cpu':
                        result=case(library,root,source,model,backend,device,'neural_only')
                        report['cases'].append(result);print(model+' '+device+' deferred-presentation passed.',flush=True)
            # case() releases each DLL handle after destroying its sessions.
            if any(sha(ROOT/name)!=digest for name,digest in inputs.items()):raise RuntimeError('Portable pipeline sources changed during validation')
            from prepare_addon_release import safe_payload
            safe_payload('runtime/phoenix_runtime_cpp.dll',library.read_bytes())
            report['runtime_sha256']=sha(library);shutil.copy2(library,output/library.name)
            report['public_binary_audit_passed']=True
            report['source_inputs']=inputs
            report['passed']=True
    except Exception as error:
        report['error']=str(error);raise
    finally:
        (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
