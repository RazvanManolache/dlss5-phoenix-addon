"""HD portable feature isolation, run sequentially after the resolution sweep."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from compare_neural_controls import configure,advanced_request,Frame3,status
from benchmark_neural_portable import stats
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['cuda','vulkan']);args=p.parse_args()
    destination=ROOT/'artifacts/current/neural-performance'
    config=json.loads((ROOT/f'artifacts/current/neural-candidate/portable-{args.backend}.json').read_text())
    api=configure(Path(config['runtime_root'])/'phoenix_runtime_cpp.dll');api.phoenix_runtime_frame_v3.argtypes=[C.c_void_p,C.POINTER(Frame3)]
    w,h=1920,1080;session=api.phoenix_runtime_create(json.dumps(config).encode(),w,h)
    if not session:raise RuntimeError(api.phoenix_runtime_error().decode())
    image=np.empty((h,w,3),np.float32)
    image[:,:,0]=.125+(np.arange(w,dtype=np.float32)[None,:]%128)/256
    image[:,:,1]=.25+(np.arange(h,dtype=np.float32)[:,None]%128)/256
    image[:,:,2]=.5
    output=np.empty_like(image);motion=np.zeros((h,w,2),np.float32)
    mask=np.ones((h,w,4),np.float32);mask[:,:w//3,0]=0;mask[:,w//2:,1]=.4;mask[:,w//2:,2]=.25
    ui=np.zeros((h,w,4),np.float32);ui[:h//10,w//16:w//3]=1
    field=np.zeros((h,w,4),np.float32);field[:,:,0]=-1/w;field[:,:,2]=1/w
    report=dict(backend=args.backend,extent=[w,h],full_graph=True,passes=1,profiles=[],passed=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        for name in ['default','advanced_default','natural','cinematic','control_map','ui_overlay','distortion','combined','default_after_features']:
            metadata=dict(reset=True,has_motion=True,controls=dict(style=1 if name in ['natural','combined'] else 2 if name=='cinematic' else 0,auto_mask=True,sampling_mode='fixed'))
            advanced=name not in ['default','default_after_features']
            planes=dict(motion=motion)
            if name in ['control_map','combined']:planes['mask']=mask
            if name in ['ui_overlay','combined']:planes['ui']=ui
            if name in ['distortion','combined']:planes['distortion']=field
            def call():
                text=json.dumps(metadata).encode();request=advanced_request(image,output,metadata,**planes) if advanced else None
                start=time.perf_counter()
                code=api.phoenix_runtime_frame_v3(session,C.byref(request)) if advanced else api.phoenix_runtime_frame(session,image.ctypes.data,motion.ctypes.data,None,None,text,output.ctypes.data)
                ms=(time.perf_counter()-start)*1000
                if code:raise RuntimeError(api.phoenix_runtime_error().decode())
                return ms
            first=call();metadata['reset']=False;warm=0;start=time.perf_counter()
            while warm<4 or time.perf_counter()-start<.75:call();warm+=1
            walls=[];gpus=[];stages=[];start=time.perf_counter()
            while (len(walls)<12 or time.perf_counter()-start<2) and time.perf_counter()-start<15:
                walls.append(call());r=status(api,session);gpus.append(r['operator_ms'])
                stages.append({k:r.get(k,0) for k in ['canonicalize_ms','presentation_ms','prepare_ms','resolve_ms','execute_with_transfers_ms']})
            assert np.isfinite(output).all()
            row=dict(profile=name,first_call_ms=first,wall=stats(walls),gpu=stats(gpus),
                wall_samples_ms=walls,gpu_samples_ms=gpus,stages_mean_ms={k:float(np.mean([s[k] for s in stages])) for k in stages[0]})
            report['profiles'].append(row);(destination/f'portable-feature-cost-{args.backend}.json').write_text(json.dumps(report,indent=2)+'\n')
            print(args.backend,name,round(row['wall']['median_ms'],2),'ms',flush=True)
        report['passed']=True
        (destination/f'portable-feature-cost-{args.backend}.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:api.phoenix_runtime_destroy(session)
if __name__=='__main__':main()
