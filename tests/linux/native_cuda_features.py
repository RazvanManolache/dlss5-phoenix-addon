"""Explicit GPU behavior checks for native CUDA temporal, V3 and scheduling APIs."""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('profile',ROOT/'tools/profile_linux.py')
profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)


class V3(C.Structure):
    _fields_=[('base',profile.Frame),('ui',profile.Plane),('alpha',profile.Plane),
              ('backbuffer',profile.Plane),('distortion',profile.Plane)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('package','config','image','report'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.package.resolve()
    if (root/'lib').is_dir():root=root/'lib'
    library=root/('phoenix_native_cuda.dll' if sys.platform=='win32' else 'libphoenix_native_cuda.so')
    config=json.loads(args.config.read_text());config['runtime_root']=str(root)
    lib=C.CDLL(str(library));lib.phoenix_runtime_create.argtypes=[C.c_char_p,C.c_uint,C.c_uint];lib.phoenix_runtime_create.restype=C.c_void_p
    lib.phoenix_runtime_destroy.argtypes=[C.c_void_p];lib.phoenix_runtime_error.restype=C.c_char_p
    lib.phoenix_runtime_frame_v2.argtypes=[C.c_void_p,C.POINTER(profile.Frame)]
    lib.phoenix_runtime_frame_v3.argtypes=[C.c_void_p,C.POINTER(V3)]
    lib.phoenix_runtime_status.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];lib.phoenix_runtime_status.restype=C.c_size_t
    lib.phoenix_runtime_replace_model.argtypes=[C.c_void_p,C.c_char_p]
    with Image.open(args.image) as image:source=np.asarray(image.convert('RGB'),dtype=np.float32)/np.float32(255)
    height,width,_=source.shape
    sessions=[];checks=[];measurements={}
    def require(value,message):
        if not value:raise AssertionError(message)
    def checked(name):checks.append(name);print(name,flush=True)
    def create():
        handle=lib.phoenix_runtime_create(json.dumps(config).encode(),width,height)
        require(handle,lib.phoenix_runtime_error().decode());sessions.append(handle);return handle
    def status(handle):
        buffer=C.create_string_buffer(lib.phoenix_runtime_status(handle,None,0));lib.phoenix_runtime_status(handle,buffer,len(buffer));return json.loads(buffer.value)
    def run(handle,color=source,motion=None,depth=None,validity=None,mask=None,ui=None,alpha=None,distortion=None,metadata=None,reject=False):
        owned=[]
        def plane(value,channels):
            if value is None:return profile.Plane()
            value=np.ascontiguousarray(value,dtype=np.float32);owned.append(value)
            return profile.Plane(value.ctypes.data_as(C.POINTER(C.c_float)),value.nbytes,0,width,height,channels,0)
        out=np.full_like(source,42);meta=json.dumps(metadata if metadata is not None else {'reset':True}).encode()
        request=V3();frame=request.base;advanced=any(v is not None for v in (ui,alpha,distortion))
        frame.struct_bytes=C.sizeof(V3) if advanced else C.sizeof(profile.Frame);frame.version=3 if advanced else 2
        frame.control_profile=int(mask is not None);frame.color=plane(color,3);frame.motion=plane(motion,2)
        frame.depth=plane(depth,1);frame.validity=plane(validity,1);frame.mask=plane(mask,4);frame.output=plane(out,3);frame.metadata=meta
        request.ui=plane(ui,4);request.alpha=plane(alpha,1);request.distortion=plane(distortion,4)
        result=lib.phoenix_runtime_frame_v3(handle,C.byref(request)) if advanced else lib.phoenix_runtime_frame_v2(handle,C.byref(frame))
        if reject:
            require(result==-1 and np.all(out==42),'Rejected frame wrote output or succeeded');return out
        require(result==0,lib.phoenix_runtime_error().decode());require(np.isfinite(out).all(),'Nonfinite output');return out.copy()
    try:
        a,b=create(),create();zero=np.zeros((height,width,2),np.float32)
        first=run(a,motion=zero);require(not status(a)['history_used'],'First frame used uninitialized history')
        run(b,motion=zero)
        moved=np.roll(source,1,axis=1);vectors=zero.copy();vectors[...,0]=-1
        temporal=run(a,color=moved,motion=vectors,metadata={'reset':False})
        require(status(a)['history_used'],'Motion did not activate temporal history')
        run(b,color=moved,motion=vectors,metadata={'reset':False})
        reset=run(a,color=moved,motion=vectors)
        require(np.any(temporal!=reset),'Temporal output equals independent reset output')
        measurements['temporal_reset_mae']=float(np.abs(temporal-reset).mean());checked('motion uses persistent GPU history and changes output')
        low=run(a,motion=zero,metadata={'reset':False,'controls':{'local_tone':0.,'local_structure':0.,'auto_mask':False}})
        high=run(a,motion=zero,metadata={'reset':False,'controls':{'local_tone':2.,'local_structure':2.,'auto_mask':False}})
        measurements['conditioning_mae']=float(np.abs(low-high).mean())
        require(measurements['conditioning_mae']>.001,'Conditioning did not change recurrent output')
        require(not status(a)['history_used'],'Conditioning change retained incompatible history')
        checked('live neural conditioning changes pixels and resets incompatible history')
        # Synchronize both sessions, then inject rejected frames into only one.
        run(a,motion=zero);run(b,motion=zero)
        before=status(a)['frames']
        bad=vectors.copy();bad[0,0,0]=np.nan
        run(a,motion=bad,metadata={'reset':True},reject=True)
        run(a,motion=vectors,metadata={'reset':True,'passes':11},reject=True)
        require(status(a)['frames']==before,'Rejected frame advanced completion state')
        one=run(a,color=moved,motion=vectors,metadata={'reset':False})
        two=run(b,color=moved,motion=vectors,metadata={'reset':False})
        require(np.array_equal(one,two),'Rejected frame changed subsequent history');checked('invalid motion and schedule preserve output and temporal state')
        validity=np.ones((height,width),np.float32);validity[:,width//2:]=0
        depth=np.linspace(0,1,width,dtype=np.float32)[None,:].repeat(height,axis=0)
        run(a,motion=vectors,depth=depth,validity=validity,metadata={'reset':False,'reversed_depth':True})
        checked('depth and per-pixel validity accepted with recurrent motion')
        original=run(a,metadata={'reset':True,'controls':{'intensity':0.}})
        require(np.array_equal(original,source),'Zero intensity failed to preserve input');checked('display intensity affects output independently of history')
        ui=np.zeros((height,width,4),np.float32);ui[...,3]=1;ui[...,0]=.25
        out=run(a,ui=ui);require(np.all(out[...,0]==np.float32(.25)) and np.all(out[...,1:]==0),'Opaque UI not preserved')
        checked('V3 premultiplied UI composition')
        run(a,alpha=np.full((height,width),2,np.float32),reject=True);checked('invalid V3 coverage rejected atomically')
        identity=np.zeros((height,width,4),np.float32)
        plain=run(a);warped=run(a,distortion=identity)
        require(np.array_equal(plain,warped),'Identity distortion changed output');checked('V3 canonical identity distortion')
        mask=np.ones((height,width,4),np.float32);mask[...,0]=0
        out=run(a,mask=mask);require(np.array_equal(out,source),'Mask did not preserve original pixels');checked('bounded experimental control mask')
        for blocks,repeats in [(4,[1]*8),(8,[1,0,2,1,0,1,1,1]),(8,[0]*8),(8,[4]*8),(8,[1]*8)]:
            meta={'reset':True,'transformer_blocks':blocks,'repeats':repeats}
            one=run(a,metadata=meta)
            for repeat in range(6):
                two=run(a,metadata=meta)
                require(np.array_equal(one,two),'Modified schedule is not deterministic: '+str(meta)+' repeat='+str(repeat)+' max='+str(float(np.abs(one-two).max()))+' mae='+str(float(np.abs(one-two).mean())))
            count=status(a)['neural_launches'];base=78 if config['weight_source']['model']=='small-fp8' else 154
            require(count==base-40+5*sum(repeats[:blocks]),'Native schedule launch count differs')
        checked('truncation, bypass, block repetition and full-schedule recovery')
        one=run(a,metadata={'reset':True,'passes':2});two=run(a,metadata={'reset':True,'passes':2})
        require(np.array_equal(one,two),'Whole-network passes retained stale workspace');checked('multiple whole-network passes with independent history')
        for _ in range(32):run(a,metadata={'reset':True,'adaptive_budget':{'enabled':True,'target_ms':.1,'minimum_blocks':1}})
        schedule=status(a)['network_schedule'];require(schedule['next_transformer_blocks']<8,'GPU budget never adjusted native depth')
        measurements['budget']=schedule;checked('GPU-event adaptive transformer budget')
        run(a,motion=zero)
        require(lib.phoenix_runtime_replace_model(a,json.dumps(config['weight_source']).encode())==0,lib.phoenix_runtime_error().decode())
        after=run(a,motion=zero,metadata={'reset':False})
        require(not status(a)['history_used'],'Model replacement retained incompatible history')
        checked('model replacement rebuilds immutable data and resets history')
        args.report.write_text(json.dumps({'schema':'phoenix-native-cuda-features/1','passed':True,'checks':checks,
            'measurements':measurements,'runtime_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
            'scope':'Explicit GPU behavior and deterministic recurrence; no perceptual temporal-quality or NVIDIA parity claim.'},indent=2)+'\n')
    finally:
        for handle in reversed(sessions):lib.phoenix_runtime_destroy(handle)


if __name__=='__main__':main()
