"""Quantify reset-hash flags separately from GPU-resident timing runs."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_neural_controls import NativeControls
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'artifacts/current/neural-performance'
def differences(a,b):
    left=a.view(np.float16)[:,:,:3].astype(np.float32)
    right=b.view(np.float16)[:,:,:3].astype(np.float32)
    delta=np.abs(left-right)
    return dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),changed_rgb_components=int(np.count_nonzero(delta)),
        changed_pixels=int(np.count_nonzero(np.any(delta!=0,axis=2))),rgb_components=int(delta.size),exact=bool(np.array_equal(a,b)))
def main():
    run=json.loads((DEST/'run.json').read_text())
    sizes=[(1920,1080)]+[tuple(map(int,key.split('x'))) for key,row in run['native'].items()
        if row.get('candidate',{}).get('passed') and not row.get('legacy_reset_output_exact',False)]
    report=dict(schema='phoenix-native-reset-repeatability/1',timing_benchmark=False,
        scope='Identical synthetic FP16 scene and zero motion; three explicit reset frames separated by eight accepted continuation frames per DLL/session; host transport used only for readback validation',cases=[])
    for w,h in sizes:
        color=np.empty((h,w,4),np.uint16);x=np.arange(w,dtype=np.uint32)[None,:];y=np.arange(h,dtype=np.uint32)[:,None]
        color[:,:,0]=0x3000+(x//8%16)*0x40;color[:,:,1]=0x3400+(y//8%8)*0x40
        color[:,:,2]=0x3800+((x+y)//16%8)*0x20;color[:,:,3]=0x3c00
        motion=np.zeros((h,w,2),np.uint16);output=np.empty_like(color);firsts={};case=dict(extent=[w,h],routes={})
        for label,folder in [('old','sdk/windows-x64/bin'),('candidate','neural-candidate/windows-x64/bin')]:
            path=ROOT/'artifacts/current'/folder/'phoenix_native.dll';api=C.CDLL(str(path))
            api.phoenix_native_create.argtypes=[C.c_char_p,C.c_void_p,C.c_void_p];api.phoenix_native_create.restype=C.c_void_p
            api.phoenix_native_destroy.argtypes=[C.c_void_p];api.phoenix_native_error.restype=C.c_char_p
            frame=api.phoenix_native_frame_host;frame.argtypes=[C.c_void_p]+[t for _ in range(3) for t in (C.c_void_p,C.c_size_t,C.c_size_t)]+[C.POINTER(NativeControls)]
            config=json.loads((ROOT/'artifacts/current/neural-candidate/native-rtx.json').read_text());config.update(width=w,height=h)
            session=api.phoenix_native_create(json.dumps(config).encode(),None,None)
            if not session:raise RuntimeError(api.phoenix_native_error().decode())
            row=dict(runtime_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hashes=[],reset_vs_first=[])
            try:
                for i in range(3):
                    if i:
                        continuation=NativeControls(C.sizeof(NativeControls),0,1,1,1,1,-1,1,0)
                        for _ in range(8):
                            if frame(session,color.ctypes.data,color.nbytes,w*8,motion.ctypes.data,motion.nbytes,w*4,
                                output.ctypes.data,output.nbytes,w*8,C.byref(continuation)):raise RuntimeError(api.phoenix_native_error().decode())
                    controls=NativeControls(C.sizeof(NativeControls),1,1,1,1,1,-1,1,0)
                    if frame(session,color.ctypes.data,color.nbytes,w*8,motion.ctypes.data,motion.nbytes,w*4,
                        output.ctypes.data,output.nbytes,w*8,C.byref(controls)):raise RuntimeError(api.phoenix_native_error().decode())
                    assert np.isfinite(output.view(np.float16)).all()
                    row['hashes'].append(hashlib.sha256(output.tobytes()).hexdigest())
                    if i==0:firsts[label]=output.copy()
                    else:row['reset_vs_first'].append(differences(firsts[label],output))
                case['routes'][label]=row
            finally:assert api.phoenix_native_destroy(session)==0
        case['old_vs_candidate_first_reset']=differences(firsts['old'],firsts['candidate'])
        report['cases'].append(case);(DEST/'repeatability.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'{w}x{h}',json.dumps(case['old_vs_candidate_first_reset']),flush=True)
    report['completed']=True;(DEST/'repeatability.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
