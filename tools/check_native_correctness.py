"""Explicit full-model GPU reset/continuation checks; never a timing benchmark."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_neural_controls import NativeControls

def check(api,config,w,h,resets,continuations):
    config=dict(config,width=w,height=h)
    color=np.empty((h,w,4),np.uint16)
    x=np.arange(w,dtype=np.uint32)[None,:];y=np.arange(h,dtype=np.uint32)[:,None]
    color[:,:,0]=0x3000+(x//8%16)*0x40;color[:,:,1]=0x3400+(y//8%8)*0x40
    color[:,:,2]=0x3800+((x+y)//16%8)*0x20;color[:,:,3]=0x3c00
    motion=np.zeros((h,w,2),np.uint16);output=np.empty_like(color)
    session=api.phoenix_native_create(json.dumps(config).encode(),None,None)
    if not session:raise RuntimeError(api.phoenix_native_error().decode())
    row=dict(extent=[w,h],passed=False,reset_hashes=[],reset_differences=[],frames=0)
    try:
        def evaluate(reset):
            controls=NativeControls(C.sizeof(NativeControls),int(reset),1,1,1,1,-1,1,0)
            if api.phoenix_native_frame_host(session,color.ctypes.data,color.nbytes,w*8,
                motion.ctypes.data,motion.nbytes,w*4,output.ctypes.data,output.nbytes,w*8,C.byref(controls)):
                raise RuntimeError(api.phoenix_native_error().decode())
            if not np.isfinite(output.view(np.float16)).all():raise RuntimeError('Nonfinite native output')
            row['frames']+=1
        for trial in range(resets):
            if trial:
                for _ in range(continuations):evaluate(False)
            evaluate(True)
            row['reset_hashes'].append(hashlib.sha256(output.tobytes()).hexdigest())
            if trial==0:first=output.copy()
            else:
                delta=np.abs(output.view(np.float16)[:,:,:3].astype(np.float32)-first.view(np.float16)[:,:,:3].astype(np.float32))
                row['reset_differences'].append(dict(exact=bool(np.array_equal(first,output)),
                    max_abs=float(delta.max()),mean_abs=float(delta.mean())))
        row['passed']=all(d['exact'] for d in row['reset_differences'])
        if not row['passed']:row['error']='Identical resets are not bitwise repeatable'
    except Exception as error:row['error']=str(error)
    finally:
        if api.phoenix_native_destroy(session):raise RuntimeError(api.phoenix_native_error().decode())
    return row

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resolutions',nargs='+',default=['1023x767','1024x768','1025x769',
        '1920x1080','2560x1440','3440x1440','3840x1600','3839x2159','3840x2160','3841x2161','7680x4320'])
    parser.add_argument('--resets',type=int,default=3)
    parser.add_argument('--continuations',type=int,default=8)
    parser.add_argument('--sessions',type=int,default=1)
    args=parser.parse_args()
    if args.resets<2 or args.continuations<0 or args.sessions<1:parser.error('Need >=2 resets, >=0 continuations and >=1 sessions')
    config=json.loads(args.config.read_text());config=config.get('config',config)
    api=C.CDLL(str(args.runtime.resolve()))
    api.phoenix_native_create.argtypes=[C.c_char_p,C.c_void_p,C.c_void_p];api.phoenix_native_create.restype=C.c_void_p
    api.phoenix_native_destroy.argtypes=[C.c_void_p];api.phoenix_native_error.restype=C.c_char_p
    api.phoenix_native_frame_host.argtypes=[C.c_void_p]+[t for _ in range(3) for t in (C.c_void_p,C.c_size_t,C.c_size_t)]+[C.POINTER(NativeControls)]
    report=dict(passed=False,completed=False,full_graph=True,network_passes=1,
        timing_benchmark=False,runtime_sha256=hashlib.sha256(args.runtime.read_bytes()).hexdigest(),
        resets_per_session=args.resets,continuations_between_resets=args.continuations,cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for resolution in args.resolutions:
        w,h=map(int,resolution.split('x'))
        for session in range(args.sessions):
            row=check(api,config,w,h,args.resets,args.continuations);row['session']=session
            report['cases'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(resolution,'session',session,'PASS' if row['passed'] else 'FAIL',row.get('error',''),flush=True)
    report['completed']=True;report['passed']=all(c['passed'] for c in report['cases'])
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()
