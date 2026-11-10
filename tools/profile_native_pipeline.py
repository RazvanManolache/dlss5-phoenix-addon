"""Explicit GPU benchmark of the complete shared Windows host-frame call."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('package','config','image','report'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--samples',type=int,default=20)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--host-transfer',action='store_true',help='Force the compatibility transport for an A/B comparison')
    args=parser.parse_args();root=args.package.resolve()
    assert 1<=args.warmup<=100 and 3<=args.samples<=100
    runner=json.loads(args.config.read_text());runner.update(runtime_root=str(root),runtime_sha256=hashlib.sha256((root/'phoenix_native_cuda.dll').read_bytes()).hexdigest())
    if args.host_transfer:runner['d3d12_interop']=False
    source=np.asarray(Image.open(args.image).convert('RGBA'),dtype=np.float16)/np.float16(255)
    height,width,_=source.shape;output=np.empty_like(source)
    lib=C.CDLL(str(root/'phoenix_native.dll'))
    lib.phoenix_native_create.argtypes=[C.c_char_p,C.c_void_p,C.c_void_p];lib.phoenix_native_create.restype=C.c_void_p
    lib.phoenix_native_error.restype=C.c_char_p;lib.phoenix_native_destroy.argtypes=[C.c_void_p]
    lib.phoenix_native_frame_host.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t,C.c_size_t,C.c_void_p]
    lib.phoenix_native_status.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];lib.phoenix_native_status.restype=C.c_size_t
    config=dict(width=width,height=height,portable_runtime=runner,weight_source=runner['weight_source'],
        tuning=dict(local_tone=1.,local_structure=1.,auto_mask=False),
        pipeline=dict(MotionEstimation=True,SceneDetection=True,PipelineMonitoring=1))
    session=lib.phoenix_native_create(json.dumps(config).encode(),None,None)
    assert session,lib.phoenix_native_error()
    timings=[];hashes=[];first=None
    try:
        for index in range(args.warmup+args.samples):
            started=time.perf_counter_ns()
            result=lib.phoenix_native_frame_host(session,source.ctypes.data,source.nbytes,width*8,None,0,0,
                output.ctypes.data,output.nbytes,width*8,None)
            elapsed=(time.perf_counter_ns()-started)*1e-6
            assert result==0,lib.phoenix_native_error()
            assert np.isfinite(output).all(),'Nonfinite SDK output'
            if index==0:first=output.copy()
            if index>=args.warmup:
                timings.append(elapsed);hashes.append(hashlib.sha256(output.tobytes()).hexdigest())
        buffer=C.create_string_buffer(lib.phoenix_native_status(session,None,0));lib.phoenix_native_status(session,buffer,len(buffer));status=json.loads(buffer.value)
    finally:
        assert lib.phoenix_native_destroy(session)==0,lib.phoenix_native_error()
    comparison=None
    if args.reference:
        baseline=json.loads(args.reference.read_text())
        assert baseline['input_sha256']==hashlib.sha256(source.tobytes()).hexdigest(),'Benchmark input differs'
        assert baseline['warmup']==args.warmup and len(baseline['samples_ms'])==args.samples,'Benchmark sequence differs'
        with np.load(args.reference.with_suffix('.npz')) as reference:
            comparison={}
            for name,actual in [('reset',first),('temporal',output)]:
                difference=np.abs(actual.astype(np.float32)-reference[name].astype(np.float32))
                comparison[name]=dict(exact=bool(np.array_equal(actual,reference[name])),maximum=float(difference.max()),mae=float(difference.mean()))
        assert all(x['exact'] for x in comparison.values()),'Transport changed fixed-input output: '+str(comparison)
        comparison['all_measured_frames_exact']=hashes==baseline['output_sha256']
        assert comparison['all_measured_frames_exact'],'Intermediate temporal output changed'
    args.report.parent.mkdir(parents=True,exist_ok=True)
    if not args.reference:np.savez_compressed(args.report.with_suffix('.npz'),reset=first,temporal=output)
    report=dict(schema='phoenix-native-pipeline-profile/1',passed=True,width=width,height=height,
        warmup=args.warmup,samples_ms=timings,median_ms=statistics.median(timings),p95_ms=float(np.percentile(timings,95)),
        output_sha256=hashes,comparison=comparison,status=status,
        input_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        runtime_sha256=runner['runtime_sha256'],sdk_sha256=hashlib.sha256((root/'phoenix_native.dll').read_bytes()).hexdigest(),
        scope='Complete synchronous SDK host call with optical flow and temporal history; excludes game capture/presentation. Fixed input, unmanaged desktop GPU clocks.')
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':True,'median_ms':report['median_ms'],'p95_ms':report['p95_ms'],'comparison':comparison}))


if __name__=='__main__':main()
