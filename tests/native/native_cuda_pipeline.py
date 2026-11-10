"""Explicit Windows GPU test of the CUDA leaf through the shared host SDK."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('package','config','image','report'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.package.resolve()
    runner=json.loads(args.config.read_text());runner.update(runtime_root=str(root),
        runtime_sha256=hashlib.sha256((root/'phoenix_native_cuda.dll').read_bytes()).hexdigest())
    source=np.asarray(Image.open(args.image).convert('RGBA'),dtype=np.float16)/np.float16(255)
    height,width,_=source.shape
    # Match the addon's spatial-to-shared transition, including unloading the
    # initial CUDA runtime before the SDK opens its neural leaf.
    initial=C.CDLL(str(root/'phoenix_native_cuda.dll'))
    initial.phoenix_runtime_create.argtypes=[C.c_char_p,C.c_uint,C.c_uint];initial.phoenix_runtime_create.restype=C.c_void_p
    initial.phoenix_runtime_destroy.argtypes=[C.c_void_p]
    warm=initial.phoenix_runtime_create(json.dumps(runner).encode(),width,height)
    assert warm
    initial.phoenix_runtime_destroy(warm)
    kernel=C.WinDLL('kernel32');kernel.FreeLibrary.argtypes=[C.c_void_p]
    assert kernel.FreeLibrary(initial._handle)
    del initial
    lib=C.CDLL(str(root/'phoenix_native.dll'))
    lib.phoenix_native_create.argtypes=[C.c_char_p,C.c_void_p,C.c_void_p];lib.phoenix_native_create.restype=C.c_void_p
    lib.phoenix_native_error.restype=C.c_char_p;lib.phoenix_native_destroy.argtypes=[C.c_void_p]
    lib.phoenix_native_set_tuning.argtypes=[C.c_void_p,C.c_char_p]
    lib.phoenix_native_frame_host.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t,C.c_size_t,C.c_void_p,C.c_size_t,C.c_size_t,C.c_void_p]
    lib.phoenix_native_status.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];lib.phoenix_native_status.restype=C.c_size_t
    rows=[]
    for motion in (False,True):
        config=dict(width=width,height=height,portable_runtime=runner,weight_source=runner['weight_source'],
            tuning=dict(local_tone=0.,local_structure=0.,auto_mask=False),
            pipeline=dict(MotionEstimation=motion,SceneDetection=motion,PipelineMonitoring=1))
        session=lib.phoenix_native_create(json.dumps(config).encode(),None,None)
        assert session,lib.phoenix_native_error()
        def frame():
            output=np.zeros_like(source)
            assert lib.phoenix_native_frame_host(session,source.ctypes.data,source.nbytes,width*8,None,0,0,
                output.ctypes.data,output.nbytes,width*8,None)==0,lib.phoenix_native_error()
            return output.astype(np.float32)
        try:
            for _ in range(20):low=frame()
            assert lib.phoenix_native_set_tuning(session,json.dumps(dict(local_tone=2.,local_structure=2.,auto_mask=False)).encode())==0,lib.phoenix_native_error()
            for _ in range(20):high=frame()
            difference=float(np.abs(low[...,:3]-high[...,:3]).mean())
            buffer=C.create_string_buffer(lib.phoenix_native_status(session,None,0));lib.phoenix_native_status(session,buffer,len(buffer))
            status=json.loads(buffer.value)
            rows.append(dict(estimated_motion=motion,conditioning_mae=difference,status=status))
            assert difference>.001,'Shared SDK dropped live conditioning: '+str(rows[-1])
        finally:
            assert lib.phoenix_native_destroy(session)==0,lib.phoenix_native_error()
    args.report.write_text(json.dumps(dict(passed=True,cases=rows,
        runtime_sha256=hashlib.sha256((root/'phoenix_native_cuda.dll').read_bytes()).hexdigest(),
        sdk_sha256=hashlib.sha256((root/'phoenix_native.dll').read_bytes()).hexdigest()),indent=2)+'\n')
    print('Shared SDK conditioning passed with and without estimated motion')


if __name__=='__main__':main()
