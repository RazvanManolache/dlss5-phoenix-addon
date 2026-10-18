"""Explicit GPU acceptance for boundary transactions; no vendor-parity claim."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from compare_neural_controls import configure, advanced_request, Frame3, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("portable_job", type=Path)
    parser.add_argument("native_job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.portable_job.read_text())
    native_job = json.loads(args.native_job.read_text())
    api = configure(job["runtime"])
    api.phoenix_runtime_frame_v3.argtypes = [C.c_void_p, C.POINTER(Frame3)]
    api.phoenix_runtime_replace_model.argtypes = [C.c_void_p, C.c_char_p]
    h, w = 48, 64
    image = np.linspace(.1, .8, h*w*3, dtype=np.float32).reshape(h,w,3)
    mask = np.ones((h,w,4),np.float32)
    mask[:,w//2:,1:3] = .5
    motion = np.zeros((h,w,2),np.float32)
    report = {"gpu_used": True, "full_graph": True, "vendor_parity": False}

    def frame(session, reset=False, alias=False):
        color=image.copy(); out=color if alias else np.empty_like(color)
        request=advanced_request(color,out,{"reset":reset,"has_motion":True,
            "controls":{"sampling_mode":"advancing"}},mask=mask,motion=motion)
        if api.phoenix_runtime_frame_v3(session,C.byref(request)):
            raise RuntimeError(api.phoenix_runtime_error().decode())
        return out.copy()

    def open_session(config):
        session=api.phoenix_runtime_create(json.dumps(config).encode(),w,h)
        if not session: raise RuntimeError(api.phoenix_runtime_error().decode())
        return session

    session=open_session(job["config"])
    try:
        fresh=frame(session,True)
        continued=frame(session)
        frame(session,True)
        bad=dict(job["config"]["weight_source"],sha256="0"*64)
        assert api.phoenix_runtime_replace_model(session,json.dumps(bad).encode()) == -1
        assert np.array_equal(continued,frame(session)),"failed model switch changed continuation"
        assert api.phoenix_runtime_replace_model(session,json.dumps(job["config"]["weight_source"]).encode()) == 0
        assert status(api,session)["state"]=="model_replaced"
        assert np.array_equal(fresh,frame(session)),"successful model switch did not reset history/noise"
        assert np.array_equal(fresh,frame(session,True,True)),"owned input/output alias changed result"
        report.update(failed_model_switch_preserves_continuation=True,
            successful_model_switch_resets_history=True,bounded_input_output_alias_exact=True)
    finally: api.phoenix_runtime_destroy(session)
    config=dict(job["config"],device_features=False)
    session=open_session(config)
    try:
        packed=frame(session,True)
        report["host_vs_device_map_max_error"]=float(np.max(np.abs(packed-fresh)))
        assert np.array_equal(packed,fresh),"host/device map features differ"
    finally: api.phoenix_runtime_destroy(session)

    native=C.CDLL(native_job["runtime"])
    native.phoenix_native_create.argtypes=[C.c_char_p,C.c_void_p,C.c_void_p]
    native.phoenix_native_create.restype=C.c_void_p
    native.phoenix_native_reconfigure.argtypes=[C.POINTER(C.c_void_p),C.c_char_p]
    native.phoenix_native_frame_host_v2.argtypes=[C.c_void_p,C.POINTER(Frame3)]
    native.phoenix_native_destroy.argtypes=[C.c_void_p]
    native.phoenix_native_error.restype=C.c_char_p
    session=C.c_void_p(native.phoenix_native_create(json.dumps(native_job["config"]).encode(),None,None))
    if not session.value: raise RuntimeError(native.phoenix_native_error().decode())
    try:
        original=session.value
        invalid=dict(native_job["config"],width=0)
        assert native.phoenix_native_reconfigure(C.byref(session),json.dumps(invalid).encode())==-1
        assert session.value==original
        resized=dict(native_job["config"],width=320,height=320)
        if native.phoenix_native_reconfigure(C.byref(session),json.dumps(resized).encode()):
            raise RuntimeError(native.phoenix_native_error().decode())
        color=np.linspace(.1,.8,320*320*3,dtype=np.float32).reshape(320,320,3)
        out=np.empty_like(color);zero=np.ones((320,320,4),np.float32);zero[:,:,0]=0
        request=advanced_request(color,out,{"reset":True},mask=zero)
        if native.phoenix_native_frame_host_v2(session,C.byref(request)):
            raise RuntimeError(native.phoenix_native_error().decode())
        error=float(np.max(np.abs(out-color)))
        assert error<=1/4096,"resized zero intensity region exceeded half tolerance"
        report.update(invalid_native_resize_retains_session=True,native_resize=[320,320],
            native_resized_zero_mask_max_error=error)
    finally:
        assert native.phoenix_native_destroy(session)==0
    report["passed"]=True
    destination=Path(job["output"])/"transaction-checks.json"
    destination.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
