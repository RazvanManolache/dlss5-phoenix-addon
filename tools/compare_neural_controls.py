"""Full-model before/after fixture runner. Input pixels are an explicit SDR proxy.

Runs a bounded three-frame camera translation through the unchanged SDK and the
candidate. The baseline is never regenerated from the candidate implementation.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw

class Plane(C.Structure):
    _fields_=[("data",C.c_void_p),("capacity",C.c_uint64),("stride",C.c_uint64),
              ("width",C.c_uint32),("height",C.c_uint32),("channels",C.c_uint32),("reserved",C.c_uint32)]
class Frame2(C.Structure):
    _fields_=([("size",C.c_uint32),("version",C.c_uint32),("profile",C.c_uint32),("reserved",C.c_uint32)]+
        [(name,Plane) for name in ("color","motion","depth","validity","mask","output")]+[("metadata",C.c_char_p)])
class Frame3(C.Structure):
    _fields_=[("base",Frame2)]+[(name,Plane) for name in ("ui","alpha","backbuffer","distortion")]
class NativeControls(C.Structure):
    _fields_=[("size",C.c_uint32),("reset",C.c_uint32)]+[(n,C.c_float) for n in
        ("motion_x","motion_y","tone","structure","skin")]+[("automatic",C.c_uint32),("inverted",C.c_uint32)]

def run_native_legacy(runtime,config,image,destination,name):
    api=C.CDLL(str(runtime));create=api.phoenix_native_create
    create.argtypes=[C.c_char_p,C.c_void_p,C.c_void_p];create.restype=C.c_void_p
    api.phoenix_native_destroy.argtypes=[C.c_void_p]
    api.phoenix_native_error.restype=C.c_char_p
    api.phoenix_native_status.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];api.phoenix_native_status.restype=C.c_size_t
    evaluate=api.phoenix_native_frame_host
    evaluate.argtypes=[C.c_void_p]+[t for _ in range(3) for t in (C.c_void_p,C.c_size_t,C.c_size_t)]+[C.POINTER(NativeControls)]
    session=create(json.dumps(config).encode(),None,None)
    if not session:raise RuntimeError(api.phoenix_native_error().decode())
    h,w=image.shape[:2];motion=np.zeros((h,w,2),np.float16);motion[:,:,0]=-2
    frames=[];rows=[]
    try:
        for i in range(3):
            rgba=np.ones((h,w,4),np.float16);rgba[:,:,:3]=np.roll(image,i*2,axis=1)
            output=np.empty_like(rgba);controls=NativeControls(C.sizeof(NativeControls),i==0,1,1,1,1,-1,1,0)
            start=time.perf_counter()
            result=evaluate(session,rgba.ctypes.data,rgba.nbytes,w*8,motion.ctypes.data,motion.nbytes,w*4,
                output.ctypes.data,output.nbytes,w*8,C.byref(controls))
            if result:raise RuntimeError(api.phoenix_native_error().decode())
            assert np.isfinite(output).all()
            frames.append(output[:,:,:3].astype(np.float32))
            text=C.create_string_buffer(api.phoenix_native_status(session,None,0));api.phoenix_native_status(session,text,len(text))
            row=json.loads(text.value);row["call_ms"]=(time.perf_counter()-start)*1000;rows.append(row)
            print(name,i,round(row["call_ms"],2),"ms",flush=True)
    finally:api.phoenix_native_destroy(session)
    np.savez_compressed(destination/(name+".npz"),output=np.asarray(frames))
    Image.fromarray(np.uint8(np.clip(frames[-1],0,1)*255+.5)).save(destination/(name+".png"))
    (destination/(name+".json")).write_text(json.dumps(dict(frames=rows,runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
        config=config,full_model_inference=True,vendor_reference=False),indent=2)+"\n")


def view(array):
    if array is None:return Plane()
    h,w=array.shape[:2]
    channels=array.shape[2] if array.ndim==3 else 1
    assert array.dtype==np.float32 and array.flags.c_contiguous
    return Plane(array.ctypes.data,array.nbytes,0,w,h,channels,0)


def advanced_request(image,output,metadata,**planes):
    request=Frame3()
    request.base.size=C.sizeof(request);request.base.version=3
    request.base.color=view(image);request.base.output=view(output)
    request.base.metadata=json.dumps(metadata).encode()
    for name,plane in planes.items():
        if name in ("mask","motion","depth","validity"):setattr(request.base,name,view(plane))
        else:setattr(request,name,view(plane))
    request.base.profile=int(planes.get("mask") is not None)
    return request


def run_advanced(runtime,config,image,destination,native=False):
    if native:
        api=C.CDLL(str(runtime))
        create=api.phoenix_native_create;create.argtypes=[C.c_char_p,C.c_void_p,C.c_void_p];create.restype=C.c_void_p
        destroy=api.phoenix_native_destroy;destroy.argtypes=[C.c_void_p]
        evaluate=api.phoenix_native_frame_host_v2
        error=api.phoenix_native_error;error.restype=C.c_char_p
        get_status=api.phoenix_native_status;get_status.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];get_status.restype=C.c_size_t
        session=create(json.dumps(config).encode(),None,None)
    else:
        api=configure(runtime);create=api.phoenix_runtime_create;destroy=api.phoenix_runtime_destroy
        evaluate=api.phoenix_runtime_frame_v3;error=api.phoenix_runtime_error;get_status=api.phoenix_runtime_status
        session=create(json.dumps(config).encode(),image.shape[1],image.shape[0])
    evaluate.argtypes=[C.c_void_p,C.POINTER(Frame3)]
    if not session:raise RuntimeError(error().decode())
    h,w=image.shape[:2]
    mask=np.ones((h,w,4),np.float32);mask[:,:w//3,0]=0;mask[:,w//2:,1]=.4;mask[:,w//2:,2]=.25
    zero_mask=mask.copy();zero_mask[:,:,0]=0
    overlay=Image.new("RGBA",(w,h));draw=ImageDraw.Draw(overlay)
    draw.rectangle((12,8,min(180,w-1),34),fill=(38,38,38,204))
    draw.text((20,14),"HUD 01 - protected",fill=(255,255,255,255))
    draw.rectangle((min(166,w-14),14,min(174,w-6),22),fill=(255,255,255,255))
    ui=np.ascontiguousarray(np.asarray(overlay,np.float32)/255)
    ui[:,:,:3]*=ui[:,:,3:4]
    alpha=ui[:,:,3].copy();bb=np.ascontiguousarray(ui[:,:,:3]+image*(1-alpha[:,:,None]))
    identity=np.zeros((h,w,4),np.float32)
    translation=identity.copy();translation[:,:,0]=-1/w;translation[:,:,2]=1/w
    cases=[("default",{},{}),("natural",{"style":1},{}),("cinematic",{"style":2},{}),
        ("masked",{},dict(mask=mask)),("mask_zero",{},dict(mask=zero_mask)),
        ("ui",{},dict(ui=ui)),("ui_alpha",{},dict(alpha=alpha,backbuffer=bb)),
        ("identity",{},dict(distortion=identity)),("translated",{},dict(distortion=translation))]
    results={};rows=[]
    try:
        for name,controls,planes in cases:
            output=np.empty_like(image)
            metadata={"reset":True,"controls":{"auto_mask":True,**controls}}
            request=advanced_request(image,output,metadata,**planes)
            start=time.perf_counter();code=evaluate(session,C.byref(request))
            if code:raise RuntimeError(name+": "+error().decode())
            assert np.isfinite(output).all(),name
            text=C.create_string_buffer(get_status(session,None,0));get_status(session,text,len(text))
            report=json.loads(text.value);report.update(case=name,call_ms=(time.perf_counter()-start)*1000)
            results[name]=output.copy();rows.append(report)
            prefix="native_" if native else (config["backend"]+"_" if config["backend"]!="vulkan" else "")
            Image.fromarray(np.uint8(np.clip(output,0,1)*255+.5)).save(destination/(prefix+name+".png"))
            print("native" if native else "portable",name,round(report["call_ms"],2),"ms",flush=True)
    finally:
        destroy(session)
    prefix="native_" if native else (config["backend"]+"_" if config["backend"]!="vulkan" else "")
    np.savez_compressed(destination/(prefix+"controls.npz"),**results)
    checks={"mask_zero_max_error":float(np.max(np.abs(results["mask_zero"]-image))),
        "protected_region_max_error":float(np.max(np.abs(results["masked"][:,:w//3]-image[:,:w//3]))),
        "identity_max_error":float(np.max(np.abs(results["identity"]-results["default"]))),
        "opaque_ui_max_error":float(np.max(np.abs(results["ui"][ui[:,:,3]==1]-1))),
        "ui_overlay_vs_alpha_backbuffer_max_error":float(np.max(np.abs(results["ui"]-results["ui_alpha"])))}
    (destination/(prefix+"controls.json")).write_text(json.dumps(dict(frames=rows,checks=checks,
        runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),config=config,
        vendor_parity=False,full_model_inference=True),indent=2)+"\n")


def configure(path):
    api = C.CDLL(str(path))
    api.phoenix_runtime_create.argtypes = [C.c_char_p, C.c_uint, C.c_uint]
    api.phoenix_runtime_create.restype = C.c_void_p
    api.phoenix_runtime_destroy.argtypes = [C.c_void_p]
    api.phoenix_runtime_frame.argtypes = [C.c_void_p] + [C.c_void_p] * 4 + [C.c_char_p, C.c_void_p]
    api.phoenix_runtime_error.restype = C.c_char_p
    api.phoenix_runtime_status.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    api.phoenix_runtime_status.restype = C.c_size_t
    return api


def status(api, session):
    text = C.create_string_buffer(api.phoenix_runtime_status(session, None, 0))
    api.phoenix_runtime_status(session, text, len(text))
    return json.loads(text.value)


def run(runtime, config, image, destination, name):
    destination.mkdir(parents=True, exist_ok=True)
    api = configure(runtime)
    height, width = image.shape[:2]
    start = time.perf_counter()
    session = api.phoenix_runtime_create(json.dumps(config).encode(), width, height)
    if not session:
        raise RuntimeError(api.phoenix_runtime_error().decode())
    outputs, reports = [], []
    try:
        motion = np.zeros((height, width, 2), np.float32)
        motion[:, :, 0] = -2
        for frame in range(3):
            color = np.ascontiguousarray(np.roll(image, frame * 2, axis=1))
            output = np.empty_like(color)
            metadata = {"reset": frame == 0, "has_motion": True,
                        "controls": {"seed": 0, "sampling_mode": "fixed", "auto_mask": True}}
            tick = time.perf_counter()
            code = api.phoenix_runtime_frame(session, color.ctypes.data, motion.ctypes.data,
                None, None, json.dumps(metadata).encode(), output.ctypes.data)
            if code:
                raise RuntimeError(api.phoenix_runtime_error().decode())
            assert np.isfinite(output).all()
            report = status(api, session)
            report["call_ms"] = (time.perf_counter() - tick) * 1000
            reports.append(report)
            outputs.append(output.copy())
            print(name, frame, round(report["call_ms"], 2), "ms", flush=True)
    finally:
        api.phoenix_runtime_destroy(session)
    np.savez_compressed(destination / (name + ".npz"), output=np.asarray(outputs))
    Image.fromarray(np.uint8(np.clip(outputs[-1], 0, 1) * 255 + .5)).save(destination / (name + ".png"))
    receipt = {"runtime_sha256": hashlib.sha256(Path(runtime).read_bytes()).hexdigest(),
        "config": config, "frames": reports, "elapsed_ms": (time.perf_counter() - start) * 1000,
        "full_model_inference": True, "reference_producer": "unchanged SDK Phoenix" if name == "baseline" else "candidate Phoenix",
        "vendor_reference": False}
    (destination / (name + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("name")
    parser.add_argument("--advanced",action="store_true")
    parser.add_argument("--native",action="store_true")
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    image = Image.open(job["image"]).convert("RGB").resize(tuple(job["extent"]), Image.Resampling.LANCZOS)
    image=np.ascontiguousarray(np.asarray(image,np.float32)/255)
    if args.advanced:
        run_advanced(Path(job["runtime"]),job["config"],image,Path(job["output"]),args.native)
    else:
        (run_native_legacy if args.native else run)(Path(job["runtime"]), job["config"], image, Path(job["output"]), args.name)


if __name__ == "__main__":
    main()
