"""One isolated portable GPU timing case for the resolution sweep."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from compare_neural_controls import configure,advanced_request,Frame3,status

def stats(values):
    a=np.asarray(values)
    return dict(samples=len(a),mean_ms=float(a.mean()),median_ms=float(np.median(a)),
        p95_ms=float(np.percentile(a,95)),min_ms=float(a.min()),max_ms=float(a.max()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("job",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--warmup",type=float,default=.75);parser.add_argument("--measure",type=float,default=2.)
    parser.add_argument("--cap",type=float,default=15.)
    args=parser.parse_args();job=json.loads(args.job.read_text());w,h=job["extent"]
    if args.warmup<=0 or args.measure<=0 or args.cap<args.measure:parser.error("Invalid timing windows")
    api=configure(job["runtime"])
    if job["advanced"]:api.phoenix_runtime_frame_v3.argtypes=[C.c_void_p,C.POINTER(Frame3)]
    t=time.perf_counter();session=api.phoenix_runtime_create(json.dumps(job["config"]).encode(),w,h)
    if not session:raise RuntimeError(api.phoenix_runtime_error().decode())
    create_ms=(time.perf_counter()-t)*1000
    image=np.empty((h,w,3),np.float32)
    image[:,:,0]=.125+(np.arange(w,dtype=np.float32)[None,:]%128)/256
    image[:,:,1]=.25+(np.arange(h,dtype=np.float32)[:,None]%128)/256
    image[:,:,2]=.5
    motion=np.zeros((h,w,2),np.float32);output=np.empty_like(image)
    mask=ui=field=None
    if job["advanced"]:
        mask=np.ones((h,w,4),np.float32);mask[:,:w//3,0]=0;mask[:,w//2:,1]=.4;mask[:,w//2:,2]=.25
        ui=np.zeros((h,w,4),np.float32);ui[:h//10,w//16:w//3]=1
        field=np.zeros((h,w,4),np.float32);field[:,:,0]=-1/w;field[:,:,2]=1/w
    metadata=dict(reset=True,has_motion=True,controls=dict(auto_mask=True,sampling_mode="fixed",style=1 if job["advanced"] else 0))
    def call():
        text=json.dumps(metadata).encode()
        request=advanced_request(image,output,metadata,mask=mask,ui=ui,distortion=field,motion=motion) if job["advanced"] else None
        t=time.perf_counter()
        code=api.phoenix_runtime_frame_v3(session,C.byref(request)) if request is not None else api.phoenix_runtime_frame(
            session,image.ctypes.data,motion.ctypes.data,None,None,text,output.ctypes.data)
        ms=(time.perf_counter()-t)*1000
        if code:raise RuntimeError(api.phoenix_runtime_error().decode())
        return ms
    try:
        cold=call();metadata["reset"]=False
        warm=0;t=time.perf_counter()
        while warm<4 or time.perf_counter()-t<args.warmup:call();warm+=1
        walls=[];gpu=[];stages=[];t=time.perf_counter()
        while (len(walls)<12 or time.perf_counter()-t<args.measure) and time.perf_counter()-t<args.cap:
            walls.append(call());row=status(api,session)
            gpu.append(row["operator_ms"])
            stages.append({key:row.get(key,0) for key in ("canonicalize_ms","presentation_ms","prepare_ms","execute_with_transfers_ms","resolve_ms")})
        duration=time.perf_counter()-t;row=status(api,session)
        assert np.isfinite(output).all()
        report=dict(passed=True,extent=[w,h],route=job["route"],advanced=job["advanced"],
            create_ms=create_ms,first_call_ms=cold,warmup_frames=warm,warmup_seconds_min=args.warmup,
            measured_duration_seconds=duration,measured_seconds_min=args.measure,measured_seconds_cap=args.cap,minimum_samples=12,
            wall=stats(walls),gpu=stats(gpu),wall_samples_ms=walls,gpu_samples_ms=gpu,
            gpu_timing_scope="neural backend GPU operator events only; excludes host preparation/resolve and display",
            wall_scope="complete synchronous host-plane frame API; includes upload/readback, bounded snapshots and host presentation",
            stages_mean_ms={key:float(np.mean([v[key] for v in stages])) for key in stages[0]},
            finite_output=True,last_output_sha256=hashlib.sha256(output.tobytes()).hexdigest(),runtime_status=row)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(f"{job['route']} {w}x{h}: GPU {report['gpu']['median_ms']:.2f} ms, call {report['wall']['median_ms']:.2f} ms",flush=True)
    finally:api.phoenix_runtime_destroy(session)

if __name__=="__main__":main()
