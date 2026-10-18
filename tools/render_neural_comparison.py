"""Render measured Phoenix results, never synthetic predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
FONT=Path("C:/Windows/Fonts/segoeui.ttf")
def font(size): return ImageFont.truetype(str(FONT),size)
def rgb(array): return Image.fromarray(np.uint8(np.clip(array,0,1)*255+.5))
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(a,b):
    delta=np.abs(a-b)
    return dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),exact=bool(np.array_equal(a,b)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("directory",type=Path)
    args=parser.parse_args();dest=args.directory
    load=lambda name: np.load(dest/(name+".npz"))
    baseline=load("baseline")["output"];candidate=load("candidate")["output"]
    native=load("native_baseline")["output"];nc=load("native_compat")["output"]
    controls=load("native_controls")
    job=json.loads((dest/"candidate-job.json").read_text())
    photo=np.asarray(Image.open(job["image"]).convert("RGB").resize(tuple(job["extent"]),Image.Resampling.LANCZOS),np.float32)/255
    rgb(photo).save(dest/"input.png")
    colors={"bg":"#101721","fg":"#e5eef9","muted":"#adbbce","accent":"#71d9cd"}
    canvas=Image.new("RGB",(1232,806),colors["bg"]);draw=ImageDraw.Draw(canvas)
    draw.text((24,15),"Phoenix: actual full-model GPU output",font=font(28),fill=colors["fg"])
    draw.text((24,55),"RTX 5080 · 384 × 256 SDR proxy · all 8 blocks · one pass · reset frame",font=font(17),fill=colors["muted"])
    cells=[("Input",photo,"Source photo; no model"),
        ("Current native / candidate default",native[0],"V1 compatibility: exact across all 3 frames"),
        ("Surface controls",controls["masked"],"Left third: intensity 0; right: tone .4 / structure .25"),
        ("Natural style",controls["natural"],"Neural style input + recovered display grade"),
        ("Cinematic style",controls["cinematic"],"Neural style input + recovered display grade"),
        ("Protected HUD",controls["ui"],"Premultiplied overlay; opaque text preserved")]
    for i,(title,values,caption) in enumerate(cells):
        x=24+(i%3)*400;y=94+(i//3)*346
        draw.text((x,y),title,font=font(18),fill=colors["accent"])
        canvas.paste(rgb(values),(x,y+32))
        if i==2:
            draw.line((x+128,y+32,x+128,y+287),fill="#71d9cd",width=2)
            draw.line((x+192,y+32,x+192,y+287),fill="#71d9cd",width=1)
        draw.text((x,y+294),caption,font=font(12),fill=colors["muted"])
    draw.text((24,782),"Experimental Phoenix contract. These are Phoenix comparisons; no NVIDIA-output or game-quality certification.",font=font(14),fill=colors["muted"])
    canvas.save(dest/"comparison.png")

    frame=2;delta=np.abs(baseline[frame]-candidate[frame]);heat=np.max(delta,axis=2)
    heat=np.stack([np.clip(heat/.04,0,1),np.clip((heat-.01)/.03,0,1),np.zeros_like(heat)],axis=2)
    canvas=Image.new("RGB",(1232,433),colors["bg"]);draw=ImageDraw.Draw(canvas)
    draw.text((24,14),"Portable temporal resolve: current vs candidate",font=font(26),fill=colors["fg"])
    m=metrics(baseline[frame],candidate[frame])
    draw.text((24,53),f"Third translated frame · mean difference {m['mean_abs']:.6f} · max {m['max_abs']:.6f} (0–1 RGB)",font=font(17),fill=colors["muted"])
    for i,(title,values) in enumerate([("Current: unit history gain",baseline[frame]),("Candidate: model gain 0.739746",candidate[frame]),("Absolute difference (amplified)",heat)]):
        x=24+i*400;draw.text((x,88),title,font=font(18),fill=colors["accent"]);canvas.paste(rgb(values),(x,120))
    draw.text((24,393),"The correction changes temporal blending. Difference from the old output is measured; improved quality needs a reference.",font=font(15),fill=colors["muted"])
    canvas.save(dest/"temporal-comparison.png")

    canvas=Image.new("RGB",(1232,701),colors["bg"]);draw=ImageDraw.Draw(canvas)
    draw.text((24,16),"What happens between Phoenix model calls",font=font(28),fill=colors["fg"])
    def box(x,y,w,h,title,lines):
        draw.rounded_rectangle((x,y,x+w,y+h),12,fill="#1c2b3a",outline=colors["accent"],width=1)
        draw.text((x+14,y+12),title,font=font(19),fill=colors["accent"])
        for j,line in enumerate(lines):draw.text((x+14,y+44+j*23),line,font=font(16),fill=colors["fg"])
    def arrow(x,y,xx,yy):
        draw.line((x,y,xx,yy),fill=colors["accent"],width=3)
        if xx>x:draw.polygon(((xx,yy),(xx-9,yy-6),(xx-9,yy+6)),fill=colors["accent"])
        else:draw.polygon(((xx,yy),(xx-6,yy-9),(xx+6,yy-9)),fill=colors["accent"])
    box(24,78,366,140,"1. Validate and prepare",["RGB + motion/depth/validity + map", "Optional field → canonical coordinates", "Warp accepted previous half history"])
    box(432,78,366,140,"2. Pack 16 input features",["RGB / noise / warped RGB + validity", "G × tone, B × structure", "Map mode disables automatic skin inputs"])
    box(840,78,366,140,"3. Encoder → global → decoder",["Multi-scale features + encoder skips", "Eight central blocks may repeat", "Head returns RGB correction + gate"])
    arrow(390,148,432,148);arrow(798,148,840,148)
    box(840,274,366,140,"4. Temporal resolve",["Correct current RGB; blend valid history", "Portable gate uses model blend_scale", "Store half RGB for the next frame"])
    arrow(1023,218,1023,274)
    box(432,274,366,140,"5. Optional whole-network pass",["Feed resolved RGB into the next pass", "Same weights; separate per-pass history", "No extra diffusion sampler or VAE"])
    draw.line((840,344,798,344),fill=colors["accent"],width=3)
    box(24,274,366,140,"6. Return to display domain",["Inverse field sampling if provided", "Recovered style grade; intensity × map R", "UI alpha protects original/backbuffer"])
    draw.line((432,344,390,344),fill=colors["accent"],width=3)
    box(24,468,774,128,"Output and next accepted frame",["Display operations do not enter recurrent history.","Async helper returns raw prediction; current UI/map are applied after reprojection.","Object selection comes from engine IDs/materials, then the RGBA map producer."])
    arrow(207,414,207,468)
    draw.text((24,638),"Repeating blocks, whole-network passes, and recurrence across video frames are separate mechanisms.",font=font(18),fill=colors["fg"])
    draw.text((24,671),"Learned blocks have computational roles above; assigning specific objects/materials to each layer requires ablation evidence.",font=font(14),fill=colors["muted"])
    canvas.save(dest/"pipeline.png")
    result={"schema":"phoenix-neural-comparison/1","fixture":"SDR proxy, not a rendered game",
        "full_model_inference":True,"gpu":"RTX 5080","extent":job["extent"],
        "repeats":[1]*8,"passes":1,"vendor_parity":False,
        "native_legacy_compatibility":metrics(native,nc),
        "portable_legacy_compatibility":metrics(baseline,load("legacy_compat")["output"]),
        "portable_cuda_vs_vulkan":metrics(candidate,load("cuda_candidate")["output"]),
        "native_new_default_vs_old_reset":metrics(native[0],controls["default"]),
        "portable_migration_per_frame":[metrics(a,b) for a,b in zip(baseline,candidate)],
        "input_source_sha256":digest(Path(job["image"])),
        "feature_checks":{name:json.loads((dest/(name+"controls.json")).read_text())["checks"] for name in ("","native_","cuda_")},
        "transactions":json.loads((dest/"transaction-checks.json").read_text()),
        "standalone_replay":json.loads((dest/"cli-checks.json").read_text()),
        "transport":{b:json.loads((dest/("transport-"+b+".json")).read_text()) for b in ("x64","x86")}}
    (dest/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:result[key] for key in ("native_legacy_compatibility","portable_legacy_compatibility","portable_cuda_vs_vulkan","native_new_default_vs_old_reset")},indent=2))

if __name__=="__main__":main()
