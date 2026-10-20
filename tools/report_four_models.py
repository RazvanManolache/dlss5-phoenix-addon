"""Summarize matched timings and RGB outputs; no GPU execution."""
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/current/four-model-comparison"


def metrics(a, b):
    difference = np.abs(a.astype(np.float32) - b.astype(np.float32))
    mse = float(np.mean(difference ** 2))
    return dict(mean_abs=float(difference.mean()), mean_abs_8bit=float(difference.mean()*255),
        max_abs=float(difference.max()), rmse=math.sqrt(mse), psnr_db=-10*math.log10(mse) if mse else None,
        exact_fraction=float(np.mean(difference == 0)), p99_abs=float(np.quantile(difference,.99)))


def main():
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    assert run["passed"]
    routes = [("phoenix","original"),("nvidia","original"),("phoenix","small-fp8"),("nvidia","small-fp8")]
    labels = ["Phoenix original", "NVIDIA original", "Phoenix lightweight", "NVIDIA lightweight"]
    colors = ["#137f99", "#3f6077", "#ef9b43", "#ad6035"]
    rows = []
    matched_inputs = {}
    for case in run["cases"]:
        row = json.loads((OUT / case["file"]).read_text())
        assert row["passed"] and row["reset_repeatable"] and row["debug_errors"] == 0
        key = (row["width"],row["height"])
        identity = (row["input_sha256"],row["motion_sha256"],row["model_dll_sha256"],json.dumps(row["effective_controls"],sort_keys=True))
        assert matched_inputs.setdefault(key,identity) == identity
        if row["engine"] == "nvidia":
            assert row["render_preset"] == (1 if row["model_id"] == "original" else 2)
            assert row["observed_cuda_launches"] == (156 if row["model_id"] == "original" else 80)
        rows.append(dict(engine=row["engine"],model=row["model_id"],width=row["width"],height=row["height"],
            gpu_median_ms=row["gpu"]["median_ms"],gpu_p95_ms=row["gpu"]["p95_ms"],gpu_mean_ms=row["gpu"]["mean_ms"],
            wall_median_ms=row["wall"]["median_ms"],samples=row["gpu"]["samples"],create_ms=row["create_ms"],
            reset_repeatable=True,finite=True))
    w,h=1920,1080
    source=np.asarray(Image.open(OUT/"input.png"),np.float32)/255
    images={}
    for engine,model in routes:
        for frame in ("reset_a","history"):
            path=OUT/f"{engine}-{model}-{w}x{h}-{frame}.rgba16f"
            images[(engine,model,frame)]=np.fromfile(path,np.float16).reshape(h,w,4)[:,:,:3].astype(np.float32)
    comparisons=[]
    hash_comparisons=[]
    for size in dict.fromkeys((r["width"],r["height"]) for r in rows):
        for model in ("original","small-fp8"):
            a,b=[json.loads((OUT/f"{engine}-{model}-{size[0]}x{size[1]}.json").read_text()) for engine in ("phoenix","nvidia")]
            hash_comparisons.append(dict(model=model,width=size[0],height=size[1],
                frame_matches={frame:a["captures"][frame]["sha256"]==b["captures"][frame]["sha256"] for frame in ("reset_a","history","reset_b")}))
    parity=all(all(row["frame_matches"].values()) for row in hash_comparisons)
    for model in ("original","small-fp8"):
        for frame in ("reset_a","history"):
            a,b=images[("phoenix",model,frame)],images[("nvidia",model,frame)]
            comparisons.append(dict(model=model,frame=frame,**metrics(a,b)))
    model_difference={engine:metrics(images[(engine,"small-fp8","reset_a")],images[(engine,"original","reset_a")]) for engine in ("phoenix","nvidia")}
    temporal={f"{engine}-{model}":metrics(images[(engine,model,"history")],images[(engine,model,"reset_a")]) for engine,model in routes}
    effects={f"{engine}-{model}":metrics(images[(engine,model,"reset_a")],source) for engine,model in routes}
    for engine,model in routes:
        rgb=images[(engine,model,"reset_a")]
        Image.fromarray(np.uint8(np.clip(rgb,0,1)*255+.5)).save(OUT/f"{engine}-{model}.png")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    resolutions=list(dict.fromkeys((row["width"],row["height"]) for row in rows))
    fig,axes=plt.subplots(1,len(resolutions),figsize=(13,4.7),layout="constrained")
    for ax,size in zip(np.atleast_1d(axes),resolutions):
        values=[next(r for r in rows if (r["engine"],r["model"],r["width"],r["height"])==(*route,*size))["gpu_median_ms"] for route in routes]
        ax.barh(labels[::-1],values[::-1],color=colors[::-1],height=.65)
        for y,value in enumerate(values[::-1]): ax.text(value+.04,y,f"{value:.3f}",va="center",fontsize=10)
        ax.set_xlim(0,max(values)*1.25);ax.set_xlabel("Median GPU time (ms)\nLower is faster");ax.set_title(f"{size[0]} x {size[1]}",weight="bold");ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("Both models, both implementations · RTX 5080",fontsize=16,weight="bold")
    fig.savefig(OUT/"timings.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,7.5),layout="constrained")
    panels=[("Shared input",source)]+[(label,images[(*route,"reset_a")]) for route,label in zip(routes,labels)]
    for ax,(label,array) in zip(axes.flat,panels):ax.imshow(np.clip(array,0,1));ax.set_title(label,weight="bold");ax.axis("off")
    axes.flat[-1].axis("off");axes.flat[-1].text(.02,.75,"1080p reset frame\nSame FP16 input and neutral controls\nSDR rendered museum scene proxy; zero motion\nNumerical differences are reported separately\nNo game, depth, masks or UI guides",va="top",linespacing=1.8)
    fig.savefig(OUT/"results.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,7.3),layout="constrained")
    crop=(slice(250,700),slice(600,1200))
    for row,model in enumerate(("original","small-fp8")):
        a,b=images[("phoenix",model,"reset_a")],images[("nvidia",model,"reset_a")]
        for col,(label,array) in enumerate((("Phoenix",a),("NVIDIA",b))):axes[row,col].imshow(np.clip(array[crop],0,1));axes[row,col].set_title(f"{label} {model}");axes[row,col].axis("off")
        diff=np.max(np.abs(a-b),axis=2);im=axes[row,2].imshow(diff[crop],vmin=0,vmax=.05,cmap="magma");axes[row,2].set_title("Maximum RGB error per pixel (0–0.05)");axes[row,2].axis("off");fig.colorbar(im,ax=axes[row,2],shrink=.8)
    fig.savefig(OUT/"differences.png",dpi=150);plt.close(fig)
    summary=dict(schema="phoenix-nvidia-four-model-results/1",passed=True,gpu=run["gpu_before"],timings=rows,
        comparison_resolution=[w,h],pixel_comparisons=comparisons,original_vs_small=model_difference,
        temporal_change=temporal,input_effect=effects,all_reset_outputs_repeatable=True,all_outputs_finite=True,
        nvidia_presets_verified_by_launch_count=True,matched_inputs_verified=True,matched_controls_verified=True,
        pixel_parity=parity,full_rgba_hash_comparisons=hash_comparisons,fixture=run["fixture"],
        scope="Production native RTX graph recorder vs NVIDIA feature 18; shared queue/inputs/timestamp scope; no game FPS or perceptual superiority claim")
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    lines=["# Phoenix and NVIDIA: both models","",f"GPU: `{run['gpu_before']}`.","",
        "Both models use the same 310.9.16 DLL. NVIDIA preset 1 produced 156 CUDA launches; preset 2 produced 80. Phoenix ran the full original 154-launch or smaller 78-launch production graph, plus matching output transport. One network pass, all eight central blocks once. All four routes passed a D3D12 debug smoke and finite/repeatable-reset checks. Automatic masking is explicitly enabled and inverted depth explicitly disabled on both implementations; neither route inherits these settings from a constructor default.","",
        "GPU timestamps bracket each complete engine recording and the same output copy, in a single submitted command list. Upload/readback, initialization, warmup, and CPU gaps are excluded. Wall timings include command recording and a synchronous queue fence. The DLL/input hashes, samples and telemetry are retained in individual receipts. This measures the production graph recorder; it is not the end-to-end addon/game frame rate.","",
        "| Implementation | Model | Resolution | GPU median ms | GPU p95 ms | Wall median ms | Samples |","|---|---|---|---:|---:|---:|---:|"]
    for r in rows: lines.append(f"| {r['engine']} | {r['model']} | {r['width']}x{r['height']} | {r['gpu_median_ms']:.3f} | {r['gpu_p95_ms']:.3f} | {r['wall_median_ms']:.3f} | {r['samples']} |")
    lines += ["","![Timings](timings.png)","","## Output comparison","","Same resized SDR rendered museum scene proxy, neutral controls and zero motion. The 1080p captures include a reset frame and its following history frame. RGBA16F data and displayed PNGs are retained. Pixel metrics use RGB before display clipping. Differences between learned models are expected; neither output is ground truth.","","| Model | Frame | Mean absolute RGB difference (8-bit levels) | Maximum difference (0–1) | PSNR dB |","|---|---|---:|---:|---:|"]
    for r in comparisons:
        psnr=f"{r['psnr_db']:.2f}" if r['psnr_db'] is not None else "exact"
        lines.append(f"| {r['model']} | {r['frame']} | {r['mean_abs_8bit']:.3f} | {r['max_abs']:.5f} | {psnr} |")
    conclusion=("Both Phoenix models match their corresponding NVIDIA model byte-for-byte (full RGBA16F) for reset, history, and repeated reset captures at all three resolutions." if parity else "Some Phoenix/NVIDIA captures differ; inspect the full hash comparisons and RGB metrics.")
    lines += ["",conclusion,"","The initial report was invalid: it enabled automatic masking for NVIDIA but left it disabled for Phoenix, then attributed the resulting differences to implementation parity. The corrected harness sets and records the same controls explicitly. The affected runs, images, and numerical conclusions have been replaced by this corrected comparison. See `harness-correction.json`.","","This static zero-motion rendered museum scene test does not validate game-motion quality, disocclusion, HDR, supplied masks/UI, or visual superiority.","","![Outputs](results.png)","","![Crops and error maps](differences.png)",""]
    if (OUT/"person-closeup/README.md").exists():
        lines += ["[Person close-up: original vs Lightweight FP8](person-closeup/README.md).", ""]
    if (OUT/"woman-closeup/README.md").exists():
        lines += ["[Woman close-up: original vs Lightweight FP8](woman-closeup/README.md).", ""]
    (OUT/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(dict(passed=True,timings=len(rows),comparisons=comparisons)))


if __name__=="__main__":main()
