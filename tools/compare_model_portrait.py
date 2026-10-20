"""Capture original/FP8 on a native-resolution face fixture and verify NVIDIA parity."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from compare_four_models import ROOT, sha, write, telemetry
from report_four_models import metrics


OUT = ROOT / "artifacts/current/four-model-comparison/person-closeup"
WORK = ROOT / "build/four_model_comparison"
SOURCE = ROOT.parent / "dlss5-phoenix-gallery/docs/readme001/images/person11_face__beauty__off.jpg"
ROUTES = [("phoenix", "original"), ("nvidia", "original"),
          ("phoenix", "small-fp8"), ("nvidia", "small-fp8")]
CROPS = {"portrait": [570, 260, 1150, 1080], "face": [625, 440, 990, 950]}
SUBJECT = "man"
PROFILES = {
    "man": dict(folder="person-closeup", source="docs/readme001/images/person11_face__beauty__off.jpg",
                crops=CROPS, context="face and hood", label="Person close-up",
                observations="The original output has a cooler, more purple cast across the forehead, cheeks and lips. "
                "Lightweight FP8 looks warmer and slightly more contrasty in fine skin texture, especially around the forehead "
                "and eyes. Facial structure and the hood weave look very similar. The smaller model does not show a clear "
                "loss of facial detail in this example."),
    "woman": dict(folder="woman-closeup", source=ROOT / "artifacts/current/four-model-comparison/woman-closeup/input.png",
                  crops={"portrait": [730, 295, 1180, 1000], "face": [770, 415, 1135, 880]},
                  context="face and hair", label="Woman close-up",
                  observations="With skin diffusion disabled in the source renderer, both models retain visible eyebrows, "
                  "irises, lip edges and fine skin texture. The original looks darker and cooler across the cheek and nose "
                  "shadows; Lightweight FP8 looks brighter and slightly pinker. Facial shape and hair structure remain very similar."),
}


def capture():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    binaries = OUT.parent / "bin"
    executable, forwarder = binaries / "four_model_probe.exe", binaries / "phoenix-nvngx.dll"
    baseline = json.loads((OUT.parent / "run.json").read_text(encoding="utf-8"))
    assert sha(executable) == baseline["executable_sha256"]
    assert sha(forwarder) == baseline["forwarder_sha256"]
    model = ROOT.parent / "research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll"
    assert sha(model) == baseline["model_dll_sha256"]
    core = baseline["ngx_core"]
    assert sha(Path(core)) == baseline["ngx_core_sha256"]
    image = Image.open(SOURCE).convert("RGB")
    w, h = image.size
    assert (w, h) == (1920, 1080), "Do not resize the source fixture"
    rgba = np.ones((h, w, 4), np.float16)
    rgba[:, :, :3] = np.asarray(image, np.float32) / 255
    color, motion = WORK / "portrait.rgba16f", WORK / "portrait.rg16f"
    color.write_bytes(rgba.tobytes())
    motion.write_bytes(np.zeros((h, w, 2), np.float16).tobytes())
    Image.fromarray(np.uint8(np.clip(rgba[:, :, :3].astype(np.float32), 0, 1)*255+.5)).save(OUT / "input.png")
    source_render = OUT / "source-render.json"
    if SUBJECT == "woman":
        provenance = json.loads(source_render.read_text(encoding="utf-8"))
        assert provenance["source_only"] and not provenance["network_enabled"] and provenance["own_launches"] == 0
        assert provenance["source_skin_diffusion"] is False
        assert sha(SOURCE) == provenance["input_png_sha256"]
    receipt = dict(schema="phoenix-person-closeup/1", passed=False,
                   subject=SUBJECT,
                   started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   source_image=str(SOURCE), source_sha256=sha(SOURCE), resolution=[w, h],
                   input_sha256=sha(color), motion_sha256=sha(motion),
                   model_dll=str(model), model_dll_sha256=sha(model), ngx_core=core,
                   ngx_core_sha256=baseline["ngx_core_sha256"], executable_sha256=sha(executable),
                   forwarder_sha256=sha(forwarder), gpu_before=telemetry(), cases=[],
                   fixture="Native 1920x1080 SDR rendered character close-up; sRGB values in RGBA16F; zero RG16F motion; no depth/UI/mask guides",
                   controls=dict(auto_mask=True, inverted_depth=False, skin=-1, tone=1, structure=1, style=0),
                   full_graph=True, passes=1, repeats=[1]*8,
                   timing_scope="Short bounded capture runs only; use the parent report for performance comparisons",
                   games_launched=False, installed_files_modified=False)
    if SUBJECT == "woman":
        receipt.update(source_render_file=source_render.name, source_render_sha256=sha(source_render))
        receipt["fixture"] += "; gallery Beauty skin diffusion disabled to preserve face texture"
    for engine, model_id in ROUTES:
        name = f"{engine}-{model_id}"
        job, output = WORK / "portrait-job.json", OUT / (name + ".json")
        config = dict(width=w, height=h, engine=engine,
                      weight_source=dict(kind="nvidia_dll", path=str(model), sha256=receipt["model_dll_sha256"], model=model_id),
                      bundle_path=baseline["native_bundle"], ngx_core=core, forwarder=str(forwarder),
                      cache=str(WORK / "ngx-cache"), color_file=str(color), motion_file=str(motion),
                      capture=engine == "phoenix", debug=False, warmup_seconds=.1, measure_seconds=.1)
        write(job, config)
        write(OUT / "run.json", receipt)
        result = subprocess.run([str(executable), str(job), str(output)], cwd=WORK,
                                capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f"{name} failed: {result.stderr[-4000:]}")
        row = json.loads(output.read_text(encoding="utf-8"))
        assert row["passed"] and row["reset_repeatable"] and row["debug_errors"] == 0
        assert (row["engine"], row["model_id"], row["width"], row["height"]) == (engine, model_id, w, h)
        assert row["effective_controls"] == receipt["controls"]
        assert row["input_sha256"] == receipt["input_sha256"]
        assert row["motion_sha256"] == receipt["motion_sha256"]
        assert row["model_dll_sha256"] == receipt["model_dll_sha256"]
        receipt["cases"].append(dict(engine=engine, model=model_id, file=output.name))
        print(name, "PASS", flush=True)
    receipt.update(passed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), gpu_after=telemetry())
    write(OUT / "run.json", receipt)


def board(arrays, labels, box, filename, title):
    x0, y0, x1, y1 = box
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.8), layout="constrained", facecolor="#f5f5f5")
    for ax, rgb, label in zip(axes, arrays, labels):
        ax.imshow(np.clip(rgb[y0:y1, x0:x1], 0, 1), interpolation="nearest")
        ax.set_title(label, fontsize=15, weight="bold", pad=12)
        ax.axis("off")
    fig.suptitle(title, fontsize=18, weight="bold")
    fig.supxlabel("Same input and settings · 1080p reset frame · identical crop and display mapping", fontsize=11)
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)


def report():
    run = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    assert run["passed"]
    rows = {route: json.loads((OUT / f"{route[0]}-{route[1]}.json").read_text(encoding="utf-8")) for route in ROUTES}
    for row in rows.values():
        assert row["passed"] and row["reset_repeatable"] and row["debug_errors"] == 0
        assert row["effective_controls"] == run["controls"]
        for key in ("input_sha256", "motion_sha256", "model_dll_sha256"):
            assert row[key] == run[key]
    parity = {}
    for model in ("original", "small-fp8"):
        phoenix, nvidia = rows[("phoenix", model)], rows[("nvidia", model)]
        assert nvidia["render_preset"] == (1 if model == "original" else 2)
        assert nvidia["observed_cuda_launches"] == (156 if model == "original" else 80)
        parity[model] = {frame: phoenix["captures"][frame]["sha256"] == nvidia["captures"][frame]["sha256"]
                         for frame in ("reset_a", "history", "reset_b")}
        assert all(parity[model].values()), "Phoenix/NVIDIA outputs differ"
    w, h = run["resolution"]
    images = {}
    for model in ("original", "small-fp8"):
        for frame in ("reset_a", "history"):
            path = OUT / rows[("phoenix", model)]["captures"][frame]["file"]
            assert sha(path) == rows[("phoenix", model)]["captures"][frame]["sha256"]
            array = np.fromfile(path, np.float16).reshape(h, w, 4).astype(np.float32)
            assert np.all(np.isfinite(array))
            images[(model, frame)] = array[:, :, :3]
        Image.fromarray(np.uint8(np.clip(images[(model, "reset_a")], 0, 1)*255+.5)).save(OUT / f"{model}.png")
    source = np.asarray(Image.open(OUT / "input.png"), np.float32) / 255
    arrays = [source, images[("original", "reset_a")], images[("small-fp8", "reset_a")]]
    labels = ["Input", "Original", "Lightweight FP8"]
    board(arrays, labels, CROPS["face"], "face-comparison.png", "Original vs Lightweight FP8 · face close-up")
    board(arrays, labels, CROPS["portrait"], "portrait-comparison.png", "Original vs Lightweight FP8 · " + PROFILES[SUBJECT]["context"])
    differences = {}
    for name, box in {"full_frame": [0, 0, w, h], **CROPS}.items():
        x0, y0, x1, y1 = box
        differences[name] = {frame: metrics(images[("original", frame)][y0:y1, x0:x1], images[("small-fp8", frame)][y0:y1, x0:x1])
                             for frame in ("reset_a", "history")}
    summary = dict(schema="phoenix-person-closeup-results/1", passed=True, pixel_parity=True,
                   subject=SUBJECT,
                   visual_observations=PROFILES[SUBJECT]["observations"],
                   full_rgba_hash_matches=parity, nvidia_presets_verified_by_launch_count=True,
                   crops_xyxy=CROPS, original_vs_small=differences,
                   display="Same clipping to [0,1] and 8-bit rounding for each image; nearest-neighbor enlargement; no sharpening, exposure or color changes",
                   displayed_frame="reset_a", temporal_scope="Reset and one following zero-motion history frame; no moving-character sequence",
                   source_kind="Rendered character, not a photograph", model_weights_included=False)
    write(OUT / "summary.json", summary)
    lines = ["# Original vs Lightweight FP8: " + PROFILES[SUBJECT]["label"].lower(), "",
             "Fresh full-graph inference on the native 1920x1080 unprocessed rendered-character fixture. "
             "The gallery source files and all pre-existing room and man comparison outputs remain untouched. "
             "Source/model/executable hashes and controls are recorded in `run.json`.", "",
             "Phoenix and NVIDIA match byte-for-byte in full RGBA16F for each model on reset, following history, "
             "and repeated reset captures. NVIDIA presets 1/2 were independently checked through 156/80 observed CUDA launches. "
             "The figures therefore show one output per model; each represents both implementations.", "",
             "The images share the same input, neutral controls, automatic masking enabled, inverted depth disabled, "
             "zero motion, no supplied depth/mask/UI guides, and one full network pass. "
             "The short capture runs are not a replacement for the parent timing benchmark.", "",
             "Visual inspection of this fixture: " + PROFILES[SUBJECT]["observations"] + " This is a subjective "
             "observation of one rendered character, not a general quality ranking.", "",
             "## Face close-up", "", "![Face comparison](face-comparison.png)", "",
             f"Crop coordinates (left, top, right, bottom): `{CROPS['face']}`. Nearest-neighbor enlargement; "
             "same SDR display mapping for every panel. No sharpening or separate exposure/color correction.", "",
             "## Wider portrait", "", "![Portrait comparison](portrait-comparison.png)", "",
             f"Crop coordinates: `{CROPS['portrait']}`. Full-frame PNG outputs and raw FP16 reset/history captures are retained.", "",
             "Metrics use unclipped FP16 RGB values; neither model output is ground truth. "
             "This single static rendered person does not establish relative face accuracy, motion stability, "
             "or general quality across people and lighting.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    if SUBJECT == "woman":
        with (OUT / "README.md").open("a", encoding="utf-8") as stream:
            stream.write("\n## Source-render correction\n\nThe initial Beauty source blurred facial textures before either model ran. "
                         "A matched ablation with only skin diffusion disabled restored eyebrows, irises and skin detail. "
                         "The corrected comparison uses this unprocessed no-diffusion source. Full resolution and full character geometry "
                         "were already enabled; upscaling was not the cause. The gallery renderer and its installed build are unchanged. "
                         "`source-render.json` retains both source settings and provenance.\n\n![Source diagnosis](source-diagnosis.png)\n")
    manifest = {str(path.relative_to(OUT)).replace("\\", "/"): dict(bytes=path.stat().st_size, sha256=sha(path))
                for path in sorted(OUT.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    write(OUT / "manifest.json", dict(schema="phoenix-person-closeup-artifacts/1", model_weights_included=False, files=manifest))
    parent_readme = OUT.parent / "README.md"
    parent_text = parent_readme.read_text(encoding="utf-8")
    portrait_link = f"[{PROFILES[SUBJECT]['label']}: original vs Lightweight FP8]({OUT.name}/README.md)"
    if portrait_link not in parent_text:
        parent_readme.write_text(parent_text.rstrip() + "\n\n" + portrait_link + ".\n", encoding="utf-8")
    parent_manifest_path = OUT.parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_manifest["files"] = {
        path.relative_to(OUT.parent).as_posix(): dict(bytes=path.stat().st_size, sha256=sha(path))
        for path in sorted(OUT.parent.rglob("*")) if path.is_file() and path != parent_manifest_path}
    write(parent_manifest_path, parent_manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=PROFILES, default="man")
    parser.add_argument("--report-only", action="store_true", help="Regenerate figures without GPU execution")
    args = parser.parse_args()
    SUBJECT = args.subject
    profile = PROFILES[SUBJECT]
    OUT = ROOT / "artifacts/current/four-model-comparison" / profile["folder"]
    SOURCE = ROOT.parent / "dlss5-phoenix-gallery" / profile["source"]
    CROPS = profile["crops"]
    if not args.report_only:
        capture()
    report()
