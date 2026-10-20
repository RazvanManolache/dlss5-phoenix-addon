"""Render one unprocessed woman close-up with the existing gallery assets."""
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from test_model_selection import ROOT, toolchain, run
from compare_four_models import sha, write


def main():
    gallery = ROOT.parent / "dlss5-phoenix-gallery"
    work = ROOT / "build/four_model_comparison"
    out = ROOT / "artifacts/current/four-model-comparison/woman-closeup"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    compiler, env = toolchain(64)
    executable = work / "capture_portrait_source.exe"
    source = ROOT / "tools/capture_portrait_source.cpp"
    sdk = ROOT.parent / "external/Streamline/external/ngx-sdk/include"
    command = [compiler, "/nologo", "/std:c++17", "/O2", "/EHsc", "/MD", "/DNOMINMAX",
               "/I" + str(sdk), "/I" + str(gallery / "external/phoenix/third_party/nvapi"),
               "/sourceDependencies", str(work / "portrait-source-dependencies.json"), str(source),
               f"/Fo:{work / 'capture_portrait_source.obj'}", f"/Fe:{executable}", "/link", "/INCREMENTAL:NO",
               "d3d12.lib", "dxgi.lib", "d3dcompiler.lib", "user32.lib", "bcrypt.lib", "psapi.lib", "gdi32.lib",
               "windowscodecs.lib", "ole32.lib", "uxtheme.lib"]
    print(run(command, cwd=work, env=env), flush=True)
    shader = gallery / "gallery_desktop/ray_lighting.dxil"
    shutil.copyfile(shader, work / shader.name)
    raw, receipt = work / "woman-source.rgba16f", out / "source-render.json"
    print(run([executable, gallery, raw, receipt], cwd=work, env=env), flush=True)
    sys.path.insert(0, str(gallery))
    from analyze_detail import linear, picture
    selected_raw = work / "woman-no-diffusion.rgba16f"
    picture(linear(selected_raw)).save(out / "input.png")
    for name in ("woman-albedo", "woman-no-diffusion"):
        picture(linear(work / (name + ".rgba16f"))).save(work / (name + ".png"))
    blurred = json.loads(receipt.read_text(encoding="utf-8"))
    row = blurred.pop("no_diffusion_diagnostic")
    assert not row["network_enabled"] and row["own_launches"] == 0 and row["detail_flags"] == 8192
    row.update(source_only=True, presentation=False, windows_hidden=True, desktop_capture=False,
               source_skin_diffusion=False, blurred_reference=blurred,
               blurred_reference_raw_sha256=sha(raw),
               diagnosis="Beauty skin diffusion blurs textured facial color, including iris/eyebrow detail; existing no-diffusion ablation restores it before model inference")
    fig, axes = plt.subplots(1, 2, figsize=(9, 6.8), layout="constrained")
    for ax, path, label in zip(axes, (raw, selected_raw), ("Skin smoothing enabled", "Skin smoothing disabled")):
        ax.imshow(np.asarray(picture(linear(path)))[415:880, 770:1135], interpolation="nearest")
        ax.set_title(label, weight="bold")
        ax.axis("off")
    fig.suptitle("Source-render diagnosis · neither model applied", weight="bold")
    fig.savefig(out / "source-diagnosis.png", dpi=150)
    plt.close(fig)
    dependencies = json.loads((work / "portrait-source-dependencies.json").read_text(encoding="utf-8-sig"))["Data"]
    row.update(source=source.name, executable_sha256=sha(executable), raw_input_sha256=sha(selected_raw),
               input_png_sha256=sha(out / "input.png"), shader_sha256=sha(shader),
               conversion="Gallery analyze_detail.linear and picture: read linear FP16 RGB and encode sRGB",
               source_files={str(path): sha(Path(path)) for path in [dependencies["Source"], *dependencies["Includes"]]})
    write(receipt, row)
    print("Fresh 1080p woman source: zero neural launches, zero D3D12 errors", flush=True)


if __name__ == "__main__":
    main()
