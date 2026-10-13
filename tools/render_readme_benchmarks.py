"""Export the checked-in benchmark snapshot as README images (Pillow required).

These are static chart exports of measured data, not browser screenshots.
Run from any working directory: python tools/render_readme_benchmarks.py
"""

import argparse
import datetime
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--snapshot", type=Path, default=ROOT / "docs/benchmarks/2026-10-13")
OUT = parser.parse_args().snapshot.resolve()
DATA = json.loads((OUT / "measurements.json").read_text(encoding="utf-8"))
ROWS = {row["id"]: row for row in DATA["records"]}
MIB = 1024**2
BG, PANEL, LINE = "#0e1420", "#161f2d", "#2b3a50"
INK, MUTED, GREEN, BLUE, GOLD = "#e9f0f9", "#acbbce", "#79dbb4", "#93bcff", "#f5c879"
COLORS = [GREEN, BLUE, GOLD, "#c29aff", "#ef9bb0", "#68c9e1"]


def measured_date(row):
    return datetime.date.fromisoformat(row.get("measured_date", DATA["date"]))


def font(size, bold=False):
    """Prefer the report's Windows font; allow regeneration on Linux too."""
    names = (
        ["C:/Windows/Fonts/segoeuib.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold
        else ["C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


class Panel:
    def __init__(self, title, subtitle, height, date=None):
        self.image = Image.new("RGB", (1600, height), BG)
        self.draw = ImageDraw.Draw(self.image)
        date = date or datetime.date.fromisoformat(DATA["date"]).strftime("%d %b %Y").upper()
        self.text(48, 28, f"DLSS5 PHOENIX  /  LOCAL BENCHMARK  /  {date}", 21, GREEN, True)
        self.text(48, 67, title, 44, INK, True)
        self.text(48, 130, subtitle, 25, MUTED)

    def text(self, x, y, text, size=26, color=INK, bold=False):
        self.draw.text((x, y), str(text), font=font(size, bold), fill=color)

    def right(self, x, y, text, size=26, color=INK, bold=False):
        face = font(size, bold)
        self.draw.text((x - self.draw.textlength(text, font=face), y), text, font=face, fill=color)

    def box(self, left, top, right, bottom):
        self.draw.rounded_rectangle((left, top, right, bottom), radius=15, fill=PANEL, outline=LINE, width=2)

    def wrap(self, x, y, text, width, size=24, color=MUTED):
        words, line = text.split(), ""
        for word in words:
            trial = (line + " " + word).strip()
            if line and self.draw.textlength(trial, font=font(size)) > width:
                self.text(x, y, line, size, color)
                y += size + 9
                line = word
            else:
                line = trial
        if line:
            self.text(x, y, line, size, color)
        return y + size + 9

    def save(self, name):
        self.image.save(OUT / name, optimize=True)


def native():
    phoenix, nvidia = ROWS["phoenix"], ROWS["nvidia"]
    p = Panel("Native RTX: near NVIDIA at 1080p", "RTX 5080 · driver 616.64 · GPU inference latency; lower is better", 1040, measured_date(phoenix).strftime("%d %b %Y").upper())
    for left, row, color in [(48, phoenix, GREEN), (815, nvidia, BLUE)]:
        p.box(left, 196, left + 735, 396)
        p.text(left + 28, 216, "Phoenix native RTX" if row["id"] == "phoenix" else "NVIDIA DLSS NR", 29, color, True)
        p.text(left + 28, 258, f"{row['core_ms']:.3f} ms", 66, INK, True)
        p.text(left + 28, 350, f"Process-median range: {row['process_median_min_ms']:.3f}–{row['process_median_max_ms']:.3f} ms", 23, MUTED)
    p.box(48, 424, 810, 836)
    p.text(76, 447, "Matched benchmark extents", 28, INK, True)
    p.text(76, 499, "Resolution", 23, MUTED)
    p.right(480, 499, "Phoenix", 23, GREEN)
    p.right(625, 499, "NVIDIA", 23, BLUE)
    p.right(780, 499, "Δ latency", 23, MUTED)
    for index, row in enumerate(row for row in DATA["resolution_matrix"] if row["width"] >= 1280):
        y = 551 + index * 48
        p.text(76, y, f"{row['width']} × {row['height']}", 26)
        p.right(480, y, f"{row['phoenix']['gpu_ms']:.3f}", 26)
        p.right(625, y, f"{row['nvidia']['gpu_ms']:.3f}", 26)
        p.right(780, y, f"{row['phoenix_latency_difference_percent']:+.2f}%", 25)
    p.box(835, 424, 1550, 836)
    p.text(863, 447, "1080p process GPU-local residency", 28, INK, True)
    for index, (row, color) in enumerate([(phoenix, GREEN), (nvidia, BLUE)]):
        y = 509 + index * 91
        label = "Phoenix" if index == 0 else "NVIDIA"
        value = row["device_local_bytes"] / MIB
        p.text(863, y, label, 25, color)
        p.right(1522, y, f"{value:.1f} MiB", 27, INK, True)
        p.draw.rounded_rectangle((863, y + 40, 863 + 650 * value / 750, y + 64), radius=5, fill=color)
    saving = 100 * (1 - phoenix["device_local_bytes"] / nvidia["device_local_bytes"])
    p.text(863, 712, f"{saving:.1f}% less in this matched probe", 28, GREEN, True)
    p.wrap(48, 865, "Two fresh processes per renderer at every extent; 96 measured frames each. Average of process medians. Identical RGBA16F color / RG16F motion, controls and model; D3D12 debug enabled for both. Upload/readback excluded. Inference timings, not game FPS.", 1500, 24)
    p.wrap(48, 979, "Residency includes the benchmark harness and driver. NVIDIA's internal memory breakdown is unknown.", 1500, 23)
    p.save("native-rtx.png")


def resolution_scaling(rows, title, filename):
    """Use shared linear axes so latency and memory scaling stay comparable."""
    bottom = 325 + len(rows) * 153
    p = Panel(title, "Phoenix / NVIDIA DLSS NR · same fixture and model · exact reset + final outputs", bottom + 180)
    p.box(48, 190, 790, bottom)
    p.box(815, 190, 1550, bottom)
    p.text(76, 214, "GPU inference · ms", 30, INK, True)
    p.text(843, 214, "Process GPU-local residency · MiB", 29, INK, True)
    max_time = max(row[route]["gpu_ms"] for row in rows for route in ("phoenix", "nvidia")) * 1.22
    max_memory = max(row[route]["device_local_bytes"] / MIB for row in rows for route in ("phoenix", "nvidia")) * 1.22
    for index, row in enumerate(rows):
        top = 282 + index * 153
        for left in (76, 843):
            p.text(left, top, f"{row['width']} × {row['height']}", 26, INK, True)
        for offset, (route, label, color) in enumerate((("phoenix", "Phoenix", GREEN), ("nvidia", "NVIDIA", BLUE))):
            y = top + 43 + offset * 39
            for left, key, divisor, maximum in ((76, "gpu_ms", 1, max_time), (843, "device_local_bytes", MIB, max_memory)):
                value = row[route][key] / divisor
                p.text(left, y - 4, label, 21, color)
                start = left + 100
                end = start + 490 * value / maximum
                p.draw.rectangle((start, y + 3, end, y + 24), fill=color)
                p.text(end + 10, y - 4, f"{value:.2f}" if divisor == 1 else f"{value:,.1f}", 23)
    p.wrap(48, bottom + 30, "RTX 5080 16 GiB · driver 616.64 · Ryzen 9950X3D · 96 GiB RAM. Two processes per renderer and extent; 128 frames each, discard 32. Means of process medians; clocks and scheduling introduce variation.", 1500, 24)
    p.wrap(48, bottom + 110, "GPU inference excludes upload/readback. Residency includes harness/driver. Synthetic validation; not gameplay FPS or moving-scene quality.", 1500, 23)
    p.save(filename)


def resolution_report():
    """A local, dependency-free HTML view of the same checked-in measurements."""
    rows = DATA["resolution_matrix"]
    table = ""
    options = ""
    for row in rows:
        key = f"{row['width']}x{row['height']}"
        options += f'<option value="{key}">{row["width"]} × {row["height"]}</option>'
        values = [f'{row["width"]} × {row["height"]}',
                  f'{row["phoenix"]["gpu_ms"]:.3f}', f'{row["nvidia"]["gpu_ms"]:.3f}',
                  f'{row["phoenix"]["device_local_bytes"] / MIB:,.1f}',
                  f'{row["nvidia"]["device_local_bytes"] / MIB:,.1f}', 'Exact reset + final']
        table += '<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DLSS5 Phoenix — resolution benchmarks</title>
<style>body{background:#0e1420;color:#e9f0f9;font:18px Segoe UI,system-ui;margin:32px auto;padding:0 24px;max-width:1600px}
a{color:#93bcff}h1{color:#79dbb4}img{max-width:100%;height:auto;border-radius:12px}table{width:100%;border-collapse:collapse}
td,th{padding:13px;text-align:left;border-bottom:1px solid #2b3a50}select{background:#161f2d;color:#e9f0f9;padding:10px;font:inherit}
.scroll{overflow:auto}.muted{color:#acbbce}section{margin:36px 0}</style>
<h1>DLSS5 Phoenix: resolution benchmarks</h1>
<p>13 October 2026 · RTX 5080 16 GiB · NVIDIA 616.64 · Ryzen 9 9950X3D, 32 threads · 96 GiB RAM.</p>
<p>Identical synthetic RGBA16F color / RG16F motion, model and controls; full eight blocks, one pass.
Two fresh processes per renderer and extent, 128 frames each; first 32 discarded. Figures average the process medians.
GPU inference excludes upload/readback; GPU residency includes harness and driver. These are offline checks, not game FPS.</p>
<section class="scroll"><table><thead><tr><th>Resolution</th><th>Phoenix ms</th><th>NVIDIA ms</th>
<th>Phoenix GPU MiB</th><th>NVIDIA GPU MiB</th><th>Output</th></tr></thead><tbody>''' + table + '''</tbody></table></section>
<p class="muted">Reset and frame 127 match exactly at every listed extent. This fixture does not certify moving-scene quality.
Timing ranges, host memory and numerical receipts: <a href="measurements.json">measurements</a> ·
<a href="native-resolution-validation.json">validation</a> · <a href="README.md">methodology and limitations</a>.</p>
<section><img src="native-resolution-scaling.png" alt="Latency and GPU memory from 720p through 8K"></section>
<section><img src="native-lower-resolutions.png" alt="Lower-resolution latency and GPU memory"></section>
<section><h2>Actual output comparisons</h2><p>Final frame 127; fixed RGB range 0–1, no automatic exposure or contrast.
Left: Phoenix. Middle: NVIDIA. Right: absolute difference ×8. Synthetic fixture, not a game screenshot.</p>
<label for="extent">Resolution </label><select id="extent">''' + options + '''</select>
<p><img id="comparison" alt="Phoenix, NVIDIA and amplified difference" src="output-''' + f'{rows[0]["width"]}x{rows[0]["height"]}' + '''.png"></p></section>
<script>document.getElementById('extent').addEventListener('change',event=>{
document.getElementById('comparison').src='output-'+event.target.value+'.png';});</script></html>'''
    (OUT / "resolutions.html").write_text(page, encoding="utf-8")


def portable():
    p = Panel("The portability lab: experimental alternate runners", "1920 × 1080 · separate from native RTX · current results; lower is better", 1040)
    p.box(48, 190, 1550, 837)
    for x, label in [(76, "Runner / tested device"), (750, "Model ms"), (1005, "Full frame ms"), (1260, "Graph MiB"), (1520, "Host RAM MiB")]:
        if x == 76:
            p.text(x, 214, label, 24, MUTED)
        else:
            p.right(x, 214, label, 24, MUTED)
    names = [
        ("cuda", "CUDA C++", "RTX 5080"),
        ("vulkan_nvidia", "Vulkan C++", "RTX 5080"),
        ("cpu", "CPU C++", "Ryzen 9950X3D · 32 workers"),
        ("hip_amd", "HIP C++", "AMD gfx1036 integrated GPU"),
        ("vulkan_amd", "Vulkan C++", "AMD gfx1036 integrated GPU"),
        ("sycl_cpu", "SYCL C++", "Ryzen CPU · Intel CPU runtime"),
    ]
    for index, (identifier, name, hardware) in enumerate(names):
        row, y, color = ROWS[identifier], 275 + index * 91, COLORS[index]
        p.text(76, y, name, 28, color, True)
        p.text(76, y + 37, hardware + " · " + measured_date(row).strftime("%d %b"), 22, MUTED)
        for x, value in [(750, row["core_ms"]), (1005, row["wall_ms"]), (1260, row["owned_bytes"] / MIB), (1520, row["host_working_set_bytes"] / MIB)]:
            p.right(x, y + 5, f"{value:,.2f}", 27)
        # A common logarithmic axis keeps both the sub-second and 30-second routes visible.
        length = 650 * (math.log10(row["wall_ms"]) - 2) / (math.log10(40000) - 2)
        p.draw.rectangle((620, y + 49, 620 + length, y + 58), fill=color)
    p.wrap(48, 866, "Thin bars show complete-frame latency on a shared logarithmic scale. Full frame includes preparation, transfers, inference, resolve and broker round trip. Host RAM is worker resident working set; graph allocations and host RAM overlap and must not be added.", 1500, 24)
    gpu, cpu = ROWS["cuda"], ROWS["cpu"]
    p.wrap(48, 971, f"RTX: {gpu['processes']} × {gpu['samples'] // gpu['processes']} measured temporal frames. CPU/HIP: {cpu['processes']} × {cpu['samples'] // cpu['processes']} measured reset frames. Older SYCL/Vulkan AMD rows retain their dates. CPU uses host timing; discrete AMD/Intel GPUs untested.", 1500, 23)
    p.save("portable-runners.png")


def memory():
    dates = [datetime.date.fromisoformat(DATA.get("direct_memory_date", ROWS["phoenix"]["measured_date"]))]
    date_range = dates[0].strftime("%d %b %Y") if len(dates) == 1 else dates[0].strftime("%d %b") + " – " + dates[-1].strftime("%d %b %Y")
    p = Panel("Memory: shared work, less duplication", "Native full adapter · 1920 × 1080 · owned resources, separate from process residency", 1030, date_range.upper())
    p.box(48, 190, 790, 820)
    p.text(76, 214, "One-pass adapter: 493.94 MiB owned", 30, INK, True)
    parts = [("Weights", 140.9375), ("Workspace", 203.125), ("Temporal history", 33.75), ("Bridge images", 115.8125), ("Exposure / readback", 0.3125)]
    total, angle = sum(value for _, value in parts), -90
    for index, (_, value) in enumerate(parts):
        end = angle + value / total * 360
        p.draw.pieslice((92, 285, 416, 609), angle, end, fill=COLORS[index])
        angle = end
    p.draw.ellipse((174, 367, 334, 527), fill=PANEL)
    p.text(199, 415, "493.9", 37, INK, True)
    for index, (name, value) in enumerate(parts):
        y = 300 + index * 58
        p.draw.rectangle((449, y + 8, 463, y + 22), fill=COLORS[index])
        p.text(475, y, name, 23)
        p.right(760, y + 27, f"{value:.2f} MiB", 22, MUTED)
    p.wrap(76, 665, "Weights and temporary workspace are shared by serial refinement passes. Each extra pass adds its own temporal history.", 680, 25)
    p.box(815, 190, 1550, 820)
    p.text(843, 214, "Refinement passes: owned allocations", 29, INK, True)
    p.text(843, 261, "Packed weights + workspace + history + bridge", 22, MUTED)
    rows = DATA["direct_memory_1080"][:3]
    for index, row in enumerate(rows):
        y, left = 325 + index * 119, 843
        p.text(left, y, f"{row['passes']} pass" + ("es" if row["passes"] != 1 else ""), 27)
        p.right(1520, y, f"{row['owned_bytes'] / MIB:.2f} MiB", 29, GREEN, True)
        components = [140.9375, 203.125, 33.75 * row["passes"], 115.8125, 0.3125]
        for color, value in zip(COLORS, components):
            width = value / 650 * 670
            p.draw.rectangle((left, y + 48, left + width, y + 78), fill=color)
            left += width
    p.wrap(843, 709, "All tracked owned allocations reached zero after shutdown; capture readbacks retired too.", 670, 24)
    gpu, cpu = ROWS["cuda"], ROWS["cpu"]
    p.wrap(48, 854, f"The current native inner-network probe owns {ROWS['phoenix']['owned_bytes'] / MIB:.2f} MiB with RG16F motion; the adapter measurements above are from 11 October. Alternate-runner graphs request {gpu['owned_bytes'] / MIB:.2f} MiB on all six routes: {gpu['memory_parts']['Packed weights'] / MIB:.2f} MiB weights + {gpu['memory_parts']['Scratch / activations'] / MIB:.2f} MiB scratch/activations. These are different scopes.", 1500, 24)
    p.wrap(48, 958, "Owned resources, GPU residency and host working set overlap. Keep them separate; NVIDIA's internal component sizes remain unknown.", 1500, 23)
    p.save("memory-breakdown.png")


if __name__ == "__main__":
    native()
    resolution_scaling([row for row in DATA["resolution_matrix"] if row["width"] >= 1280],
                       "Native RTX: 720p through 8K", "native-resolution-scaling.png")
    resolution_scaling([row for row in DATA["resolution_matrix"] if row["width"] < 1280],
                       "Native RTX: lower resolutions", "native-lower-resolutions.png")
    resolution_report()
    portable()
    memory()
    print(f"Exported five benchmark images and the resolution report to {OUT.relative_to(ROOT)}")
