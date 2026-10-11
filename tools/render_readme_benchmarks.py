"""Export the checked-in benchmark snapshot as README images (Pillow required).

These are static chart exports of measured data, not browser screenshots.
Run from any working directory: python tools/render_readme_benchmarks.py
"""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks" / "2026-10-11"
DATA = json.loads((OUT / "measurements.json").read_text(encoding="utf-8"))
ROWS = {row["id"]: row for row in DATA["records"]}
MIB = 1024**2
BG, PANEL, LINE = "#0e1420", "#161f2d", "#2b3a50"
INK, MUTED, GREEN, BLUE, GOLD = "#e9f0f9", "#acbbce", "#79dbb4", "#93bcff", "#f5c879"
COLORS = [GREEN, BLUE, GOLD, "#c29aff", "#ef9bb0", "#68c9e1"]


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
    def __init__(self, title, subtitle, height):
        self.image = Image.new("RGB", (1600, height), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.text(48, 28, "DLSS5 PHOENIX  /  LOCAL BENCHMARK  /  11 OCT 2026", 21, GREEN, True)
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
    p = Panel("Native RTX: effectively tied at 1080p", "RTX 5080 · driver 616.64 · GPU inference latency; lower is better", 980)
    for left, row, color in [(48, phoenix, GREEN), (815, nvidia, BLUE)]:
        p.box(left, 196, left + 735, 396)
        p.text(left + 28, 216, "Phoenix native RTX" if row["id"] == "phoenix" else "NVIDIA DLSS NR", 29, color, True)
        p.text(left + 28, 258, f"{row['core_ms']:.3f} ms", 66, INK, True)
        p.text(left + 28, 350, f"Process-median range: {row['process_median_min_ms']:.3f}–{row['process_median_max_ms']:.3f} ms", 23, MUTED)
    p.box(48, 424, 810, 786)
    p.text(76, 447, "All tested production extents", 28, INK, True)
    p.text(76, 499, "Resolution", 23, MUTED)
    p.right(480, 499, "Phoenix", 23, GREEN)
    p.right(625, 499, "NVIDIA", 23, BLUE)
    p.right(780, 499, "Δ latency", 23, MUTED)
    for index, row in enumerate(DATA["resolution_matrix"]):
        y = 551 + index * 54
        p.text(76, y, f"{row['width']} × {row['height']}", 26)
        p.right(480, y, f"{row['phoenix']['gpu_ms']:.3f}", 26)
        p.right(625, y, f"{row['nvidia']['gpu_ms']:.3f}", 26)
        p.right(780, y, f"{row['phoenix_latency_difference_percent']:+.2f}%", 25)
    p.box(835, 424, 1550, 786)
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
    p.wrap(48, 815, "Eight fresh processes per route at 1080p, 96 measured frames each; other extents: two processes per route. Identical FP16 fixture, controls and model; D3D12 debug enabled for both. Upload/readback excluded. These are inference timings, not game FPS.", 1500, 24)
    p.wrap(48, 919, "Residency includes the benchmark harness and driver. NVIDIA's internal memory breakdown is unknown.", 1500, 23)
    p.save("native-rtx.png")


def portable():
    p = Panel("Portable runners: working, with plenty to optimize", "1920 × 1080 · six measured history frames per route · lower is better", 1040)
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
        p.text(76, y + 37, hardware, 22, MUTED)
        for x, value in [(750, row["core_ms"]), (1005, row["wall_ms"]), (1260, row["owned_bytes"] / MIB), (1520, row["host_working_set_bytes"] / MIB)]:
            p.right(x, y + 5, f"{value:,.2f}", 27)
        # A common logarithmic axis keeps both the sub-second and 30-second routes visible.
        length = 650 * (math.log10(row["wall_ms"]) - 2) / (math.log10(40000) - 2)
        p.draw.rectangle((620, y + 49, 620 + length, y + 58), fill=color)
    p.wrap(48, 866, "Thin bars show complete-frame latency on a shared logarithmic scale. Full frame includes preparation, transfers, inference, resolve and broker round trip. Host RAM is worker resident working set; graph allocations and host RAM overlap and must not be added.", 1500, 24)
    p.wrap(48, 971, "Portable inputs and timing boundaries differ from the native comparison. AMD iGPU is not a discrete Radeon benchmark; SYCL CPU is not an Intel GPU benchmark.", 1500, 23)
    p.save("portable-runners.png")


def memory():
    p = Panel("Memory: shared work, less duplication", "Native full adapter · 1920 × 1080 · owned resources, separate from process residency", 1030)
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
    p.wrap(48, 854, "The inner-network probe owns 409.69 MiB; the full adapter adds bridge and exposure resources. Portable graphs request 467.01 MiB: 144.51 MiB weights + 322.50 MiB scratch/activations. These are different ownership scopes.", 1500, 24)
    p.wrap(48, 958, "Owned resources, GPU residency and host working set overlap. Keep them separate; NVIDIA's internal component sizes remain unknown.", 1500, 23)
    p.save("memory-breakdown.png")


if __name__ == "__main__":
    native()
    portable()
    memory()
    print(f"Exported three benchmark images to {OUT.relative_to(ROOT)}")
