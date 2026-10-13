"""Render the measured resolution-crossover investigation (Pillow required)."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from render_vr_estimates import font

ROOT = Path(__file__).resolve().parents[1]
BG, CARD, EDGE = "#0e1420", "#161f2d", "#2b3a50"
TEXT, MUTED, GREEN, BLUE, GOLD = "#e9f0f9", "#acbbce", "#79dbb4", "#93bcff", "#f5c879"


class Chart:
    def __init__(self, title, subtitle, height):
        self.image = Image.new("RGB", (1600, height), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.text(48, 27, "DLSS5 PHOENIX / DIAGNOSTIC BASELINE BEFORE TUNING / 13 OCT 2026", 22, GREEN, True)
        self.text(48, 72, title, 42, TEXT, True)
        self.text(48, 132, subtitle, 25, MUTED)

    def text(self, x, y, text, size=26, color=TEXT, bold=False):
        self.draw.text((x, y), str(text), font=font(size, bold), fill=color)

    def right(self, x, y, text, size=26, color=TEXT, bold=False):
        self.text(x-self.draw.textlength(str(text), font=font(size, bold)), y, text, size, color, bold)

    def card(self, top, bottom):
        self.draw.rounded_rectangle((48, top, 1550, bottom), radius=15, fill=CARD, outline=EDGE, width=2)

    def wrap(self, y, text, size=24, color=MUTED):
        line = ""
        for word in text.split():
            trial = (line + " " + word).strip()
            if line and self.draw.textlength(trial, font=font(size)) > 1500:
                self.text(48, y, line, size, color)
                y += size+10
                line = word
            else:
                line = trial
        self.text(48, y, line, size, color)
        return y+size+10


def overview(data, output):
    chart = Chart("Baseline: the resolution crossover before tuning",
                  "RTX 5080 · matched input/model · full network · GPU inference only", 1080)
    chart.card(190, 690)
    for x, name in [(76, "Resolution"), (650, "Phoenix ms"), (945, "NVIDIA ms"), (1250, "Difference ms")]:
        chart.text(x, 215, name, 25, MUTED)
    for index, case in enumerate(data["cases"]):
        top = 285+index*95
        chart.text(76, top, f"{case['width']} × {case['height']}", 30, TEXT, True)
        chart.right(850, top, f"{case['whole_frame']['phoenix']['gpu_ms']:.3f}", 31, GREEN, True)
        chart.right(1140, top, f"{case['whole_frame']['nvidia']['gpu_ms']:.3f}", 31, BLUE, True)
        color = GREEN if case["difference_ms"] < 0 else BLUE
        chart.right(1515, top, f"{case['difference_ms']:+.3f}", 31, color, True)
        chart.right(1515, top+41, f"Phoenix {case['difference_percent']:+.2f}%", 22, MUTED)
    chart.text(48, 725, "Negative difference: Phoenix faster. Positive: NVIDIA faster.", 27, TEXT, True)
    y = chart.wrap(782, "Four fresh processes per renderer and resolution in balanced interleaved order; 256 frames each, discard 64. Values are means of process medians. Per-launch profiling is disabled in this diagnostic baseline comparison.")
    y = chart.wrap(y+12, "A separate clock-window pass repeats the crossover. Clocks were observed, not locked; NVIDIA's median observed SM clock was about 1.1% higher at 4K and 0.4% higher at 8K. Clocks alone do not establish the cause of a timing difference.")
    chart.wrap(y+12, "8K timings varied during sustained load: Phoenix process medians 58.47–62.17 ms; NVIDIA 56.56–61.18 ms. Treat the exact margin as variable. This is an offline fixture, not game FPS.", color=GOLD)
    chart.image.save(output / "native-scaling-crossover.png", optimize=True)


def kernel_crossover(data, output):
    chart = Chart("Baseline: which kernels lost time at large inputs",
                  "Per-launch attribution · timings can disturb overlap · not headline speed figures", 1340)
    chart.card(190, 600)
    chart.text(76, 212, "512-channel Q/K/V projection — aggregate of 16 launches", 30, TEXT, True)
    chart.text(76, 263, "Resolution", 24, MUTED)
    chart.right(900, 263, "Phoenix ms", 24, GREEN)
    chart.right(1200, 263, "NVIDIA ms", 24, BLUE)
    chart.right(1515, 263, "Difference ms", 24, MUTED)
    for index, case in enumerate(data["cases"]):
        family = next(row for row in case["kernel_families"] if row["name"] == "cc_split_swin_16h_qkv_512_chained_fp8")
        top = 318+index*62
        chart.text(76, top, f"{case['width']} × {case['height']}", 28)
        chart.right(900, top, f"{family['phoenix_ms']:.3f}", 28, GREEN)
        chart.right(1200, top, f"{family['nvidia_ms']:.3f}", 28, BLUE)
        chart.right(1515, top, f"{family['difference_ms']:+.3f}", 28,
                    GREEN if family["difference_ms"]<0 else BLUE)
    chart.card(625, 1120)
    chart.text(76, 648, "8K attribution: losses and retained wins", 30, TEXT, True)
    families = {row["name"]:row for row in data["cases"][-1]["kernel_families"]}
    selected = [
        ("Global attention (8 calls)", "cc_vit_1d_attention_chained_fp8"),
        ("512-channel projection (14 calls)", "cc_split_swin_16h_proj_512_chained_fp8"),
        ("512-channel Q/K/V (16 calls)", "cc_split_swin_16h_qkv_512_chained_fp8"),
        ("64-channel downsample transition", "cc_tinlayout_fused_swin_2h_64_2_ds_wait_fp8"),
        ("Input preparation", "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"),
        ("Output reconstruction", "cc_tinlayout_fused_post_block_swin_1h_32_fp8"),
    ]
    chart.right(900, 701, "Phoenix ms", 24, GREEN)
    chart.right(1200, 701, "NVIDIA ms", 24, BLUE)
    chart.right(1515, 701, "Difference ms", 24, MUTED)
    for index, (label, name) in enumerate(selected):
        family, top = families[name], 755+index*55
        chart.text(76, top, label, 25)
        chart.right(900, top, f"{family['phoenix_ms']:.3f}", 27, GREEN)
        chart.right(1200, top, f"{family['nvidia_ms']:.3f}", 27, BLUE)
        chart.right(1515, top, f"{family['difference_ms']:+.3f}", 27,
                    GREEN if family["difference_ms"]<0 else BLUE)
    y = chart.wrap(1150, "Two fresh instrumented processes per renderer/resolution; 256 frames each, discard 128. The 154 core launches align by order, grid and block size. This does not imply identical machine code. All final output hashes match.")
    chart.wrap(y+12, "Instrumentation can change relative timings, especially at low resolution. These are kernel-family sums of per-launch medians; do not add them to reconstruct headline whole-frame latency. GPU hardware counters were unavailable.", color=GOLD)
    chart.image.save(output / "native-scaling-kernels.png", optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "docs/benchmarks/2026-10-13")
    output = parser.parse_args().snapshot.resolve()
    data = json.loads((output / "native-scaling-diagnosis.json").read_text(encoding="utf-8"))
    overview(data, output)
    kernel_crossover(data, output)
    print("Rendered native-scaling comparison and kernel attribution charts")


if __name__ == "__main__":
    main()
