"""Plot hypothetical stereo inference costs from measured single-view timings.

These estimates are deliberately separate from measured benchmark records.
They assume two independent views; no unmeasured batching speedup is assigned.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BG, CARD, EDGE = "#0e1420", "#161f2d", "#2b3a50"
TEXT, MUTED = "#e9f0f9", "#acbbce"
GREEN, BLUE, GOLD, RED = "#79dbb4", "#93bcff", "#f5c879", "#ef9bb0"
HEADSETS = [
    {"name": "PSVR2", "width": 2000, "height": 2040,
     "refresh_hz": [90, 120],
     "source": "https://www.playstation.com/en-us/ps-vr2/ps-vr2-tech-specs/"},
    {"name": "Steam Frame", "width": 2160, "height": 2160,
     "refresh_hz_range": [72, 144], "experimental_refresh_hz": 144,
     "source": "https://store.steampowered.com/hardware/steamframe"},
]


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


class Figure:
    def __init__(self, title, subtitle, height):
        self.image = Image.new("RGB", (1600, height), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.text(48, 26, "DLSS5 PHOENIX  /  VR ESTIMATE  /  NOT A VR BENCHMARK", 22, GOLD, True)
        self.text(48, 69, title, 43, TEXT, True)
        self.text(48, 132, subtitle, 25, MUTED)

    def text(self, x, y, text, size=25, color=TEXT, bold=False):
        self.draw.text((x, y), str(text), font=font(size, bold), fill=color)

    def wrap(self, x, y, text, width=1500, size=24, color=MUTED):
        line = ""
        for word in text.split():
            trial = (line + " " + word).strip()
            if line and self.draw.textlength(trial, font=font(size)) > width:
                self.text(x, y, line, size, color)
                y += size + 10
                line = word
            else:
                line = trial
        self.text(x, y, line, size, color)
        return y + size + 10

    def dashed(self, x, top, bottom, color):
        for y in range(top, bottom, 14):
            self.draw.line((x, y, x, min(y + 7, bottom)), fill=color, width=2)

    def save(self, path):
        self.image.save(path, optimize=True)


def interpolate(rows, width, height, route):
    """Piecewise linear interpolation in per-eye pixels, without extrapolation."""
    pixels = width * height
    lower = max((row for row in rows if row["width"] * row["height"] <= pixels),
                key=lambda row: row["width"] * row["height"])
    upper = min((row for row in rows if row["width"] * row["height"] >= pixels),
                key=lambda row: row["width"] * row["height"])
    minimum = lower["width"] * lower["height"]
    maximum = upper["width"] * upper["height"]
    fraction = 0 if minimum == maximum else (pixels - minimum) / (maximum - minimum)
    cost = lower[route]["gpu_ms"] + fraction * (upper[route]["gpu_ms"] - lower[route]["gpu_ms"])
    return cost, [[lower["width"], lower["height"]], [upper["width"], upper["height"]]]


def cost_chart(out, records):
    figure = Figure("VR estimate: neural cost for both eyes",
                    "RTX 5080 · full network, one pass · independent-view assumption · model GPU time only", 1070)
    for index, headset in enumerate(HEADSETS):
        left = 48 + index * 767
        figure.draw.rounded_rectangle((left, 190, left + 735, 840), radius=15,
                                      fill=CARD, outline=EDGE, width=2)
        figure.text(left + 28, 211, headset["name"], 34, TEXT, True)
        figure.text(left + 28, 258, f"Panel: {headset['width']} × {headset['height']} per eye", 24, MUTED)
        plot_left, plot_width = left + 139, 450
        for mode_index, mode in enumerate(["panel", "half_linear"]):
            row = next(row for row in records if row["headset"] == headset["name"] and row["mode"] == mode)
            top = 315 + mode_index * 206
            label = "Full panel-size processing" if mode == "panel" else "Half width + height, then upscale"
            figure.text(left + 28, top, label, 26, TEXT, True)
            figure.text(left + 28, top + 35,
                        f"Input: {row['input_per_eye'][0]} × {row['input_per_eye'][1]} per eye", 23, MUTED)
            for route_index, (route, name, color) in enumerate([
                    ("phoenix", "Phoenix", GREEN), ("nvidia", "NVIDIA", BLUE)]):
                y = top + 85 + route_index * 51
                cost = row["stereo_model_ms"][route]
                figure.text(left + 28, y - 3, name, 24, color)
                end = plot_left + cost / 18 * plot_width
                figure.draw.rectangle((plot_left, y + 4, end, y + 28), fill=color)
                for hz, marker_color in [(90, GOLD), (120, MUTED)]:
                    marker = plot_left + (1000 / hz) / 18 * plot_width
                    figure.dashed(marker, y + 4, y + 28, marker_color)
                # Keep a budget marker from running through the value label.
                label = f"≈{cost:.1f} ms"
                label_width = figure.draw.textlength(label, font=font(25, True))
                figure.draw.rectangle((end + 6, y - 4, end + 14 + label_width, y + 32), fill=CARD)
                figure.text(end + 10, y - 4, label, 25, TEXT, True)
        figure.text(left + 28, 733, "Dashed lines = entire-frame budgets", 24, MUTED)
        figure.text(left + 28, 772, "90 Hz: 11.11 ms", 25, GOLD)
        figure.text(left + 375, 772, "120 Hz: 8.33 ms", 25, MUTED)
    y = figure.wrap(48, 866, "Estimate = 2 × interpolated single-eye GPU time from our 13 October 2026 native benchmarks. Interpolation uses per-eye pixel count, not headset marketing resolution. No batching or stereo-specific optimization gain is assumed.")
    figure.wrap(48, y + 12, "Not measured in VR. Game rendering, upscaling, compositor, copies and streaming are excluded. Actual SteamVR render targets can differ from panel sizes; aspect ratio, padding and scheduling can change these estimates.")
    figure.save(out / "vr-estimated-cost.png")


def budget_chart(out, records):
    figure = Figure("VR estimate: how much frame time remains?",
                    "Phoenix native on RTX 5080 · both eyes · inference first; everything else still has to fit", 1110)
    for index, hz in enumerate([90, 120]):
        left, budget = 48 + index * 767, 1000 / hz
        figure.draw.rounded_rectangle((left, 190, left + 735, 882), radius=15,
                                      fill=CARD, outline=EDGE, width=2)
        figure.text(left + 28, 213, f"{hz} Hz  /  {budget:.2f} ms total", 33, TEXT, True)
        bar_left, bar_width, maximum = left + 28, 590, 16
        marker = bar_left + budget / maximum * bar_width
        for row_index, row in enumerate(records):
            y = 284 + row_index * 141
            cost = row["stereo_model_ms"]["phoenix"]
            mode = "full size" if row["mode"] == "panel" else "half width + height"
            figure.text(left + 28, y, row["headset"] + " · " + mode, 25, TEXT, True)
            bar_top = y + 45
            infer_end = bar_left + min(cost, budget) / maximum * bar_width
            figure.draw.rectangle((bar_left, bar_top, infer_end, bar_top + 25), fill=GREEN)
            if cost < budget:
                figure.draw.rectangle((infer_end, bar_top, marker, bar_top + 25), fill=EDGE)
                caption = f"≈{cost:.1f} ms neural  |  ≈{budget - cost:.1f} ms left"
                color = MUTED
            else:
                end = bar_left + cost / maximum * bar_width
                figure.draw.rectangle((marker, bar_top, end, bar_top + 25), fill=RED)
                caption = f"≈{cost:.1f} ms neural  |  ≈{cost - budget:.1f} ms OVER"
                color = RED
            figure.draw.line((marker, bar_top - 9, marker, bar_top + 33), fill=GOLD, width=3)
            figure.text(left + 28, y + 85, caption, 24, color)
    figure.text(48, 909, "GREEN: neural work    DARK: theoretical headroom    RED: exceeds frame budget", 25, MUTED)
    y = figure.wrap(48, 957, "The remaining time is not a game-performance prediction. It must cover the game, upscaling and compositor work; streaming adds encoding and transport. Half width and height means one quarter of the neural input pixels, not half.")
    figure.wrap(48, y + 8, "Two independent views are an estimation baseline. Proper stereo batching may share weights and scheduling, but no speedup has been measured. Both eyes still need their own motion/depth inputs and temporal histories.")
    figure.save(out / "vr-estimated-budget.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "docs/benchmarks/2026-10-13")
    out = parser.parse_args().snapshot.resolve()
    measured_path = out / "measurements.json"
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    records = []
    for headset in HEADSETS:
        for mode, scale in [("panel", 1), ("half_linear", 0.5)]:
            width, height = int(headset["width"] * scale), int(headset["height"] * scale)
            row = {"headset": headset["name"], "mode": mode, "input_per_eye": [width, height],
                   "input_pixels_both_eyes": 2 * width * height, "stereo_model_ms": {},
                   "interpolation_extents": {}}
            for route in ["phoenix", "nvidia"]:
                cost, extents = interpolate(measured["resolution_matrix"], width, height, route)
                row["stereo_model_ms"][route] = 2 * cost
                row["interpolation_extents"][route] = extents
            row["phoenix_headroom_ms"] = {
                str(hz): 1000 / hz - row["stereo_model_ms"]["phoenix"] for hz in [90, 120]}
            records.append(row)
    receipt = {
        "schema": "phoenix-vr-estimate/1", "status": "ESTIMATES, NOT VR BENCHMARKS",
        # Git normalizes text line endings. Hash the same UTF-8/LF representation
        # on Windows and Linux so the published source checksum stays verifiable.
        "source_measurements_sha256": hashlib.sha256(measured_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
        "source_hash_normalization": "UTF-8 with LF line endings",
        "source_measurement_date": measured["date"], "gpu": "RTX 5080, driver 616.64",
        "headsets": HEADSETS, "records": records,
        "method": "Stereo time = 2 * piecewise-linear interpolation in single-eye pixel count between neighboring measured native resolutions; no extrapolation or batching gain.",
        "excluded": ["game rendering", "upscaling", "compositor", "copies", "encoding/transport"],
        "limitations": ["Unmeasured eye aspect ratio and padding", "Panel resolution may differ from runtime render target", "No validated stereo integration or visual-quality result", "Batch-2 speedup unmeasured; not a timing bound or error interval"],
    }
    (out / "vr-estimates.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cost_chart(out, records)
    budget_chart(out, records)
    print(json.dumps({"images": ["vr-estimated-cost.png", "vr-estimated-budget.png"],
                      "status": receipt["status"], "estimated_stereo_ms": [row["stereo_model_ms"] for row in records]}))


if __name__ == "__main__":
    main()
