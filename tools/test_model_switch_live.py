"""Opt-in bounded addon test in one existing Museum D3D12 x64 fixture.

Uses DLSS Everything's addon bridge; restores the fixture addon and configuration
on exit. No retail game, external model copy or desktop interaction is needed.
"""
import argparse
import configparser
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "application", "addon", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--nvidia", action="store_true", help="Also exercise NVIDIA feature creation and retirement")
    args = parser.parse_args()
    fixture, output = args.fixture.resolve(), args.output.resolve()
    testing = fixture.parents[1]
    metadata = json.loads((fixture / "test-game.json").read_text())
    if metadata.get("api") != "dx12" or metadata.get("architecture") != "x64":
        parser.error("Select the existing dx12-x64 Museum fixture")
    exe = testing / metadata["executable"]
    for process in psutil.process_iter(["exe"]):
        if process.info["exe"] and Path(process.info["exe"]).resolve() == exe.resolve():
            raise RuntimeError("The selected fixture is already running")
    sys.path.insert(0, str(testing / "tools"))
    from run_live_regressions import ApplicationControl, wait_for_bridge_endpoint
    from run_backends import close_fixture

    output.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "fixture": metadata["id"], "retail_game": False, "stages": []}
    receipt = output / "live.json"
    saved = {p: p.read_bytes() for p in fixture.glob("*.ini")}
    config_path, installed = fixture / "Phoenix.json", fixture / "Phoenix.addon64"
    saved[config_path] = config_path.read_bytes()
    process = None
    try:
        with tempfile.TemporaryDirectory(prefix="live_switch_", dir=ROOT / "build") as temporary:
            previous = Path(temporary) / "Phoenix.addon64"
            os.replace(installed, previous)
            try:
                shutil.copy2(args.addon, installed)
                ini = configparser.ConfigParser(interpolation=None, strict=False)
                ini.optionxform = str
                ini.read_string(saved[fixture / "ReShade.ini"].decode("utf-8-sig"))
                model_path = str(args.model.resolve())
                settings = {"Backend": "3", "ExecutionDevice": "native", "ModelId": "original",
                    "ModelSource": "nvidia_dll", "ModelDll": model_path, "ModelFolder": "",
                    "NvidiaDllFolder": "", "Experimental": "0", "RefinementPasses": "1",
                    "ResolutionMode": "0", "ProcessingEnabled": "0", "Comparison": "0"}
                for name, value in settings.items(): ini["Phoenix"][name] = value
                ini["Phoenix.Compatibility"]["ExecutionMode"] = "in_process"
                ini["Phoenix.Compatibility"]["Enabled"] = "1"
                buffer = io.StringIO(); ini.write(buffer)
                (fixture / "ReShade.ini").write_text(buffer.getvalue(), encoding="utf-8")
                config = json.loads(saved[config_path])
                config["weight_source"] = {"kind": "nvidia_dll", "path": model_path,
                    "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest()}
                config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
                log = (output / "fixture.log").open("w", encoding="utf-8")
                process = subprocess.Popen([str(exe), "--background", "--live-control", "--seconds", "180",
                    "--width", "1280", "--height", "720", "--freeze", "--fixed-time", "--no-animations",
                    "--no-capture", "--output", str(output / "museum")], cwd=fixture, stdout=log, stderr=log)
                try:
                    wait_for_bridge_endpoint(exe, process)
                    control = ApplicationControl(args.application.resolve(), exe, process, output)
                    control.wait({"Backend": 3}, timeout=40, route="in_process")
                    images = {}
                    stages = [("input", {"Backend": 0}),
                        ("original", {"Backend": 3, "ModelId": "original"}),
                        ("lightweight", {"Backend": 3, "ModelId": "small-fp8"})]
                    if args.nvidia: stages.append(("nvidia", {"Backend": 1}))
                    stages.append(("original_again", {"Backend": 3, "ModelId": "original"}))
                    for label, values in stages:
                        snapshot = control.set(values, route="in_process")
                        time.sleep(.4)
                        image = control.screenshot(); image.save(output / (label + ".png"))
                        images[label] = np.asarray(image).astype(np.float32)
                        report["stages"].append({"stage": label, "snapshot": snapshot})
                        print(label, snapshot.get("execution_mode"), snapshot.get("delivered"), flush=True)
                    # Exclude borders and the diagnostic HUD; assess actual scene pixels.
                    def difference(a, b):
                        h, w = a.shape[:2]
                        return float(np.abs(a[h//5:4*h//5, w//5:4*w//5] - b[h//5:4*h//5, w//5:4*w//5]).mean())
                    report["scene_difference_rgb255"] = {name: difference(images[name], images["input"])
                        for name in ("original", "lightweight", "original_again")}
                    if any(value < .1 for value in report["scene_difference_rgb255"].values()):
                        raise AssertionError("Phoenix scene difference is absent in final presented screenshots")
                    report["model_difference_rgb255"] = difference(images["original"], images["lightweight"])
                    if report["model_difference_rgb255"] < .01:
                        raise AssertionError("Model switching did not change final presented pixels")
                    if args.nvidia:
                        nvidia = next(stage["snapshot"]["in_process"]["nvidia"] for stage in report["stages"]
                                      if stage["stage"] == "nvidia")
                        final = report["stages"][-1]["snapshot"]["in_process"]["nvidia"]
                        if not nvidia["evaluates"] or final["ready"]:
                            raise AssertionError("NVIDIA feature did not evaluate and then retire")
                    report["passed"] = True
                finally:
                    close_fixture(process)
                    log.close()
                    shutil.copy2(fixture / "ReShade.log", output / "ReShade.log")
            finally:
                installed.unlink(missing_ok=True)
                os.replace(previous, installed)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        for path, data in saved.items(): path.write_bytes(data)
        report["fixture_restored"] = True
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
