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


def temporal_samples(control, executable, count):
    """Compare same-frame pairs; a frozen fixture must not invent changes."""
    rows = []
    input_reference = output_reference = None
    for _ in range(count):
        control.call('screenshot_pair')
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            pair = control.call()['snapshot'].get('screenshot_pair', {})
            if pair.get('state') == 'error':
                raise AssertionError(pair.get('error', 'Paired screenshot failed'))
            if pair.get('state') == 'saved':
                break
            time.sleep(.05)
        else:
            raise AssertionError('Paired screenshot did not complete')
        directory = Path(pair['directory']).resolve()
        if not directory.is_relative_to((executable.parent / 'PhoenixOutput').resolve()):
            raise AssertionError('Paired screenshot escaped the fixture')
        from PIL import Image
        with Image.open(directory / 'original.png') as image:
            before = np.asarray(image.convert('RGB')).copy()
        with Image.open(directory / 'modified.png') as image:
            after = np.asarray(image.convert('RGB')).copy()
        # The fixture's diagnostic HUD is outside the scene region.
        height, width = before.shape[:2]
        before = before[height//5:4*height//5, width//5:4*width//5]
        after = after[height//5:4*height//5, width//5:4*width//5]
        if input_reference is None:
            input_reference, output_reference = before, after
        rows.append({'frame': pair.get('frame'),
                     'input_mae': float(np.abs(before.astype(float) - input_reference).mean()),
                     'output_mae': float(np.abs(after.astype(float) - output_reference).mean())})
        shutil.rmtree(directory)
    if any(row['input_mae'] for row in rows):
        raise AssertionError('Temporal fixture is not frozen; input changed')
    if any(row['output_mae'] for row in rows):
        raise AssertionError('Processed output changed despite identical frozen input')
    return {'passed': True, 'sample_count': count, 'scene_pixels_identical': True, 'samples': rows}


def loading_screenshot(control, executable):
    """The normal image helper rejects black frames; loading must be black."""
    control.call('screenshot')
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = control.call()['snapshot'].get('screenshot', {})
        path = Path(state['path']) if state.get('path') else None
        if path and not state.get('pending') and path.is_file():
            if not path.resolve().is_relative_to((executable.parent / 'PhoenixOutput').resolve()):
                raise AssertionError('Loading screenshot escaped fixture')
            from PIL import Image
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB')).copy()
            path.unlink()
            return pixels
        time.sleep(.05)
    raise AssertionError('Loading screenshot did not complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "application", "addon", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--nvidia", action="store_true", help="Also exercise NVIDIA feature creation and retirement")
    parser.add_argument("--temporal-samples", type=int, default=0, help="Capture 2..32 frozen same-frame pairs for both Phoenix models")
    parser.add_argument('--presentation-gate', action='store_true', help='Force missing-guide startup and warm refusal; verify raw pixels stay hidden')
    parser.add_argument('--estimated-motion', action='store_true', help='Exercise hardware motion preparation and resumed feeder bindings on both models')
    args = parser.parse_args()
    if args.estimated_motion and args.presentation_gate:
        parser.error('Run motion and forced-guide refusal checks separately')
    if args.temporal_samples and not 2 <= args.temporal_samples <= 32:
        parser.error('--temporal-samples must be 0 or between 2 and 32')
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
                if args.estimated_motion:
                    ini['Phoenix'].update({'ProcessingEnabled':'1','HookStage':'3','ReuseDlssUpscalingResources':'1',
                        'MotionEstimation':'1','MotionProvider':'2','PipelineMonitoring':'1'})
                if args.presentation_gate:
                    ini['Phoenix'].update({'ProcessingEnabled':'1','HookStage':'3','ReuseDlssUpscalingResources':'2'})
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
                    if args.presentation_gate:
                        deadline = time.monotonic() + 20
                        while time.monotonic() < deadline:
                            state = control.call()['snapshot']
                            if state.get('presentation_gate', {}).get('loading', 0) > 2: break
                            time.sleep(.1)
                        else: raise AssertionError('Missing-guide startup did not activate presentation gate')
                        pixels = loading_screenshot(control, exe)
                        h,w=pixels.shape[:2]
                        if pixels[h//5:4*h//5,w//5:4*w//5].max() != 0:
                            raise AssertionError('Raw scene pixels escaped cold-start presentation gate')
                        report['presentation_gate']={'cold_start_hides_raw':True}
                        control.call('set',{'ReuseDlssUpscalingResources':1})
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
                        if args.estimated_motion and label in ('original','lightweight','original_again'):
                            deadline=time.monotonic()+15
                            while time.monotonic()<deadline:
                                motion_snapshot=control.call()['snapshot']
                                provider=motion_snapshot.get('in_process',{}).get('pipeline_status',{}).get('motion_provider')
                                if provider=='NVIDIA Optical Flow':break
                                time.sleep(.1)
                            else:raise AssertionError('Requested hardware motion did not become active: '+str(provider))
                            report.setdefault('estimated_motion',{})[label]={'provider':provider,'snapshot':motion_snapshot}
                        if args.temporal_samples and label in ('original', 'lightweight'):
                            report.setdefault('temporal_stability', {})[label] = temporal_samples(control, exe, args.temporal_samples)
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
                    if args.presentation_gate:
                        before=control.call()['snapshot']
                        control.call('set',{'ReuseDlssUpscalingResources':2})
                        deadline=time.monotonic()+15
                        while time.monotonic()<deadline:
                            state=control.call()['snapshot']
                            if state.get('presentation_gate',{}).get('held',0)>before.get('presentation_gate',{}).get('held',0):break
                            time.sleep(.05)
                        else:raise AssertionError('Warm refusal did not hold completed output')
                        pixels=np.asarray(control.screenshot()).astype(np.float32)
                        if difference(pixels,images['original_again']) != 0:
                            raise AssertionError('Warm refusal exposed raw or changed processed pixels')
                        report['presentation_gate']['warm_refusal_holds_output']=True
                        control.set({'ReuseDlssUpscalingResources':1},route='in_process')
                        report['presentation_gate']['recovers']=True
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
