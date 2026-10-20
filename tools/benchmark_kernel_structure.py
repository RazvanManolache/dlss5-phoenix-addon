"""Explicit paired native-kernel refactor validation. Never installs or launches games."""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from compare_four_models import ROOT, build, sha, telemetry, write


def compiled_resources(bundle, kernel):
    tool = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/cuobjdump.exe'
    text = subprocess.check_output([tool, '--dump-resource-usage', str(bundle / kernel['binary'])], text=True)
    functions = {}
    for name, fields in re.findall(r'Function ([^:\n]+):\s*\n([^\n]+)', text):
        functions[name.strip()] = {key: int(value) for key, value in
            re.findall(r'\b(REG|STACK|SHARED|LOCAL):(\d+)', fields)}
    if not functions or any(len(fields) != 4 for fields in functions.values()):
        raise RuntimeError('Incomplete compiled resource report for ' + kernel['name'])
    return functions


def machine_comparison(before, after):
    tool = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/cuobjdump.exe'
    manifest = json.loads((after / 'native_manifest.json').read_text())
    rows = []
    for kernel in manifest['kernels']:
        code = [subprocess.check_output([tool, '--dump-sass', str(folder / kernel['binary'])])
                for folder in (before, after)]
        hashes = [hashlib.sha256(value).hexdigest() for value in code]
        rows.append(dict(name=kernel['name'], identical=hashes[0] == hashes[1],
                         before_sha256=hashes[0], after_sha256=hashes[1],
                         before_resources=compiled_resources(before, kernel),
                         after_resources=compiled_resources(after, kernel)))
    return rows


def build_identity(folder):
    manifest = json.loads((folder / 'native_manifest.json').read_text())
    return dict(native_graph_sha256=manifest['native_graph_sha256'],
                kernels=[{key: row[key] for key in ('name', 'entry', 'binary_sha256', 'parameter_bytes', 'block')}
                         for row in manifest['kernels']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--rounds', type=int, default=3, choices=range(1, 6))
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/current/kernel-structure')
    args = parser.parse_args()
    (ROOT / 'build').mkdir(exist_ok=True)
    bundles = dict(before=args.before.resolve(), after=args.after.resolve())
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, games_launched=False, installed_files_modified=False,
                  scope='Native Phoenix DLSS5 kernels only; matched full schedules, fixed seed, zero motion',
                  fixture='Deterministic RGB gradients and sinusoidal detail, RGBA16F; no model data copied',
                  gpu_before=telemetry(), cases=[], machine_code=machine_comparison(**bundles),
                  builds={label: build_identity(folder) for label, folder in bundles.items()})
    output = destination / 'paired.json'
    write(output, report)
    model = ROOT.parent / 'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
    with tempfile.TemporaryDirectory(prefix='kernel-paired-', dir=ROOT / 'build') as folder:
        work = Path(folder)
        executable, forwarder = build(work)
        for width, height in ((320, 180), (801, 603), (1920, 1080), (3840, 2160)):
            y, x = np.mgrid[:height, :width].astype(np.float32)
            pixels = np.ones((height, width, 4), dtype=np.float16)
            pixels[:, :, 0] = x / width
            pixels[:, :, 1] = y / height
            pixels[:, :, 2] = .5 + .25 * np.sin(x * .17) * np.cos(y * .11)
            color, motion = work / 'color.rgba16f', work / 'motion.rg16f'
            color.write_bytes(pixels.tobytes())
            motion.write_bytes(np.zeros((height, width, 2), dtype=np.float16).tobytes())
            for model_id in ('original', 'small-fp8'):
                repetitions = args.rounds if width >= 1920 else 1
                results = dict(before=[], after=[])
                for repeat in range(repetitions):
                    # Alternate order to reduce clock/temperature ordering bias.
                    for label in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
                        config = dict(width=width, height=height, engine='phoenix',
                            weight_source=dict(kind='nvidia_dll', path=str(model), sha256=sha(model), model=model_id),
                            bundle_path=str(bundles[label]), forwarder=str(forwarder), cache=str(work / 'cache'),
                            color_file=str(color), motion_file=str(motion), capture=False,
                            debug=width < 1920, warmup_seconds=.5, measure_seconds=1.)
                        job, result_file = work / 'job.json', work / 'result.json'
                        write(job, config)
                        result_file.unlink(missing_ok=True)
                        process = subprocess.run([str(executable), str(job), str(result_file)],
                            cwd=work, capture_output=True, text=True, timeout=180)
                        if process.returncode:
                            raise RuntimeError(f'{label} {model_id} {width}x{height}: {process.stderr}')
                        result = json.loads(result_file.read_text())
                        if not result.get('passed'):
                            raise RuntimeError(f'{label} probe did not pass')
                        results[label].append(dict(gpu_median_ms=result['gpu']['median_ms'],
                            wall_median_ms=result['wall']['median_ms'], captures=result['captures'],
                            debug_errors=result['debug_errors'], reset_repeatable=result['reset_repeatable']))
                        print(label, model_id, width, height, result['gpu']['median_ms'], flush=True)
                reference = results['before'][0]['captures']
                exact = all(row['captures'] == reference for values in results.values() for row in values)
                before_ms = statistics.median(row['gpu_median_ms'] for row in results['before'])
                after_ms = statistics.median(row['gpu_median_ms'] for row in results['after'])
                report['cases'].append(dict(model=model_id, width=width, height=height, exact=exact,
                    before_ms=before_ms, after_ms=after_ms, change_percent=(after_ms / before_ms - 1) * 100,
                    rounds=results))
                write(output, report)
                if not exact:
                    raise RuntimeError(f'Numerical regression: {model_id} {width}x{height}')
    report.update(passed=True, gpu_after=telemetry(), temporary_builds_removed=True,
                  note='Passed means exact tested outputs and successful probes, not a statistical guarantee of speed.')
    write(output, report)


if __name__ == '__main__':
    main()
