"""Explicit isolated GPU comparison of older kernel strategies on a newer GPU.

Uses the production catalogue, full trained graph and identical input/control
settings. Timing is warmed GPU work, never cold compilation or game FPS.
Temporary executables and raw readbacks are removed even if a case fails.
"""
import argparse
import datetime
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from compare_four_models import ROOT, build, sha, telemetry, write

STYLES = ('auto', 'rtx40', 'rtx30', 'rtx20')
MODELS = ('original', 'small-fp8')


def output_difference(reference, actual):
    """Compare RGB only; alpha and packing must not hide a colour difference."""
    reference = reference.reshape(-1, 4)[:, :3].astype(np.float32)
    actual = actual.reshape(-1, 4)[:, :3].astype(np.float32)
    difference = np.abs(reference - actual)
    return dict(exact=bool(np.array_equal(reference, actual)),
                mean_abs=float(difference.mean()), max_abs=float(difference.max()),
                changed_fraction=float(np.mean(reference != actual)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--model-dll', type=Path, required=True)
    parser.add_argument('--resolutions', nargs='+', default=['1920x1080'],
                        help='Default is 1080p; larger workloads require an explicit selection')
    parser.add_argument('--rounds', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/current/kernel-style-benchmarks')
    args = parser.parse_args()
    bundle, model = args.bundle.resolve(), args.model_dll.resolve()
    sizes = [tuple(map(int, value.split('x'))) for value in args.resolutions]
    if any(len(size) != 2 or not all(0 < axis <= 8192 for axis in size) for size in sizes):
        parser.error('Use bounded WIDTHxHEIGHT resolutions')
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / 'run.json'
    report = dict(schema='phoenix-kernel-style-benchmark/1', passed=False, state='building',
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  games_launched=False, installed_files_modified=False,
                  scope='Phoenix DLSS5 full graph + matching output copy, GPU-resident input; not game FPS or older GPU emulation',
                  controls=dict(passes=1, repeats=[1]*8, tone=1, structure=1, skin=-1, automatic_mask=True),
                  bundle_manifest_sha256=sha(bundle / 'native_manifest.json'), model_dll_sha256=sha(model),
                  gpu_before=telemetry(), cases=[], summary=[])
    write(report_path, report)
    try:
        (ROOT / 'build').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='kernel-style-benchmark-', dir=ROOT / 'build') as temporary:
            work = Path(temporary)
            executable, forwarder = build(work)
            report['probe_sha256'] = sha(executable)
            report['state'] = 'running'
            write(report_path, report)
            for width, height in sizes:
                y, x = np.mgrid[:height, :width].astype(np.float32)
                pixels = np.ones((height, width, 4), dtype=np.float16)
                pixels[..., 0] = x / width
                pixels[..., 1] = y / height
                pixels[..., 2] = .5 + .25 * np.sin(x * .17) * np.cos(y * .11)
                color, motion = work / 'color.rgba16f', work / 'motion.rg16f'
                color.write_bytes(pixels.tobytes())
                motion.write_bytes(np.zeros((height, width, 2), dtype=np.float16).tobytes())
                del x, y, pixels
                for model_id in MODELS:
                    baseline, differences = {}, {}
                    for repeat in range(args.rounds):
                        # Reverse order every other round to reduce systematic
                        # warmup/temperature bias. Each cell is a fresh process.
                        order = STYLES if repeat % 2 == 0 else tuple(reversed(STYLES))
                        for style in order:
                            key = f'{model_id}-{width}x{height}-{style}-r{repeat}'
                            job, output = work / 'job.json', work / 'result.json'
                            config = dict(width=width, height=height, engine='phoenix',
                                          weight_source=dict(kind='nvidia_dll', path=str(model),
                                                             sha256=report['model_dll_sha256'], model=model_id),
                                          bundle_path=str(bundle), native_kernel_style=style,
                                          forwarder=str(forwarder), color_file=str(color), motion_file=str(motion),
                                          capture=repeat == 0, debug=False, warmup_seconds=1., measure_seconds=3.)
                            write(job, config)
                            output.unlink(missing_ok=True)
                            report['current_case'] = key
                            write(report_path, report)
                            process = subprocess.run([str(executable), str(job), str(output)], cwd=work,
                                                     capture_output=True, text=True, timeout=180)
                            result = json.loads(output.read_text()) if output.exists() else dict(passed=False)
                            if process.returncode or not result.get('passed'):
                                report['cases'].append(dict(key=key, passed=False, error=result.get('error', process.stderr)))
                                raise RuntimeError(f'{key}: {result.get("error", process.stderr)}')
                            if result.get('native_kernel_style') != style or result.get('physical_cuda_architecture') != 120:
                                raise RuntimeError('This experiment requires verified style selection on the RTX 50 target')
                            if repeat == 0:
                                differences[style] = {}
                                for sequence in ('reset_a', 'history'):
                                    capture = work / result['captures'][sequence].pop('file')
                                    actual = np.fromfile(capture, dtype=np.float16)
                                    capture.unlink()
                                    if style == 'auto':
                                        baseline[sequence] = actual
                                    differences[style][sequence] = output_difference(baseline[sequence], actual)
                            # Preserve the receipt/samples, not duplicate screenshots,
                            # model paths, temporary binaries or private host paths.
                            row = dict(key=key, model=model_id, width=width, height=height, style=style, round=repeat,
                                       passed=True, gpu=result['gpu'], wall=result['wall'],
                                       gpu_samples_ms=result['gpu_samples_ms'], wall_samples_ms=result['wall_samples_ms'],
                                       reset_repeatable=result['reset_repeatable'], captures=result['captures'],
                                       physical_cuda_architecture=result['physical_cuda_architecture'],
                                       memory=result['memory'], output_vs_native=differences[style], gpu_after=telemetry())
                            report['cases'].append(row)
                            write(report_path, report)
                            print(f'{key}: GPU {row["gpu"]["median_ms"]:.4f} ms; wall {row["wall"]["median_ms"]:.4f} ms', flush=True)
                    native = statistics.median(row['gpu']['median_ms'] for row in report['cases']
                                               if row['model'] == model_id and row['width'] == width and row['height'] == height and row['style'] == 'auto')
                    for style in STYLES:
                        rows = [row for row in report['cases'] if row['model'] == model_id and row['width'] == width
                                and row['height'] == height and row['style'] == style]
                        measured = [row['gpu']['median_ms'] for row in rows]
                        value = statistics.median(measured)
                        report['summary'].append(dict(model=model_id, width=width, height=height, style=style,
                                                      gpu_median_ms=value, round_medians_ms=measured,
                                                      slowdown_percent=(value / native - 1) * 100,
                                                      wall_median_ms=statistics.median(row['wall']['median_ms'] for row in rows),
                                                      output_vs_native=differences[style]))
                    write(report_path, report)
        report.update(passed=True, state='completed', gpu_after=telemetry())
    except Exception as error:
        report.update(state='failed', error=str(error))
        raise
    finally:
        report['temporary_builds_removed'] = True
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(report_path, report)


if __name__ == '__main__':
    main()
