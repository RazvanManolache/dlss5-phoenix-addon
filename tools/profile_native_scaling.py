"""Locate native resolution thresholds with paired plain/kernel-instrumented runs.

Isolated GPU inference only. No games or installed packages are changed. Kernel
timestamps deliberately split launches, so their attribution is diagnostic;
plain runs remain the source for total latency and proportional comparisons.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from benchmark_readme_case import fixture
from compare_four_models import ROOT, build, sha, telemetry, write
from experiment_native_tuning import MODEL

EXTENTS = ['1280x720', '1600x900', '1920x960', '1920x1024', '1920x1032',
           '1920x1080', '2560x1440', '3840x2160']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/current/native-scaling-profile/run.json')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--extents', nargs='+', default=EXTENTS)
    parser.add_argument('--deep-stages-only', action='store_true', help='Keep native dispatch batches intact; mark prepare/network/copy regions only')
    args = parser.parse_args()
    if not 1 <= args.rounds <= 5:
        parser.error('Use one to five fresh-process rounds')
    bundle = ROOT / 'artifacts/current/native-kernel-styles'
    cores = list(Path('C:/Windows/System32/DriverStore/FileRepository').rglob('_nvngx.dll'))
    if len(cores) != 1:
        raise RuntimeError('An unambiguous installed NGX core is required')
    report = dict(passed=False, state='building', measured_date=datetime.date.today().isoformat(),
        rounds=args.rounds, extents=args.extents, deep_stages_only=args.deep_stages_only, model_sha256=sha(MODEL),
        native_manifest_sha256=sha(bundle / 'native_manifest.json'),
        source_sha256=sha(ROOT / 'tests/native/four_model_probe.cpp'),
        games_launched=False, installed_files_modified=False, gpu_before=telemetry(), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        write(args.output, report)
    save()
    expected = {}
    with tempfile.TemporaryDirectory(prefix='native-scaling-profile-', dir=ROOT / 'build') as temporary:
        work = Path(temporary).resolve()
        if work.parent != (ROOT / 'build').resolve():
            raise RuntimeError('Unsafe temporary cleanup root')
        executable, forwarder = build(work)
        report.update(state='running', executable_sha256=sha(executable))
        save()
        for extent in args.extents:
            width, height = map(int, extent.split('x'))
            rgba = np.ones((height, width, 4), np.float16)
            rgba[..., :3] = fixture(width, height)
            (work / 'color.rgba16f').write_bytes(rgba.tobytes())
            (work / 'motion.rg16f').write_bytes(np.zeros((height, width, 2), np.float16).tobytes())
            del rgba
            for model in ['original', 'small-fp8']:
                for repeat in range(args.rounds):
                    variants = [('phoenix', False)] if args.deep_stages_only else [('phoenix', False), ('phoenix', True)]
                    if not args.deep_stages_only and extent in ['1280x720', '1920x1024', '1920x1032', '1920x1080']:
                        variants.append(('nvidia', False))
                    if repeat % 2:
                        variants.reverse()
                    for engine, instrumented in variants:
                        report['current'] = dict(extent=extent, model=model, round=repeat,
                                                 engine=engine, instrumented=instrumented)
                        save()
                        config = dict(width=width, height=height, engine=engine,
                            weight_source=dict(kind='nvidia_dll', path=str(MODEL), sha256=report['model_sha256'], model=model),
                            bundle_path=str(bundle), native_kernel_style='auto', ngx_core=str(cores[0]),
                            forwarder=str(forwarder), cache=str(work / 'ngx-cache'),
                            color_file=str(work / 'color.rgba16f'), motion_file=str(work / 'motion.rg16f'),
                            capture=False, debug=False, warmup_seconds=1., measure_seconds=1.,
                            minimum_measured_frames=40, maximum_measure_seconds=30.,
                            profile_individual_kernels=instrumented,
                            profile_gpu_stages=instrumented or args.deep_stages_only,
                            deep_profile_gpu_stages=args.deep_stages_only,
                            report_kernel_selection=True)
                        write(work / 'job.json', config)
                        output = work / 'result.json'
                        output.unlink(missing_ok=True)
                        process = subprocess.run([str(executable), str(work / 'job.json'), str(output)],
                            cwd=work, capture_output=True, text=True, timeout=150)
                        result = json.loads(output.read_text(encoding='utf-8')) if output.exists() else {}
                        if process.returncode or not result.get('passed'):
                            raise RuntimeError(str(report['current']) + ': ' + str(result.get('error', process.stderr)))
                        if result['gpu']['samples'] < 40 or not all(c['finite'] for c in result['captures'].values()):
                            raise RuntimeError('Invalid samples or nonfinite output')
                        hashes = {name: value['sha256'] for name, value in result['captures'].items()}
                        key = (extent, model, engine)
                        if key in expected and hashes != expected[key]:
                            raise RuntimeError('Instrumentation or fresh process changed temporal outputs')
                        expected[key] = hashes
                        # Per-frame samples stay private in this compact local receipt.
                        result.update(extent=[width, height], model=model, round=repeat,
                                      instrumented=instrumented, engine=engine, stage_only=args.deep_stages_only)
                        report['cases'].append(result)
                        save()
                        print(extent, model, engine, 'profile' if instrumented else 'plain',
                              f"{result['gpu']['median_ms']:.4f} ms", flush=True)
        report.update(passed=True, state='completed', gpu_after=telemetry())
        report.pop('current', None)
        save()
    report['temporary_builds_removed'] = True
    save()


if __name__ == '__main__':
    main()
