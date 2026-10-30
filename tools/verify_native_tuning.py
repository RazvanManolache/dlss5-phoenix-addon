"""Explicit paired GPU validation of a finished bundle, with exact output checks.

Process medians are paired in alternating order to reduce clock/desktop drift.
The whole graph and matching output copy are timed; this is not game FPS.
Only compact receipts survive. Models and temporary input images are not copied.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile

import numpy as np
from compare_four_models import ROOT, build, sha, telemetry, write
from experiment_native_tuning import MODEL


def fixture(work, width, height):
    """Use the same deterministic FP16 detail fixture at each geometry."""
    y, x = np.mgrid[:height, :width].astype(np.float32)
    rgba = np.ones((height, width, 4), np.float16)
    rgba[:, :, 0] = x / width
    rgba[:, :, 1] = y / height
    rgba[:, :, 2] = .5 + .25 * np.sin(x * .17) * np.cos(y * .11)
    (work / 'color.rgba16f').write_bytes(rgba.tobytes())
    (work / 'motion.rg16f').write_bytes(np.zeros((height, width, 2), np.float16).tobytes())


def summarize(cases, rounds):
    """Pair by explicit round, never by sorting timings or pooling engines."""
    control = {r['round']: r for r in cases if r['label'] == 'control'}
    candidate = {r['round']: r for r in cases if r['label'] == 'candidate'}
    if set(control) != set(range(rounds)) or set(candidate) != set(control):
        raise ValueError('Missing paired rounds')
    captures = control[0]['captures']
    exact = all(r['captures'] == captures and r['reset_repeatable'] for r in cases)
    pairs = [(candidate[i]['gpu']['median_ms'] / control[i]['gpu']['median_ms'] - 1) * 100
             for i in range(rounds)]
    memory = {}
    for field in ('device_local_bytes', 'device_nonlocal_bytes', 'host_private_bytes',
                  'host_working_set_bytes', 'host_peak_working_set_bytes'):
        memory[field] = statistics.median(candidate[i]['memory'][field] - control[i]['memory'][field]
                                          for i in range(rounds))
    return dict(exact=exact, paired_changes_percent=pairs,
                paired_median_change_percent=statistics.median(pairs),
                control_ms=statistics.median(r['gpu']['median_ms'] for r in control.values()),
                candidate_ms=statistics.median(r['gpu']['median_ms'] for r in candidate.values()),
                paired_memory_delta_bytes=memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--extents', nargs='+', default=['1920x1080', '320x180', '801x603'])
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('At least one paired round is required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bundles = dict(control=args.baseline.resolve(), candidate=args.candidate.resolve())
    report = dict(schema='phoenix-native-tuning-validation/1', passed=False, state='building',
                  games_launched=False, installed_files_modified=False, older_device_validated=False,
                  timing_scope='Full model plus matching output copy; no upload/readback/CPU gaps',
                  analysis='Median of matched-pair changes; separate process medians also retained',
                  model_sha256=sha(MODEL),
                  runner_sha256=sha(Path(__file__)), probe_source_sha256=sha(ROOT/'tests/native/four_model_probe.cpp'),
                  manifests={k: json.loads((v/'native_manifest.json').read_text()) for k,v in bundles.items()},
                  gpu_before=telemetry(), cases=[], decisions=[])
    write(args.output, report)
    try:
        with tempfile.TemporaryDirectory(prefix='verify-native-tuning-', dir=ROOT/'build') as temporary:
            work = Path(temporary)
            executable, forwarder = build(work)
            report.update(executable_sha256=sha(executable), forwarder_sha256=sha(forwarder), state='running')
            for extent in args.extents:
                width, height = map(int, extent.split('x'))
                fixture(work, width, height)
                rounds = args.rounds if extent == '1920x1080' else 1
                for style in ('auto', 'rtx40', 'rtx30', 'rtx20'):
                    for model in ('original', 'small-fp8'):
                        cases = []
                        for repeat in range(rounds):
                            labels = ('control','candidate') if repeat % 2 == 0 else ('candidate','control')
                            for label in labels:
                                config = dict(width=width, height=height, engine='phoenix',
                                    weight_source=dict(kind='nvidia_dll', path=str(MODEL),
                                                       sha256=report['model_sha256'], model=model),
                                    bundle_path=str(bundles[label]), native_kernel_style=style,
                                    forwarder=str(forwarder), color_file=str(work/'color.rgba16f'),
                                    motion_file=str(work/'motion.rg16f'), capture=False, debug=False,
                                    warmup_seconds=1. if rounds > 1 else .1,
                                    measure_seconds=2. if rounds > 1 else .1,
                                    minimum_measured_frames=30 if rounds > 1 else 3)
                                write(work/'job.json', config)
                                output = work/'result.json'
                                output.unlink(missing_ok=True)
                                process = subprocess.run([str(executable),str(work/'job.json'),str(output)],
                                                         cwd=work,capture_output=True,text=True,timeout=120)
                                result = json.loads(output.read_text()) if output.exists() else {}
                                if process.returncode or not result.get('passed'):
                                    raise RuntimeError(result.get('error', process.stderr or 'Probe failed without receipt'))
                                row = dict(extent=extent, style=style, model=model, label=label, round=repeat,
                                           **{k: result[k] for k in ('gpu','memory','captures','reset_repeatable')})
                                cases.append(row)
                                report['cases'].append(row)
                                write(args.output, report)
                        decision = dict(extent=extent,style=style,model=model,rounds=rounds,**summarize(cases,rounds))
                        report['decisions'].append(decision)
                        write(args.output,report)
                        print(json.dumps(decision),flush=True)
                        if not decision['exact']:
                            raise RuntimeError('Full-model reset/history outputs changed')
            report.update(passed=True,state='completed',gpu_after=telemetry())
    except Exception as error:
        report.update(state='failed',error=str(error))
        raise
    finally:
        report['temporary_builds_removed']=True
        write(args.output,report)


if __name__ == '__main__':
    main()
