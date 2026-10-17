"""Serial DX12 fixture measurements; preserves installed configs and dashboard history."""
import argparse
import configparser
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CASES = [('native_x64','dx12-x64','phoenix','in_process'),
         ('portable_x86','dx12-x86','phoenix','in_process_portable'),
         ('helper_x86','dx12-x86','phoenix','helper'),
         ('nvidia_native_x64','dx12-x64','nvidia','in_process'),
         ('nvidia_x64','dx12-x64','nvidia','helper'),
         ('nvidia_x86','dx12-x86','nvidia','helper')]


def execution_setting(expected):
    if expected not in ('in_process', 'in_process_portable', 'helper'):
        raise ValueError('Unknown benchmark mechanism')
    return 'portable' if expected == 'in_process_portable' else expected


def stats(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v > 0)
    if not values:
        return None
    return dict(samples=len(values), mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                p95_ms=values[max(0, math.ceil(.95*len(values))-1)])


def summarize(result, snapshots, expected):
    if not result.get('passed'):
        return dict(passed=False, error=result.get('error', 'Incomplete fixture run'))
    if result.get('execution_mode') != expected:
        return dict(passed=False, error='Execution mechanism differed: '+str(result.get('execution_mode')))
    window = result['measurement']
    samples = []
    seen = set()
    for row in snapshots:
        if not window['start_render_seconds'] <= row.get('seconds', -1) <= window['end_render_seconds']:
            continue
        if row.get('execution_mode') != expected:
            return dict(passed=False, error='Mechanism changed during measurement')
        if expected == 'in_process' and (not result.get('fixture_pid') or
                row.get('inference_pid') != result['fixture_pid'] or row.get('helper_pid') != 0):
            return dict(passed=False, error='Native inference PID differs from fixture, or a helper is active')
        delivered = row.get('delivered', 0)
        if delivered in seen or not delivered:
            continue
        seen.add(delivered)
        samples.append(row)
    if len(samples) < 3:
        return dict(passed=False, error='Insufficient distinct completed-frame timing samples')
    gpu, route, worker, roundtrip = [], [], [], []
    for sample in samples:
        info = sample.get('details') or {}
        for row in (sample.get('gpu_timing') or {}).get('rows', []):
            if row.get('mode') != (1 if result['backend'] == 'nvidia' else 3) or (
                row.get('width'), row.get('height')) != (1920, 1080):
                continue
            field = 'nvidia' if result['backend'] == 'nvidia' else 'phoenix'
            if (row.get(field) or {}).get('last_ms', 0) <= 0:
                continue
            gpu.append((row.get(field) or {}).get('last_ms', 0))
            route.append((row.get('route') or {}).get('last_ms', 0))
        runtime = info.get('runtime') or {}
        if runtime.get('operator_clock') == 'GPU events':
            gpu.append(runtime.get('operator_ms', 0))
        worker.append(info.get('frame_wall_ms', 0))
        roundtrip.append(sample.get('processing_roundtrip_ms', 0))
    fps = result['neural_fps']
    if not isinstance(fps, (float, int)) or not math.isfinite(fps) or fps <= 0:
        return dict(passed=False, error='No completed-output throughput')
    return dict(passed=True, execution_mode=expected, completed_output_fps=fps,
        output_interval_ms=1000/fps, gpu_model=stats(gpu), gpu_route=stats(route),
        worker_wall=stats(worker), processing_roundtrip=stats(roundtrip),
        timing_scope='GPU command timestamps, sampled worker/IPC wall latency, and fresh-output interval are separate; not game FPS.',
        sampled_latency_scope='One latest completed frame per status poll, deduplicated by delivered counter; percentiles are sampled, not all-frame percentiles.',
        measured_seconds=20, warmup_seconds=5, frames=result['measurement'].get('frames_rendered'),
        pixel_comparison=result.get('pixel_comparison'), inference_pid=samples[-1].get('inference_pid'),
        helper_pid=samples[-1].get('helper_pid'), settings=samples[-1].get('applied_settings'),
        sampled_frames=len(samples))


def run_enhanced(runner, game, backend, folder, output, reference):
    """Warm the model, then wait five seconds, then measure exactly twenty."""
    output.mkdir(parents=True, exist_ok=True)
    command = runner.fixture_command(folder/Path(game['executable']).name, backend, output)
    process, mail = None, None
    snapshots, start, end = [], None, None
    result = dict(passed=False, backend=backend)
    number = 1 if backend == 'nvidia' else 3
    try:
        with (output/'process.log').open('w') as log:
            process = subprocess.Popen(command, cwd=folder, env=os.environ.copy(), stdout=log, stderr=log)
            result['fixture_pid'] = process.pid
            mail = runner.Mailbox(folder, process)
            deadline = time.monotonic()+120
            while process.poll() is None and time.monotonic() < deadline:
                response = mail.command(timeout=3)
                snapshot = response.get('snapshot', {})
                if snapshot:
                    if snapshot.get('phase') == 5: raise RuntimeError(snapshot.get('status', 'Processing failed'))
                    row = dict(seconds=snapshot['tick_ms']/1000, **runner.compact(snapshot))
                    snapshots.append(row)
                    if runner.ready(snapshot, number):
                        if start is None: start, end = row['seconds']+5, row['seconds']+25
                    elif start is not None and row['seconds'] >= start:
                        raise RuntimeError('Processing stopped or changed settings during measurement')
                    if end is not None and row['seconds'] >= end:
                        break
                time.sleep(.25)
            if process.poll() is not None: raise RuntimeError('Fixture exited before measurement completed')
            if start is None or not snapshots or snapshots[-1]['seconds'] < end:
                raise RuntimeError('Bounded readiness/measurement deadline exceeded')
            window = dict(start_render_seconds=start, end_render_seconds=end, complete=True,
                enabled=True, requested_seconds=20, warmup_seconds=5)
            rates = runner.completed_fps(snapshots, window)
            result.update(execution_mode=snapshots[-1]['execution_mode'], neural_fps=rates['average_fps'],
                measurement=rates['measurement'])
            shot = mail.command('screenshot')
            path = Path(shot.get('screenshot_path', ''))
            if not shot.get('screenshot_path') or not path.resolve().is_relative_to((folder/'PhoenixOutput').resolve()):
                raise RuntimeError('Screenshot did not return an owned output path')
            try:
                until = time.monotonic()+15
                while process.poll() is None and time.monotonic() < until:
                    snapshot = mail.command(timeout=3).get('snapshot', {})
                    state = snapshot.get('screenshot') or {}
                    if state.get('error'): raise RuntimeError(state['error'])
                    if state.get('path') == str(path) and not state.get('pending') and path.is_file(): break
                    time.sleep(.1)
                if not path.is_file(): raise RuntimeError('Presented output capture missing')
                record = runner.read(reference/'captures.json')['captures'][0]
                with runner.Image.open(path) as image, runner.Image.open(reference/record['png']) as baseline:
                    a, b = runner.np.asarray(image.convert('RGB')).astype(float), runner.np.asarray(baseline.convert('RGB')).astype(float)
                if a.shape != (1080,1920,3) or a.shape != b.shape: raise RuntimeError('Output extent differs')
                d = abs(a[270:810,480:1440]-b[270:810,480:1440])
                comparison = dict(central_mae=float(d.mean()), changed_fraction_gt2=float((d.max(axis=2)>2).mean()))
                if comparison['central_mae'] <= .25 or comparison['changed_fraction_gt2'] <= .01:
                    raise RuntimeError('No measurable enhanced image change')
                result.update(passed=True, pixel_comparison=[comparison])
            finally: path.unlink(missing_ok=True)
    except Exception as error:
        result.update(passed=False, error=str(error), processing=snapshots[-1] if snapshots else {})
    finally:
        if process: runner.close_fixture(process)
        if mail: mail.stop_helpers()
    return result, snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=ROOT/'artifacts/current/addon-portable-candidate')
    parser.add_argument('--run-gpu', action='store_true', required=True)
    parser.add_argument('--only', nargs='+', choices=[case[0] for case in CASES])
    args = parser.parse_args()
    suite, candidate = args.suite.resolve(), args.candidate.resolve()
    sys.path.insert(0, str(suite/'tools'))
    runner = importlib.import_module('run_backends')
    original_compact = runner.compact
    def compact(snapshot):
        value = original_compact(snapshot)
        value.update({key:snapshot.get(key) for key in ('inference_pid', 'processing_roundtrip_ms', 'timing_provenance')})
        info = snapshot.get('in_process') or snapshot.get('helper') or {}
        value['details'] = {key: info.get(key) for key in ('runtime', 'frame_wall_ms', 'processing_location')}
        return value
    runner.compact = compact
    games = {game['id']:game for game in runner.read(suite/'matrix.json')['games']}
    cases = list(CASES)
    if args.only: cases = [case for case in cases if case[0] in args.only]
    report = dict(schema='phoenix-dx12-method-benchmark/1', width=1920, height=1080, vsync=False,
        warmup_seconds=5, measured_seconds=20, warmup_scope='Five seconds AFTER first verified processed presentation; model/device startup excluded', runs={}, dashboard_modified=False,
        scope='Real DX12 x86/x64 fixtures with ReShade and candidate add-ons. Baseline used only for output verification. No retail games.',
        wow64_native=dict(passed=False, error='No proven complete model execution; not assigned a timing'))
    receipt = ROOT/'artifacts/current/dx12-method-benchmark.json'
    if args.only and receipt.is_file():
        previous = json.loads(receipt.read_text())
        for key in ('schema', 'width', 'height', 'vsync', 'warmup_scope', 'measured_seconds'):
            if previous.get(key) != report[key]: raise ValueError('Existing benchmark profile differs; rerun the complete matrix')
        for key, row in previous.get('runs', {}).items():
            bits = 32 if row.get('fixture', '').endswith('x86') else 64
            if row.get('addon_sha256') != runner.digest(candidate/f'Phoenix.addon{bits}'):
                raise ValueError('Preserved benchmark uses another addon build')
        report['runs'] = previous.get('runs', {})
    try:
        report['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError): report['gpu'] = 'Unavailable'
    def save(): receipt.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    save()
    with tempfile.TemporaryDirectory(prefix='dx12-benchmark-', dir=ROOT/'artifacts') as temporary:
        staging = Path(temporary)
        baselines = {}
        for name, game_id, backend, expected in cases:
            game = games[game_id]
            folder = suite/game['install_directory']
            clean = suite/'clean'/game['install_directory']
            exe = folder/Path(game['executable']).name
            if runner.digest(exe) != game['sha256'] or runner.digest(clean/exe.name) != game['sha256']:
                raise ValueError('Mismatched clean/working DX12 executable')
            if game_id not in baselines:
                baseline = staging/game_id/'native'
                result = runner._run_case(game, 'native', clean, baseline, 20, os.environ.copy(), baseline)
                if not result.get('passed'): raise RuntimeError('Clean DX12 baseline failed: '+str(result.get('error')))
                baselines[game_id] = baseline
            addon = folder/('Phoenix.addon32' if game_id.endswith('x86') else 'Phoenix.addon64')
            settings_files = [folder/'ReShade.ini', folder/'ReShadeVR.ini', addon]
            originals = {path:path.read_bytes() if path.exists() else None for path in settings_files}
            try:
                shutil.copyfile(candidate/addon.name, addon)
                game['addon_sha256'] = runner.digest(addon)
                runner.ini_backend(folder, 1 if backend == 'nvidia' else 3)
                for path in settings_files[:2]:
                    ini = configparser.ConfigParser(interpolation=None, strict=False)
                    ini.optionxform = str
                    ini.read(path, encoding='utf-8-sig')
                    if not ini.has_section('Phoenix.Compatibility'): ini.add_section('Phoenix.Compatibility')
                    ini['Phoenix.Compatibility'].update(ExecutionMode=execution_setting(expected), HelperTransport='readback',
                        PortableFallback='0', PortableAllowUnvalidated='1', PortableDevice='auto',
                        PortableRuntime=str(candidate/'PhoenixRuntime'/('win-x86' if game_id.endswith('x86') else 'win-x64')))
                    with path.open('w', encoding='utf-8') as stream: ini.write(stream)
                output = staging/game_id/name
                print('Running '+name, flush=True)
                result, rows = run_enhanced(runner, game, backend, folder, output, baselines[game_id])
                summary = summarize(result, rows, expected)
                summary.update(fixture=game_id, addon_sha256=game['addon_sha256'])
                if not summary['passed']:
                    summary['diagnostics'] = {key:result.get(key) for key in ('execution_mode', 'loaded_components', 'processing')}
                report['runs'][name] = summary
                save()
                print(json.dumps(dict(method=name, **{k:v for k,v in summary.items() if k not in ('settings','diagnostics')})), flush=True)
            finally:
                for path, data in originals.items():
                    if data is None: path.unlink(missing_ok=True)
                    else: path.write_bytes(data)
    return 0 if all(row.get('passed') for row in report['runs'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
