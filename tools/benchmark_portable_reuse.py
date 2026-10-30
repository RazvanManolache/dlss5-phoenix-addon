"""Paired full-model validation of portable operand/frame reuse changes.

Control and candidate must be independently built, model-free runtimes. Models
stay external. The receipt retains timings and temporal hashes, not frame dumps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from test_model_selection import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--extent', default='1920x1080')
    parser.add_argument('--routes', nargs='+', default=['cuda', 'vulkan_nvidia'])
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--scalar-matrices', action='store_true', help='Exercise the scalar GPU fallback in both builds')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    width, height = map(int, args.extent.split('x'))
    if args.pairs < 1:
        parser.error('At least one paired process required')
    inventory = json.loads(subprocess.check_output([
        str(args.candidate / 'phoenix_cpp_devices.exe'), str(args.candidate.resolve())], text=True))
    devices = inventory['devices']
    cuda = next((device['id'] for device in devices if device['id'].startswith('cuda:')), None)
    vulkan = {device['label']: device['id'] for device in devices if device['id'].startswith('vulkan:')}
    routes = {'cpu32': ('cpu', 'cpu', True, 0)}
    if cuda:
        routes['cuda'] = ('cuda', cuda, False, 0x10de)
    for name, device in vulkan.items():
        if 'NVIDIA' in name:
            routes['vulkan_nvidia'] = ('vulkan', device, False, 0x10de)
        elif 'AMD' in name or 'Radeon' in name:
            routes['vulkan_amd'] = ('vulkan', device, True, 0x1002)
    missing = set(args.routes) - routes.keys()
    if missing:
        raise RuntimeError('Requested backends are unavailable: ' + ', '.join(sorted(missing)))
    source = ROOT.parent / 'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    report = dict(passed=False, state='running', extent=[width, height], paired_processes=args.pairs,
                  scalar_matrices=args.scalar_matrices,
                  fixture='structured SDR RGB rounded through binary16, fixed seed, zero motion',
                  model_sha256=source_sha, measurements=[], summary=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    with tempfile.TemporaryDirectory(prefix='portable-reuse-cases-', dir=ROOT / 'build') as folder:
        work = Path(folder).resolve()
        if work.parent != (ROOT / 'build').resolve():
            raise RuntimeError('Invalid cleanup root')
        for route in args.routes:
            backend, device, slow, vendor = routes[route]
            for model in ['original', 'small-fp8']:
                expected = None
                for pair in range(args.pairs):
                    for label in (['control', 'candidate'] if pair % 2 == 0 else ['candidate', 'control']):
                        root = getattr(args, label).resolve()
                        config = dict(runtime_root=str(root), backend=backend, device=device,
                                      allow_unvalidated=True, cpu_threads=32,
                                      weight_source=dict(kind='nvidia_dll', path=str(source), sha256=source_sha, model=model))
                        if args.scalar_matrices:
                            config['scalar_matrices'] = True
                        job = dict(extent=[width, height], model=model, route=route, profile='default',
                                   process_index=pair, config=config, runtime=str(root / 'phoenix_runtime_cpp.dll'),
                                   slow=slow, vendor_id=vendor)
                        job_path, result_path = work / 'job.json', work / 'result.json'
                        job_path.write_text(json.dumps(job), encoding='utf-8')
                        process = subprocess.run([sys.executable, str(ROOT / 'tools/benchmark_readme_case.py'),
                                                  str(job_path), str(result_path)], capture_output=True,
                                                 text=True, timeout=900)
                        result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else {}
                        if process.returncode or result.get('passed') is not True:
                            report.update(state='failed', failure=dict(route=route, model=model, label=label,
                                          error=result.get('error', process.stderr[-3000:])))
                            save()
                            raise RuntimeError(report['failure'])
                        hashes = result['output_hashes']
                        if expected is None:
                            expected = hashes
                        if hashes != expected:
                            report.update(state='failed', failure=dict(route=route, model=model, label=label,
                                                                      error='Temporal hashes differ'))
                            save()
                            raise RuntimeError(report['failure'])
                        result.update(variant=label, pair=pair)
                        report['measurements'].append(result)
                        save()
                        print(route, model, label, f"{result['wall']['median_ms']:.3f} ms", flush=True)
                rows = [row for row in report['measurements'] if row['route'] == route and row['model'] == model]
                summary = dict(route=route, model=model, temporal_hashes_exact=True)
                for label in ['control', 'candidate']:
                    selected = [row for row in rows if row['variant'] == label]
                    summary[label] = dict(wall_ms=statistics.median(row['wall']['median_ms'] for row in selected),
                        gpu_ms=statistics.median(row['gpu']['median_ms'] for row in selected) if selected[0]['gpu'] else None,
                        prepare_ms=statistics.median(row['stages_mean_ms']['prepare_ms'] for row in selected))
                summary['wall_reduction_percent'] = 100 * (1 - summary['candidate']['wall_ms'] / summary['control']['wall_ms'])
                report['summary'].append(summary)
                save()
    report.update(passed=True, state='completed')
    save()


if __name__ == '__main__':
    main()
