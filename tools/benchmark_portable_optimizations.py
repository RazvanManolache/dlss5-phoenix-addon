"""Matched 1080p portable A/B runs with exact temporal hashes and owned-memory reporting.

Uses explicit external runtimes/model fixtures. No model is copied into the report.
The caller must keep other GPU workloads idle; utilization observations are retained.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'runner', 'jobs', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[128, 256, 512, 1024])
    parser.add_argument('--processes', type=int, default=3)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--motion-mode', choices=('zero', 'fractional'), default='zero')
    parser.add_argument('--routes', choices=('cuda', 'vulkan_nvidia'), nargs='+',
                        default=['cuda', 'vulkan_nvidia'])
    parser.add_argument('--candidate-options', type=json.loads, default={},
                        help='JSON object of candidate runtime options for ablation')
    args = parser.parse_args()
    if args.frames < 5 or args.processes < 2 or any(n < 16 or n > 1024 for n in args.batches):
        parser.error('At least five frames, two processes and batches in 16..1024 required')
    if not isinstance(args.candidate_options, dict):
        parser.error('--candidate-options must be a JSON object')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Generate motion/history planes once. Input RGB remains the saved fixture.
    import numpy as np
    motion = np.zeros((1080, 1920, 2), dtype='<f4')
    if args.motion_mode == 'fractional':
        # Exercise interpolation and history rejection, not only the zero-motion fast path.
        motion[..., 0] = 0.375
        motion[..., 1] = -0.625
    depth = np.broadcast_to(np.linspace(.1, .9, 1920, dtype='<f4'), (1080, 1920))
    valid = np.ones((1080, 1920), dtype='<f4')
    for name, plane in [('motion', motion), ('depth', depth), ('validity', valid)]:
        plane.tofile(out / (name + '.bin'))
    routes = args.routes
    measurements, golden = [], {}
    receipt = {'schema': 'phoenix-portable-optimization-ab/1', 'passed': False,
               'hardware': {'gpu': 'RTX 5080', 'cpu': 'Ryzen 9 9950X3D'},
               'resolution': [1920, 1080], 'frames_per_process': args.frames,
               'motion_mode': args.motion_mode, 'golden_sha256': golden,
               'candidate_options': args.candidate_options,
               'discarded_per_process': 2, 'measurements': measurements,
               'scope': 'Matched RGB32F host/broker boundary; initialization excluded. Owned GPU graph bytes exclude driver caches and host histories.'}
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    receipt['runner_sha256'] = digest(args.runner)
    receipt['runtime_binaries_sha256'] = {
        label: {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*'))
                if path.is_file() and path.suffix.lower() in ('.dll', '.exe', '.spv')}
        for label, root in [('baseline', args.baseline), ('candidate', args.candidate)]}
    receipt['plane_sha256'] = {name: digest(out / (name + '.bin'))
                              for name in ('motion', 'depth', 'validity')}
    def save():
        (out / 'report.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    def utilization():
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        if result.returncode: raise RuntimeError('GPU utilization query failed')
        fields = result.stdout.strip().splitlines()[0].split(',')
        return {'gpu_percent': int(fields[0]), 'adapter_memory_mib': int(fields[1])}
    try:
        for iteration in range(args.processes):
            variants = [('baseline', None)] + [('batch' + str(n), n) for n in args.batches]
            # First establish each route's baseline temporal hashes. Later rounds
            # rotate the order so warmup/clock drift does not always favor one path.
            if iteration: variants = variants[iteration:] + variants[:iteration]
            for variant, batch in variants:
                for route in routes:
                    # nvidia-smi reports a recent sampling window. Allow the
                    # previous measured process to leave that window first.
                    deadline = time.monotonic() + 15
                    idle_samples = 0
                    while True:
                        before = utilization()
                        idle_samples = idle_samples + 1 if before['gpu_percent'] <= 25 else 0
                        if idle_samples >= 2: break
                        if time.monotonic() >= deadline:
                            raise RuntimeError('GPU busy before measurement; pause other workloads and use a fresh output')
                        time.sleep(1)
                    job = json.loads((args.jobs / ('hd_a_' + route + '.json')).read_text(encoding='utf-8'))
                    config = job['config']
                    config.update(runtime_root=str((args.baseline if batch is None else args.candidate).resolve()), execution='broker')
                    if batch is not None:
                        config.update(args.candidate_options)
                        config['window_batch_limit'] = batch
                    first = job['cases'][0]
                    first['save_output'] = False
                    job['cases'] = [first]
                    for frame in range(1, args.frames):
                        task = copy.deepcopy(first)
                        task.pop('expected_sha256', None)
                        task['name'] = 'history_' + str(frame)
                        task['metadata']['reset'] = False
                        task['motion'] = str(out / 'motion.bin')
                        task['depth'] = str(out / 'depth.bin')
                        task['validity'] = str(out / 'validity.bin')
                        task['metadata']['motion_from_frame'] = frame - 1
                        job['cases'].append(task)
                    if route in golden:
                        for task, sha in zip(job['cases'], golden[route]): task['expected_sha256'] = sha
                    name = f'{iteration}_{variant}_{route}'
                    job_path = out / (name + '.json')
                    job_path.write_text(json.dumps(job, indent=2) + '\n', encoding='utf-8')
                    started = time.monotonic()
                    with (out / (name + '.log')).open('w', encoding='utf-8') as log:
                        result = subprocess.run([str(args.runner.resolve()), str(job_path), str(out / name)],
                                                stdout=log, stderr=subprocess.STDOUT, timeout=180)
                    report = json.loads((out / name / 'report.json').read_text(encoding='utf-8'))
                    if result.returncode or not report['passed']:
                        raise RuntimeError(name + ': ' + report.get('error', 'runner failed'))
                    hashes = [frame['output_sha256'] for frame in report['cases']]
                    if route not in golden: golden[route] = hashes
                    if hashes != golden[route]: raise RuntimeError(name + ': temporal output hash mismatch')
                    frames = report['cases'][2:]
                    row = {'name': name, 'route': route, 'variant': variant, 'process': iteration,
                           'exact_frames': len(hashes), 'gpu_before': before, 'gpu_after': utilization(),
                           'seconds': time.monotonic() - started,
                           'frame_ms': statistics.median(frame['call_ms'] for frame in frames),
                           'model_ms': statistics.median(frame['operator_ms'] for frame in frames),
                           'prepare_ms': statistics.median(frame['prepare_ms'] for frame in frames),
                           'resolve_ms': statistics.median(frame['resolve_ms'] for frame in frames),
                           'resident_bytes': frames[-1]['resident_bytes'],
                           'scratch_bytes': frames[-1]['resident_scratch_bytes'],
                           'command_count': frames[-1]['command_count'],
                           'host_working_set_bytes': frames[-1]['memory']['working_set_bytes'],
                           'host_private_bytes': frames[-1]['memory']['private_bytes'],
                           'host_frame_buffers': frames[-1]['host_frame_buffers'],
                           'timing_intervals': frames[-1]['timing_intervals']}
                    measurements.append(row); save()
                    print(name, round(row['frame_ms'], 2), round(row['model_ms'], 2), row['resident_bytes'], flush=True)
        receipt.update(passed=True, exact_frames=sum(row['exact_frames'] for row in measurements))
        save()
    except Exception as error:
        receipt['error'] = str(error); save(); raise


if __name__ == '__main__':
    main()
