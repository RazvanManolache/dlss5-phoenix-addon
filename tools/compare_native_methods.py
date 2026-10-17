"""Summarize validated current method benchmarks without launching anything."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_comparison(native, host, helper):
    for report, profile in ((native, native.get('benchmark_profile', {})),
                            (host, host.get('benchmark_profile', {})), (helper, helper)):
        if (profile.get('width'), profile.get('height')) != (1920, 1080):
            raise ValueError('All methods must use 1920x1080')
        if profile.get('warmup_seconds') != 5 or profile.get('measured_seconds') != 20:
            raise ValueError('Measurement windows differ')
        if report.get('gpu') != native.get('gpu') or not native.get('gpu'):
            raise ValueError('GPU or driver differs across measurements')
    if native['benchmark_profile'].get('reset_each_frame') is not True:
        raise ValueError('Native model probes must use reset frames for route 3 comparison')
    if not host.get('x86_host_benchmark_conclusions', {}).get('passed'):
        raise ValueError('Route 3 has not completed validated timing')
    rows = []

    def append(backend, method, timing, gpu_key, wall_key, scope, fps_key):
        gpu, wall = timing.get(gpu_key), timing.get(wall_key)
        if not gpu or not wall:
            raise ValueError('Missing real GPU/wall timing')
        numbers = [gpu.get('mean_ms'), wall.get('mean_ms'), timing.get(fps_key)]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in numbers):
            raise ValueError('Invalid timing/throughput')
        rows.append(dict(backend=backend, method=method, available=True, gpu_mean_ms=numbers[0],
            wall_mean_ms=numbers[1], completed_outputs_per_second=numbers[2], wall_scope=scope,
            gpu_p95_ms=gpu['p95_ms'], wall_p95_ms=wall['p95_ms'], samples=gpu['samples']))

    for backend in ('phoenix', 'nvidia'):
        proof = native.get('benchmark_conclusions', {}).get(backend, {})
        if not proof.get('passed'):
            raise ValueError('Native/WOW64 model timing failed: '+backend)
        for method, prefix in ((1, 'native'), (2, 'thread')):
            case = proof['cases'][prefix]
            if not case.get('passed') or case.get('timing',{}).get('reset_each_frame') is not True:
                raise ValueError('Incomplete native/WOW64 case')
            append(backend, method, case['timing'], 'gpu_processing', 'synchronized_call',
                'Isolated GPU-resident synchronized API call; route 2 worker entered once, not per-frame x86 handoff.',
                'completed_output_fps')
        if backend == 'phoenix':
            case = host['x86_host_benchmark_conclusions']['cases']['thread_driver_host']
            if not case.get('passed'):
                raise ValueError('Incomplete route 3 case')
            append(backend, 3, case['timing'], 'gpu_processing', 'synchronized_call',
                'GPU-resident processing plus per-frame x86/native dispatch-packet handoff; no host image transfers.',
                'completed_output_fps')
        else:
            rows.append(dict(backend=backend, method=3, available=False,
                reason='No x86 NVIDIA neural host implementation/source. Route 2 instead runs its x64 library in-process.'))
        case = helper.get('runs', {}).get('helper_x86' if backend == 'phoenix' else 'nvidia_x86', {})
        if (not case.get('passed') or case.get('execution_mode') != 'helper' or
                not case.get('helper_pid') or case.get('inference_pid') != case.get('helper_pid') or
                case.get('fixture') != 'dx12-x86'):
            raise ValueError('Unverified helper case: '+backend)
        append(backend, 4, case, 'gpu_model', 'processing_roundtrip',
            'Real x86 DX12 fixture + external x64 helper; sampled processing round-trip includes readback/IPC/return.',
            'completed_output_fps')
    return dict(schema='phoenix-native-method-comparison/1', gpu=native.get('gpu'),
        width=1920, height=1080, warmup_seconds=5, measured_seconds=20, rows=rows,
        game_fps=False, gpu_timing_scope='GPU model/route processing; Phoenix native probes include output copy.',
        limitations=[
            'Wall time scope differs by method; read wall_scope before comparing.',
            'Methods 1-3 use isolated synthetic input; method 4 uses the DX12 fixture image, not identical pixels.',
            'Methods 1-3 collect every completed call; method 4 latency is status-poll sampled.',
            'Single serial run per case; not a statistical estimate of architectural overhead.',
            'Completed output throughput is not game FPS and includes each harness scheduling overhead.'])


def main():
    paths = ['wow64-method-benchmark.json', 'wow64-x86-host-benchmark.json', 'dx12-method-benchmark.json']
    data = [(ROOT/'artifacts/current'/name).read_bytes() for name in paths]
    result = build_comparison(*(json.loads(value) for value in data))
    result['sources'] = {name: hashlib.sha256(value).hexdigest() for name, value in zip(paths, data)}
    output = ROOT/'artifacts/current/native-method-comparison.json'
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    for row in result['rows']:
        if row['available']:
            print(f"{row['backend']:7} #{row['method']} GPU={row['gpu_mean_ms']:.3f} ms wall={row['wall_mean_ms']:.3f} ms outputs/s={row['completed_outputs_per_second']:.2f}")
        else:
            print(f"{row['backend']:7} #{row['method']} unavailable: {row['reason']}")


if __name__ == '__main__':
    main()
