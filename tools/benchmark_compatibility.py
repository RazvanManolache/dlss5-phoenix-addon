"""Compare old/new embedded helper probes, including sampled Windows memory.

GPU work is serial. Timings are helper round trips, not whole-game frame times.
Memory monitoring additionally needs psutil; no model data is copied.
"""
import argparse
import ctypes as ct
from ctypes import wintypes as wt
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
import psutil


class CounterValue(ct.Structure):
    class Number(ct.Union):
        _fields_ = [('double', ct.c_double), ('large', ct.c_longlong)]
    _fields_ = [('status', wt.DWORD), ('number', Number)]


class CounterItem(ct.Structure):
    _fields_ = [('name', wt.LPWSTR), ('value', CounterValue)]


class GpuMemory:
    def __init__(self):
        self.api = ct.WinDLL('pdh')
        self.query = ct.c_void_p()
        self.counter = ct.c_void_p()
        self.api.PdhOpenQueryW.argtypes = [wt.LPCWSTR, ct.c_size_t, ct.POINTER(ct.c_void_p)]
        self.api.PdhAddEnglishCounterW.argtypes = [ct.c_void_p, wt.LPCWSTR, ct.c_size_t, ct.POINTER(ct.c_void_p)]
        self.api.PdhCollectQueryData.argtypes = [ct.c_void_p]
        self.api.PdhGetFormattedCounterArrayW.argtypes = [ct.c_void_p, wt.DWORD, ct.POINTER(wt.DWORD), ct.POINTER(wt.DWORD), ct.c_void_p]
        self.api.PdhCloseQuery.argtypes = [ct.c_void_p]
        if self.api.PdhOpenQueryW(None, 0, ct.byref(self.query)) != 0:
            raise RuntimeError('Cannot open GPU memory performance counters')
        if self.api.PdhAddEnglishCounterW(self.query, r'\GPU Process Memory(*)\Dedicated Usage', 0, ct.byref(self.counter)) != 0:
            self.close()
            raise RuntimeError('GPU process-memory counters unavailable')

    def sample(self):
        if self.api.PdhCollectQueryData(self.query) != 0:
            return {}
        size, count = wt.DWORD(), wt.DWORD()
        self.api.PdhGetFormattedCounterArrayW(self.counter, 0x200, ct.byref(size), ct.byref(count), None)
        if not size.value:
            return {}
        storage = ct.create_string_buffer(size.value)
        if self.api.PdhGetFormattedCounterArrayW(self.counter, 0x200, ct.byref(size), ct.byref(count), storage) != 0:
            return {}
        items = ct.cast(storage, ct.POINTER(CounterItem))
        result = {}
        for index in range(count.value):
            item = items[index]
            if item.value.status not in (0, 1) or not item.name.startswith('pid_'):
                continue
            pid = int(item.name.split('_')[1])
            result[pid] = result.get(pid, 0) + int(item.value.number.double)
        return result

    def close(self):
        if self.query.value:
            self.api.PdhCloseQuery(self.query)
            self.query.value = None


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def run_probe(executable, source, mode, width, height, frames, folder):
    folder.mkdir(parents=True, exist_ok=False)
    peak = dict(helper_private_bytes=0, helper_working_set_bytes=0,
                helper_dedicated_gpu_bytes=0, client_dedicated_gpu_bytes=0)
    samples = 0
    helper_pids = set()
    counters = GpuMemory()
    try:
        with (folder / 'stdout.log').open('w') as output, (folder / 'stderr.log').open('w') as errors:
            process = subprocess.Popen([str(executable), str(source), mode, str(width), str(height), str(frames)],
                cwd=folder, stdout=output, stderr=errors, creationflags=subprocess.CREATE_NO_WINDOW)
            client = psutil.Process(process.pid)
            deadline = time.monotonic() + 150
            while process.poll() is None:
                if time.monotonic() > deadline:
                    process.kill()
                    process.wait()
                    raise TimeoutError('Helper benchmark exceeded 150 seconds')
                gpu = counters.sample()
                peak['client_dedicated_gpu_bytes'] = max(peak['client_dedicated_gpu_bytes'], gpu.get(process.pid, 0))
                try:
                    helpers = [child for child in client.children(recursive=True)
                               if child.name().lower() == 'phoenix_graphics_host.exe']
                    for helper in helpers:
                        helper_pids.add(helper.pid)
                        info = helper.memory_info()
                        peak['helper_private_bytes'] = max(peak['helper_private_bytes'], info.private)
                        peak['helper_working_set_bytes'] = max(peak['helper_working_set_bytes'], info.rss)
                        peak['helper_dedicated_gpu_bytes'] = max(peak['helper_dedicated_gpu_bytes'], gpu.get(helper.pid, 0))
                        samples += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                time.sleep(0.05)
            if process.returncode:
                raise RuntimeError(f'Probe failed ({process.returncode}): {(folder / "stderr.log").read_text()[-1000:]}')
        result = json.loads((folder / 'stdout.log').read_text())
        result.update(sampled_peak_memory=peak, helper_pids=sorted(helper_pids), memory_samples=samples,
                      memory_scope='Observed process peaks including startup; WDDM dedicated bytes; shared resources may appear in both processes, so do not sum client/helper.')
        if samples == 0 or not peak['helper_dedicated_gpu_bytes']:
            raise RuntimeError('No usable helper memory observations; refusing zero-memory claim')
        (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    finally:
        counters.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-probe', type=Path, required=True)
    parser.add_argument('--candidate-probe', type=Path, required=True)
    parser.add_argument('--weight-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--frames', type=int, default=64)
    args = parser.parse_args()
    if not 2 <= args.rounds <= 10 or not 16 <= args.frames <= 1000:
        parser.error('Use 2..10 rounds and 16..1000 frames')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    probes = {'baseline': args.baseline_probe.resolve(), 'candidate': args.candidate_probe.resolve()}
    report = dict(passed=False, scope='Serial alternating helper-only benchmarks; game capture/display excluded',
                  frames=args.frames, rounds=args.rounds, probes={key: sha(path) for key, path in probes.items()},
                  runs=[], summary=[], model_included=False)
    receipt = root / 'report.json'
    try:
        for mode, width, height in [('cpu_transport', 1920, 1080), ('cpu_phoenix', 257, 255),
                                    ('cpu_phoenix', 1920, 1080), ('cpu_transport', 3840, 2160)]:
            runs = []
            for index in range(args.rounds):
                order = ['baseline', 'candidate'] if index % 2 == 0 else ['candidate', 'baseline']
                for variant in order:
                    folder = root / f'{mode}_{width}x{height}' / f'{index}_{variant}'
                    result = run_probe(probes[variant], args.weight_source.resolve(), mode, width, height, args.frames, folder)
                    result.update(variant=variant, round=index)
                    runs.append(result)
                    report['runs'].append(result)
                    receipt.write_text(json.dumps(report, indent=2) + '\n')
                    print(json.dumps({key: result[key] for key in ['variant', 'round', 'mode', 'width', 'height', 'warm_roundtrip_median_ms', 'sampled_peak_memory']}), flush=True)
            if len({item['output_sha256'] for item in runs}) != 1:
                raise RuntimeError('Baseline and candidate pixel outputs differ')
            summary = dict(mode=mode, width=width, height=height, output_sha256=runs[0]['output_sha256'])
            for variant in probes:
                group = [item for item in runs if item['variant'] == variant]
                summary[variant] = dict(roundtrip_ms=statistics.median(item['warm_roundtrip_median_ms'] for item in group),
                    observed_peak_memory={key: int(statistics.median(item['sampled_peak_memory'][key] for item in group))
                                          for key in group[0]['sampled_peak_memory']})
            summary['timing_change_percent'] = (summary['candidate']['roundtrip_ms'] / summary['baseline']['roundtrip_ms'] - 1) * 100
            report['summary'].append(summary)
        report['passed'] = True
    finally:
        receipt.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
