"""Paired resident matrix tests for CUDA half and Vulkan scalar fallbacks."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
from test_model_selection import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    devices = json.loads(subprocess.check_output([str(args.candidate / 'phoenix_cpp_devices.exe'),
                                                  str(args.candidate.resolve())], text=True))['devices']
    routes = [dict(route='cuda_half', backend='cuda', options=0,
                   device=next(d['id'] for d in devices if d['id'].startswith('cuda:'))),
              dict(route='amd_scalar', backend='vulkan', options=(1 << 22) | (1 << 21) | 1,
                   device=next(d['id'] for d in devices if d['id'].startswith('vulkan:') and d.get('vendor') == 0x1002))]
    report = dict(passed=False, scope='resident matrices, not whole-frame time', pairs=3, measurements=[], summary=[])
    with tempfile.TemporaryDirectory(prefix='reuse-matrices-', dir=ROOT / 'build') as folder:
        work = Path(folder).resolve()
        if work.parent != (ROOT / 'build').resolve():
            raise RuntimeError('Invalid cleanup root')
        for route in routes:
            expected = None
            for pair in range(3):
                for label in (['control', 'candidate'] if pair % 2 == 0 else ['candidate', 'control']):
                    root = getattr(args, label).resolve()
                    output = work / f'{route["route"]}-{pair}-{label}.json'
                    subprocess.run([str(root / 'phoenix_cpp_matrix_benchmark.exe'), str(root), route['backend'],
                                    route['device'], str(output), str(route['options']), '80'], check=True, timeout=300)
                    result = json.loads(output.read_text(encoding='utf-8'))
                    hashes = [row['output_sha256'] for row in result['rows']]
                    if expected is None:
                        expected = hashes
                    if not result['passed'] or hashes != expected:
                        raise RuntimeError('Resident matrix outputs differ')
                    report['measurements'].append(dict(route=route['route'], variant=label, pair=pair,
                                                       rows=result['rows']))
                    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            rows = [row for row in report['measurements'] if row['route'] == route['route']]
            for index, shape in enumerate(rows[0]['rows']):
                summary = dict(route=route['route'], shape=[shape[k] for k in ['m', 'n', 'k']], exact=True)
                for label in ['control', 'candidate']:
                    summary[label + '_ms'] = statistics.median(row['rows'][index]['median_ms']
                        for row in rows if row['variant'] == label)
                summary['speedup'] = summary['control_ms'] / summary['candidate_ms']
                report['summary'].append(summary)
    report['passed'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
