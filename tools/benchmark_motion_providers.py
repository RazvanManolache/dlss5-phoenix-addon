"""Benchmark selected motion providers through a packaged x64 helper and x86 client."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import tempfile

from test_model_selection import ROOT, run, toolchain
from validate_portable_bridge import resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--addon32', type=Path, required=True)
    parser.add_argument('--weight-source', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=76)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--heights', type=int, nargs='+', default=[180, 360, 747])
    args = parser.parse_args()
    if not 24 <= args.frames <= 1000 or not 1 <= args.rounds <= 5:
        parser.error('Use 24..1000 frames and 1..5 rounds')
    sources = [path.resolve() for path in args.weight_source]
    package = args.addon32.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    names = {2: 'NVIDIA Optical Flow', 3: 'Zero', 4: 'FidelityFX', 5: 'OpenCV DIS', 6: 'D3D12 Video'}
    report = {'passed': False, 'workload': '2560x1080, two refinement passes, mixed motion, no depth or scene detection',
              'transport_scope': 'x86 capture fixture to x64 helper; game capture and presentation excluded',
              'warmup': 12, 'cases': [], 'summary': []}

    def save():
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    with tempfile.TemporaryDirectory(prefix='motion-bench-', dir=ROOT / 'build') as temporary:
        work = Path(temporary)
        compiler, env = toolchain(32)
        env['PHOENIX_DATA_ROOT'] = str(work / 'data')
        recipe, _ = resources(package, work)
        rc = shutil.which('rc.exe', path=env['PATH'])
        if not rc and env.get('WINDOWSSDKDIR') and env.get('WINDOWSSDKVERSION'):
            rc = Path(env['WINDOWSSDKDIR']) / 'bin' / env['WINDOWSSDKVERSION'].rstrip('\\/') / 'x86/rc.exe'
            if not rc.is_file():
                rc = None
        if not rc:
            raise RuntimeError('Windows SDK resource compiler unavailable')
        resource = work / 'probe.res'
        run([rc, '/nologo', '/fo', resource, recipe], cwd=work, env=env)
        executable = work / 'probe.exe'
        run([compiler, '/nologo', '/std:c++20', '/EHsc', '/MT', '/O2',
             '/I' + str(ROOT / 'third_party'), '/I' + str(ROOT / 'third_party/reshade'),
             ROOT / 'tests/native/compatibility_bridge_probe.cpp', '/Fe:' + str(executable),
             '/link', resource, 'd3d11.lib', 'd3d12.lib', 'dxgi.lib', 'd3dcompiler.lib', 'user32.lib',
             'bcrypt.lib', 'psapi.lib', 'cabinet.lib', 'shell32.lib', 'ole32.lib', 'uuid.lib', 'windowscodecs.lib'],
            cwd=work, env=env)
        for config in sources:
            model = json.loads(config.read_text(encoding='utf-8-sig'))
            model = model.get('weight_source', model).get('model', 'original')
            cases = [(provider, height) for height in args.heights for provider in (2, 4, 5, 6)]
            cases.append((3, args.heights[0]))
            for repeat in range(args.rounds):
                for provider, height in (cases if repeat % 2 == 0 else reversed(cases)):
                    result = json.loads(run([executable, config, 'cpu_motion', '2560', '1080',
                        args.frames, provider, height], cwd=work, env=env))
                    result.update(model=model, provider=names[provider], round=repeat)
                    report['cases'].append(result)
                    save()
                    print(f"{model} {names[provider]} {height}: {result['roundtrip_median_ms']:.2f} ms", flush=True)
        for group in sorted({(row['model'], row['source'], row['analysis_height']) for row in report['cases']}):
            rows = [row for row in report['cases'] if (row['model'], row['source'], row['analysis_height']) == group]
            report['summary'].append({'model': group[0], 'provider': names[group[1]], 'analysis_height': group[2],
                **{key: statistics.median(row[key] for row in rows)
                   for key in ('roundtrip_median_ms', 'roundtrip_p95_ms')},
                **{'motion_' + key + '_ms': statistics.median(row['motion']['motion_wall_ms'][key] for row in rows)
                   for key in ('median', 'p95')}})
        report['passed'] = True
        save()


if __name__ == '__main__':
    main()
