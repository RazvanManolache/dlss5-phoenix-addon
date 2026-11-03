"""Check the versioned WOW64 pixel contract; GPU checks require --gpu."""
import argparse
import json
from pathlib import Path
import tempfile
from test_model_selection import ROOT, toolchain, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Run isolated D3D12 conversions, never models or games')
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix='wow64-pixels-', dir=ROOT / 'build') as directory:
        work = Path(directory).resolve()
        for bits in (32, 64):
            compiler, env = toolchain(bits)
            flags = [compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/O2', '/fp:strict', '/EHsc', '/MT', '/bigobj', '/DNOMINMAX']
            cases = ['wow64_pixel_contract_tests']
            if args.gpu and bits == 64:
                cases.append('wow64_pixel_gpu_tests')
            for name in cases:
                exe = work / f'{name}-{bits}.exe'
                run(flags + [ROOT / f'tests/native/{name}.cpp', '/Fo:' + str(work / f'{name}-{bits}.obj'),
                    '/Fe:' + str(exe), 'd3d12.lib', 'dxgi.lib', 'd3dcompiler.lib', 'user32.lib'], cwd=work, env=env)
                results.append({'bits': bits, 'test': name, 'result': json.loads(run([exe], cwd=work, env=env, timeout=90))})
    report = {'passed': True, 'gpu_used': args.gpu, 'models_run': False, 'games_run': False, 'results': results}
    output = ROOT / 'artifacts/current/wow64-pixel-transport-validation.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
