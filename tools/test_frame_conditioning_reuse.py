"""Check exact packed frame inputs and compare control/candidate CPU costs."""
import argparse
import json
from pathlib import Path
import tempfile
from test_model_selection import ROOT, run, toolchain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compiler, env = toolchain(64)
    build = (ROOT / 'build').resolve()
    source = (ROOT / 'tests/portable_cpp/frame_conditioning_reuse_tests.cpp').read_text(encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='frame-conditioning-', dir=build) as folder:
        work = Path(folder).resolve()
        if work.parent != build:
            raise RuntimeError('Invalid cleanup root')
        executables = {}
        for label, header in [('control', args.control.resolve()), ('candidate', ROOT / 'native_runtime/frame.h')]:
            text = source.replace('../../native_runtime/frame.h', header.as_posix()).replace(
                '../../src/native/model/weight_source.h', (ROOT / 'src/native/model/weight_source.h').as_posix())
            cpp, exe = work / (label + '.cpp'), work / (label + '.exe')
            cpp.write_text(text, encoding='utf-8')
            run([compiler, '/nologo', '/std:c++20', '/EHsc', '/O2', '/MD', '/DNOMINMAX',
                 '/I' + str(ROOT / 'native_runtime'), cpp, '/Fe' + str(exe), '/link', 'bcrypt.lib'], cwd=work, env=env)
            executables[label] = exe
        # Alternating process order exposes run-to-run scheduling variation.
        results = {label: [] for label in executables}
        for pair in range(3):
            for label in (['control', 'candidate'] if pair % 2 == 0 else ['candidate', 'control']):
                results[label].append(json.loads(run([executables[label]], cwd=work, env=env, timeout=180)))
        expected = [case['sha256'] for case in results['control'][0]['cases']]
        for runs in results.values():
            for result in runs:
                if [case['sha256'] for case in result['cases']] != expected:
                    raise RuntimeError('Conditioning optimization changed packed frame inputs')
        receipt = dict(passed=True, exact_cases=len(expected), paired_processes=3, results=results)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        print(f'{len(expected)} packed-input cases exact across all six processes')


if __name__ == '__main__':
    main()
