"""Compare CPU operand loaders to a read-only pre-change source directory."""
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
    with tempfile.TemporaryDirectory(prefix='cpu-operands-', dir=build) as folder:
        work = Path(folder).resolve()
        if work.parent != build:
            raise RuntimeError('Invalid cleanup root')
        objects = []
        for style, arch in [('avx2', 'AVX2'), ('avx512', 'AVX512')]:
            for label, root in [('control', args.control.resolve()), ('candidate', ROOT / 'native_runtime')]:
                obj = work / f'{label}_{style}.obj'
                command = [compiler, '/nologo', '/std:c++20', '/EHsc', '/O2', '/MD',
                           '/DNOMINMAX', f'/arch:{arch}', '/c', root / f'cpu_{style}.cpp',
                           '/I' + str(ROOT / 'native_runtime'), '/Fo' + str(obj)]
                if label == 'control':
                    command += [f'/Dphoenix_cpu_{style}=control_{style}',
                                f'/Dphoenix_cpu_{style}_range=control_{style}_range']
                run(command, cwd=work, env=env)
                objects.append(obj)
        executable = work / 'cpu_operands.exe'
        run([compiler, '/nologo', '/std:c++20', '/EHsc', '/O2', '/MD', '/DNOMINMAX',
             ROOT / 'tests/portable_cpp/cpu_operand_reuse_tests.cpp', *objects,
             '/Fe' + str(executable)], cwd=work, env=env)
        output = run([executable], cwd=work, env=env, timeout=180)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(passed=True, scope='single-thread resident GEMM operators',
                                               output=output), indent=2) + '\n', encoding='utf-8')
        print(output)


if __name__ == '__main__':
    main()
