"""Bounded model-free CPU contracts, with GPU probes only when --gpu is supplied.

Full-model inference is a separate explicit operation. This runner never loads
weights, launches games, packages a release or modifies installed files.
"""
import argparse
import json
from pathlib import Path
import tempfile

from test_model_selection import ROOT, run, toolchain
from test_kernel_structure import gpu_contracts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bits', nargs='+', type=int, choices=(32, 64), default=[32, 64])
    parser.add_argument('--gpu', action='store_true', help='Explicitly execute three isolated CUDA probes')
    parser.add_argument('--output', type=Path, help='Optional compact JSON receipt')
    args = parser.parse_args()
    report = dict(passed=False, gpu_requested=args.gpu, inference_executed=False,
                  games_launched=False, installed_files_modified=False, cases=[])
    build_root = (ROOT / 'build').resolve()
    build_root.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='numeric-contracts-', dir=build_root) as folder:
            work = Path(folder).resolve()
            # Verify the absolute cleanup target before TemporaryDirectory's
            # recursive removal. This folder is created exclusively for this run.
            if work.parent != build_root or not work.name.startswith('numeric-contracts-'):
                raise RuntimeError('Temporary build target escaped workspace')
            for bits in args.bits:
                compiler, env = toolchain(bits)
                for group, name in (('native', 'integer_geometry_tests'),
                                    ('native', 'kernel_word_contract_tests'),
                                    ('portable_cpp', 'numeric_contract_tests')):
                    stem = name + str(bits)
                    executable = work / (stem + '.exe')
                    run([compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/EHsc',
                         '/O2', '/fp:strict', '/DNOMINMAX', ROOT / f'tests/{group}/{name}.cpp',
                         '/Fo:' + str(work / (stem + '.obj')), '/Fe:' + str(executable)],
                        cwd=work, env=env)
                    result = json.loads(run([executable], cwd=work, env=env))
                    if result.get('passed') is not True:
                        raise RuntimeError(name + ' did not pass')
                    report['cases'].append(dict(bits=bits, test=name, result=result))
            if args.gpu:
                _, env = toolchain(64)
                report['gpu'] = gpu_contracts(work, env)
        report.update(passed=True, temporary_builds_removed=True)
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
