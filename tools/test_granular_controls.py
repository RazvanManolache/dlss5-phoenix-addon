"""Focused Original/Lightweight granular-control regressions; no GPU device or game."""
import argparse
import hashlib
import json
from test_model_selection import ROOT, toolchain, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    work = ROOT / 'build/four_model_comparison'
    results = []
    for bits in (32, 64):
        compiler, env = toolchain(bits)
        name = f'granular_controls_tests{bits}'
        executable = work / (name + '.exe')
        run([compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/EHsc', '/O2', '/MT',
             '/DNOMINMAX', ROOT / 'tests/native/granular_controls_tests.cpp',
             '/Fo:' + str(work / (name + '.obj')), '/Fe:' + str(executable)], cwd=work, env=env)
        results.append({'bits': bits, **json.loads(run([executable], cwd=work, env=env))})
    from pathlib import Path
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = ['src/native/support/budget.h', 'src/native/support/stage_queries.h',
               'src/native/support/gpu_timing.h', 'src/native/model/native_layout.h',
               'src/native/support/processing.h', 'src/native/support/dispatch.h',
               'src/integration/reshade/shared_panel.h', 'src/integration/reshade/addon.cpp',
               'tests/native/granular_controls_tests.cpp']
    report = {'passed': all(r['passed'] for r in results), 'gpu_executed': False,
              'cases': results, 'source_sha256': {
                  p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
