"""Check native file-backed weights under a strict host-allocation ceiling."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from test_model_selection import ROOT, run, toolchain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='External dual-model DLL, read only')
    parser.add_argument('--bundle', type=Path, required=True, help='Model-free native kernel bundle')
    args = parser.parse_args()
    sources = ('src/native/model/weight_view.h', 'src/native/model/native_weight_reader.h',
               'src/native/model/weight_source.h', 'src/native/model/portable_model_io.h',
               'src/native/model/bundle_loader.h', 'tests/native/streamed_weight_tests.cpp')
    report = {'passed': False, 'gpu_used': False, 'cases': [],
              'sources': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}}
    build = ROOT/'build'
    build.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='streamed-weights-', dir=build) as temporary:
            work = Path(temporary)
            for bits in (32, 64):
                compiler, environment = toolchain(bits)
                executable = work/f'weights{bits}.exe'
                run([compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/EHsc', '/bigobj', '/O2',
                     '/MT', '/DNOMINMAX', ROOT/'tests/native/streamed_weight_tests.cpp',
                     '/Fo:'+str(work/f'weights{bits}.obj'), '/Fe:'+str(executable), 'bcrypt.lib'],
                    cwd=work, env=environment)
                for model in ('original', 'small-fp8'):
                    result = json.loads(run([executable, args.model.resolve(), args.bundle.resolve(), model],
                                            cwd=work, env=environment))
                    report['cases'].append({'bits': bits, **result})
                    print(json.dumps(report['cases'][-1]), flush=True)
        report['passed'] = len(report['cases']) == 4 and all(case['passed'] for case in report['cases'])
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        report['temporary_build_removed'] = True
        (ROOT/'artifacts/current/streamed-weight-validation.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
