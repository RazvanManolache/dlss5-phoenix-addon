"""Focused x86/x64 path, cache and child-process CPU tests; never uses the GPU."""
import argparse
import json
import tempfile
from pathlib import Path
from test_model_selection import ROOT, toolchain, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/current/driver-discovery-tests.json')
    args = parser.parse_args()
    results = []
    for bits in (32, 64):
        compiler, env = toolchain(bits)
        with tempfile.TemporaryDirectory(prefix='driver_tests_', dir=ROOT/'build') as temporary:
            work = Path(temporary)
            for name in ('wow64_driver_discovery_tests', 'wow64_driver_process_tests'):
                executable = work/(name+'.exe')
                run([compiler, '/nologo', '/std:c++20', '/EHsc', '/O2', '/MD', '/DNOMINMAX',
                     ROOT/'tests/native'/(name+'.cpp'), '/Fo:'+str(work)+'\\', '/Fe:'+str(executable),
                     'bcrypt.lib', 'advapi32.lib', 'shell32.lib', 'ole32.lib', 'cabinet.lib', 'uuid.lib'], cwd=work, env=env)
                command = [executable]
                if name.endswith('process_tests'):
                    command.append(work/'fixture')
                result = json.loads(run(command, cwd=work, env=env))
                if result.get('passed') is not True:
                    raise RuntimeError('Driver discovery test failed')
                results.append({'test': name, 'bits': bits, 'result': result})
                print(json.dumps(results[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'passed': True, 'gpu_used': False, 'results': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
