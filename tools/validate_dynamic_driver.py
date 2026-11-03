"""Explicit isolated driver startup/copy validation; no games or model inference."""
import argparse
import json
import tempfile
from pathlib import Path
from test_model_selection import ROOT, toolchain, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--allow-gpu-startup', action='store_true', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/current/dynamic-driver-validation.json')
    args = parser.parse_args()
    compiler, env = toolchain(32)
    with tempfile.TemporaryDirectory(prefix='driver_validation_', dir=ROOT/'build') as temporary:
        work = Path(temporary)
        executable = work/'validation.exe'
        run([compiler, '/nologo', '/std:c++20', '/EHsc', '/O2', '/MD', '/bigobj', '/DNOMINMAX',
             ROOT/'tests/native/wow64_driver_validation.cpp', '/Fo:'+str(work)+'\\', '/Fe:'+str(executable),
             'bcrypt.lib', 'advapi32.lib', 'shell32.lib', 'ole32.lib', 'cabinet.lib', 'uuid.lib'], cwd=work, env=env)
        result = json.loads(run([executable, args.runtime.resolve()], cwd=work, env=env, timeout=60))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
