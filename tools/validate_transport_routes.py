"""Compile and run x86/x64 route/availability contracts without loading GPU runtimes."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ngx-sdk', type=Path, required=True)
    args = parser.parse_args()
    sdk = args.ngx_sdk.resolve()
    vswhere = Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    results = {}
    with tempfile.TemporaryDirectory(prefix='transport-contracts-', dir=ROOT/'build') as temporary:
        work = Path(temporary)
        for bits in (64, 32):
            lines = [f'@call "{studio}/VC/Auxiliary/Build/vcvars{bits}.bat" >nul']
            for name in ('backbuffer_route_tests', 'transport_availability_tests'):
                target = work/f'{name}{bits}.exe'
                lines += [f'cl /nologo /std:c++20 /EHsc /MD /DNOMINMAX /I"{ROOT}/third_party" '
                    f'/I"{ROOT}/third_party/nvapi" /I"{sdk}" "{ROOT}/tests/native/{name}.cpp" '
                    f'/Fo:"{work}/{name}{bits}.obj" /Fe:"{target}" /link bcrypt.lib cabinet.lib shell32.lib '
                    f'ole32.lib uuid.lib user32.lib > "{work}/{name}{bits}.log" 2>&1',
                    '@if errorlevel 1 exit /b 1', f'"{target}" > "{work}/{name}{bits}.json"',
                    '@if errorlevel 1 exit /b 2']
            recipe = work/f'contracts{bits}.cmd'
            recipe.write_text('\n'.join(lines))
            result = subprocess.run(['cmd', '/d', '/c', str(recipe)], cwd=work, text=True, capture_output=True)
            if result.returncode:
                for path in work.glob(f'*{bits}.log'):
                    print(path.name, path.read_text(errors='replace'))
                for path in work.glob(f'*{bits}.json'):
                    print(path.name, path.read_text(errors='replace'))
                raise RuntimeError('Transport contracts failed')
            results[str(bits)] = {name: json.loads((work/f'{name}{bits}.json').read_text())
                for name in ('backbuffer_route_tests', 'transport_availability_tests')}
    output = ROOT/'artifacts/current/transport-routing'
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(passed=True, cpu_only=True, architectures=results)
    (output/'CONTRACTS.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
