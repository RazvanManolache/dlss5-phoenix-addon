"""Run independent candidate CPU inference validation in both addon bitnesses."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--weight-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'build'):
        parser.error('Outputs must be inside build')
    output.mkdir(parents=True, exist_ok=False)
    finder = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = subprocess.check_output([str(finder), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    report = {'passed': False, 'gpu_executed': False, 'cases': []}
    try:
        for bits in (32, 64):
            folder = output / str(bits)
            folder.mkdir()
            resources = args.package.resolve() / f'embedded{bits}/runtime.res'
            recipe = folder / 'build.cmd'
            recipe.write_text(f'@echo off\ncall "{studio}/VC/Auxiliary/Build/vcvars{bits}.bat" > "{folder}/toolchain.log" 2>&1\n'
                f'cl /nologo /std:c++20 /Zc:__cplusplus /O2 /EHsc /MT /DNOMINMAX "{ROOT}/tests/native/model_validation_probe.cpp" '
                f'/Fo:"{folder}/probe.obj" /Fe:"{folder}/probe.exe" /link "{resources}" '
                f'user32.lib bcrypt.lib cabinet.lib shell32.lib ole32.lib uuid.lib > "{folder}/build.log" 2>&1\n'
                'exit /b %errorlevel%\n')
            subprocess.run(['cmd', '/d', '/c', str(recipe)], check=True)
            result = subprocess.run([str(folder / 'probe.exe'), str(args.weight_source.resolve())],
                cwd=folder, capture_output=True, text=True, timeout=200)
            (folder / 'stdout.log').write_text(result.stdout)
            (folder / 'stderr.log').write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f'{bits}-bit validation failed: {result.stderr}')
            proof = json.loads(result.stdout)
            if not proof['passed']:
                raise RuntimeError('Candidate validation failed')
            report['cases'].append(proof)
            print(json.dumps(proof), flush=True)
        report['passed'] = True
    finally:
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
