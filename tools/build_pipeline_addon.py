"""Compile the evolving addon with its contract suites in disposable staging."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'artifacts/current/pipeline-addon'
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pipeline_addon_', dir=ROOT / 'build') as temporary:
        name = Path(temporary).name + '_core'
        core = ROOT / 'build' / name
        try:
            subprocess.run([sys.executable, ROOT / 'tools/build_addon.py', name, '--production-only',
                '--ngx-sdk', ROOT.parent / 'external/DLSS/include'], cwd=ROOT, check=True)
            report = json.loads((core / 'report.json').read_text())
            if not report.get('passed'): raise RuntimeError('Addon build did not pass')
            shutil.copy2(core / 'phoenix.addon64', output / 'Phoenix.addon64')
            (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
            (output / 'README.txt').write_text('Build/check artifact only. No embedded runtime or model bundle.\nFinal packaging follows integration validation.\n')
            (output / 'build-errors.json').unlink(missing_ok=True)
            print('Addon compilation and contract checks passed.', flush=True)
        finally:
            if core.exists():
                if core.resolve().parent != (ROOT / 'build').resolve() or core.is_symlink() or core.is_junction():
                    raise RuntimeError('Unsafe addon staging path')
                if any(p.is_symlink() or p.is_junction() for p in core.rglob('*')):
                    raise RuntimeError('Addon staging contains a link')
                errors = {}
                for path in core.glob('*.build.log'):
                    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
                    relevant = [line for line in lines if 'error ' in line or 'fatal error' in line]
                    if relevant: errors[path.name] = relevant[-25:]
                if errors:
                    (output / 'build-errors.json').write_text(json.dumps(errors, indent=2) + '\n')
                    print(json.dumps(errors), flush=True)
                failed_checks={}
                for path in core.glob('*.result.json'):
                    try:
                        result=json.loads(path.read_text(encoding='utf-8-sig'))
                        if not result.get('passed'):failed_checks[path.name]=result
                    except (ValueError,AttributeError):failed_checks[path.name]=path.read_text(encoding='utf-8',errors='replace')[-4000:]
                if failed_checks:
                    (output/'failed-checks.json').write_text(json.dumps(failed_checks,indent=2)+'\n')
                    print(json.dumps(failed_checks),flush=True)
                shutil.rmtree(core)


if __name__ == '__main__':
    main()
