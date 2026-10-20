"""Build the current dual-architecture addon package without retaining staging."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ngx-sdk', 'runtime', 'native', 'reshade64'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--experimental-native', action='store_true',
                        help='Build explicitly experimental multi-architecture kernels')
    args = parser.parse_args()
    build = ROOT / 'build'
    with tempfile.TemporaryDirectory(prefix='current_', dir=build) as temporary:
        anchor = Path(temporary)
        core, package = [build / (anchor.name + suffix) for suffix in ('_core', '_package')]
        try:
            runtime = anchor / 'runtime'
            # The builder recompiles its own forwarding shim from current source.
            shutil.copytree(args.runtime.resolve(), runtime,
                            ignore=shutil.ignore_patterns('phoenix-nvngx.dll'))
            subprocess.run([sys.executable, str(ROOT / 'tools/build_addon.py'), core.name,
                            '--ngx-sdk', str(args.ngx_sdk.resolve()), '--production-only',
                            '--embed-runtime', str(runtime),
                            '--embed-native', str(args.native.resolve()),
                            *(['--experimental-native'] if args.experimental_native else [])], cwd=ROOT, check=True)
            subprocess.run([sys.executable, str(ROOT / 'tools/build_compatibility.py'), package.name,
                            '--core-build', str(core), '--reshade64', str(args.reshade64.resolve()),
                            '--ngx-sdk', str(args.ngx_sdk.resolve())],
                           cwd=ROOT, check=True)
            report = json.loads((package / 'report.json').read_text())
            if report.get('passed') is not True:
                raise RuntimeError('Package verification failed')
            current = ROOT / 'artifacts/current' / ('addon-experimental-rtx' if args.experimental_native else 'addon')
            current.mkdir(parents=True, exist_ok=True)
            for name in ('Phoenix.addon32', 'Phoenix.addon64', 'phoenix_graphics_host.exe', 'report.json'):
                shutil.copy2(package / name, current / name)
            shutil.copy2(core / 'report.json', current / 'core-build.json')
            print('Promoted verified x86/x64 package to ' + str(current.relative_to(ROOT)), flush=True)
        finally:
            for path in (core, package):
                if path.exists():
                    if path.is_symlink() or path.is_junction() or path.resolve().parent != build.resolve():
                        raise RuntimeError('Unsafe build staging cleanup target')
                    shutil.rmtree(path)


if __name__ == '__main__':
    main()
