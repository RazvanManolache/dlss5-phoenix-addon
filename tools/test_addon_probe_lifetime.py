"""Exercise temporary-device unload/reload with the real addon and CPU-only NGX exports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from test_model_selection import toolchain, run

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = {'passed': False, 'cpu_only': True, 'architectures': {}}
    try:
        with tempfile.TemporaryDirectory(prefix='probe_lifetime_', dir=ROOT/'build') as temporary:
            work = Path(temporary)
            for bits in (32, 64):
                directory = work/str(bits)
                directory.mkdir()
                compiler, env = toolchain(bits)
                flags = [compiler, '/nologo', '/std:c++20', '/EHsc', '/MT', '/O2']
                stub = directory/'_nvngx.dll'
                host = directory/'probe.exe'
                run(flags+['/LD', ROOT/'tests/native/ngx_probe_stub.cpp',
                    f'/Fo:{directory/"stub.obj"}', f'/Fe:{stub}'], cwd=directory, env=env)
                run(flags+[ROOT/'tests/native/addon_probe_lifetime_tests.cpp',
                    f'/Fo:{directory/"host.obj"}', f'/Fe:{host}'], cwd=directory, env=env)
                addon = (args.package/f'Phoenix.addon{bits}').resolve()
                result = subprocess.run([str(host), str(addon), str(stub)], cwd=directory,
                    env=env, capture_output=True, text=True, timeout=30)
                row = {'returncode': result.returncode, 'stdout': result.stdout,
                    'stderr': result.stderr, 'addon_sha256': hashlib.file_digest(addon.open('rb'), 'sha256').hexdigest()}
                report['architectures'][str(bits)] = row
                if result.returncode:
                    raise RuntimeError(f'{bits}-bit addon probe lifetime failed: {result.stderr}')
                row['result'] = json.loads(result.stdout)
                if row['result'].get('passed') is not True:
                    raise RuntimeError(f'{bits}-bit addon probe receipt failed')
                sl_stub = directory/'sl.interposer.dll'
                sl_host = directory/'sl_tags.exe'
                run(flags+['/LD', ROOT/'tests/native/sl_probe_stub.cpp',
                    f'/Fo:{directory/"sl_stub.obj"}', f'/Fe:{sl_stub}'], cwd=directory, env=env)
                run(flags+[ROOT/'tests/native/sl_tag_safety_tests.cpp',
                    f'/Fo:{directory/"sl_host.obj"}', f'/Fe:{sl_host}'], cwd=directory, env=env)
                tagged = subprocess.run([str(sl_host), str(addon), str(sl_stub)], cwd=directory,
                    env=env, capture_output=True, text=True, timeout=30)
                row['streamline_tags'] = {'returncode': tagged.returncode,
                    'stdout': tagged.stdout, 'stderr': tagged.stderr}
                if tagged.returncode or json.loads(tagged.stdout).get('passed') is not True:
                    raise RuntimeError(f'{bits}-bit foreign Streamline resource test failed')
                provider = directory/'second_provider.dll'
                switch_host = directory/'provider_switch.exe'
                run(flags+['/LD', ROOT/'tests/native/reshade_provider_stub.cpp',
                    f'/Fo:{directory/"provider.obj"}', f'/Fe:{provider}'], cwd=directory, env=env)
                run(flags+[ROOT/'tests/native/addon_provider_switch_tests.cpp',
                    f'/Fo:{directory/"provider_host.obj"}', f'/Fe:{switch_host}'], cwd=directory, env=env)
                switched = subprocess.run([str(switch_host), str(addon), str(provider)],
                    cwd=directory, env=env, capture_output=True, text=True, timeout=30)
                row['provider_switch'] = {'returncode': switched.returncode,
                    'stdout': switched.stdout, 'stderr': switched.stderr}
                if switched.returncode or json.loads(switched.stdout).get('passed') is not True:
                    raise RuntimeError(f'{bits}-bit ReShade provider switch failed: {switched.stderr}')
                fixture = directory/'fixture_options.exe'
                run(flags+[ROOT/'tests/native/graphics_fixture_options_tests.cpp',
                    f'/Fo:{directory/"fixture.obj"}', f'/Fe:{fixture}'], cwd=directory, env=env)
                row['fixture_options'] = json.loads(run([fixture], cwd=directory, env=env))
        report['passed'] = True
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
