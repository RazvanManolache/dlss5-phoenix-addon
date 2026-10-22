"""Build and smoke-test the optional x86/x64 background-input companion."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from audit_release import ROOT
from prepare_addon_release import safe_payload, sha
from test_model_selection import toolchain, run
from build_compatibility import pe_exports, pe_machine


def read_json(path, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            time.sleep(0.03)
    raise RuntimeError('Input companion mailbox timed out: ' + path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/current/input-addon')
    args = parser.parse_args()
    source = ROOT / 'src/integration/input/phoenix_input_addon.cpp'
    bindings = {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in
        (source, Path(__file__), ROOT / 'tests/native/input_addon_load_tests.cpp')}
    results, binaries = {}, {}
    with tempfile.TemporaryDirectory(prefix='input-addon-', dir=ROOT / 'build') as temp:
        work = Path(temp)
        for bits in (32, 64):
            folder = work / str(bits)
            folder.mkdir()
            compiler, env = toolchain(bits)
            target = folder / f'PhoenixInput.addon{bits}'
            minhook = ROOT / 'third_party/minhook'
            sources = [minhook / 'src' / n for n in
                       ('hook.c', 'buffer.c', 'trampoline.c', f'hde/hde{bits}.c')]
            common = [compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/O2', '/EHsc',
                      '/MT', '/DNOMINMAX', '/experimental:deterministic', f'/pathmap:{ROOT}=PROJECT_SOURCE']
            # Publish architecture-neutral names for ReShade's optional callbacks.
            definition = folder / 'input.def'
            definition.write_text('EXPORTS\nAddonInit\nAddonUninit\n')
            run(common + ['/LD', '/I' + str(ROOT / 'third_party'),
                '/I' + str(ROOT / 'third_party/reshade'), '/I' + str(minhook / 'include'),
                source, *sources, f'/Fo:{folder}\\', f'/Fe:{target}', '/link', '/INCREMENTAL:NO',
                f'/DEF:{definition}', 'user32.lib', 'dxguid.lib'], cwd=folder, env=env)
            assert pe_machine(target) == (0x14c if bits == 32 else 0x8664)
            assert {'AddonInit', 'AddonUninit', 'NAME', 'DESCRIPTION'}.issubset(pe_exports(target))
            safe_payload(target.name, target.read_bytes())
            host = folder / 'input_test.exe'
            run(common + [ROOT / 'tests/native/input_addon_load_tests.cpp',
                f'/Fo:{folder / "input_test.obj"}', f'/Fe:{host}', 'user32.lib'], cwd=folder, env=env)
            process = subprocess.Popen([str(host), str(target)], cwd=folder, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                mailbox = folder / 'PhoenixOutput/input'
                endpoint = read_json(mailbox / 'endpoint.json')
                assert endpoint['schema'] == 'phoenix-input/1'
                # Status requests exercise session matching without sending input.
                for request_id, session, expected in [('status', endpoint['session'], True),
                                                       ('stale', 'stale-session', False)]:
                    request = dict(schema='phoenix-input/1', session=session, id=request_id,
                                   ops=[dict(op='status')])
                    staged = mailbox / 'request.tmp'
                    staged.write_text(json.dumps(request))
                    staged.replace(mailbox / 'request.json')
                    deadline = time.monotonic() + 10
                    while True:
                        response = read_json(mailbox / 'response.json')
                        if response['id'] == request_id:
                            break
                        if time.monotonic() > deadline:
                            raise RuntimeError('Input request was not acknowledged')
                        time.sleep(0.03)
                    assert response['ok'] is expected
                output, error = process.communicate('stop\n', timeout=20)
                if process.returncode:
                    raise RuntimeError('Input addon load/unload failed: ' + error)
                result = json.loads(output.splitlines()[-1])
                assert result['passed']
                results[str(bits)] = dict(**result, mailbox_status=True, stale_session_rejected=True,
                                         game_input_injection_validated=False)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            binaries[target.name] = sha(target.read_bytes())
        for name, digest in bindings.items():
            assert sha((ROOT / name).read_bytes()) == digest, name
        args.output.mkdir(parents=True, exist_ok=True)
        for bits in (32, 64):
            shutil.copy2(work / str(bits) / f'PhoenixInput.addon{bits}', args.output)
        receipt = dict(passed=True, gpu_used=False, games_launched=False, input_macros_sent=False,
                       binaries=binaries, source_inputs=bindings, cases=results)
        (args.output / 'VALIDATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
