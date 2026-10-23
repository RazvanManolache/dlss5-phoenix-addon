"""Check final release DLLs and nested payloads without launching a game."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from audit_release import ROOT
from prepare_addon_release import audit_embedded, decompress
from test_model_selection import toolchain, run
from validate_portable_bridge import resources


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_payload(binary, destination, candidate, baseline_payload, core_hash):
    destination.mkdir()
    resources(binary, destination)
    manifest = json.loads((destination/'1001.bin').read_bytes())
    verified = {}
    for item in manifest['files']:
        raw = (destination/f"{item['resource']}.bin").read_bytes()
        if item['algorithm'] == 'xpress_huff':
            raw = decompress(raw, item['bytes'])
        elif item['algorithm'] != 'raw':
            raise ValueError('Unknown compression')
        actual = hashlib.sha256(raw).hexdigest()
        if len(raw) != item['bytes'] or actual != item['sha256']:
            raise ValueError('Embedded file changed: '+item['path'])
        name = item['path']
        if name.startswith(('native/', 'runtime/')):
            if baseline_payload.get(name) != actual:
                raise ValueError('Retained inference payload changed: '+name)
            verified[name] = actual
        if name == 'compatibility/phoenix_graphics_host.exe':
            if digest(candidate/'phoenix_graphics_host.exe') != actual:
                raise ValueError('Embedded graphics helper differs from the final helper')
        if name == 'compatibility/Phoenix.addon64':
            # The x86 helper loads the freshly built *core* addon. The public
            # standalone x64 addon is relinked with additional helper resources,
            # so its file hash intentionally differs from that embedded core.
            if core_hash != actual:
                raise ValueError('Nested x64 addon differs from the verified core build')
    expected = {name:value for name,value in baseline_payload.items()
                if name.startswith(('native/', 'runtime/'))}
    if verified != expected:
        raise ValueError('Retained inference payload inventory differs from the released baseline')
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline-validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    build = json.loads((candidate/'report.json').read_text())
    core = json.loads((candidate/'core-build.json').read_text())
    if not build.get('passed') or not core.get('passed'):
        raise ValueError('A successful core and package build is required')
    # Bind compiled sources, rather than unrelated packaging documentation or
    # tools which may acquire a version bump after an identical DLL was tested.
    for name in core['project_dependencies']:
        if digest(ROOT/name) != core['source_inputs'].get(name):
            raise ValueError('Compiled source differs: '+name)
    for group, minimum in (('cpu_tests', 24), ('software_d3d12_tests', 3)):
        if len(core[group]) < minimum or not all(row.get('passed') for row in core[group].values()):
            raise ValueError('Incomplete build tests: '+group)
    binaries = {name:digest(candidate/name) for name in
                ('Phoenix.addon32','Phoenix.addon64','phoenix_graphics_host.exe')}
    if any(build['binaries'][name]['sha256'] != value for name,value in binaries.items()):
        raise ValueError('Final binary differs from its build receipt')
    baseline = json.loads(args.baseline_validation.read_text())['retained_payload_sha256']
    report = dict(passed=False, release_payload_checked=False, release_channel=build['release_channel'],
        gpu_used=False, games_launched=False, installed_files_modified=False,
        older_gpu_inference_validated=False, binaries=binaries,
        cpu_tests=core['cpu_tests'], software_d3d12_tests=core['software_d3d12_tests'],
        dll_load={}, embedded={}, extraction={}, retained_payload_sha256={})
    try:
        with tempfile.TemporaryDirectory(prefix='release_check_', dir=ROOT/'build') as directory:
            work = Path(directory)
            for bits in (32,64):
                compiler, env = toolchain(bits)
                flags = [compiler,'/nologo','/std:c++20','/EHsc','/MT','/O2','/DNOMINMAX']
                loader = work/f'load{bits}.exe'
                run(flags+[ROOT/'tests/native/packaged_addon_load_tests.cpp',
                    f'/Fo:{work/f"load{bits}.obj"}',f'/Fe:{loader}'],cwd=work,env=env)
                binary = candidate/f'Phoenix.addon{bits}'
                report['dll_load'][str(bits)] = json.loads(run(
                    [loader,binary,build['release_channel']],cwd=work,env=env))
                extractor = work/f'extract{bits}.exe'
                run(flags+[ROOT/'tests/native/embedded_runtime_tests.cpp',
                    f'/Fo:{work/f"extract{bits}.obj"}',f'/Fe:{extractor}',
                    'bcrypt.lib','cabinet.lib','shell32.lib','ole32.lib'],cwd=work,env=env)
                extracted = work/f'extracted{bits}.json'
                run([extractor,binary,work/f'cache{bits}',extracted],cwd=work,env=env)
                result = json.loads(extracted.read_text())
                result.pop('installed_root',None)
                report['extraction'][str(bits)] = result
                report['embedded'][str(bits)] = audit_embedded(binary,work/f'audit{bits}')
                retained = inspect_payload(binary,work/f'payload{bits}',candidate,baseline,
                                           core['binaries']['phoenix.addon64'])
                if report['retained_payload_sha256'] and retained != report['retained_payload_sha256']:
                    raise ValueError('x86/x64 retained inference payloads differ')
                report['retained_payload_sha256'] = retained
                print(json.dumps(dict(bits=bits,passed=True,channel=build['release_channel'])),flush=True)
        report.update(passed=True, release_payload_checked=True, fresh_nested_addon_and_helper_verified=True)
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
