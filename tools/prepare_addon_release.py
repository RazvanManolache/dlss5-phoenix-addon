"""Assemble audited add-on release assets from the current verified candidate."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import zipfile

from audit_release import ROOT, audit, validate_public_content
from validate_portable_bridge import resources

VERSION = '0.81'
CURRENT = ROOT/'artifacts/current'
CANDIDATE = CURRENT/'addon-portable-candidate'
FORBIDDEN = {'.log', '.dmp', '.pdb', '.bmp', '.jpg', '.png', '.pt', '.pth',
             '.safetensors', '.onnx', '.npy', '.npz'}
LEGACY_DIAGNOSTIC_COMPONENTS = {
    'runtime/phoenix_cpp_devices.exe', 'runtime/phoenix_native_broker.exe',
    'runtime/phoenix_runtime_cpp.dll', 'runtime/phoenix_sycl_cpp.dll'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_payload(name, raw):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive name: '+name)
    if (rel.suffix.lower() in FORBIDDEN or rel.name.lower().startswith('nvngx') or
            any(part.lower() in {'logs', 'captures', 'models', '.git'} for part in rel.parts)):
        raise ValueError('Excluded binary-package file: '+name)
    scanned = raw
    compact = raw.replace(b'\x00', b'')
    if name in LEGACY_DIAGNOSTIC_COMPONENTS and raw[:2] == b'MZ':
        # The retained binaries have __FILE__ assertion locations at their former
        # project source root. This is not a user/game/config path. Never extend
        # this exception to arbitrary files, usernames or diagnostic JSON.
        old_root = b'E:'+bytes([92])+b'Games'+bytes([92])+b'Reshade'+bytes([92])+b'dlss5-phoenix-addon'+bytes([92])
        scanned = scanned.replace(old_root, b'PROJECT_SOURCE/')
        compact = compact.replace(old_root, b'PROJECT_SOURCE/')
    validate_public_content(name, scanned)
    # Windows binaries can contain UTF-16 strings as well as ASCII strings.
    validate_public_content(name, compact)


def decompress(raw, count):
    cabinet = ctypes.WinDLL('cabinet', use_last_error=True)
    cabinet.CreateDecompressor.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    cabinet.Decompress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    cabinet.CloseDecompressor.argtypes = [ctypes.c_void_p]
    handle = ctypes.c_void_p()
    if not cabinet.CreateDecompressor(4, None, ctypes.byref(handle)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        output = ctypes.create_string_buffer(count)
        written = ctypes.c_size_t()
        if not cabinet.Decompress(handle, ctypes.create_string_buffer(raw), len(raw),
                                  output, count, ctypes.byref(written)) or written.value != count:
            raise ValueError('Invalid embedded compression payload')
        return output.raw
    finally:
        cabinet.CloseDecompressor(handle)


def audit_embedded(binary, directory):
    directory.mkdir()
    _, retained = resources(binary, directory)
    manifest = json.loads((directory/'1001.bin').read_bytes())
    if manifest.get('model_data_included') is not False:
        raise ValueError('Embedded payload is not declared model-free')
    nested = []
    for entry in manifest['files']:
        raw = (directory/f"{entry['resource']}.bin").read_bytes()
        if entry['algorithm'] == 'xpress_huff':
            raw = decompress(raw, entry['bytes'])
        elif entry['algorithm'] != 'raw':
            raise ValueError('Unknown embedded compression')
        if len(raw) != entry['bytes'] or sha(raw) != entry['sha256']:
            raise ValueError('Embedded hash mismatch: '+entry['path'])
        safe_payload(entry['path'], raw)
        if PurePosixPath(entry['path']).suffix in {'.addon64', '.addon32'}:
            path = directory/f"nested-{entry['resource']}.dll"
            path.write_bytes(raw)
            nested.append(audit_embedded(path, directory/f"nested-{entry['resource']}"))
    return dict(resources=len(manifest['files']), nested=nested,
                retained_payload_sha256=retained)


def write_archive(path, files):
    inventory = {}
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(files.items()):
            safe_payload(name, raw)
            inventory[name] = sha(raw)
            archive.writestr(name, raw)
        archive.writestr('FILES_SHA256.json', json.dumps(inventory, indent=2)+'\n')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip():
            raise ValueError('Archive CRC failure: '+path.name)
        for name, expected in inventory.items():
            if sha(archive.read(name)) != expected:
                raise ValueError('Archive content mismatch: '+name)
    return dict(files=len(inventory), bytes=path.stat().st_size, sha256=sha(path.read_bytes()))


def common_files():
    files = {name: (ROOT/name).read_bytes() for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY.md')}
    for path in (ROOT/'third_party').rglob('*'):
        if path.is_file() and (path.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING', 'MIT_LICENSE'))
                               or path.name == 'ReShade-LICENSE.txt'):
            files['licenses/'+path.relative_to(ROOT/'third_party').as_posix()] = path.read_bytes()
    for name in ('EXECUTION_MECHANISMS.md', 'WOW64_EXPERIMENT.md', 'X86_NATIVE_INVESTIGATION.md', 'X86_VULKAN.md', 'RELEASE_0.81.md', 'VALIDATION_0.81.md'):
        files['docs/'+name] = (ROOT/'docs'/name).read_bytes()
    files['README.txt'] = (
        'DLSS5 Phoenix add-on 0.81\n'
        'Copy the matching Phoenix.addon32 OR Phoenix.addon64 beside your game ReShade DLL.\n'
        'Keep the graphics helper and notices alongside it. Select your external model in Phoenix.\n'
        'Leave Advanced transport on Automatic initially. Optional runtimes are separate downloads.\n'
        'Experimental native x86 paths are Windows/driver-profile specific, not universally validated.\n'
        'See docs/RELEASE_0.81.md for installation, features, evidence and limits.\n'
        'No NVIDIA model DLLs, model weights, user configurations, logs or captures are included.\n'
        'https://github.com/RazvanManolache/dlss5-phoenix-addon/releases/tag/0.81\n').encode()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wow64ext', type=Path, required=True, help='Pinned unmodified WOW64Ext source checkout')
    parser.add_argument('--ref', default='HEAD', help='Publication commit with exactly the current source tree')
    args = parser.parse_args()
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=ROOT).returncode:
        raise ValueError('Commit maintained source before assembling a release')
    commit = subprocess.check_output(['git', 'rev-parse', args.ref+'^{commit}'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'rev-parse', commit+'^{tree}'], cwd=ROOT) != subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT):
        raise ValueError('Publication ref must have the exact current source tree')
    manifest, source = audit()
    tracked = set(filter(None, subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')))
    if set(source) != tracked:
        raise ValueError('Source archive must equal the complete audited tracked tree')
    hashes = json.loads((CANDIDATE/'FILES_SHA256.json').read_text())
    for name, expected in hashes.items():
        path = CANDIDATE/name
        if not path.resolve().is_relative_to(CANDIDATE.resolve()) or sha(path.read_bytes()) != expected:
            raise ValueError('Candidate hash mismatch: '+name)
        safe_payload(name, path.read_bytes())
    compile_receipt = json.loads((CURRENT/'portable-bridge-validation.json').read_text())
    if not compile_receipt.get('passed'):
        raise ValueError('CPU integration/build verification did not pass')
    for name, expected in compile_receipt['source_sha256'].items():
        if sha((ROOT/name).read_bytes()) != expected:
            raise ValueError('Add-on source changed after compilation: '+name)
    output = CURRENT/f'release-{VERSION}'
    if output.exists():
        raise ValueError('Release output already exists; inspect it before replacement')
    with tempfile.TemporaryDirectory(prefix='release-', dir=ROOT/'artifacts') as temporary:
        staging = Path(temporary)
        embedded = {}
        for bits in (32, 64):
            embedded[str(bits)] = audit_embedded(CANDIDATE/f'Phoenix.addon{bits}', staging/str(bits))
            row = compile_receipt['architectures'][str(bits)]
            if hashes[f'Phoenix.addon{bits}'] != row['addon_sha256'] or embedded[str(bits)]['retained_payload_sha256'] != row['retained_payload_sha256']:
                raise ValueError('Add-on/embedded payload differs from compile evidence')
        assets = {}
        main_files = common_files()
        for name in ('Phoenix.addon32', 'Phoenix.addon64', 'phoenix_graphics_host.exe'):
            main_files[name] = (CANDIDATE/name).read_bytes()
        assets[f'DLSS5-Phoenix-{VERSION}-Windows.zip'] = write_archive(staging/f'DLSS5-Phoenix-{VERSION}-Windows.zip', main_files)
        portable = common_files()
        native = common_files()
        for name in hashes:
            if name.startswith(('PhoenixRuntime/win-x86/', 'PhoenixRuntime/win-x64/')):
                portable[name] = (CANDIDATE/name).read_bytes()
            elif name.startswith('PhoenixRuntime/wow64/'):
                native[name] = (CANDIDATE/name).read_bytes()
        from probe_wow64_native import WOW64EXT_COMMIT
        extension = args.wow64ext.resolve()
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=extension, text=True).strip() != WOW64EXT_COMMIT:
            raise ValueError('Wrong WOW64Ext source revision')
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=extension, text=True).strip():
            raise ValueError('WOW64Ext source is modified')
        for name in filter(None, subprocess.check_output(['git', 'ls-files', '-z'], cwd=extension).decode().split('\0')):
            if not name.startswith('bin/'):
                native['WOW64Ext-source/'+name] = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=extension)
        native['PhoenixRuntime/wow64/licenses/lgpl-3.0.txt'] = (extension/'lgpl-3.0.txt').read_bytes()
        for suffix, files in (('Portable-Runtimes', portable), ('Experimental-WOW64', native)):
            name = f'DLSS5-Phoenix-{VERSION}-{suffix}.zip'
            assets[name] = write_archive(staging/name, files)
        source_name = f'DLSS5-Phoenix-{VERSION}-Source.zip'
        source_path = staging/source_name
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip', f'--prefix=DLSS5-Phoenix-{VERSION}/',
                        f'--output={source_path}', commit], cwd=ROOT, check=True)
        with zipfile.ZipFile(source_path) as archive:
            for name, row in manifest['files'].items():
                raw = archive.read(f'DLSS5-Phoenix-{VERSION}/'+name)
                if sha(raw) != row['sha256']:
                    # Git archives use repository line endings, not worktree CRLF.
                    expected = subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT)
                    if raw != expected:
                        raise ValueError('Source archive mismatch: '+name)
                validate_public_content(name, raw)
            if archive.testzip():
                raise ValueError('Source archive CRC failure')
        assets[source_name] = dict(files=len(source), bytes=source_path.stat().st_size, sha256=sha(source_path.read_bytes()))
        validation = dict(schema='phoenix-addon-release/1', version=VERSION, commit=commit,
            source_audit_passed=True, archive_hashes_verified=True,
            private_paths_secrets_models_and_logs_excluded=True,
            retained_source_location_diagnostics='Four embedded legacy runtime binaries retain a project-source checkout root in __FILE__ assertions; no user or game configuration is included.',
            release_preparation_used_gpu=False, retail_games_launched=False,
            experimental_x86_game_presentation_validated=False,
            standalone_runtimes='Retained hash-verified builds; not rebuilt for this add-on release.',
            gpu_evidence='Pre-release isolated/package-client evidence, not a fresh release-binary game run.',
            cpu_contracts={bits: {name: dict(passed=row['passed']) for name, row in value['tests'].items()}
                for bits, value in compile_receipt['architectures'].items()},
            embedded_resources={bits: row['resources'] for bits, row in embedded.items()}, assets=assets)
        (staging/'VALIDATION.json').write_text(json.dumps(validation, indent=2)+'\n')
        (staging/'RELEASE_NOTES.md').write_text((ROOT/'docs/RELEASE_0.81.md').read_text(encoding='utf-8'), encoding='utf-8')
        upload_names = [*assets, 'VALIDATION.json']
        (staging/'SHA256SUMS.txt').write_text(''.join(sha((staging/name).read_bytes())+'  '+name+'\n' for name in upload_names), encoding='ascii')
        output.mkdir()
        for name in [*upload_names, 'SHA256SUMS.txt', 'RELEASE_NOTES.md']:
            (staging/name).replace(output/name)
    print(json.dumps(dict(passed=True, version=VERSION, commit=commit, assets=assets), indent=2))


if __name__ == '__main__':
    main()
