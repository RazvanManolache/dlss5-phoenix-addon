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

VERSION = '0.19'
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
        if path.is_file() and (path.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING', 'MIT_LICENSE')) or path.name == 'ReShade-LICENSE.txt'):
            files['licenses/'+path.relative_to(ROOT/'third_party').as_posix()] = path.read_bytes()
    for name in (f'RELEASE_{VERSION}.md', f'VALIDATION_{VERSION}.md', 'RELEASE_0.10.md', 'VALIDATION_0.10.md', 'RELEASE_0.9.md', 'VALIDATION_0.9.md', 'NEURAL_CONTROLS.md', 'NATIVE_CORRECTNESS_0.82.md',
                 'FEEDER_API.md', 'LIBRARY.md', 'NATIVE_LIBRARY.md', 'SUPPORT.md',
                 'MULTI_MODEL.md', 'SMALL_FP8.md', 'SHARED_MODELS.md', 'NATIVE_ARCHITECTURES.md',
                 'EXECUTION_MECHANISMS.md', 'PORTABLE_MODEL_CONTROLS.md',
                 'PROCESSING_PIPELINE.md', 'PROCESSING_CHAIN.md', 'overlay-ui.md', 'model-switching.md', 'PAIRED_SCREENSHOTS.md', 'PHOENIX_INPUT.md', 'UPSTREAM_CREDITS.md'):
        files['docs/'+name] = (ROOT/'docs'/name).read_bytes()
    files['README.txt'] = (f'Phoenix DLSS5 add-on {VERSION}\n'
        'Use a current ReShade build with full add-on support. Copy only the matching add-on and graphics helper.\n'
        'Select your separately obtained compatible model in the Phoenix panel. New artistic inputs are experimental.\n'
        f'See docs/RELEASE_{VERSION}.md for installation and validation boundaries.\n'
        'No model weights, NVIDIA model DLLs or private configurations are included.\n').encode()
    return files


def input_package(candidate):
    """Keep the optional input hooks outside the neural-rendering packages."""
    candidate = Path(candidate).resolve()
    receipt = json.loads((candidate/'VALIDATION.json').read_text(encoding='utf-8'))
    expected = {'PhoenixInput.addon32', 'PhoenixInput.addon64'}
    required_sources = {'src/integration/input/phoenix_input_addon.cpp',
                        'tools/build_input_addon.py', 'tests/native/input_addon_load_tests.cpp'}
    if receipt.get('passed') is not True or set(receipt.get('binaries', {})) != expected:
        raise ValueError('Input companion build validation is incomplete')
    sources = receipt.get('source_inputs', {})
    if not required_sources.issubset(sources):
        raise ValueError('Input companion source bindings are incomplete')
    for name, digest in sources.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha(path.read_bytes()) != digest:
            raise ValueError('Input companion source changed: '+name)
    for bits in ('32', '64'):
        case = receipt.get('cases', {}).get(bits, {})
        if not all(case.get(key) is True for key in ('passed', 'actual_dll_loaded',
                'balanced_attach_detach', 'mailbox_status', 'stale_session_rejected')):
            raise ValueError('Input companion smoke validation is incomplete: '+bits)
    files = {name: (ROOT/name).read_bytes() for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY.md')}
    for name in expected:
        raw = (candidate/name).read_bytes()
        if sha(raw) != receipt['binaries'][name]:
            raise ValueError('Input companion changed: '+name)
        safe_payload(name, raw)
        files[name] = raw
    for name in ('PHOENIX_INPUT.md', f'VALIDATION_{VERSION}.md'):
        files['docs/'+name] = (ROOT/'docs'/name).read_bytes()
    for name in ('minhook/LICENSE.txt', 'reshade/LICENSE.txt',
                 'nlohmann/LICENSE.MIT', 'nlohmann/json.hpp'):
        files['licenses/'+name] = (ROOT/'third_party'/name).read_bytes()
    files['VALIDATION.json'] = (json.dumps(receipt, indent=2)+'\n').encode()
    files['README.txt'] = (
        f'Phoenix Input {VERSION} - optional experimental background-input companion\n'
        'Copy only PhoenixInput.addon32 OR PhoenixInput.addon64, matching the game architecture, beside ReShade.\n'
        'The rendering addon does not require this companion. Virtual focus is off by default.\n'
        'See docs/PHOENIX_INPUT.md for setup, mailbox operations, limits and removal.\n'
        'Smoke checks cover loading, status/session handling and clean unload, not broad gameplay compatibility.\n').encode()
    return files, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', default='HEAD')
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-x86', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--experimental-candidate', type=Path)
    parser.add_argument('--experimental-validation', type=Path)
    parser.add_argument('--input-candidate', type=Path, required=True,
                        help='Separately built and smoke-tested optional input companion')
    args = parser.parse_args()
    if bool(args.experimental_candidate) != bool(args.experimental_validation):
        parser.error('Supply both the experimental candidate and its validation receipt')
    if subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=ROOT).returncode:
        raise ValueError('Commit maintained source before assembling a release')
    commit = subprocess.check_output(['git', 'rev-parse', args.ref+'^{commit}'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'rev-parse', commit+'^{tree}'], cwd=ROOT) != subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT):
        raise ValueError('Publication ref must match this source tree')
    source_manifest, source = audit()
    tracked = set(filter(None, subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')))
    if set(source) != tracked:
        raise ValueError('Source archive must equal the complete audited tracked tree')
    tests = json.loads(args.validation.read_text())
    if not tests.get('passed') or not tests.get('release_payload_checked'):
        raise ValueError('Fresh release payload validation is incomplete')
    candidate=args.candidate.resolve()
    build=json.loads((candidate/'report.json').read_text())
    if not build.get('passed'):
        raise ValueError('Add-on package build failed')
    for name,row in build['binaries'].items():
        if sha((candidate/name).read_bytes())!=row['sha256'] or tests.get('binaries', {}).get(name) != row['sha256']:
            raise ValueError('Packaged add-on changed: '+name)
    output=CURRENT/f'release-{VERSION}'
    if output.exists():raise ValueError('Release output exists; inspect before replacing')
    with tempfile.TemporaryDirectory(prefix='release-',dir=ROOT/'artifacts') as temporary:
        staging=Path(temporary); embedded={}
        for bits in (32,64):
            embedded[str(bits)]=audit_embedded(candidate/f'Phoenix.addon{bits}',staging/str(bits))
        files=common_files()
        for name in ('Phoenix.addon32','Phoenix.addon64','phoenix_graphics_host.exe'):
            files[name]=(candidate/name).read_bytes()
        assets={}
        input_files, input_validation = input_package(args.input_candidate)
        name=f'DLSS5-Phoenix-{VERSION}-Input-Windows.zip'
        assets[name]=write_archive(staging/name,input_files)
        name=f'DLSS5-Phoenix-{VERSION}-Windows.zip'
        assets[name]=write_archive(staging/name,files)
        experimental = None
        if args.experimental_candidate:
            experimental = json.loads(args.experimental_validation.read_text())
            if not experimental.get('passed') or not experimental.get('release_payload_checked'):
                raise ValueError('Experimental release payload validation is incomplete')
            folder = args.experimental_candidate.resolve()
            experimental_build = json.loads((folder/'report.json').read_text())
            if not experimental_build.get('passed') or experimental_build.get('release_channel') != 'experimental':
                raise ValueError('Compatibility addon must retain its experimental release channel')
            files = common_files()
            for name,row in experimental_build['binaries'].items():
                raw = (folder/name).read_bytes()
                if sha(raw) != row['sha256'] or experimental.get('binaries', {}).get(name) != row['sha256']:
                    raise ValueError('Experimental validation does not match the shipped binary: '+name)
                files[name] = raw
            for bits in (32,64):
                audit_embedded(folder/f'Phoenix.addon{bits}', staging/f'experimental-{bits}')
            files['README.txt'] += (
                '\nEXPERIMENTAL RTX 20/30/40/50 COMPATIBILITY PACKAGE\n'
                'Older RTX 20/30/40 cards have compiled kernel images, not physical-device qualification.\n'
                'Both models are supported; inference, quality and performance on those cards need testing.\n'
                'Do not install alongside another Phoenix addon; replace the matching architecture instead.\n').encode()
            files['COMPATIBILITY_VALIDATION.json'] = json.dumps(experimental,indent=2).encode()
            name=f'DLSS5-Phoenix-{VERSION}-RTX-Compatibility-Windows.zip'
            assets[name]=write_archive(staging/name,files)
        files=common_files()
        for architecture,folder in (('win-x64',args.runtime),('win-x86',args.runtime_x86)):
            for path in folder.rglob('*'):
                if path.is_file():files['PhoenixRuntime/'+architecture+'/'+path.relative_to(folder).as_posix()]=path.read_bytes()
        name=f'DLSS5-Phoenix-{VERSION}-Portable-Runtimes.zip'
        assets[name]=write_archive(staging/name,files)
        name=f'DLSS5-Phoenix-{VERSION}-Source.zip';prefix=f'DLSS5-Phoenix-{VERSION}/'
        subprocess.run(['git','-c','core.autocrlf=false','archive','--format=zip','--prefix='+prefix,'--output='+str(staging/name),commit],cwd=ROOT,check=True)
        index={}
        for record in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).decode().split('\0'):
            if record:
                meta,path=record.split('\t',1);index[path]=meta.split()[1]
        algorithm=subprocess.check_output(['git','rev-parse','--show-object-format'],cwd=ROOT,text=True).strip()
        with zipfile.ZipFile(staging/name) as archive:
            archived={p[len(prefix):] for p in archive.namelist() if not p.endswith('/')}
            if archived!=tracked:raise ValueError('Source ZIP differs from the tracked tree')
            for path in archived:
                raw=archive.read(prefix+path);h=hashlib.new(algorithm)
                h.update(b'blob '+str(len(raw)).encode()+b'\0'+raw)
                if h.hexdigest()!=index[path]:raise ValueError('Source ZIP differs from committed blob: '+path)
                validate_public_content(path,raw)
            if archive.testzip():raise ValueError('Source ZIP CRC error')
        assets[name]=dict(files=len(tracked),bytes=(staging/name).stat().st_size,sha256=sha((staging/name).read_bytes()))
        validation=dict(schema='phoenix-addon-release/2',version=VERSION,commit=commit,passed=True,
            source_audit_passed=True,archive_hashes_verified=True,model_free=True,
            private_paths_secrets_models_and_logs_excluded=True,retail_games_launched=False,
            vendor_parity=False,release_validation=tests,
            experimental_compatibility_validation=experimental,
            input_companion_validation=input_validation,
            embedded_resources={bits:row['resources'] for bits,row in embedded.items()},assets=assets)
        (staging/'VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
        (staging/'RELEASE_NOTES.md').write_text((ROOT/f'docs/RELEASE_{VERSION}.md').read_text(encoding='utf-8'),encoding='utf-8')
        names=[*assets,'VALIDATION.json']
        (staging/'SHA256SUMS.txt').write_text(''.join(sha((staging/n).read_bytes())+'  '+n+'\n' for n in names),encoding='ascii')
        output.mkdir(parents=True)
        for n in [*names,'SHA256SUMS.txt','RELEASE_NOTES.md']:(staging/n).replace(output/n)
    print(json.dumps(dict(passed=True,version=VERSION,commit=commit,assets=assets),indent=2))


if __name__ == '__main__':main()
