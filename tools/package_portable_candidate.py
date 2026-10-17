"""Assemble the local alternate addon/runtime candidate; no downloads, models or GPU."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT/'artifacts/current'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    validation = json.loads((CURRENT/'portable-bridge-validation.json').read_text())
    if not validation.get('passed') or validation.get('game_presentation_validated'):
        raise ValueError('Expected a successful CPU integration receipt with bounded claims')
    for name, expected in validation['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Integration source changed after compilation: ' + name)
    candidate = CURRENT/'addon-portable-candidate'
    if candidate.is_symlink() or candidate.is_junction() or candidate.resolve().parent != CURRENT.resolve():
        raise ValueError('Unsafe candidate directory')
    for bits in (32, 64):
        if digest(candidate/f'Phoenix.addon{bits}') != validation['architectures'][str(bits)].get('addon_sha256'):
            raise ValueError('Compile both full addons before packaging')
    receipt = json.loads((CURRENT/'portable-x86-validation.json').read_text())
    if not receipt.get('passed') or not receipt.get('inference_validated'):
        raise ValueError('The x86 runtime needs standalone inference validation')
    source = CURRENT/'portable-x86'
    for name, expected in receipt['candidate_sha256'].items():
        if digest(source/name) != expected:
            raise ValueError('Standalone runtime changed after validation: ' + name)
    sdk = CURRENT/'sdk/windows-x64'
    sdk_hashes = json.loads((sdk/'FILES_SHA256.json').read_text())
    runtime_source = sdk/'bin'
    for path in runtime_source.rglob('*'):
        if path.is_symlink() or path.is_junction():
            raise ValueError('Runtime source contains a redirect')
        if path.is_file() and digest(path) != sdk_hashes.get(path.relative_to(sdk).as_posix()):
            raise ValueError('x64 SDK payload changed: ' + path.name)
    with tempfile.TemporaryDirectory(prefix='portable-package-', dir=ROOT/'artifacts') as temporary:
        from validate_portable_bridge import resources
        for bits in (32, 64):
            extraction = Path(temporary)/str(bits)
            extraction.mkdir()
            _, retained = resources(candidate/f'Phoenix.addon{bits}', extraction)
            if retained != validation['architectures'][str(bits)]['retained_payload_sha256']:
                raise ValueError('Candidate changed the retained helper payload')
        staging = Path(temporary)/'PhoenixRuntime'
        shutil.copytree(source, staging/'win-x86')
        shutil.copytree(runtime_source, staging/'win-x64')
        shutil.copytree(sdk/'licenses', staging/'win-x64/licenses')
        native = candidate/'PhoenixRuntime/wow64'
        if native.exists():
            if native.is_symlink() or native.is_junction():
                raise ValueError('Unsafe native runtime package')
            native_receipt=json.loads((CURRENT/'wow64-runtime-build.json').read_text())
            if not native_receipt.get('passed'): raise ValueError('Native package build did not pass')
            for name,expected in native_receipt['files'].items():
                if digest(native/name)!=expected: raise ValueError('Native runtime changed: '+name)
            shutil.copytree(native,staging/'wow64')
        for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY.md'):
            shutil.copy2(sdk/name, staging/'win-x64'/name)
        for bits in (32, 64):
            # Only the x86 candidate's exact binaries are bound to this inference
            # receipt. Do not transfer those claims to a different x64 SDK build.
            validated = [receipt['architectures']['x86']['device']['id']] if bits == 32 else []
            config = dict(validated_devices=validated, allow_unvalidated=False)
            (staging/f'win-x{bits if bits == 64 else "86"}'/'phoenix_native_cpp.json').write_text(
                json.dumps(config, indent=2)+'\n', encoding='utf-8')
        (staging/'win-x86/README.txt').write_text(
            'Experimental in-process x86 Vulkan runtime for the alternate addon.\n'
            'Standalone neural execution was validated separately. The addon integration\n'
            'has CPU/headless contract evidence, not game-presentation validation.\n'
            'See ../../EXECUTION_MECHANISMS.md for configuration and limitations.\n'
            'No weights or NVIDIA DLLs are included. The x64 helper remains available.\n', encoding='utf-8')
        destination = candidate/'PhoenixRuntime'
        if destination.exists():
            if destination.is_symlink() or destination.is_junction() or destination.resolve().parent != candidate.resolve():
                raise ValueError('Unsafe existing candidate runtime')
            shutil.rmtree(destination)
        shutil.move(str(staging), str(destination))
    shutil.copy2(ROOT/'docs/EXECUTION_MECHANISMS.md', candidate/'EXECUTION_MECHANISMS.md')
    (candidate/'README.txt').write_text(
        'Local experimental candidate. Not promoted to the production package.\n'
        'Use matching Phoenix.addon32 or Phoenix.addon64 and keep PhoenixRuntime beside it.\n'
        'Retains the previous embedded x64 helper and automatic routing.\n'
        'Choose Advanced transport > In-process portable to try the alternate route.\n'
        'The x86 addon also offers x64 library and x86 host/native driver execution.\n'
        'These native x86 routes require the optional locally verified PhoenixRuntime/wow64 package.\n'
        'Default runtime paths work relative to the addon; select your own model source.\n'
        'x86 Vulkan standalone inference was validated separately; game presentation was not.\n'
        'x64 portable devices require their own validation catalogue or explicit UI opt-in.\n'
        'No model/NVIDIA DLLs, personal game profiles, logs or captures are included.\n', encoding='utf-8')
    files = {}
    for path in candidate.rglob('*'):
        if path.is_file():
            if path.suffix.lower() in ('.log', '.bmp', '.jpg', '.png') or path.name.lower().startswith('nvngx'):
                raise ValueError('Excluded file in candidate: ' + path.name)
            if path.name != 'FILES_SHA256.json':
                files[path.relative_to(candidate).as_posix()] = digest(path)
    (candidate/'FILES_SHA256.json').write_text(json.dumps(files, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=True, candidate=str(candidate), files=len(files),
        models_included=False, gpu_used=False, production_package_modified=False)))


if __name__ == '__main__':
    main()
