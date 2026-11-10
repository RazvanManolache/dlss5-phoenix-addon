"""Promote a source-verified local Windows CUDA preview; no release or upload."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    current = ROOT/'artifacts/current'
    runtime,addon = current/'windows-cuda',current/'addon-portable-candidate'
    build = json.loads((runtime/'BUILD.json').read_text())
    bridge = json.loads((current/'portable-bridge-validation.json').read_text())
    if not build['passed'] or not bridge['passed']:
        raise ValueError('A successful runtime build and addon validation are required')
    for receipt,key in ((build,'source_inputs'),(bridge,'source_sha256')):
        for name,sha in receipt[key].items():
            if digest(ROOT/name)!=sha: raise ValueError('Source changed since build: '+name)
    destination = current/'windows-linux-integration'
    if destination.is_symlink() or destination.is_junction(): raise ValueError('Linked package destination')
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='windows-preview-package-',dir=ROOT/'build') as temporary:
        staged = Path(temporary)
        for bits in (32,64):
            name = f'Phoenix.addon{bits}'
            if digest(addon/name)!=bridge['architectures'][str(bits)]['addon_sha256']:
                raise ValueError('Addon differs from validated output: '+name)
            shutil.copy2(addon/name,staged/name)
        shutil.copy2(addon/'phoenix_graphics_host.exe',staged/'phoenix_graphics_host.exe')
        for name,sha in build['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name:
                raise ValueError('Unsafe runtime inventory member')
            if digest(runtime/name)!=sha: raise ValueError('Runtime output changed: '+name)
            target = staged/('include' if relative.suffix=='.h' else 'PhoenixRuntime/win-x64')/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(runtime/name,target)
        # The shared Windows frame-services SDK accompanies the CUDA leaf.
        # Its public header is source-bound by runtime_build.source_inputs.
        shutil.copy2(ROOT/'native_runtime/phoenix_native.h',staged/'include/phoenix_native.h')
        (staged/'PhoenixRuntime/win-x64/phoenix_native_cpp.json').write_text(json.dumps({
            'validated_devices':['cpu'],'allow_unvalidated':False},indent=2)+'\n')
        for name in ('LICENSE','NOTICE','THIRD_PARTY.md'):
            shutil.copy2(ROOT/name,staged/name)
        shutil.copy2(ROOT/'docs/WINDOWS_CUDA.md',staged/'WINDOWS_CUDA.md')
        (staged/'BUILD.json').write_text(json.dumps({'schema':'phoenix-windows-cuda-preview/1',
            'release':False,'models_included':False,'runtime_build':build,
            'addon_validation':bridge},indent=2)+'\n')
        inventory = {p.relative_to(staged).as_posix():digest(p) for p in staged.rglob('*') if p.is_file()}
        (staged/'FILES_SHA256.json').write_text(json.dumps(inventory,indent=2)+'\n')
        previous = destination/'FILES_SHA256.json'
        if previous.is_file():
            for name,sha in json.loads(previous.read_text()).items():
                relative = PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name:
                    raise ValueError('Unsafe previous package member')
                obsolete = destination/name
                if name not in inventory and obsolete.is_file():
                    if any(p.is_symlink() or p.is_junction() for p in (obsolete,*obsolete.parents) if p.is_relative_to(destination)):
                        raise ValueError('Linked obsolete package member')
                    if digest(obsolete)!=sha: raise ValueError('Modified obsolete package member: '+name)
                    obsolete.unlink()
        # Replace only owned current members, retaining unrelated existing data.
        for path in staged.rglob('*'):
            if path.is_file():
                target = destination/path.relative_to(staged)
                if target.is_symlink() or target.is_junction(): raise ValueError('Linked package member')
                target.parent.mkdir(parents=True,exist_ok=True)
                if any(p.is_symlink() or p.is_junction() for p in target.parents if p.is_relative_to(destination)):
                    raise ValueError('Linked package parent')
                shutil.copy2(path,target)
    print(json.dumps({'passed':True,'package':str(destination),'release':False,'models_included':False}))


if __name__=='__main__': main()
