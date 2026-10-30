"""Add older instruction styles to an existing model-free native kernel bundle.

Physical targets stay explicit: RTX20 style on 30/40/50, RTX30 on 40/50,
RTX40 on 50. Native binaries are retained byte-for-byte, never replaced.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from production_build_native_kernels import ROOT, digest

STYLE_TARGETS = {"rtx20": ["sm_86", "sm_89", "sm_120"],
                 "rtx30": ["sm_89", "sm_120"], "rtx40": ["sm_120"]}


def copy_pinned(source, destination, relative, pin):
    relative_path = Path(relative)
    if relative_path.is_absolute() or '..' in relative_path.parts or ':' in relative or '\\' in relative:
        raise ValueError('Kernel bundle requires a safe relative path')
    path = (source / relative).resolve()
    if not path.is_relative_to(source.resolve()) or digest(path) != pin:
        raise ValueError("Kernel bundle path or digest mismatch: " + relative)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-native', required=True, type=Path)
    parser.add_argument('--compiler', required=True, type=Path)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/current/native-kernel-styles')
    args = parser.parse_args()
    base, output = args.base_native.resolve(), args.output.resolve()
    if not output.is_relative_to((ROOT/'artifacts/current').resolve()) or output.exists():
        parser.error('Choose a fresh output directory below artifacts/current')
    manifest = json.loads((base/'native_manifest.json').read_text())
    if not manifest.get('passed') or manifest.get('model_weights_included') is not False:
        parser.error('Base must be a successfully built model-free native bundle')
    if manifest.get('kernel_style', 'auto') != 'auto':
        parser.error('Base must contain native default kernels, not a standalone style build')
    staging_paths = []
    with tempfile.TemporaryDirectory(prefix='kernel_styles_', dir=ROOT/'build') as temporary:
        staging = Path(temporary)
        try:
            payload = staging/'payload'; payload.mkdir()
            for name, pin in [('native_graph.json', 'native_graph_sha256'),
                              ('native_graph_v2.json', 'native_graph_v2_sha256')]:
                if pin in manifest: copy_pinned(base, payload, name, manifest[pin])
            for row in manifest['kernels']:
                copy_pinned(base, payload, row['binary'], row['binary_sha256'])
            manifest['kernel_styles'] = {}
            for style, targets in STYLE_TARGETS.items():
                folder = ROOT/'build'/(staging.name+'_'+style)
                staging_paths.append(folder)
                subprocess.run([sys.executable, str(ROOT/'tools/production_build_native_kernels.py'),
                    folder.name, '--compiler', str(args.compiler.resolve()), '--jobs', str(args.jobs),
                    '--architectures', *targets, '--kernel-style', style], cwd=ROOT, check=True)
                built = json.loads((folder/'native_manifest.json').read_text())
                if not built.get('passed'): raise ValueError('Style compilation failed: '+style)
                records = []
                for row in built['kernels']:
                    relative = 'styles/'+style+'/'+row['binary']
                    copy_pinned(folder, payload/'styles'/style, row['binary'], row['binary_sha256'])
                    records.append({**row, 'binary': relative})
                manifest['kernel_styles'][style] = dict(architectures=targets, kernels=records,
                    validation_pending=True, source_hashes=built['source_hashes'])
            manifest.update(selection_status='compiled', validation_pending=True,
                            native_images_retained=True)
            (payload/'native_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            output.parent.mkdir(parents=True, exist_ok=True)
            payload.replace(output)
        finally:
            for path in staging_paths:
                if path.exists():
                    if path.is_symlink() or path.is_junction() or path.resolve().parent != (ROOT/'build').resolve():
                        raise ValueError('Unsafe style-build cleanup path')
                    shutil.rmtree(path)
    print(json.dumps(dict(passed=True, gpu_used=False, device_inference_validated=False,
                         styles=STYLE_TARGETS, output=str(output))))


if __name__ == '__main__': main()
