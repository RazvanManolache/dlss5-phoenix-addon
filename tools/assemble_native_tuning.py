"""Merge freshly built native/styles and rebuild only the measured spill hotspot.

This is an incremental build, not a new optimization search. Unchanged source
hashes are required, and both the previous snapshot and the exact updated source
are retained in metadata. All temporary extracted images are removed.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from compare_four_models import ROOT, sha, write
from experiment_native_tuning import CUDA, STYLES
from test_model_selection import toolchain, run

SOURCE = 'src/native/kernels/production/optimized/window_attention/channels_64/attention.cu'
NAME = 'cc_tinlayout_fused_swin_2h_64_2_chained_fp8'


def refresh_snapshot(snapshot):
    """A local annotation cannot invalidate unrelated translation units."""
    changed = {name: dict(before=pin, after=sha(ROOT/name)) for name,pin in snapshot.items()
               if sha(ROOT/name) != pin}
    if set(changed) != {SOURCE}:
        raise ValueError('Expected only the measured attention annotation to change: '+str(changed))
    return {name: sha(ROOT/name) for name in snapshot}, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--styles', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    base, fresh = args.styles.resolve(), args.native.resolve()
    if not base.is_relative_to((ROOT/'artifacts/current').resolve()):
        parser.error('Candidate must stay under artifacts/current')
    manifest = json.loads((base/'native_manifest.json').read_text())
    native = json.loads((fresh/'native_manifest.json').read_text())
    if not native['passed'] or native['model_weights_included'] or not manifest['passed']:
        raise ValueError('Require successful model-free builds')
    if native['native_graph_sha256'] != manifest['native_graph_sha256']:
        raise ValueError('Graph changed')
    native['source_hashes'], changes = refresh_snapshot(native['source_hashes'])
    for style in manifest['kernel_styles'].values():
        style['source_hashes'], changed = refresh_snapshot(style['source_hashes'])
        if changed != changes:
            raise ValueError('Style and native compilation snapshots differ')
    for row in native['kernels']:
        if sha(fresh/row['binary']) != row['binary_sha256']:
            raise ValueError('Fresh native binary hash changed')
        shutil.copy2(fresh/row['binary'],base/row['binary'])
    manifest.update({k:v for k,v in native.items() if k != 'kernel_styles'})
    manifest.update(native_images_retained=False,
                    incremental_rebuild=dict(reason='Cache-induced local-memory spill',source_changes=changes))
    _, env = toolchain(64)
    with tempfile.TemporaryDirectory(prefix='assemble-native-tuning-',dir=ROOT/'build') as temporary:
        work = Path(temporary)
        groups = [('auto',manifest,('sm_75','sm_86')),
                  ('rtx20',manifest['kernel_styles']['rtx20'],('sm_86','sm_89','sm_120')),
                  ('rtx30',manifest['kernel_styles']['rtx30'],('sm_89','sm_120'))]
        jobs = []
        rows = {}
        for style, group, targets in groups:
            row = next(r for r in group['kernels'] if r['name'] == NAME)
            folder = work/style
            folder.mkdir()
            run([CUDA/'cuobjdump.exe','--extract-elf','all',base/row['binary']],cwd=folder,env=env)
            images = {'sm_'+str((int.from_bytes(p.read_bytes()[48:52],'little')>>8)&255):p
                      for p in folder.iterdir() if p.read_bytes()[:4] == b'\x7fELF'}
            if set(images) != {r['architecture'] for r in row['architecture_images']}:
                raise ValueError('Unexpected extracted targets')
            for image in row['architecture_images']:
                if sha(images[image['architecture']]) != image['sha256']:
                    raise ValueError('Extracted image hash mismatch')
            rows[style] = row, images, folder
            jobs.extend((style,target,images[target],row) for target in targets)

        def compile_one(job):
            style,target,output,row = job
            command = [CUDA/'nvcc.exe','--entries',row['entry'],'-allow-unsupported-compiler',
                       '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH','-diag-suppress','177','-diag-suppress','550',
                       '-cubin','-arch='+target,'-O3','-std=c++17',
                       *(['-DPHOENIX_KERNEL_STYLE='+str(STYLES[style])] if style != 'auto' else []),
                       *(['--split-compile=2'] if target != 'sm_120' else []),
                       '-lineinfo','-Xptxas=-v,-O3,--suppress-debug-info',SOURCE,'-o',output]
            run(command,cwd=ROOT,env=env,timeout=900)
            print('Rebuilt attention',style,target,flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(compile_one,jobs))
        for style,(row,images,folder) in rows.items():
            packed = folder/'new.fatbin'
            run([CUDA/'fatbinary.exe','--64','--create',packed,'--compress-all',
                 *['--image3=file='+str(path)+',kind=elf,sm='+target[3:] for target,path in images.items()]],
                cwd=ROOT,env=env)
            shutil.copy2(packed,base/row['binary'])
            row.update(binary_sha256=sha(packed), architecture_images=[
                dict(architecture=target,sha256=sha(path),bytes=path.stat().st_size)
                for target,path in sorted(images.items(),key=lambda item:int(item[0][3:]))])
        if any(sha(ROOT/name) != pin for name,pin in manifest['source_hashes'].items()):
            raise ValueError('Sources changed during incremental build')
    write(base/'native_manifest.json',manifest)
    print('Merged and rebuilt seven affected device images; no GPU inference performed',flush=True)


if __name__ == '__main__':
    main()
