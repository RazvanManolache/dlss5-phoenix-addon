"""CPU/offline checks for architecture bundles. Never creates a GPU context."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from test_model_selection import ROOT, toolchain, run


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cubin_architecture(data):
    if len(data) < 52 or data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('Expected a little-endian CUDA ELF64 cubin')
    if int.from_bytes(data[18:20], 'little') != 190:
        raise ValueError('ELF is not a CUDA device image')
    return 'sm_' + str((int.from_bytes(data[48:52], 'little') >> 8) & 255)


def check_images(bundle, baseline, extractor, work):
    manifest = json.loads((bundle / 'native_manifest.json').read_text())
    reference = json.loads((baseline / 'native_manifest.json').read_text())
    expected_architectures = {'sm_75', 'sm_86', 'sm_89', 'sm_120'}
    if not manifest.get('passed') or set(manifest.get('architectures', [])) != expected_architectures:
        raise ValueError('Expected all four compiled architectures')
    kernels = manifest['kernels']
    originals = {row['name']: row for row in reference['kernels']}
    if len(kernels) != 41 or len({row['name'] for row in kernels}) != 41:
        raise ValueError('Expected 41 unique kernel modules')
    total = {arch: 0 for arch in expected_architectures}
    for index, row in enumerate(kernels):
        binary = (bundle / row['binary']).resolve()
        if not binary.is_relative_to(bundle) or sha(binary) != row['binary_sha256']:
            raise ValueError('Fatbin path/hash mismatch')
        folder = work / str(index)
        folder.mkdir()
        subprocess.run([str(extractor), '--extract-elf', 'all', str(binary)], cwd=folder,
                       check=True, capture_output=True, timeout=60)
        images = {}
        for path in folder.iterdir():
            data = path.read_bytes()
            arch = cubin_architecture(data)
            if arch in images:
                raise ValueError('Duplicate architecture image')
            images[arch] = path
            if row['entry'].encode() + b'\0' not in data or (
                'large_input_entry' in row and row['large_input_entry'].encode() + b'\0' not in data):
                raise ValueError('Missing default/large-input entry')
        declared = {image['architecture']: image for image in row['architecture_images']}
        if set(images) != expected_architectures or len(declared) != 4:
            raise ValueError('Incomplete architecture coverage')
        for arch, path in images.items():
            if sha(path) != declared[arch]['sha256'] or path.stat().st_size != declared[arch]['bytes']:
                raise ValueError('Extracted image differs from build receipt')
            total[arch] += path.stat().st_size
        base = originals[row['name']]
        base_path = (baseline / base['binary']).resolve()
        if not base_path.is_relative_to(baseline) or sha(base_path) != base['binary_sha256']:
            raise ValueError('Baseline kernel hash/path mismatch')
        if sha(images['sm_120']) != sha(base_path):
            raise ValueError('RTX 50-series code changed')
    return dict(modules=41, device_images=164, sm120_byte_exact=True, image_bytes=total,
                fatbin_bytes=sum((bundle / row['binary']).stat().st_size for row in kernels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--cuda-bin', type=Path, default=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, gpu_used=False, games_launched=False, older_gpu_inference_validated=False)
    args.output.write_text(json.dumps(report) + '\n')
    compiler, env = toolchain(64)
    try:
        with tempfile.TemporaryDirectory(prefix='native-architectures-', dir=ROOT / 'build') as temporary:
            work = Path(temporary)
            report['images'] = check_images(args.bundle.resolve(), args.baseline.resolve(),
                (args.cuda_bin / 'cuobjdump.exe').resolve(), work)
            executable = work / 'conversions.exe'
            run([args.cuda_bin / 'nvcc.exe', '-allow-unsupported-compiler',
                 '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', '-arch=sm_75', '-O2', '-std=c++17',
                 ROOT / 'tests/native/kernel_architecture_host_tests.cu', '-o', executable], cwd=work, env=env)
            report['conversions'] = json.loads(run([executable], cwd=work, env=env))
            executable = work / 'graph.exe'
            run([compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/O2', '/EHsc', '/MD', '/bigobj', '/DNOMINMAX',
                 '/I' + str(ROOT.parent / 'external/DLSS/include'),
                 '/I' + str(ROOT / 'third_party/nvapi'), ROOT / 'tests/native/native_small_model_tests.cpp',
                 '/Fo:' + str(work / 'graph.obj'), '/Fe:' + str(executable), 'd3d12.lib', 'dxgi.lib',
                 'd3dcompiler.lib', 'bcrypt.lib'], cwd=work, env=env)
            report['models'] = json.loads(run([executable, ROOT, args.bundle.resolve(), args.model.resolve()], cwd=work, env=env))
            report.update(passed=True, temporary_builds_removed=True)
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
