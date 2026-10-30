"""Stage current model-free release inputs; bounded inference uses CPU only.

Native sm_120 images are extracted byte-for-byte from the already measured
multi-architecture bundle. No model is copied; SDK paths remain build-only.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
from test_model_selection import ROOT, toolchain, run
from validate_x86_vulkan import stdcall_aliases
from validate_native import kernel_selection_digest, require_variant_coverage


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def standard_bundle(destination, bundle):
    manifest = json.loads((bundle / 'native_manifest.json').read_text(encoding='utf-8'))
    evidence = json.loads((ROOT / 'artifacts/current/native-vendor-refresh/run.json').read_text(encoding='utf-8'))
    if not evidence['passed'] or not all(row['exact'] for row in evidence['comparisons']):
        raise ValueError('Native measurement/reference evidence failed')
    if manifest['kernels'] != evidence['native_manifest']['kernels']:
        raise ValueError('Measured native images differ from release inputs')
    for name, digest in manifest['source_hashes'].items():
        if sha(ROOT / name) != digest: raise ValueError('Native kernel source changed: ' + name)
    destination.mkdir()
    compiler = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/cuobjdump.exe')
    for index, row in enumerate(manifest['kernels']):
        work = destination / f'extract_{index}'; work.mkdir()
        result = subprocess.run([str(compiler), '-xelf', 'sm_120', str(bundle / row['binary'])],
                                cwd=work, capture_output=True, text=True, timeout=60)
        images = [p for p in work.iterdir() if p.is_file()]
        pin = next(image for image in row['architecture_images'] if image['architecture'] == 'sm_120')
        if result.returncode or len(images) != 1 or sha(images[0]) != pin['sha256']:
            raise ValueError('Native extraction did not preserve the measured image: ' + row['name'])
        target = destination / row['binary']; target.parent.mkdir(parents=True, exist_ok=True)
        images[0].replace(target); work.rmdir()
        row['binary_sha256'] = sha(target); row['architecture_images'] = [pin]
    manifest.pop('kernel_styles', None)
    manifest.update(architectures=['sm_120'], architecture_validation={'sm_120': 'validated'},
                    selection_status='validated', validation_pending=False, native_images_retained=41,
                    release_input_provenance='Byte-exact sm_120 images from the paired native/vendor study; no new GPU execution')
    extents = [tuple(map(int, case['extent'].split('x'))) for case in evidence['cases']
               if case['engine'] == 'phoenix' and case['passed']]
    coverage = {(row['name'], 'large' if width * height >= row['large_input_min_pixels'] else 'default')
                for row in manifest['kernels'] if 'large_input_entry' in row
                for width, height in extents}
    # The full measured sweep executes below and above the production selector's
    # threshold. The selector itself remains bound into this policy digest.
    require_variant_coverage(manifest, coverage)
    manifest['validation'] = dict(kernel_selection_sha256=kernel_selection_digest(manifest),
        resolution_variant_coverage=[dict(name=name, mode=mode) for name, mode in sorted(coverage)],
        measured_sweep_sha256=sha(ROOT / 'artifacts/current/native-vendor-refresh/run.json'),
        scope='Prior isolated full-graph sm_120 execution, both models through 8K; extraction authenticated by image digest')
    shutil.copy2(bundle / 'native_graph.json', destination / 'native_graph.json')
    (destination / 'native_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-build', type=Path, required=True)
    parser.add_argument('--vulkan-build', type=Path, required=True)
    parser.add_argument('--deps', type=Path, required=True)
    args = parser.parse_args()
    work = args.output.resolve()
    if work.parent != (ROOT / 'build').resolve(): raise ValueError('Use a fresh owned directory immediately below build/')
    work.mkdir(exist_ok=False)
    runtime, vulkan = args.runtime_build.resolve(), args.vulkan_build.resolve()
    native = work / 'native-standard'
    standard_bundle(native, ROOT / 'artifacts/current/native-kernel-styles')
    _, env64 = toolchain(64)
    report = dict(passed=False, gpu_executed=False, games_launched=False, model_data_included=False, architectures={})
    for bits, label in [(64, 'win-x64'), (32, 'win-x86')]:
        target = work / label; target.mkdir()
        compiler, env = toolchain(bits)
        flags = [compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/O2', '/EHsc', '/MD', '/bigobj',
                 '/DNOMINMAX', '/experimental:deterministic', '/pathmap:' + str(ROOT.parent) + '=.']
        def dll(name, source, definition=None, libraries=()):
            command = flags + ['/LD', ROOT / source, '/Fo:' + str(target / (name + '.obj')),
                               '/Fe:' + str(target / (name + '.dll')), '/link']
            if definition: command.append('/DEF:' + str(ROOT / 'native_runtime' / definition))
            run(command + list(libraries), cwd=target, env=env)
        if bits == 64:
            for name in ['phoenix_cpu_cpp.dll', 'phoenix_runtime_cpp.dll', 'phoenix_native_broker.exe',
                         'phoenix_cpp_devices.exe', 'phoenix_vulkan_cpp.dll', 'phoenix_cuda_cpp.dll']:
                shutil.copy2(runtime / name, target / name)
            for name in ['phoenix_native.dll', 'phoenix-process.exe']:
                shutil.copy2(ROOT / 'artifacts/current/pipeline-native' / name, target / name)
            shutil.copytree(native, target / 'native')
        else:
            dll('phoenix_runtime_cpp', 'native_runtime/runtime.cpp', 'phoenix_runtime.def', ['bcrypt.lib'])
            # Preserve the portable x86 CPU reference; the measured AVX tuning
            # belongs to the x64 CPU plugin, not an unqualified new x86 path.
            dll('phoenix_cpu_cpp', 'native_runtime/backend_reference.cpp', 'cpu_backend.def')
            dll('phoenix_vulkan_cpp', 'native_runtime/backend_vulkan.cpp', 'vulkan_backend.def')
            run(flags + [ROOT / 'native_runtime/broker.cpp', '/Fo:' + str(target / 'broker.obj'),
                         '/Fe:' + str(target / 'phoenix_native_broker.exe'), '/link', 'bcrypt.lib'], cwd=target, env=env)
        run(flags + [ROOT / 'tools/portable_frame.cpp', '/Fo:' + str(target / 'frame.obj'),
                     '/Fe:' + str(target / 'phoenix_frame.exe'), '/link', 'bcrypt.lib'], cwd=target, env=env)
        shaders = target / 'vulkan'; shaders.mkdir()
        for source in vulkan.glob('*.spv'): shutil.copy2(source, shaders / source.name)
        if bits == 64:
            shutil.copy2(vulkan / 'phoenix_vulkan.dll', shaders / 'phoenix_vulkan.dll')
        else:
            own = ROOT / 'src/backends/vulkan'
            run([compiler, '/nologo', '/O2', '/std:c11', '/TC', '/MD', '/D_CRT_SECURE_NO_WARNINGS',
                 '/experimental:deterministic', '/pathmap:' + str(ROOT.parent) + '=.',
                 '/I' + str(args.deps.resolve() / 'headers/include'), '/I' + str(own),
                 '/c', vulkan / 'phoenix_vulkan.c', '/Fo:' + str(target / 'provider.obj')], cwd=target, env=env)
            symbols = run([Path(compiler).parent / 'dumpbin.exe', '/nologo', '/symbols', target / 'provider.obj'], cwd=target, env=env)
            functions = sorted(set(re.findall(r'\bUNDEF\b[^\n]*\|\s+_?vk(\w+?)(?:@\d+)?\s*$', symbols, re.M)))
            if not functions: raise ValueError('No Vulkan imports found')
            definition = target / 'loader.def'
            definition.write_text('LIBRARY vulkan-1.dll\nEXPORTS\n' + ''.join('vk' + name + '\n' for name in functions), encoding='ascii')
            run([Path(compiler).parent / 'lib.exe', '/nologo', '/def:' + str(definition), '/machine:x86',
                 '/out:' + str(target / 'vulkan-1.lib')], cwd=target, env=env)
            aliases = ['/alternatename:' + symbol + '=' + symbol.split('@')[0] for symbol in stdcall_aliases(symbols)]
            run([Path(compiler).parent / 'link.exe', '/nologo', '/DLL', target / 'provider.obj', target / 'vulkan-1.lib',
                 '/DEF:' + str(vulkan / 'provider.def'), '/OUT:' + str(shaders / 'phoenix_vulkan.dll'), *aliases], cwd=target, env=env)
        # This public default contains no private model path or test-device UUID.
        (target / 'phoenix_native_cpp.json').write_text(json.dumps(dict(validated_devices=['cpu'],
            allow_unvalidated=False, execution='broker', cpu_threads=0), indent=2) + '\n', encoding='utf-8')
        validation = dict(schema='phoenix-release-runtime-validation/1', passed=False, inference_validated=False,
            gpu_inference_validated=False, model_data_included=False,
            scope='Fresh CPU fixtures for both model families; GPU inference not rerun for these host binaries',
            cpu_models=[], binaries={})
        fixtures = work / ('cpu-fixtures-' + str(bits)); fixtures.mkdir()
        width = height = 32
        (fixtures / 'input.bin').write_bytes(struct.pack('<fff', .25, .5, .75) * width * height)
        source = ROOT.parent / 'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
        for model in ['original', 'small-fp8']:
            job = dict(width=width, height=height, color=str(fixtures / 'input.bin'),
                config=dict(runtime_root=str(target), backend='cpu', device='cpu', allow_unvalidated=True, cpu_threads=4,
                            weight_source=dict(kind='nvidia_dll', path=str(source), sha256=sha(source), model=model)))
            (fixtures / 'job.json').write_text(json.dumps(job), encoding='utf-8')
            outputs = []
            for repeat in range(2):
                output = fixtures / f'{model}-{repeat}.bin'
                run([target / 'phoenix_frame.exe', fixtures / 'job.json', output], cwd=target, env=env, timeout=300)
                raw = output.read_bytes()
                if len(raw) != width * height * 12 or not all(math.isfinite(value[0]) for value in struct.iter_unpack('<f', raw)):
                    raise ValueError('Invalid CPU output')
                outputs.append(sha(output))
            if outputs[0] != outputs[1]: raise ValueError('CPU reset output changed across processes')
            validation['cpu_models'].append(dict(model=model, extent=[width, height], reset_repeatable=True, output_sha256=outputs[0]))
        for source in list(target.rglob('*')):
            if source.is_file() and source.suffix.lower() in {'.obj', '.lib', '.exp', '.def'}: source.unlink()
        validation['binaries'] = {p.relative_to(target).as_posix(): sha(p) for p in target.rglob('*')
                                  if p.is_file() and p.suffix in {'.dll', '.exe', '.spv'}}
        validation.update(passed=True, inference_validated=True)
        (target / 'validation_manifest.json').write_text(json.dumps(validation, indent=2) + '\n', encoding='utf-8')
        report['architectures'][label] = validation
        (work / 'INPUT_VALIDATION.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(label + ': current binaries built; both CPU model fixtures passed', flush=True)
    report['passed'] = True
    (work / 'INPUT_VALIDATION.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Model-free release inputs ready: ' + str(work), flush=True)


if __name__ == '__main__': main()
