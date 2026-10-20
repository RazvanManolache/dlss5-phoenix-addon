"""Build/stage one current benchmark runtime. Does not run inference or install anything."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
from test_model_selection import ROOT, toolchain, run
from compare_four_models import build as build_native_probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT / 'build/four_model_comparison')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/current/readme-benchmarks')
    parser.add_argument('--native-bundle', type=Path, required=True,
                        help='Freshly source-built native kernel folder; never silently use cached kernels')
    parser.add_argument('--vulkan-bundle', type=Path,
                        help='Fresh source-built Vulkan provider/shaders; otherwise retain the historical provider')
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    stage = work / 'readme_runtime'
    stage.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler, env = toolchain(64)
    native = ROOT / 'native_runtime'
    common = [compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/O2', '/EHsc', '/MD', '/DNOMINMAX']
    for name, flag in [('cpu_avx2', '/arch:AVX2'), ('cpu_avx512', '/arch:AVX512')]:
        run(common + [flag, '/c', native / (name + '.cpp'), '/Fo:' + str(work / ('readme_' + name + '.obj'))], cwd=work, env=env)
    run(common + ['/LD', native / 'backend_cpu.cpp', work / 'readme_cpu_avx2.obj', work / 'readme_cpu_avx512.obj',
                  '/Fo:' + str(work / 'readme_cpu.obj'), '/Fe:' + str(stage / 'phoenix_cpu_cpp.dll')], cwd=work, env=env)
    run(common + ['/LD', native / 'runtime.cpp', '/Fo:' + str(work / 'readme_runtime.obj'),
                  '/Fe:' + str(stage / 'phoenix_runtime_cpp.dll'), 'bcrypt.lib'], cwd=work, env=env)
    run(common + ['/LD', native / 'backend_vulkan.cpp', '/Fo:' + str(work / 'readme_vulkan.obj'),
                  '/Fe:' + str(stage / 'phoenix_vulkan_cpp.dll')], cwd=work, env=env)
    shaders = args.vulkan_bundle.resolve() if args.vulkan_bundle else ROOT / 'artifacts/current/neural-candidate/windows-x64/bin/vulkan'
    shutil.copytree(shaders, stage / 'vulkan', dirs_exist_ok=True)
    cuda = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe')
    text = '#define PHX_CUDA\n#define DEVICE_NAME "cuda-cpp"\n#define DEVICE_PREFIX "cuda:"\n' + (native / 'backend_gpu.cpp').read_text()
    text = text.replace('hip', 'cuda').replace('cudaUUID uuid', 'cudaUUID_t uuid')
    generated = work / 'readme_cuda.cu'
    generated.write_text(text)
    print('CPU provider built; compiling the current CUDA provider', flush=True)
    run([cuda, '-allow-unsupported-compiler', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', '-std=c++20', '-O2',
         '--fmad=false', '-arch=sm_120', '-shared', '-Xcompiler=/MD,/EHsc,/DNOMINMAX', '-I' + str(native),
         generated, '-o', stage / 'phoenix_cuda_cpp.dll'], cwd=work, env=env, timeout=600)
    query = work / 'readme_device_query.exe'
    run(common + [native / 'device_query.cpp', '/Fo:' + str(work / 'readme_device_query.obj'), '/Fe:' + str(query)], cwd=work, env=env)
    run(common + ['/LD', ROOT / 'tests/native/benchmark_memory.cpp', '/Fo:' + str(work / 'readme_memory.obj'),
                  '/Fe:' + str(stage / 'readme_memory.dll'), 'dxgi.lib', 'psapi.lib'], cwd=work, env=env)
    executable, forwarder = build_native_probe(work)
    appearance = work / 'appearance_memory_probe.exe'
    run(common + ['/I' + str(ROOT.parent / 'external/DLSS/include'),
                  '/I' + str(ROOT / 'third_party/nvapi'), ROOT / 'tests/native/appearance_memory_probe.cpp',
                  '/Fo:' + str(work / 'appearance.obj'), '/Fe:' + str(appearance),
                  'd3d12.lib', 'dxgi.lib', 'd3dcompiler.lib', 'bcrypt.lib'], cwd=work, env=env)
    appearance_result = json.loads(run([appearance, 'current'], cwd=work, env=env))
    (output / 'appearance-memory.json').write_text(json.dumps(appearance_result, indent=2) + '\n')
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'passed': True, 'gpu_inference_executed': False, 'runtime_root': str(stage),
              'native_bundle': str(args.native_bundle.resolve()), 'work_root': str(work),
              'native_bundle_manifest_sha256': sha(args.native_bundle / 'native_manifest.json'),
              'native_probe': str(executable), 'forwarder': str(forwarder), 'device_query': str(query),
              'binaries': {str(p.relative_to(stage)): sha(p) for p in sorted(stage.rglob('*')) if p.is_file()},
              'native_probe_sha256': sha(executable), 'forwarder_sha256': sha(forwarder),
            'sources': {str(p.relative_to(ROOT)): sha(p) for base in
                [ROOT / 'native_runtime', ROOT / 'src/native', ROOT / 'src/backends', ROOT / 'src/integration/reshade',
                 ROOT / 'tests/native'] for p in sorted(base.rglob('*'))
                if p.is_file() and p.suffix in ('.cpp', '.c', '.h', '.hpp', '.inc', '.cuh', '.cu', '.comp', '.glsl', '.metal', '.json')},
            'reused_vulkan_provider': ('None: low-level Vulkan provider, SPIR-V, adapter and runtime rebuilt from current source'
                if args.vulkan_bundle else 'Low-level Vulkan provider and SPIR-V staged from retained neural-candidate artifact; current adapter/runtime rebuilt'),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'dirty_checkout': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True))}
    (output / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Benchmark runtime compiled; no inference or installation performed', flush=True)


if __name__ == '__main__':
    main()
