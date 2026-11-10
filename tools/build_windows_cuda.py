"""Build optional Windows CUDA runtimes and model-free SM120 kernels.

Compilation only. No driver probe, model loading, inference or game launch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from test_model_selection import ROOT, toolchain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/current/windows-cuda')
    parser.add_argument('--ngx-sdk',type=Path,required=True,help='NGX SDK include folder for the shared Windows pipeline')
    parser.add_argument('--session',action='store_true',help='Keep one temporary incremental build until quit; commands: build, quit')
    parser.add_argument('--vulkan',action='store_true',help='Include the portable Vulkan provider and shaders')
    parser.add_argument('--vulkan-headers',type=Path,default=ROOT.parent/'dlss5-everything-testing/vendor')
    parser.add_argument('--glslang',type=Path,default=ROOT.parent/'external/renodx/bin/glslang.exe')
    args = parser.parse_args()
    compiler, env = toolchain(64)
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT/'artifacts/current'):
        parser.error('Choose a current artifact directory inside this project')
    cmake = shutil.which('cmake', path=env['PATH'])
    ninja = shutil.which('ninja', path=env['PATH'])
    nvcc = args.cuda.resolve()/'bin/nvcc.exe'
    if not cmake or not ninja or not nvcc.is_file():
        parser.error('CMake, Ninja and a Windows CUDA Toolkit are required')
    destination.mkdir(parents=True, exist_ok=True)
    def snapshot():
        inputs = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('native_runtime','src/native','src/pipeline','src/backends/hip_sycl',*(('src/backends/vulkan','third_party/intel_nr/src/gpu') if args.vulkan else ()))
        for p in (ROOT/folder).rglob('*') if p.is_file() and (p.suffix in ('.h','.cpp','.cu','.cuh','.json','.inc','.hlsl','.c','.comp','.glsl','.def') or p.name=='CMakeLists.txt')}
        for name in ('tools/build_windows_cuda.py','tools/build_native_cuda.py','tools/portable_frame.cpp'):
            inputs[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if args.vulkan:
            for name in ('tools/generate_standalone_vulkan.py','tools/build_vulkan.py','tools/vulkan_phoenix_math.py','tools/vulkan_compatibility.py','tools/vulkan_fast_math.py','tools/source_patches.py','third_party/vulkan-loader.def','third_party/amd_nr/import_manifest.json','third_party/intel_nr/import_manifest.json'):
                inputs[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        return inputs
    receipt = destination/'BUILD.json'
    def build(work):
        inputs=snapshot()
        report = {'passed': False, 'gpu_executed': False, 'models_included': False}
        receipt.write_text(json.dumps(report)+'\n')
        def run(name, command):
            with (destination/(name+'.log')).open('w', encoding='utf-8') as log:
                result = subprocess.run([str(x) for x in command], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} failed: see {destination/(name+".log")}')
        vulkan_options=[]
        if args.vulkan:
            if not (args.vulkan_headers/'vulkan/vulkan.h').is_file() or not args.glslang.is_file():
                raise RuntimeError('Vulkan headers and glslang are required')
            run('vulkan-import',[shutil.which('lib.exe',path=env['PATH']),'/nologo','/def:'+str(ROOT/'third_party/vulkan-loader.def'),'/machine:x64','/out:'+str(work/'vulkan-1.lib')])
            vulkan_options=['-DPHOENIX_ENABLE_VULKAN=ON','-DPHOENIX_VULKAN_HEADERS='+str(args.vulkan_headers.resolve()),'-DPHOENIX_GLSLANG='+str(args.glslang.resolve()),'-DPHOENIX_VULKAN_LOADER='+str(work/'vulkan-1.lib')]
        run('configure', [cmake, '-S', ROOT/'native_runtime', '-B', work, '-G', 'Ninja',*vulkan_options,
            '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=OFF', '-DPHOENIX_ENABLE_METAL=OFF',
            '-DPHOENIX_ENABLE_CUDA=ON', '-DPHOENIX_ENABLE_NATIVE_CUDA=ON',
            '-DPHOENIX_ENABLE_NATIVE_RTX=ON','-DPHOENIX_NGX_INCLUDE='+str(args.ngx_sdk.resolve()),
            '-DCMAKE_C_COMPILER='+compiler, '-DCMAKE_CXX_COMPILER='+compiler,
            '-DCMAKE_CUDA_COMPILER='+str(nvcc), '-DCMAKE_CUDA_HOST_COMPILER='+compiler,
            '-DCMAKE_CUDA_ARCHITECTURES=120',
            '-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
            '-DPHOENIX_CUDA_LICENSE='+str(args.cuda.resolve()/'LICENSE'),
            '-DCMAKE_MAKE_PROGRAM='+ninja])
        run('compile', [cmake, '--build', work, '--parallel', '4', '--target',
            'phoenix_runtime_cpp', 'phoenix_cpu_cpp', 'phoenix_cuda_cpp', 'phoenix_native_cuda', 'phoenix_frame','phoenix_native','phoenix_native_broker',*(['phoenix_vulkan_cpp'] if args.vulkan else [])])
        files = []
        for name in ('phoenix_runtime_cpp.dll', 'phoenix_cpu_cpp.dll',
                     'phoenix_cuda_cpp.dll', 'phoenix_native_cuda.dll', 'phoenix_frame.exe','phoenix_native.dll','phoenix_native_broker.exe'):
            shutil.copy2(work/name, destination/name)
            files.append(name)
        if args.vulkan:
            shutil.copy2(work/'phoenix_vulkan_cpp.dll',destination/'phoenix_vulkan_cpp.dll');files.append('phoenix_vulkan_cpp.dll')
            for path in (work/'vulkan').iterdir():
                if path.is_file() and (path.suffix in ('.dll','.spv') or path.name=='standalone_manifest.json'):
                    target=destination/'vulkan'/path.name;target.parent.mkdir(exist_ok=True);shutil.copy2(path,target);files.append('vulkan/'+path.name)
        bundle = destination/'native-cuda'
        bundle.mkdir(exist_ok=True)
        for path in (work/'native-cuda').rglob('*'):
            if path.is_file() and (path.suffix == '.cubin' or path.name in ('native_manifest.json','native_graph.json')):
                relative = path.relative_to(work/'native-cuda')
                target = bundle/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path,target)
                files.append('native-cuda/'+relative.as_posix())
        shutil.copy2(args.cuda/'LICENSE', destination/'CUDA-LICENSE.txt')
        files.append('CUDA-LICENSE.txt')
        for name in ('phoenix.h','phoenix_rgba8.h','phoenix_linux.h','phoenix_native.h','phoenix_cuda_d3d12.h'):
            if not (ROOT/'native_runtime'/name).is_file():continue
            shutil.copy2(ROOT/'native_runtime'/name,destination/name)
            files.append(name)
        if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest for name,digest in inputs.items()):
            raise RuntimeError('Source changed during compilation; rebuild from a stable tree')
        report.update(passed=True, source_inputs=inputs, files={name:hashlib.sha256((destination/name).read_bytes()).hexdigest()
            for name in files}, source_revision=subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        receipt.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':True,'output':str(destination),'gpu_executed':False}),flush=True)
    with tempfile.TemporaryDirectory(prefix='windows-cuda-', dir=ROOT/'build') as temporary:
        work = Path(temporary)
        if not args.session:build(work)
        else:
            print(json.dumps({'temporary_build':str(work),'commands':['build','quit']}),flush=True)
            for line in iter(input,'quit'):
                if line.strip()!='build':continue
                try:build(work)
                except Exception as error:print(json.dumps({'passed':False,'error':str(error)}),flush=True)


if __name__ == '__main__':
    main()
