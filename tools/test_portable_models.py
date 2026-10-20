"""CPU-only multi-model scheduling/graph contracts and portable Vulkan adapter builds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
from test_model_selection import ROOT, toolchain, run


def verify_adapter(path, bits, compiler, env):
    data = path.read_bytes()
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    machine = struct.unpack_from('<H', data, pe + 4)[0]
    if machine != (0x14c if bits == 32 else 0x8664):
        raise RuntimeError('Vulkan adapter architecture mismatch')
    listing = run([Path(compiler).parent / 'dumpbin.exe', '/exports', path], cwd=ROOT, env=env)
    names = {line.split()[-1] for line in listing.splitlines() if line.split()}
    for name in ['phoenix_backend_create', 'phoenix_backend_create_utf8',
                 'phoenix_backend_create_configured', 'phoenix_backend_destroy', 'phoenix_backend_features']:
        if name not in names:
            raise RuntimeError('Missing Vulkan backend ABI export: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'validation.json').write_text(json.dumps({'passed': False, 'state': 'incomplete', 'gpu_executed': False}) + '\n')
    work = ROOT / 'build/four_model_comparison'
    cases = []
    adapters = []
    for bits in (32, 64):
        compiler, env = toolchain(bits)
        common = [compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/EHsc', '/bigobj',
                  '/O2', '/MT', '/DNOMINMAX']
        for name, arguments in [('network_schedule_tests', []),
                                ('model_backend_graph_tests', [args.model.resolve()])]:
            executable = work / f'{name}{bits}.exe'
            run(common + [ROOT / f'tests/portable_cpp/{name}.cpp',
                          '/Fo:' + str(work / f'{name}{bits}.obj'), '/Fe:' + str(executable),
                          'bcrypt.lib'], cwd=work, env=env)
            result = json.loads(run([executable, *arguments], cwd=work, env=env))
            cases.append({'bits': bits, 'test': name, 'result': result})
            print(json.dumps({'bits': bits, 'test': name, 'passed': result['passed'],
                              'checks': result['checks']}), flush=True)
        adapter = work / f'portable_vulkan{bits}.dll'
        run(common + ['/LD', ROOT / 'native_runtime/backend_vulkan.cpp',
                      '/Fo:' + str(work / f'portable_vulkan{bits}.obj'), '/Fe:' + str(adapter),
                      '/link', '/IMPLIB:' + str(work / f'portable_vulkan{bits}.lib')], cwd=work, env=env)
        verify_adapter(adapter, bits, compiler, env)
        destination = output / f'runtime{bits}/phoenix_vulkan_cpp.dll'
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(adapter, destination)
        adapters.append({'bits': bits, 'artifact': str(destination.relative_to(output)),
                         'compiled_and_linked': True, 'exports_verified': True,
                         'sha256': hashlib.sha256(adapter.read_bytes()).hexdigest()})
    # Compile and simulate the generated Metal operator body on the host. This
    # explicitly does not compile Objective-C++, Apple AIR, or execute Metal.
    run(['python', ROOT / 'tools/generate_metal_operators.py', '--output', work / 'phoenix_reference.metal'], cwd=ROOT, env=env)
    compiler, env = toolchain(64)
    executable = work / 'portable_metal_host_tests.exe'
    run([compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/EHsc', '/O2', '/MT', '/DNOMINMAX',
         '/I' + str(ROOT), '/I' + str(work), ROOT / 'tests/portable_cpp/platform_tests.cpp',
         '/Fo:' + str(work / 'portable_metal_host_tests.obj'), '/Fe:' + str(executable), 'bcrypt.lib'], cwd=work, env=env)
    run([executable], cwd=work, env=env)
    report = {'passed': all(c['result']['passed'] for c in cases), 'gpu_executed': False,
              'model_inference_executed': False, 'cases': cases, 'vulkan_adapters': adapters,
              'metal_host_arithmetic': 'passed; host simulation only',
              'metal_apple_compiler_executed': False, 'hip_compiler_executed': False,
              'sycl_compiler_executed': False,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [ROOT / 'native_runtime/network_schedule.h',
                                          ROOT / 'native_runtime/runtime.cpp', ROOT / 'native_runtime/backend_metal.mm',
                                          ROOT / 'native_runtime/backend_vulkan.cpp', ROOT / 'native_runtime/backend_gpu.cpp',
                                          ROOT / 'src/native/support/worker_budget.h',
                                          ROOT / 'tests/portable_cpp/network_schedule_tests.cpp',
                                          ROOT / 'tests/portable_cpp/model_backend_graph_tests.cpp']}}
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'gpu_executed': False, 'vulkan_adapters_built': len(adapters)}))


if __name__ == '__main__':
    main()
