"""Bounded structure/ABI regressions; isolated integer GPU probe is explicit."""
import argparse
import json
import tempfile
from pathlib import Path
from test_model_selection import ROOT, toolchain, run


def cmake_contracts(work, env):
    build = work / 'cmake'
    run(['cmake', '-S', ROOT / 'native_runtime', '-B', build, '-G', 'Ninja',
         '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=ON',
         '-DPHOENIX_ENABLE_VULKAN=OFF', '-DPHOENIX_ENABLE_CUDA=OFF'], cwd=work, env=env)
    run(['cmake', '--build', build, '--target', 'kernel_word_contract_tests'], cwd=work, env=env)
    result = run(['ctest', '--test-dir', build, '-R', '^kernel_(word_contracts|structure_guard|structure_tools)$',
                  '--output-on-failure'], cwd=work, env=env)
    return dict(bits=64, test='cmake_kernel_contracts', passed=True, output=result.strip())


def gpu_contracts(work, env):
    nvcc = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe')
    result = {}
    cases = []
    probes = ('kernel_division_contract_tests', 'kernel_boundary_clear_tests', 'kernel_numeric_contract_tests')
    for name in probes:
        executable = work / (name + '.exe')
        run([nvcc, '-allow-unsupported-compiler', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
             '-arch=sm_120', '-O3', '-std=c++20', ROOT / ('tests/native/' + name + '.cu'),
             '-o', executable], cwd=work, env=env)
        probe = json.loads(run([executable], cwd=work, env=env))
        if probe.get('passed') is not True:
            raise RuntimeError(name + ' did not pass')
        result.update(probe)
        cases.append(dict(test=name, result=probe))
    return dict(bits=64, test='kernel_division_contract_tests',
                probes_executed=list(probes), cases=cases, **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/current/kernel-structure')
    args = parser.parse_args()
    (ROOT / 'build').mkdir(exist_ok=True)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, games_launched=False, installed_files_modified=False, cases=[])
    output = destination / 'contracts.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    model = ROOT.parent / 'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
    for bits in (32, 64):
        compiler, env = toolchain(bits)
        with tempfile.TemporaryDirectory(prefix='kernel-structure-', dir=ROOT / 'build') as folder:
            work = Path(folder)
            flags = [compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/O2', '/fp:strict',
                     '/EHsc', '/MD', '/bigobj', '/DNOMINMAX',
                     '/I' + str(ROOT.parent / 'external/DLSS/include'),
                     '/I' + str(ROOT / 'third_party/nvapi')]
            cases = [
                ('native', 'kernel_word_contract_tests', []),
                ('portable_cpp', 'frame_boundary_tests', []),
                ('portable_cpp', 'advanced_frame_tests', []),
                ('portable_cpp', 'model_selection_tests', []),
                ('native', 'generated_plan_tests', [ROOT]),
            ]
            if bits == 64:
                cases.append(('native', 'native_small_model_tests',
                              [ROOT, ROOT / 'artifacts/current/native-kernels', model]))
            for group, name, arguments in cases:
                executable = work / (name + '.exe')
                run(flags + [ROOT / f'tests/{group}/{name}.cpp', '/Fo:' + str(work / (name + '.obj')),
                             '/Fe:' + str(executable), 'd3d12.lib', 'dxgi.lib', 'd3dcompiler.lib',
                             'bcrypt.lib', 'cabinet.lib', 'shell32.lib', 'ole32.lib', 'uuid.lib', 'user32.lib'],
                    cwd=work, env=env)
                result = run([executable, *arguments], cwd=work, env=env)
                report['cases'].append(dict(bits=bits, test=name, passed=True, output=result.strip()))
                output.write_text(json.dumps(report, indent=2) + '\n')
                print(bits, name, 'PASS', flush=True)
            if bits == 64 and args.gpu:
                report['cases'].append(gpu_contracts(work, env))
                print('GPU integer scheduling PASS', flush=True)
            if bits == 64:
                report['cases'].append(cmake_contracts(work, env))
                print('CMake/CTest kernel contracts PASS', flush=True)
    report.update(passed=True, gpu_integer_probe_executed=args.gpu, gpu_inference_executed=False,
                  temporary_builds_removed=True)
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
