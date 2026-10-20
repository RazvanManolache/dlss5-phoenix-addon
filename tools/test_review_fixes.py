"""Bounded x86/x64 review regressions and frontend builds. No games or GPU inference."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from test_model_selection import ROOT, toolchain, run


def main():
    output = ROOT / 'artifacts/current/review-fixes/tests'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'validation.json').write_text(json.dumps(dict(passed=False, state='incomplete')) + '\n')
    model = ROOT.parent / 'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
    sdk = ROOT.parent / 'external/DLSS/include'
    cases = []
    for bits in [32, 64]:
        compiler, env = toolchain(bits)
        with tempfile.TemporaryDirectory(prefix=f'review-tests-{bits}-', dir=ROOT / 'build') as folder:
            work = Path(folder)
            base = [compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/O2', '/EHsc', '/MD', '/bigobj', '/DNOMINMAX',
                    '/I' + str(sdk), '/I' + str(ROOT / 'third_party/nvapi')]
            libraries = ['d3d12.lib', 'dxgi.lib', 'd3dcompiler.lib', 'bcrypt.lib', 'cabinet.lib', 'shell32.lib', 'ole32.lib', 'uuid.lib', 'user32.lib']
            for group, name, arguments in [
                ('portable_cpp', 'resident_weight_tests', [model]),
                ('portable_cpp', 'network_schedule_tests', []),
                ('portable_cpp', 'model_backend_graph_tests', [model]),
                ('portable_cpp', 'small_model_refactor_probe', [ROOT, model]),
                ('native', 'processing_tests', []),
                ('native', 'granular_controls_tests', []),
                ('native', 'generated_plan_tests', [ROOT]),
                ('native', 'direct_tests', []),
                ('native', 'production_capture_tests', []),
            ]:
                executable = work / (name + '.exe')
                run(base + [ROOT / f'tests/{group}/{name}.cpp', '/Fo:' + str(work / (name + '.obj')),
                            '/Fe:' + str(executable), *libraries], cwd=work, env=env)
                result = json.loads(run([executable, *arguments], cwd=work, env=env))
                if not result.get('passed'):
                    raise RuntimeError(f'{bits} {name} did not pass')
                cases.append(dict(bits=bits, test=name, result=result))
                print(bits, name, 'PASS', flush=True)
            if bits == 64:
                name = 'native_small_model_tests'
                executable = work / (name + '.exe')
                run(base + [ROOT / f'tests/native/{name}.cpp', '/Fo:' + str(work / (name + '.obj')),
                            '/Fe:' + str(executable), *libraries], cwd=work, env=env)
                result = json.loads(run([executable, ROOT, ROOT / 'artifacts/current/native-kernels', model], cwd=work, env=env))
                cases.append(dict(bits=bits, test=name, result=result))
                print(bits, name, 'PASS', flush=True)
            # Link each frontend, but do not install or advertise a release.
            frontend = ROOT / ('src/integration/reshade/addon.cpp' if bits == 64 else 'src/integration/compatibility/addon32.cpp')
            hook = 'hde64.c' if bits == 64 else 'hde32.c'
            hooks = [ROOT / f'third_party/minhook/src/{p}' for p in ['hook.c', 'buffer.c', 'trampoline.c', 'hde/' + hook]]
            artifact = work / f'phoenix.addon{bits}'
            run(base + ['/LD', '/DImTextureID=ImU64', '/I' + str(ROOT / 'third_party/minhook/include'),
                        '/I' + str(ROOT / 'third_party/imgui_reshade'), '/I' + str(ROOT / 'third_party/reshade'),
                        frontend, *hooks, '/Fe:' + str(artifact), *libraries, 'd3d11.lib', 'psapi.lib', 'windowscodecs.lib'],
                cwd=work, env=env, timeout=600)
            shutil.copy2(artifact, output / artifact.name)
            cases.append(dict(bits=bits, test='frontend_link', passed=True,
                              sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))
            print(bits, 'frontend_link PASS', flush=True)
    report = dict(passed=True, gpu_inference_executed=False, software_d3d12_executed=True, games_launched=False,
                  installed_files_modified=False, temporary_builds_removed=True, cases=cases)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
