"""Build development addons/runtimes and helpers; never run a game or graphics device."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from test_model_selection import ROOT, toolchain, run
from build_compatibility import pe_exports, pe_machine


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build/current')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/current/processing')
    args = parser.parse_args()
    work, destination = args.build_dir.resolve(), args.output.resolve()
    if not work.is_relative_to(ROOT / 'build') or not destination.is_relative_to(ROOT / 'artifacts'):
        parser.error('Use project build/ and artifacts/ folders')
    work.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    sources = {path.relative_to(ROOT).as_posix(): digest(path)
               for folder in ('src', 'native_runtime', 'third_party')
               for path in (ROOT / folder).rglob('*')
               if path.is_file() and path.suffix in {'.h', '.hpp', '.cpp', '.c', '.inc'}}
    report = dict(passed=False, gpu_executed=False, games_launched=False, deployed=False,
                  embedded_runtime=False, model_data_included=False, cases=[], source_inputs=sources,
                  build_tool_sha256=digest(Path(__file__)))
    receipt = destination / 'build.json'
    receipt.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    libraries = ['d3d11.lib', 'd3d12.lib', 'dxgi.lib', 'd3dcompiler.lib', 'user32.lib', 'bcrypt.lib',
                 'psapi.lib', 'cabinet.lib', 'shell32.lib', 'ole32.lib', 'uuid.lib', 'windowscodecs.lib']
    for bits in (32, 64):
        compiler, environment = toolchain(bits)
        common = [compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/EHsc', '/O2', '/MT',
                  '/bigobj', '/DNOMINMAX', *['/I' + str(path) for path in
                  (ROOT.parent / 'external/DLSS/include', ROOT / 'third_party/nvapi',
                   ROOT / 'third_party/minhook/include', ROOT / 'third_party/imgui_reshade',
                   ROOT / 'third_party/reshade')]]

        def build(label, command):
            try:
                output = run(command, cwd=work, env=environment, timeout=300)
            except RuntimeError as error:
                (work / (label + '.build.log')).write_text(str(error), encoding='utf-8')
                raise RuntimeError(f'{label} failed; inspect {work / (label + ".build.log")}') from None
            (work / (label + '.build.log')).write_text(output, encoding='utf-8')

        def publish(path, name, exports=()):
            assert pe_machine(path) == (0x14c if bits == 32 else 0x8664)
            assert set(exports).issubset(pe_exports(path))
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            report['cases'].append(dict(artifact=name, bits=bits, compiled_and_linked=True,
                                        exports_verified=bool(exports), sha256=digest(target)))

        hooks = []
        for name in ('hook', 'buffer', 'trampoline', 'hde/hde' + str(bits)):
            obj = work / f'processing_{Path(name).name}{bits}.obj'
            build('hook_' + name.replace('/', '_') + str(bits), common + ['/c',
                  ROOT / f'third_party/minhook/src/{name}.c', '/Fo:' + str(obj)])
            hooks.append(obj)
        obj = work / f'processing_addon{bits}.obj'
        addon = work / f'processing.addon{bits}'
        build(f'processing_addon{bits}', common + ['/DImTextureID=ImU64', '/DPHOENIX_BUILD_TAG=processing',
              '/c', ROOT / 'src/integration/reshade/addon.cpp', '/Fo:' + str(obj)])
        link = [str(Path(compiler).parent / 'link.exe'), '/DLL', '/INCREMENTAL:NO',
                '/OUT:' + str(addon), obj, *hooks, *libraries]
        build(f'processing_link{bits}', link)
        if bits == 32:
            aliases = []
            for name in sorted(pe_exports(addon)):
                match = re.fullmatch(r'_(.+)@(\d+)', name)
                if match:
                    aliases.append('    ' + match[1] + '=' + name)
            definition = work / 'processing_addon32.def'
            definition.write_text('EXPORTS\n' + '\n'.join(aliases) + '\n')
            build('processing_addon32_exports', [*link, '/DEF:' + str(definition)])
        publish(addon, f'Phoenix.addon{bits}', ['AddonInit', 'AddonUninit', 'CnrPHOENIXFeederEvaluate',
                                              'CnrPHOENIXInProcessBackbufferVersion'])
        runtime = work / f'processing_runtime{bits}.dll'
        link_options = ['/link', '/IMPLIB:' + str(work / f'processing_runtime{bits}.lib')]
        if bits == 32:
            link_options.append('/DEF:' + str(ROOT / 'native_runtime/phoenix_runtime.def'))
        build(f'processing_runtime{bits}', common + ['/LD', ROOT / 'native_runtime/runtime.cpp',
              '/Fo:' + str(work / f'processing_runtime{bits}.obj'), '/Fe:' + str(runtime),
              'bcrypt.lib', *link_options])
        publish(runtime, f'runtime{bits}/phoenix_runtime_cpp.dll', ['phoenix_runtime_create', 'phoenix_runtime_frame'])
        print(f'Addon and runtime {bits} compiled, linked and exports verified', flush=True)
        if bits == 64:
            shim = work / 'processing-nvngx.dll'
            build('processing_model64', common + ['/LD', ROOT / 'src/integration/compatibility/wow64/model64.cpp',
                  ROOT / 'src/integration/reshade/nr_forwarder.cpp', '/Fo:' + str(work) + '\\',
                  '/Fe:' + str(shim), '/link', 'kernel32.lib', 'd3d12.lib', 'dxgi.lib', 'user32.lib',
                  'gdi32.lib', 'advapi32.lib', 'bcrypt.lib'])
            publish(shim, 'phoenix-nvngx.dll', ['PhoenixModelServe64', 'PhoenixNrCreate'])
            host = work / 'processing_graphics_host.exe'
            build('processing_host', common + [ROOT / 'src/integration/compatibility/host/dlss5-feed-host64.cpp',
                  '/Fo:' + str(work / 'processing_host.obj'), '/Fe:' + str(host), '/link', 'd3d12.lib',
                  'dxgi.lib', 'd3dcompiler.lib', 'version.lib', 'user32.lib', 'gdi32.lib', 'advapi32.lib',
                  'ole32.lib', 'windowscodecs.lib'])
            publish(host, 'phoenix_graphics_host.exe')
    for name, value in sources.items():
        if digest(ROOT / name) != value:
            raise RuntimeError('Source changed while building: ' + name)
    report['passed'] = True
    receipt.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('All source identities verified. No GPU execution, installation or packaging.', flush=True)


if __name__ == '__main__':
    main()
