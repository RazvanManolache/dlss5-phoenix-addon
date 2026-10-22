"""CPU-only integration contracts and optional full addon compilation, no games/GPU."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def resources(binary, destination):
    """Read the existing helper payload as data, never execute the old addon."""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    kernel.LoadLibraryExW.restype = ctypes.c_void_p
    kernel.FindResourceW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    kernel.FindResourceW.restype = ctypes.c_void_p
    kernel.SizeofResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.SizeofResource.restype = ctypes.c_uint32
    kernel.LoadResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.LoadResource.restype = ctypes.c_void_p
    kernel.LockResource.argtypes = [ctypes.c_void_p]
    kernel.LockResource.restype = ctypes.c_void_p
    kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
    module = kernel.LoadLibraryExW(str(binary), None, 0x22)
    if not module:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        def read(resource_id):
            entry = kernel.FindResourceW(module, resource_id, 10)
            if not entry:
                raise ValueError('Missing embedded resource ' + str(resource_id))
            size = kernel.SizeofResource(module, entry)
            address = kernel.LockResource(kernel.LoadResource(module, entry))
            if not size or not address:
                raise ValueError('Unreadable embedded resource')
            return ctypes.string_at(address, size)
        manifest = json.loads(read(1001))
        lines = []
        hashes = {}
        for identity in [1001, *[entry['resource'] for entry in manifest['files']]]:
            data = read(identity)
            path = destination / f'{identity}.bin'
            path.write_bytes(data)
            lines.append(f'{identity} RCDATA "{path.as_posix()}"')
            hashes[str(identity)] = hashlib.sha256(data).hexdigest()
        recipe = destination / 'runtime.rc'
        recipe.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return recipe, hashes
    finally:
        kernel.FreeLibrary(module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-addons', action='store_true')
    parser.add_argument('--ngx-sdk', type=Path)
    parser.add_argument('--imgui-source', type=Path, help='Optional matching ImGui v1.92.5-docking sources for real widget tests')
    args = parser.parse_args()
    if args.build_addons and not args.ngx_sdk:
        parser.error('--build-addons requires --ngx-sdk')
    if args.imgui_source:
        def header(path):
            return '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8-sig').splitlines())
        if header(args.imgui_source/'imgui.h') != header(ROOT/'third_party/imgui_reshade/imgui.h'):
            parser.error('ImGui source headers do not match the retained ReShade headers')
    vswhere = Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    if not studio:
        raise RuntimeError('No MSVC toolchain')
    current = ROOT/'artifacts/current'
    report_path = current/'portable-bridge-validation.json'
    inputs = [*list((ROOT/'src/integration/compatibility').glob('*.h')),
        *[p for p in (ROOT/'src/integration/compatibility/wow64').rglob('*') if p.is_file()],
        *list((ROOT/'src/integration/reshade').glob('*.h')), ROOT/'src/integration/reshade/addon.cpp',
        ROOT/'native_runtime/runtime_api.h', ROOT/'tools/validate_portable_bridge.py',
        ROOT/'tests/native/portable_bridge_fake.cpp', ROOT/'tests/native/portable_bridge_tests.cpp',
        ROOT/'tests/native/backbuffer_route_tests.cpp']
    inputs.extend([ROOT/'tests/native/shared_panel_tests.cpp', ROOT/'tests/native/dashboard_tests.h'])
    bindings = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    report = dict(schema='phoenix-portable-bridge-validation/1', passed=False,
        gpu_used=False, games_launched=False, game_presentation_validated=False,
        production_package_modified=False, source_sha256=bindings, architectures={})
    def save():
        report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    libraries = 'd3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib bcrypt.lib psapi.lib cabinet.lib shell32.lib ole32.lib uuid.lib windowscodecs.lib'
    try:
        with tempfile.TemporaryDirectory(prefix='portable-bridge-', dir=ROOT/'artifacts') as temporary:
            build = Path(temporary)
            for bits in (32, 64):
                out = build/str(bits)
                out.mkdir()
                row = report['architectures'][str(bits)] = dict(builds={}, tests={})
                common = '/nologo /std:c++20 /Zc:__cplusplus /O2 /EHsc /MT /DNOMINMAX'
                def compile(name, command):
                    recipe = out/'build.cmd'
                    recipe.write_text('\n'.join(['@echo off',
                        f'call "{studio}/VC/Auxiliary/Build/vcvars{bits}.bat" >nul 2>&1',
                        'if errorlevel 1 exit /b %errorlevel%', command, 'exit /b %errorlevel%', '']))
                    result = subprocess.run(['cmd', '/d', '/c', str(recipe)], cwd=out,
                        capture_output=True, text=True, timeout=300)
                    row['builds'][name] = dict(passed=result.returncode == 0,
                        diagnostics=(result.stdout+result.stderr)[-10000:] if result.returncode else '')
                    save()
                    print(f'{bits}-bit {name}: {"compiled" if not result.returncode else "FAILED"}', flush=True)
                    if result.returncode:
                        raise RuntimeError(row['builds'][name]['diagnostics'])
                def run(name, *arguments):
                    result = subprocess.run([str(out/(name+'.exe')), *map(str, arguments)],
                        cwd=out, capture_output=True, text=True, timeout=45)
                    row['tests'][name] = dict(passed=result.returncode == 0,
                        output=(result.stdout+result.stderr)[-6000:])
                    save()
                    print(f'{bits}-bit {name}: {result.stdout.strip() or result.stderr.strip()}', flush=True)
                    if result.returncode: raise RuntimeError(row['tests'][name]['output'])
                compile('fake_runtime', f'cl {common} /LD "{ROOT}/tests/native/portable_bridge_fake.cpp" '
                    f'/Fe:"{out}/phoenix_runtime_cpp.dll" /link /INCREMENTAL:NO')
                for name in ('portable_bridge_tests', 'backbuffer_route_tests', 'compatibility_policy_tests', 'compatibility_ipc_tests'):
                    compile(name, f'cl {common} "{ROOT}/tests/native/{name}.cpp" /Fe:"{out}/{name}.exe" /link {libraries}')
                    run(name, *([out] if name == 'portable_bridge_tests' else []))
                if args.imgui_source:
                    vendor = args.imgui_source.resolve()
                    sources = ' '.join(f'"{path}"' for path in [ROOT/'tests/native/shared_panel_tests.cpp',
                        *[vendor/(name+'.cpp') for name in ('imgui','imgui_draw','imgui_tables','imgui_widgets')]])
                    compile('shared_panel_tests', f'cl {common} /DIMGUI_DISABLE_INCLUDE_IMCONFIG_H /DIMGUI_ENABLE_TEST_ENGINE '
                        f'/DImTextureID=ImU64 /I"{ROOT}/third_party/imgui_reshade" /I"{vendor}" {sources} '
                        f'/Fe:"{out}/shared_panel_tests.exe" /link {libraries}')
                    run('shared_panel_tests')
                if args.build_addons:
                    old = current/'addon'/f'Phoenix.addon{bits}'
                    resource_folder = out/'resources'
                    resource_folder.mkdir()
                    recipe, hashes = resources(old, resource_folder)
                    row['retained_payload_sha256'] = hashes
                    res = out/'runtime.res'
                    compile('resources', f'rc /nologo /fo "{res}" "{recipe}"')
                    hooks = ' '.join(f'"{ROOT}/third_party/minhook/src/{name}"' for name in
                        ('hook.c','buffer.c','trampoline.c',f'hde/hde{bits}.c'))
                    includes = ' '.join(f'/I"{path}"' for path in [args.ngx_sdk.resolve(),
                        ROOT/'third_party/nvapi', ROOT/'third_party/minhook/include',
                        ROOT/'third_party/imgui_reshade', ROOT/'third_party/reshade'])
                    compile('addon', f'cl {common} /bigobj /DImTextureID=ImU64 {includes} /LD '
                        f'"{ROOT}/src/integration/reshade/addon.cpp" {hooks} '
                        f'/Fe:"{out}/Phoenix.addon{bits}" /link /INCREMENTAL:NO "{res}" {libraries}')
                    if bits == 32:
                        from build_compatibility import pe_exports
                        import re
                        aliases = []
                        for name in sorted(pe_exports(out/'Phoenix.addon32')):
                            match = re.fullmatch(r'_(.+)@(\d+)', name)
                            if match: aliases.append(f'{match[1]}={name}')
                        definition = out/'addon.def'
                        definition.write_text('EXPORTS\n'+'\n'.join(aliases)+'\n')
                        objects = ' '.join(f'"{out}/{name}.obj"' for name in ('addon','hook','buffer','trampoline','hde32'))
                        compile('addon_exports', f'link /DLL /INCREMENTAL:NO /DEF:"{definition}" /OUT:"{out}/Phoenix.addon32" '
                            f'{objects} "{res}" {libraries}')
                    row['addon_sha256'] = hashlib.sha256((out/f'Phoenix.addon{bits}').read_bytes()).hexdigest()
            for name, digest in bindings.items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError('Source changed during verification: ' + name)
            if args.build_addons:
                candidate = current/'addon-portable-candidate'
                if candidate.exists():
                    if candidate.is_symlink() or candidate.is_junction() or candidate.resolve().parent != current.resolve():
                        raise ValueError('Unsafe candidate directory')
                candidate.mkdir(exist_ok=True)
                for bits in (32, 64):
                    shutil.copy2(build/str(bits)/f'Phoenix.addon{bits}', candidate/f'Phoenix.addon{bits}')
                shutil.copy2(current/'addon/phoenix_graphics_host.exe', candidate/'phoenix_graphics_host.exe')
            report['passed'] = True
            save()
    finally:
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
