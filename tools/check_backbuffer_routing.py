"""Compile both addon architectures and run focused CPU checks; never use a GPU.

All compiler outputs live in a temporary directory removed on success or failure.
The current packaged addon is deliberately not replaced by this compile check.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ngx-sdk', type=Path, required=True)
    args = parser.parse_args()
    sdk = args.ngx_sdk.resolve()
    if not (sdk / 'nvsdk_ngx.h').is_file():
        parser.error('NGX SDK include directory is missing')
    vswhere = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    vs = Path(subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip())
    libs = 'd3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib bcrypt.lib psapi.lib cabinet.lib shell32.lib ole32.lib uuid.lib windowscodecs.lib'
    checks = ['backbuffer_route_tests', 'backbuffer_shader_tests', 'compatibility_policy_tests',
              'feeder_api_tests', 'graphics_transport_tests']
    with tempfile.TemporaryDirectory(prefix='phoenix-backbuffer-check-') as temporary:
        work = Path(temporary)
        for bits in (64, 32):
            folder = work / str(bits)
            folder.mkdir()
            flags = (f'/nologo /std:c++20 /Zc:__cplusplus /O2 /W4 /EHsc /MT /DNOMINMAX '
                f'/I"{sdk}" /I"{ROOT}/third_party/nvapi" /I"{ROOT}/third_party/minhook/include" '
                f'/I"{ROOT}/third_party/imgui_reshade" /I"{ROOT}/third_party/reshade" /DImTextureID=ImU64')
            recipe = ['@echo off', 'setlocal', f'call "{vs}/VC/Auxiliary/Build/vcvars{bits}.bat" >nul',
                      'if errorlevel 1 exit /b 2']
            for test in checks:
                recipe += [f'cl {flags} "{ROOT}/tests/native/{test}.cpp" /Fe:"{folder}/{test}.exe" /Fo:"{folder}/" /link {libs}',
                           'if errorlevel 1 exit /b 3', f'"{folder}/{test}.exe"', 'if errorlevel 1 exit /b 4']
            minhook = ' '.join(f'"{ROOT}/third_party/minhook/src/{name}"'
                for name in ('hook.c', 'buffer.c', 'trampoline.c', f'hde/hde{bits}.c'))
            source = ROOT / 'src/integration/reshade/addon.cpp'
            extra = minhook + ' /bigobj'
            recipe += [f'cl {flags} /LD {extra} "{source}" /Fe:"{folder}/Phoenix.addon{bits}" /Fo:"{folder}/" /link /INCREMENTAL:NO {libs}',
                       'if errorlevel 1 exit /b 5']
            recipe += [f'cl {flags} "{ROOT}/tests/native/compatibility_graphics_matrix.cpp" /Fe:"{folder}/graphics_matrix.exe" /Fo:"{folder}/" /link {libs} d3d9.lib d3d10.lib d3d10_1.lib opengl32.lib gdi32.lib',
                       'if errorlevel 1 exit /b 6']
            path = folder / 'check.cmd'
            path.write_bytes(('\r\n'.join(recipe) + '\r\n').encode('utf-8'))
            result = subprocess.run(['cmd', '/d', '/c', str(path)], cwd=folder, capture_output=True, text=True)
            if result.returncode:
                errors = [line for line in result.stdout.splitlines()
                          if 'error ' in line.lower() or 'fatal error' in line.lower()]
                print('\n'.join(errors) if errors else result.stdout[-4000:])
                print(result.stderr[-4000:])
                raise RuntimeError(f'{bits}-bit compile/CPU checks failed: {result.returncode}')
            results = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            if len(results) != len(checks) or not all(row.get('passed') is True for row in results):
                raise RuntimeError('Missing or failing CPU test receipt')
            print(json.dumps({'bits': bits, 'compiled': True, 'cpu_checks': dict(zip(checks, results)),
                              'gpu_executed': False, 'model_used': False}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
