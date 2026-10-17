"""Compile the full x86 addon and inspect driver ABI without graphics/inference."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ngx-sdk', required=True, type=Path)
    args = parser.parse_args()
    vswhere = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    vs = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    with tempfile.TemporaryDirectory(prefix='phoenix-x86-core-') as temporary:
        folder = Path(temporary)
        flags = (f'/nologo /std:c++20 /Zc:__cplusplus /O2 /EHsc /MT /bigobj /DNOMINMAX '
            f'/I"{args.ngx_sdk.resolve()}" /I"{ROOT}/third_party/nvapi" '
            f'/I"{ROOT}/third_party/minhook/include" /I"{ROOT}/third_party/imgui_reshade" '
            f'/I"{ROOT}/third_party/reshade" /DImTextureID=ImU64')
        hook = ' '.join(f'"{ROOT}/third_party/minhook/src/{name}"'
            for name in ('hook.c', 'buffer.c', 'trampoline.c', 'hde/hde32.c'))
        lines = ['@echo off', f'call "{vs}/VC/Auxiliary/Build/vcvars32.bat" >nul',
                 'if errorlevel 1 exit /b 2',
                 f'cl {flags} "{ROOT}/tests/native/nvapi_arch_probe.cpp" /Fe:probe.exe',
                 'if errorlevel 1 exit /b 3', 'probe.exe', 'if errorlevel 1 exit /b 4',
                 f'cl {flags} /LD {hook} "{ROOT}/src/integration/reshade/addon.cpp" /Fe:Phoenix.addon32 '
                 '/link /INCREMENTAL:NO d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib '
                 'bcrypt.lib psapi.lib cabinet.lib shell32.lib ole32.lib uuid.lib windowscodecs.lib',
                 'exit /b %errorlevel%']
        recipe = folder / 'check.cmd'
        recipe.write_text('\n'.join(lines))
        run = subprocess.run(['cmd','/d','/c',str(recipe)], cwd=folder, capture_output=True, text=True)
        print('\n'.join(line for line in run.stdout.splitlines()
                        if line.startswith('{') or 'error ' in line.lower()))
        if run.returncode:
            print(run.stdout[-4500:])
            raise RuntimeError(f'x86 core check failed: {run.returncode}')
        print('Full x86 addon compiled; no graphics or inference executed.')


if __name__ == '__main__':
    main()
