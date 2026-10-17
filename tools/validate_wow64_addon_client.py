"""Opt-in GPU validation of the packaged x86 addon client; no games or deployment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import struct
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-gpu',action='store_true',required=True)
    parser.add_argument('--x86-proxy',type=Path,help='Optional existing x86 ReShade DXGI proxy for the DLL-search regression')
    args=parser.parse_args()
    current=ROOT/'artifacts/current'
    runtime=current/'addon-portable-candidate/PhoenixRuntime/wow64'
    vswhere=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
    studio=subprocess.check_output([str(vswhere),'-latest','-products','*','-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True).strip()
    inputs=[*list((ROOT/'src/integration/compatibility/wow64').rglob('*')),
        *list((ROOT/'src/integration/compatibility').glob('*.h')),ROOT/'tests/native/wow64_addon_runtime_probe.cpp']
    report=dict(passed=False,gpu_used=True,games_launched=False,game_presentation_validated=False,
        sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.is_file()})
    try:
        package=json.loads((current/'wow64-runtime-build.json').read_text())
        report['runtime_files']={p.relative_to(runtime).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in runtime.rglob('*') if p.is_file()}
        if not package.get('passed') or package['files']!=report['runtime_files']:
            raise ValueError('Packaged native runtime differs from its build receipt')
        with tempfile.TemporaryDirectory(prefix='wow64-addon-client-',dir=ROOT/'artifacts') as temp:
            build=Path(temp)
            recipe=build/'build.cmd'
            recipe.write_text('\n'.join(['@echo off',f'call "{studio}/VC/Auxiliary/Build/vcvars32.bat" >nul 2>&1',
                'if errorlevel 1 exit /b %errorlevel%',
                f'cl /nologo /std:c++20 /O2 /EHsc /MT /bigobj /DNOMINMAX "{ROOT}/tests/native/wow64_addon_runtime_probe.cpp" /Fe:client.exe /link '
                'kernel32.lib user32.lib gdi32.lib bcrypt.lib d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib psapi.lib cabinet.lib shell32.lib ole32.lib uuid.lib windowscodecs.lib '
                '/LARGEADDRESSAWARE /SUBSYSTEM:WINDOWS /ENTRY:wmainCRTStartup',
                'exit /b %errorlevel%']))
            built=subprocess.run(['cmd','/d','/c',str(recipe)],cwd=build,capture_output=True,text=True,timeout=180)
            if built.returncode: raise RuntimeError((built.stdout+built.stderr)[-14000:])
            if args.x86_proxy:
                data=args.x86_proxy.read_bytes()
                offset=struct.unpack_from('<I',data,60)[0]
                if struct.unpack_from('<H',data,offset+4)[0]!=0x14c: raise ValueError('Expected an x86 proxy')
                report['x86_dxgi_proxy_sha256']=hashlib.sha256(data).hexdigest()
                shutil.copy2(args.x86_proxy,build/'dxgi.dll')
            run=subprocess.run([str(build/'client.exe'),str(runtime),str(current/'weight-source.json')],
                cwd=build,capture_output=True,text=True,timeout=120)
            report['returncode']=run.returncode
            report['output']=(run.stdout+run.stderr)[-12000:]
            if run.returncode: raise RuntimeError(report['output'] or f'Client exited {run.returncode}')
            report['client']=json.loads(run.stdout.splitlines()[-1])
            for name,expected in report['sources'].items():
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
                    raise ValueError('Source changed during validation: '+name)
            report['passed']=report['client']['passed']
    finally:
        (current/'wow64-addon-client-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['client'],indent=2))


if __name__=='__main__': main()
