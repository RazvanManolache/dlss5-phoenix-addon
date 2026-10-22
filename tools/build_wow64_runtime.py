"""Build/package the opt-in in-process x86 transports. No GPU, games or model DLLs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import winreg

ROOT=Path(__file__).resolve().parents[1]
CURRENT=ROOT/'artifacts/current'


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def validate_native_bundle(bundle):
    manifest=json.loads((bundle/'native_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('model_weights_included') is not False or manifest.get('captured_plans_included') is not False:
        raise ValueError('A model-free source-built native bundle is required')
    for path in bundle.rglob('*'):
        if path.is_symlink() or path.is_junction():raise ValueError('Linked native bundle entry')
        if path.is_file() and (path.name.lower().startswith('nvngx') or path.name.lower() in ('weights.bin','model.bin')):
            raise ValueError('Model files do not belong in the native runtime package')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wow64ext',type=Path,required=True)
    parser.add_argument('--ngx-sdk',type=Path,required=True)
    parser.add_argument('--native-library',type=Path,help='Freshly built native API instead of the retained SDK')
    parser.add_argument('--native-bundle',type=Path,help='Matching model-free native bundle')
    parser.add_argument('--output',type=Path,help='Owned runtime directory below artifacts/current')
    parser.add_argument('--supported-models',nargs='+',choices=('original','small-fp8'),default=['original'])
    args=parser.parse_args()
    from probe_wow64_native import WOW64EXT_COMMIT
    extension=args.wow64ext.resolve()
    if subprocess.check_output(['git','-C',str(extension),'rev-parse','HEAD'],text=True).strip()!=WOW64EXT_COMMIT:
        raise ValueError('WOW64Ext must match the pinned research dependency')
    if subprocess.check_output(['git','-C',str(extension),'status','--porcelain','--untracked-files=no'],text=True).strip():
        raise ValueError('WOW64Ext has tracked modifications')
    baseline=json.loads((CURRENT/'wow64-x86-host-benchmark.json').read_text())
    windows=Path(os.environ['SystemRoot'])
    ntdll=windows/'System32/ntdll.dll'
    if digest(ntdll)!=baseline['system_image_sha256']: raise ValueError('Windows changed; refresh the standalone bootstrap probe first')
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SYSTEM\CurrentControlSet\Services\nvlddmkm') as key:
        image=winreg.QueryValueEx(key,'ImagePath')[0]
    if image.lower().startswith('\\systemroot\\'): image=str(windows)+'\\'+image[len('\\SystemRoot\\'):]
    if image.startswith('\\??\\'): image=image[4:]
    driver=Path(image).parent.resolve()
    if not driver.is_relative_to((windows/'System32/DriverStore/FileRepository').resolve()):
        raise ValueError('Active NVIDIA driver is not in the expected Windows driver store')
    if not (driver/'nvcuda64.dll').is_file(): raise ValueError('Verified native driver is unavailable')
    symbols=baseline['symbols']
    if not symbols.get('initializer_rva') or not symbols.get('allocator_rva') or len(symbols.get('appcompat_offsets',[]))!=4: raise ValueError('Missing verified native bootstrap symbols')
    sdk=CURRENT/'sdk/windows-x64'
    native=(args.native_library or sdk/'bin/phoenix_native.dll').resolve()
    bundle=(args.native_bundle or CURRENT/'native-kernels').resolve()
    validate_native_bundle(bundle)
    if args.native_library is None:
        sdk_hashes=json.loads((sdk/'FILES_SHA256.json').read_text())
        if digest(native)!=sdk_hashes['bin/phoenix_native.dll']: raise ValueError('Native SDK binary changed')
    vswhere=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
    studio=subprocess.check_output([str(vswhere),'-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True).strip()
    destination=(args.output or CURRENT/'addon-portable-candidate/PhoenixRuntime/wow64').resolve()
    if not destination.is_relative_to(CURRENT.resolve()) or destination.name!='wow64':
        raise ValueError('Runtime output must be a wow64 directory below artifacts/current')
    if destination.exists() and not (destination/'bootstrap.json').is_file():
        raise ValueError('Existing output is not a native runtime package')
    source=ROOT/'src/integration/compatibility/wow64'
    report=dict(schema='phoenix-wow64-runtime-build/1',passed=False,gpu_used=False,games_launched=False,
        game_presentation_validated=False,files={},sources={p.relative_to(ROOT).as_posix():digest(p) for p in source.iterdir() if p.is_file()})
    report['sources']['src/integration/reshade/nr_forwarder.cpp']=digest(ROOT/'src/integration/reshade/nr_forwarder.cpp')
    receipt=CURRENT/'wow64-runtime-build.json'
    try:
        with tempfile.TemporaryDirectory(prefix='wow64-runtime-',dir=ROOT/'artifacts') as temporary:
            build=Path(temporary)
            def compile(bits,command):
                recipe=build/'build.cmd'
                recipe.write_text('\n'.join(['@echo off',f'call "{studio}/VC/Auxiliary/Build/vcvars{bits}.bat" >nul 2>&1',
                    'if errorlevel 1 exit /b %errorlevel%',command,'exit /b %errorlevel%']))
                result=subprocess.run(['cmd','/d','/c',str(recipe)],cwd=build,capture_output=True,text=True,timeout=300)
                if result.returncode: raise RuntimeError((result.stdout+result.stderr)[-18000:])
            common=f'/nologo /std:c++20 /O2 /EHsc /MT /bigobj /DNOMINMAX /I"{args.ngx_sdk.resolve()}"'
            libs='kernel32.lib d3d12.lib dxgi.lib user32.lib gdi32.lib advapi32.lib bcrypt.lib'
            compile(64,f'cl /nologo /std:c++20 /O2 /Oi /GS- /GR- /LD "{source}/bootstrap64.cpp" /link /NODEFAULTLIB /ENTRY:DllMain /OUT:phoenix_bootstrap64.dll kernel32.lib')
            compile(64,f'cl {common} /LD "{source}/driver64.cpp" /Fe:phoenix_driver64.dll /link {libs}')
            compile(64,f'cl {common} /LD "{source}/model64.cpp" "{ROOT}/src/integration/reshade/nr_forwarder.cpp" /Fe:phoenix-nvngx.dll /link {libs}')
            compile(32,f'cl /nologo /O2 /Oi /GS- /GR- /LD /DWOW64EXT_EXPORTS /DUNICODE /D_UNICODE /DWIN32 "{extension}/src/wow64ext.cpp" /link /NODEFAULTLIB /ENTRY:DllMain /OUT:wow64ext.dll kernel32.lib')
            payload=build/'package';payload.mkdir()
            for name in ('phoenix_bootstrap64.dll','phoenix_driver64.dll','phoenix-nvngx.dll','wow64ext.dll'): shutil.copy2(build/name,payload/name)
            shutil.copy2(native,payload/'phoenix_native.dll')
            shutil.copytree(bundle,payload/'native')
            licenses=payload/'licenses';licenses.mkdir()
            for path in extension.iterdir():
                if path.is_file() and path.name.lower().startswith(('license','copying','lgpl')): shutil.copy2(path,licenses/path.name)
            (licenses/'WOW64Ext.txt').write_text('WOW64Ext by rewolf, LGPL-3.0. Dynamic unmodified dependency.\nSource: https://github.com/rwfpl/rewolf-wow64ext\nRevision: '+WOW64EXT_COMMIT+'\n',encoding='utf-8')
            profile=dict(schema='phoenix-wow64-bootstrap/3',ntdll_sha256=digest(ntdll),ntdll_timestamp=baseline['image_timestamp'],
                initializer_rva=symbols['initializer_rva'],allocator_rva=symbols['allocator_rva'],appcompat_offsets=symbols['appcompat_offsets'],
                driver_directory=driver.name,cuda_sha256=digest(driver/'nvcuda64.dll'),supported_models=args.supported_models,
                files={p.name:digest(p) for p in payload.glob('*.dll')})
            (payload/'bootstrap.json').write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8')
            (payload/'README.txt').write_text('Experimental native x86 addon transports, opt-in only.\n'
                'Select x64 library inside x86 or x86 Phoenix + native driver in Advanced transport.\n'
                'bootstrap.json is bound to the locally verified Windows image and native driver.\n'
                'A mismatch is rejected; default helper fallback stays enabled. No NVIDIA/model DLLs are bundled.\n'
                'Source: src/integration/compatibility/wow64; configuration: docs/EXECUTION_MECHANISMS.md.\n',encoding='utf-8')
            for path in payload.rglob('*'):
                if path.is_file() and (path.name.lower().startswith('nvngx') or path.suffix.lower() in ('.log','.bmp','.png')):
                    raise ValueError('Excluded model/log/capture payload')
            if destination.is_symlink() or destination.is_junction():
                raise ValueError('Unsafe runtime destination')
            if destination.exists(): shutil.rmtree(destination)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(payload),str(destination))
            report['files']={p.relative_to(destination).as_posix():digest(p) for p in destination.rglob('*') if p.is_file()}
            report['passed']=True
    finally:
        receipt.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,files=len(report['files']),gpu_used=False,destination=str(destination))))


if __name__=='__main__': main()
