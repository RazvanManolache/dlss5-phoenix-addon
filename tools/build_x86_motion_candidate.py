"""Build the shared SDK and isolated x86 motion probe; GPU checks are separate."""
import hashlib
import json
from pathlib import Path
from test_model_selection import ROOT, toolchain, run


def main():
    work=ROOT/'build/x86_motion_candidate'
    work.mkdir(parents=True,exist_ok=True)
    sdk=ROOT.parent/'external/Streamline/external/ngx-sdk/include'
    libs=['d3d11.lib','d3d12.lib','dxgi.lib','d3dcompiler.lib','bcrypt.lib','user32.lib','gdi32.lib',
          'advapi32.lib','opengl32.lib','ole32.lib','cabinet.lib','shell32.lib','uuid.lib','windowscodecs.lib']
    artifacts={}
    for bits in (64,32):
        compiler,env=toolchain(bits)
        flags=[compiler,'/nologo','/std:c++20','/Zc:__cplusplus','/O2','/EHsc','/MT','/bigobj','/DNOMINMAX',
               '/I'+str(sdk),'/I'+str(ROOT/'third_party/nvapi')]
        if bits==64:
            target=work/'phoenix_native.dll'
            run(flags+['/LD',ROOT/'native_runtime/native_api.cpp','/Fo:'+str(work/'native_api.obj'),
                       '/Fe:'+str(target),*libs],cwd=work,env=env)
        else:
            target=work/'wow64_motion_tests.exe'
            run(flags+[ROOT/'tests/native/wow64_motion_tests.cpp','/Fo:'+str(work/'motion.obj'),
                       '/Fe:'+str(target),*libs,'/link','/LARGEADDRESSAWARE',
                       '/SUBSYSTEM:WINDOWS','/ENTRY:wmainCRTStartup'],cwd=work,env=env)
        artifacts[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    (work/'build.json').write_text(json.dumps({'passed':True,'gpu_used':False,'artifacts':artifacts},indent=2))
    print(json.dumps({'passed':True,'gpu_used':False,'artifacts':artifacts}))


if __name__=='__main__':main()
