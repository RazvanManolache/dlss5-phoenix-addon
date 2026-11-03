"""CPU queue regression and model-free software D3D12 output-fence validation."""
import argparse,json,pathlib,tempfile
from test_model_selection import ROOT,toolchain,run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-dx9',action='store_true',help='Also exercise isolated real-DX9 snapshot/comparison textures')
    parser.add_argument('--gpu-motion',action='store_true',help='Also test actual GPU optical-flow ordering through the native SDK')
    args=parser.parse_args()
    results=[]
    with tempfile.TemporaryDirectory(prefix='route_optimization_',dir=ROOT/'build') as folder:
        work=pathlib.Path(folder)
        for bits in (32,64):
            compiler,env=toolchain(bits);exe=work/f'capture_queue_{bits}.exe'
            run([compiler,'/nologo','/std:c++20','/O2','/EHsc','/MT',
                 ROOT/'tests/native/capture_queue_tests.cpp','/Fo:'+str(work/f'queue_{bits}.obj'),
                 '/Fe:'+str(exe)],cwd=work,env=env)
            results.append({'test':'capture_queue','bits':bits,'result':json.loads(run([exe],cwd=work,env=env))})
        compiler,env=toolchain(64)
        sdk=ROOT.parent/'external/Streamline/external/ngx-sdk/include'
        flags=[compiler,'/nologo','/std:c++20','/Zc:__cplusplus','/O2','/EHsc','/MT','/bigobj','/DNOMINMAX',
               '/I'+str(sdk),'/I'+str(ROOT/'native_runtime')]
        libraries=['d3d11.lib','d3d12.lib','dxgi.lib','d3dcompiler.lib','bcrypt.lib','user32.lib','gdi32.lib',
                   'advapi32.lib','opengl32.lib','ole32.lib','cabinet.lib','shell32.lib','uuid.lib','windowscodecs.lib']
        run(flags+['/LD',ROOT/'native_runtime/native_api.cpp','/Fo:'+str(work/'native.obj'),
                   '/Fe:'+str(work/'phoenix_native.dll'),*libraries],cwd=work,env=env,timeout=300)
        exe=work/'native_library_tests.exe'
        run(flags+[ROOT/'tests/native/native_library_tests.cpp','/Fo:'+str(work/'api_test.obj'),
                   '/Fe:'+str(exe),work/'phoenix_native.lib',*libraries],cwd=work,env=env,timeout=180)
        results.append({'test':'native_api_and_postprocess','software_d3d12':True,
                        'result':json.loads(run([exe],cwd=work,env=env,timeout=90))})
        if args.gpu_motion:
            exe=work/'pipeline_native_tests.exe'
            run(flags+[ROOT/'tests/native/pipeline_native_tests.cpp','/Fo:'+str(work/'motion_test.obj'),
                       '/Fe:'+str(exe),work/'phoenix_native.lib',*libraries],cwd=work,env=env,timeout=180)
            results.append({'test':'native_gpu_motion_ordering','result':json.loads(run([exe,'--gpu'],cwd=work,env=env,timeout=90))})
        if args.gpu_dx9:
            for bits in (32,64):
                compiler,env=toolchain(bits);exe=work/f'd3d9_capture_queue_{bits}.exe'
                dx9_flags=[compiler,*flags[1:],'/I'+str(ROOT/'third_party/nvapi')]
                run(dx9_flags+[ROOT/'tests/native/d3d9_capture_queue_tests.cpp','/Fo:'+str(work/f'dx9_test_{bits}.obj'),
                               '/Fe:'+str(exe),*libraries,'d3d9.lib'],cwd=work,env=env,timeout=180)
                results.append({'test':'dx9_snapshot_pairing','bits':bits,
                                'result':json.loads(run([exe],cwd=work,env=env,timeout=90))})
    report={'passed':True,'models_run':False,'games_run':False,'results':results}
    (ROOT/'artifacts/current/route-optimization-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
