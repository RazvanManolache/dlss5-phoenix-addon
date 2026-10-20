"""Focused processing/persistence/math and HLSL compilation checks; no GPU device."""
import argparse, hashlib, json, subprocess
from pathlib import Path
from test_model_selection import ROOT, toolchain, run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    work=ROOT/'build/four_model_comparison'
    sdk=ROOT.parent/'external/DLSS/include'
    reports=[]
    for bits in (32,64):
        compiler,env=toolchain(bits)
        binary=work/f'processing_tests{bits}.exe'
        output=run([compiler,'/nologo','/std:c++20','/Zc:__cplusplus','/EHsc','/O2','/MT','/DNOMINMAX',
            '/I'+str(sdk),'/I'+str(ROOT/'third_party/nvapi'),
            ROOT/'tests/native/processing_tests.cpp','/Fo:'+str(work/f'processing_tests{bits}.obj'),
            '/Fe:'+str(binary),'/link','d3d12.lib','dxgi.lib','d3dcompiler.lib','bcrypt.lib','cabinet.lib','shell32.lib','ole32.lib','uuid.lib'],cwd=work,env=env)
        result=json.loads(run([binary],cwd=work,env=env))
        reports.append({'bits':bits,**result})
    report={'passed':all(r['passed'] for r in reports),'cases':reports,
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [ROOT/'src/native/support/processing.h',ROOT/'src/native/support/processing_shader.h',
                 ROOT/'src/native/support/routing.h',ROOT/'src/native/support/resolution_bridge.h',
                 ROOT/'src/native/support/production_direct.h',ROOT/'src/native/support/compute_pass.h',
                 ROOT/'src/native/support/present_guides.h',ROOT/'src/integration/compatibility/session_policy.h',
                 ROOT/'src/integration/compatibility/processing_pixels.h',ROOT/'tests/native/processing_tests.cpp']}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
