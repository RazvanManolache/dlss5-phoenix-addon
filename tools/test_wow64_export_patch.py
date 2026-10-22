"""Compile/run the CPU-only native-worker restart regression in temporary staging."""
import argparse
import subprocess
import tempfile
from pathlib import Path
from test_model_selection import ROOT, toolchain, run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-ref',
        help='Also confirm a specified pre-fix revision faults on its second install')
    parser.add_argument('--old-layout-ref',
        help='Confirm a specified revision rejects the high-image layout regression')
    args=parser.parse_args()
    compiler,environment=toolchain(64)
    source=ROOT/'tests/native/wow64_export_patch_tests.cpp'
    headers=ROOT/'src/integration/compatibility/wow64'
    with tempfile.TemporaryDirectory(prefix='wow64-export-test-',dir=ROOT/'build') as temporary:
        work=Path(temporary)
        def compile(name,includes):
            executable=work/(name+'.exe')
            run([compiler,'/nologo','/std:c++20','/O2','/EHsc','/DNOMINMAX',
                *('/I'+str(path) for path in includes),source,'/Fe:'+str(executable),
                '/Fo:'+str(work/(name+'.obj')),'kernel32.lib','user32.lib','gdi32.lib'],
                cwd=work,env=environment)
            return executable
        fixed=compile('fixed',[headers])
        print(run([fixed],cwd=work,env=environment),end='')
        if args.old_ref:
            old=work/'old';old.mkdir()
            original=subprocess.check_output(['git','show',
                args.old_ref+':src/integration/compatibility/wow64/umd_bridge.h'],cwd=ROOT)
            (old/'umd_bridge.h').write_bytes(original)
            executable=compile('old',[old,headers])
            result=subprocess.run([str(executable)],cwd=work,env=environment,
                capture_output=True,text=True,timeout=30)
            if result.returncode & 0xffffffff != 0xc0000005:
                raise RuntimeError('Old implementation did not reproduce the expected RX-write fault: '+str(result.returncode))
            print('Confirmed: pre-fix implementation faults with 0xc0000005 on restart')
        if args.old_layout_ref:
            old=work/'old-layout';old.mkdir()
            (old/'umd_bridge.h').write_bytes(subprocess.check_output(['git','show',
                args.old_layout_ref+':src/integration/compatibility/wow64/umd_bridge.h'],cwd=ROOT))
            executable=compile('old-layout',[old,headers])
            result=subprocess.run([str(executable)],cwd=work,env=environment,
                capture_output=True,text=True,timeout=30)
            if result.returncode!=1 or 'High-image export redirection failed' not in result.stderr:
                raise RuntimeError('Old implementation did not reproduce the high-image allocation failure')
            print('Confirmed: pre-fix implementation refuses a valid high-image export layout')


if __name__=='__main__':main()
