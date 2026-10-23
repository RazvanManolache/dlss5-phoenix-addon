"""Real companion DLL tests under deterministic Windows file-sharing delays."""
import argparse
import ctypes as C
from ctypes import wintypes
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from test_model_selection import ROOT, toolchain, run


def read(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}


def wait(path, predicate, seconds=4):
    deadline=time.monotonic()+seconds
    value={}
    while time.monotonic()<deadline:
        value=read(path)
        if predicate(value):return value
        time.sleep(.02)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    kernel=C.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,C.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    def lock(path,sharing):
        handle=kernel.CreateFileW(str(path),0x80000000,sharing,None,3,0x80,None)
        if handle in (None,C.c_void_p(-1).value):raise C.WinError(C.get_last_error())
        return handle
    results=dict(passed=False,pending=True,gpu_used=False,desktop_input_sent=False,cases={})
    try:
        with tempfile.TemporaryDirectory(prefix='input-mailbox-',dir=ROOT/'build') as temporary:
            for bits in (32,64):
                work=Path(temporary)/str(bits);work.mkdir();compiler,env=toolchain(bits)
                target=work/'host.exe'
                run([compiler,'/nologo','/std:c++20','/EHsc','/O2','/MT',ROOT/'tests/native/input_addon_load_tests.cpp',
                    '/Fo:'+str(work/'host.obj'),'/Fe:'+str(target),'user32.lib'],cwd=work,env=env)
                binary=args.input_directory/f'PhoenixInput.addon{bits}';addon=work/binary.name;shutil.copy2(binary,addon)
                process=subprocess.Popen([str(target),str(addon)],cwd=work,env=env,stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                result=dict(passed=False,checks={},addon_sha256=hashlib.sha256(addon.read_bytes()).hexdigest())
                try:
                    if process.stdout.readline().strip()!='ready':raise RuntimeError('Companion host did not start')
                    mailbox=work/'PhoenixOutput/input'
                    endpoint=wait(mailbox/'endpoint.json',lambda v:bool(v.get('session')))
                    if not endpoint:raise RuntimeError('No input endpoint')
                    def request(identity,ops):
                        return dict(schema='phoenix-input/1',session=endpoint['session'],id=identity,ops=ops)
                    def submit(value):
                        staged=mailbox/'request.tmp';staged.write_text(json.dumps(value),encoding='utf-8');os.replace(staged,mailbox/'request.json')
                    # Allow rename, deny a second reader. The worker can claim
                    # this file but must retry its transiently denied read.
                    identity='locked-read';staged=mailbox/'locked.tmp'
                    staged.write_text(json.dumps(request(identity,[dict(op='status')])),encoding='utf-8')
                    handle=lock(staged,4)
                    try:
                        os.replace(staged,mailbox/'request.json');time.sleep(.35)
                    finally:kernel.CloseHandle(handle)
                    reply=wait(mailbox/'response.json',lambda v:v.get('id')==identity)
                    result['checks']['claimed_request_read_retry']=reply.get('id')==identity and reply.get('ok') is True
                    result['read_reply']=reply
                    # Retain the old response as a reader without delete share.
                    # A new reply must survive replacement failure exactly once.
                    if not (mailbox/'response.json').exists():
                        (mailbox/'response.json').write_text('{}')
                    handle=lock(mailbox/'response.json',1)
                    try:
                        submit(request('locked-publish',[dict(op='status')]))
                        time.sleep(.35)
                    finally:kernel.CloseHandle(handle)
                    reply=wait(mailbox/'response.json',lambda v:v.get('id')=='locked-publish')
                    result['checks']['reply_publication_retry']=reply.get('id')=='locked-publish' and reply.get('ok') is True
                    result['publication_reply']=reply
                    # The valid prefix is future work in this windowless host;
                    # rejecting the later invalid operation must discard it.
                    submit(request('atomic-invalid',[dict(op='key',key='k',action='tap',at_ms=60000),dict(op='unknown')]))
                    reply=wait(mailbox/'response.json',lambda v:v.get('id')=='atomic-invalid')
                    result['checks']['invalid_batch_schedules_nothing']=(reply.get('id')=='atomic-invalid' and
                        reply.get('ok') is False and reply.get('status',{}).get('pending_actions')==0)
                    result['invalid_reply']=reply
                    output,error=process.communicate('stop\n',timeout=20)
                    result['exit']=process.returncode
                    result['passed']=process.returncode==0 and all(result['checks'].values())
                    if error:result['stderr']=error[-1000:]
                finally:
                    if process.poll() is None:process.kill();process.wait()
                results['cases'][str(bits)]=result
                (args.output/'report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
                print(json.dumps(dict(bits=bits,passed=result['passed'],checks=result['checks'])),flush=True)
        results['passed']=all(v['passed'] for v in results['cases'].values()) and len(results['cases'])==2
    finally:
        results['pending']=False
        (args.output/'report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return 0 if results['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
