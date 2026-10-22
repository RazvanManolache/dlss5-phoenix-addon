"""Exercise the existing native portable CLI with the new metadata contract."""
import _ctypes
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from compare_neural_controls import configure
from build_experimental_rtx_addon import extract_runtime
from test_portable_pipeline import ROOT,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--addon',type=Path,help='Check the actual updated addon payload without replacing its runtime')
    args=parser.parse_args()
    pipeline=ROOT/'artifacts/current/portable-pipeline';evidence=json.loads((pipeline/'validation.json').read_text())
    library=pipeline/'phoenix_runtime_cpp.dll';addon=args.addon or ROOT/'artifacts/current/addon/Phoenix.addon64'
    if not evidence.get('passed') or not evidence.get('public_binary_audit_passed') or sha(library)!=evidence['runtime_sha256'] or (not args.addon and sha(addon)!=evidence['source_addon_sha256']):
        raise RuntimeError('Validate the current audited portable runtime first')
    source=ROOT.parent/'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
    report={'passed':False,'runtime_sha256':sha(library),'cases':[],'python_required_at_runtime':False}
    if args.addon:report['addon_sha256']=sha(addon)
    try:
        with tempfile.TemporaryDirectory(prefix='pipeline-cli-',dir=ROOT/'build') as temporary:
            work=Path(temporary)
            payload=work/'payload';payload.mkdir();scratch=work/'resources';scratch.mkdir();extract_runtime(addon,payload,scratch)
            root=payload/evidence['packaged_backend_root']
            if args.addon:
                if sha(root/library.name)!=evidence['runtime_sha256']:raise RuntimeError('Addon contains another portable runtime')
            else:shutil.copy2(library,root/library.name)
            api=configure(root/library.name)
            try:
                yy,xx=np.mgrid[:24,:32]
                color=np.stack([.2+(xx%9)/32,.1+(yy%7)/16,.25+((xx+yy)%5)/16],axis=-1).astype(np.float16).astype(np.float32)
                source_pixels=work/'source.rgb32f';color.tofile(source_pixels);seen=set()
                for case in evidence['cases']:
                    key=(case['model'],case['backend'],case['device'])
                    if key in seen:continue
                    seen.add(key);model,backend,device=key
                    config={'runtime_root':str(root),'backend':backend,'device':device,'execution':'in_process','allow_unvalidated':True,'cpu_threads':2,
                        'weight_source':{'kind':'nvidia_dll','path':str(source),'sha256':sha(source),'model':model}}
                    metadata={'reset':True,'has_motion':False,'passes':2,'repeats':[1]*8,
                        'controls':{'auto_mask':False,'seed':0,'sampling_mode':'fixed'},
                        'pipeline':{'ProcessingChain':True,'ChainFusion':True,'MultiscaleFusion':True,'FusionLayers':2,'ChainBrightness':.02}}
                    if backend=='cpu':metadata.update(transformer_blocks=1,repeats=[1,0,0,0,0,0,0,0])
                    session=api.phoenix_runtime_create(json.dumps(config).encode(),32,24)
                    if not session:raise RuntimeError(api.phoenix_runtime_error().decode())
                    try:
                        expected=np.empty_like(color)
                        if api.phoenix_runtime_frame(session,color.ctypes.data,None,None,None,json.dumps(metadata).encode(),expected.ctypes.data):
                            raise RuntimeError(api.phoenix_runtime_error().decode())
                    finally:api.phoenix_runtime_destroy(session)
                    job=work/'job.json';job.write_text(json.dumps({'width':32,'height':24,'config':config,'color':str(source_pixels),'metadata':metadata}))
                    target=work/f'{len(seen)}.rgb32f'
                    command=[str(root/'phoenix_frame.exe'),str(job),str(target)]
                    done=subprocess.run(command,capture_output=True,text=True,timeout=180)
                    if done.returncode:raise RuntimeError(done.stderr[-1500:])
                    actual=np.fromfile(target,dtype=np.float32).reshape(color.shape)
                    if not np.array_equal(actual,expected):raise RuntimeError('Native CLI pixels differ from its C API')
                    status=json.loads(done.stdout)
                    if status['independent_temporal_stages']!=4:raise RuntimeError('CLI ignored chain or pass controls')
                    old=target.read_bytes();refused=subprocess.run(command,capture_output=True,text=True,timeout=30)
                    if refused.returncode==0 or target.read_bytes()!=old:raise RuntimeError('CLI overwrote existing pixels')
                    report['cases'].append({'model':model,'backend':backend,'device':device,'pixels_exact':True,
                        'output_preserved':True,'independent_temporal_stages':4,'output_sha256':sha(target)})
                    print(model+' '+device+' native CLI pixels exact.',flush=True)
            finally:_ctypes.FreeLibrary(api._handle)
        report['passed']=True
    finally:(pipeline/('cli-package-validation.json' if args.addon else 'cli-validation.json')).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
