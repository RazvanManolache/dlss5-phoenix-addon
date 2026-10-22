"""Check the new portable runtime against the frozen release's ordinary route."""
import _ctypes
import json
from pathlib import Path
import tempfile

import numpy as np
from compare_neural_controls import configure
from build_experimental_rtx_addon import extract_runtime
from test_portable_pipeline import ROOT,sha


def sequence(library,root,source,model,backend,device):
    api=configure(library);session=None
    config={'runtime_root':str(root),'backend':backend,'device':device,'allow_unvalidated':True,'cpu_threads':2,
        'vulkan_fused_feed_forward':False,
        'weight_source':{'kind':'nvidia_dll','path':str(source),'sha256':sha(source),'model':model}}
    try:
        w,h=32,24;session=api.phoenix_runtime_create(json.dumps(config).encode(),w,h)
        if not session:raise RuntimeError(api.phoenix_runtime_error().decode())
        yy,xx=np.mgrid[:h,:w]
        image=np.stack([.2+(xx%9)/32,.1+(yy%7)/16,.25+((xx+yy)%5)/16],axis=-1).astype(np.float16).astype(np.float32)
        vectors=np.zeros((h,w,2),np.float32);results=[]
        for frame in range(3):
            color=np.ascontiguousarray(np.roll(image,frame,axis=1));output=np.empty_like(color)
            metadata={'reset':frame!=1,'has_motion':True,'passes':2,'repeats':[1]*8,
                'controls':{'auto_mask':False,'seed':0,'sampling_mode':'fixed'}}
            if backend=='cpu':metadata.update(transformer_blocks=1,repeats=[1,0,0,0,0,0,0,0])
            code=api.phoenix_runtime_frame(session,color.ctypes.data,vectors.ctypes.data,None,None,json.dumps(metadata).encode(),output.ctypes.data)
            if code:raise RuntimeError(api.phoenix_runtime_error().decode())
            results.append(output.copy())
        return results
    finally:
        if session:api.phoenix_runtime_destroy(session)
        _ctypes.FreeLibrary(api._handle)


def main():
    output=ROOT/'artifacts/current/portable-pipeline';path=output/'validation.json'
    report=json.loads(path.read_text());library=output/'phoenix_runtime_cpp.dll'
    if not report.get('passed') or sha(library)!=report['runtime_sha256']:raise RuntimeError('Validate the current portable runtime first')
    addon=ROOT/'artifacts/current/addon/Phoenix.addon64'
    if sha(addon)!=report['source_addon_sha256']:raise RuntimeError('Frozen release differs from the tested backend payload')
    source=ROOT.parent/'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
    report['legacy_reference_equivalence']=[]
    try:
        with tempfile.TemporaryDirectory(prefix='pipeline-legacy-',dir=ROOT/'build') as temporary:
            work=Path(temporary);payload=work/'payload';payload.mkdir();scratch=work/'resources';scratch.mkdir()
            extract_runtime(addon,payload,scratch);root=payload/report['packaged_backend_root']
            reference=root/'phoenix_runtime_cpp.dll';report['legacy_reference_runtime_sha256']=sha(reference)
            seen=set()
            for case in report['cases']:
                key=(case['model'],case['backend'],case['device'])
                if key in seen:continue
                seen.add(key);model,backend,device=key
                baseline=sequence(reference,root,source,model,backend,device)
                candidate=sequence(library,root,source,model,backend,device)
                errors=[float(np.max(np.abs(a-b))) for a,b in zip(baseline,candidate)]
                exact=all(np.array_equal(a,b) for a,b in zip(baseline,candidate))
                result={'model':model,'backend':backend,'device':device,'frames':3,'reset_history_reset':True,
                    'blocks':1 if backend=='cpu' else 8,'passes':2,'pixels_exact':exact,'max_errors':errors}
                report['legacy_reference_equivalence'].append(result)
                if not exact:raise RuntimeError('Default portable output changed: '+str(result))
                print(model+' '+device+' legacy reset/history/reset pixels exact.',flush=True)
    except Exception as error:
        report.update(passed=False,legacy_error=str(error));raise
    finally:path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
