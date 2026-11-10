"""Explicit GPU regression: matched NVIDIA/Phoenix appearance on frozen pixels.

Uses a supplied image/model, all three styles and below/above-one intensity.
Temporary builds and raw captures are removed; no game or installed file edits.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from PIL import Image
from compare_four_models import ROOT, build, sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--crop',type=int,nargs=4)
    p.add_argument('--model-dll',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    image=Image.open(args.input).convert('RGB')
    if args.crop:image=image.crop(tuple(args.crop))
    width,height=image.size
    core=list(Path('C:/Windows/System32/DriverStore/FileRepository').glob('nv_*.inf_amd64_*/_nvngx.dll'))
    if len(core)!=1:raise RuntimeError('Need an unambiguous installed NGX core')
    source=dict(kind='nvidia_dll',path=str(args.model_dll.resolve()),sha256=sha(args.model_dll),model='original')
    report=dict(passed=False,gpu_executed=True,games_launched=False,installed_files_modified=False,
                input_sha256=sha(args.input),crop=args.crop,resolution=[width,height],
                model_dll_sha256=source['sha256'],cases=[],failures=[],
                tolerance=dict(max_abs=1/1024,mae=1/8192),
                scope='Fixed SDR proxy, zero motion, full graphs, reset and history; not live presentation or performance')
    result=out/'validation.json';write(result,report)
    try:
        with tempfile.TemporaryDirectory(prefix='appearance_validation_',dir=ROOT/'build') as temporary:
            work=Path(temporary);probe,forwarder=build(work)
            report.update(probe_sha256=sha(probe),forwarder_sha256=sha(forwarder))
            color=np.ones((height,width,4),np.float16);color[:,:,:3]=np.asarray(image,np.float32)/255
            (work/'color.rgba16f').write_bytes(color.tobytes())
            (work/'motion.rg16f').write_bytes(np.zeros((height,width,2),np.float16).tobytes())
            base=dict(width=width,height=height,weight_source=source,bundle_path=str(args.bundle.resolve()),
                      ngx_core=str(core[0]),forwarder=str(forwarder),cache=str(work/'ngx-cache'),
                      color_file=str(work/'color.rgba16f'),motion_file=str(work/'motion.rg16f'),
                      capture=True,debug=True,warmup_seconds=.01,measure_seconds=.01,minimum_measured_frames=1)
            for model in ('original','small-fp8'):
                for style in (0,1,2):
                    for intensity in (0.,.5,1.,2.):
                        controls=dict(intensity=intensity,style=style,tone=2.,structure=2.,skin=-1.,auto_mask=False)
                        outputs={};receipts={}
                        for engine in ('nvidia','phoenix'):
                            label=f'{model}-{style}-{intensity:g}-{engine}'
                            job=dict(base,engine=engine,controls=controls,weight_source=dict(source,model=model))
                            write(work/'job.json',job);dest=work/(label+'.json')
                            proc=subprocess.run([str(probe),str(work/'job.json'),str(dest)],cwd=work,capture_output=True,text=True,timeout=120)
                            data=json.loads(dest.read_text()) if dest.exists() else {}
                            if proc.returncode or not data.get('passed'):raise RuntimeError(label+': '+str(data.get('error',proc.stderr)))
                            if data['effective_controls']!={**controls,'inverted_depth':False}:raise RuntimeError('Effective control mismatch')
                            receipts[engine]=data
                            outputs[engine]={}
                            for frame in ('reset_a','history'):
                                path=work/data['captures'][frame]['file']
                                outputs[engine][frame]=np.fromfile(path,np.float16).reshape(height,width,4)[:,:,:3].astype(np.float32)
                                path.unlink()
                            dest.unlink()
                        differences={}
                        for frame in ('reset_a','history'):
                            delta=np.abs(outputs['phoenix'][frame]-outputs['nvidia'][frame])
                            differences[frame]=dict(mae=float(delta.mean()),max_abs=float(delta.max()))
                            if not np.isfinite(delta).all() or delta.max()>1/1024 or delta.mean()>1/8192:
                                report['failures'].append(dict(model=model,style=style,intensity=intensity,
                                                               frame=frame,difference=differences[frame]))
                        report['cases'].append(dict(model=model,controls=controls,differences=differences,
                            hashes={engine:{frame:row['captures'][frame]['sha256'] for frame in ('reset_a','history','reset_b')}
                                    for engine,row in receipts.items()}))
                        if style==2 and intensity==2:
                            for engine in outputs:
                                rgb=np.uint8(np.clip(outputs[engine]['reset_a'],0,1)*255+.5)
                                Image.fromarray(rgb).save(out/f'{model}-{engine}.png')
                        write(result,report)
                        passed=not any(x['model']==model and x['style']==style and x['intensity']==intensity for x in report['failures'])
                        print(model,style,intensity,'PASS' if passed else 'FAIL',differences,flush=True)
            report['passed']=not report['failures']
            if not report['passed']:raise RuntimeError('Appearance comparisons exceeded the FP16 rounding tolerance')
    finally:
        report['temporary_builds_removed']=True;write(result,report)


if __name__=='__main__':main()
