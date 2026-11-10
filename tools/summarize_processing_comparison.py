"""Aggregate source-bound common-boundary runs without mixing timing scopes."""
import argparse,hashlib,json,statistics
from pathlib import Path

ORDER=['nvidia','native_rtx','native-cuda','cuda','vulkan_nvidia','cpu32','vulkan_amd']
SHORT={'vulkan_amd':'Vulkan / AMD iGPU'}
MODELS=['original','small-fp8']
def aggregate(report,controls=None):
    rows=[]
    for route in ORDER:
        cases=[c for c in report['cases'] if c['route']==route]
        control_row=False
        if not cases and controls:
            assert route in ['nvidia','native_rtx','native-cuda'],'Changed engines require current measurements'
            cases=[c for c in controls['cases'] if c['route']==route];control_row=True
        assert cases and all(c['passed'] for c in cases),route+' incomplete'
        row=dict(id=route,label=cases[0]['label'],short=SHORT.get(route,cases[0]['label']))
        row['measurement_source']='unchanged engine control run' if control_row else 'current run'
        for model in MODELS:
            cells=[c for c in cases if c['model']==model]
            assert len(cells)==report['rounds'] and len({c['output_sha256'] for c in cells})==1
            medians=[c['wall']['median_ms'] for c in cells]
            row[model]=dict(median_ms=statistics.median(medians),min_ms=min(medians),max_ms=max(medians),
                trial_ms=medians,measured_frames=sum(len(c['samples_ms']) for c in cells),output_sha256=cells[0]['output_sha256'])
        rows.append(row)
    for route,reason in report['unavailable'].items():
        rows.append(dict(id=route,label={'hip':'HIP / AMD','sycl':'SYCL','metal':'Metal'}[route],original=None,**{'small-fp8':None},reason=reason))
    return dict(rows=rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['before','after','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--controls',type=Path,help='Earlier same-session common-boundary measurements of unchanged control engines')
    a=p.parse_args();before=json.loads(a.before.read_text());after=json.loads(a.after.read_text())
    assert before['passed'] and after['passed'] and before['state']==after['state']=='complete'
    for key in ['width','height','input_rgba16f_sha256','model_sha256','timing_scope']:assert before[key]==after[key],key+' changed'
    controls=json.loads(a.controls.read_text()) if a.controls else None
    if controls:
        assert controls['passed'] and controls['state']=='complete'
        for key in ['width','height','input_rgba16f_sha256','model_sha256','timing_scope','rounds']:assert after[key]==controls[key],key+' differs in controls'
    first,last=aggregate(before),aggregate(after,controls);changes=[]
    for old,new in zip(first['rows'],last['rows']):
        assert old['id']==new['id']
        if old.get('original') is None:continue
        for model in MODELS:
            assert old[model]['output_sha256']==new[model]['output_sha256'],'Output changed: '+old['id']+'/'+model
            b,c=old[model]['median_ms'],new[model]['median_ms']
            changes.append(dict(route=old['id'],model=model,before_ms=b,after_ms=c,speedup=b/c,percent_reduction=100*(1-c/b)))
    result=dict(schema='phoenix-processing-before-after/1',passed=True,before=first,after=last,changes=changes,
        timing_scope=after['timing_scope'],receipts={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.before,a.after]+([a.controls] if a.controls else [])},
        measured_frames=sum(len(c['samples_ms']) for c in after['cases']),cases=len(after['cases']),
        notes=['Medians of fresh process medians; min/max are descriptive trial ranges, not confidence intervals.',
               'Dynamic GPU/CPU clocks and other desktop work are unmanaged. Control routes were remeasured.',
               'Native/CUDA/Vulkan/CPU selectors retain their own neural executors. No reduced blocks/passes or approximate arithmetic.'])
    if controls:result['notes'].append('Unchanged Native RTX, NVIDIA and Native CUDA controls were remeasured earlier in this session, with the same boundary, inputs and samples. Their separate source/binary receipt is retained.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'cases':result['cases'],'measured_frames':result['measured_frames'],'changes':[c for c in changes if c['route'] in ['cuda','vulkan_nvidia','cpu32','vulkan_amd']]},indent=2))

if __name__=='__main__':main()
