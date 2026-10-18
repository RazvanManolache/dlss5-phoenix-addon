"""Summarize measured resolution results; never extrapolate failed cases."""
import csv
import json
import math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"artifacts/current/neural-performance"
def read(p):return json.loads(p.read_text())
def median(values):
    s=sorted(values);n=len(s);return (s[n//2]+s[(n-1)//2])/2
def load_case(meta):
    if not meta or not meta.get('passed'):return None
    return read(DEST/meta['file'])
def n(value):return f'{value:.2f}' if isinstance(value,(float,int)) else '—'
def main():
    run=read(DEST/"run.json");native=[];portable=[];errors=[]
    checks=read(DEST/'rechecks.json') if (DEST/'rechecks.json').exists() else {}
    if not checks.get('completed'):checks={}
    for w,h in run['resolutions']:
        key=f'{w}x{h}';meta=run['native'].get(key,{})
        before=load_case(meta.get('before'));after=load_case(meta.get('after'));candidate=load_case(meta.get('candidate'))
        row=dict(resolution=key,width=w,height=h,megapixels=w*h/1e6)
        if before and after and candidate:
            p={r['profile']:r for r in candidate['profiles']}
            old=[before['profiles'][0],after['profiles'][0]]
            initial_default=p['default'];initial_combined=p['combined'];initial_old=old
            replacements=checks.get('cases',{}).get(key,{})
            re_before=load_case(replacements.get('before'));re_after=load_case(replacements.get('after'));re_candidate=load_case(replacements.get('candidate'))
            replaced_profiles=[]
            if re_before and re_after and re_candidate:
                old=[re_before['profiles'][0],re_after['profiles'][0]]
                replaced_profiles=[r['profile'] for r in re_candidate['profiles']]
                p.update({r['profile']:r for r in re_candidate['profiles']})
            row.update(timing_source='5 s warmup / 5 s measurement recheck' if replaced_profiles else 'original sweep',
                initial_new_gpu_ms=initial_default['gpu']['median_ms'],initial_new_wall_ms=initial_default['wall']['median_ms'],
                initial_combined_gpu_ms=initial_combined['gpu']['median_ms'],initial_combined_wall_ms=initial_combined['wall']['median_ms'])
            row.update(passed=True,old_gpu_ms=median(sum([r['gpu_samples_ms'] for r in old],[])),
                old_wall_ms=median(sum([r['wall_samples_ms'] for r in old],[])),
                new_gpu_ms=p['default']['gpu']['median_ms'],new_wall_ms=p['default']['wall']['median_ms'],
                combined_gpu_ms=p['combined']['gpu']['median_ms'],combined_wall_ms=p['combined']['wall']['median_ms'],
                new_gpu_p95_ms=p['default']['gpu']['p95_ms'],new_wall_p95_ms=p['default']['wall']['p95_ms'],
                combined_gpu_p95_ms=p['combined']['gpu']['p95_ms'],combined_wall_p95_ms=p['combined']['wall']['p95_ms'],
                default_after_features_gpu_ms=p['default_after_features']['gpu']['median_ms'],
                default_after_features_wall_ms=p['default_after_features']['wall']['median_ms'],
                native_compatibility_exact=meta.get('legacy_reset_output_exact',False),
                baseline_repeat_exact=old[0]['reset_output_sha256']==old[1]['reset_output_sha256'],
                candidate_return_exact=p['default']['reset_output_sha256']==p['default_after_features']['reset_output_sha256'],
                candidate_matches_a_legacy_run=p['default']['reset_output_sha256'] in [q['reset_output_sha256'] for q in old],
                baseline_gpu_drift_percent=(old[1]['gpu']['median_ms']/old[0]['gpu']['median_ms']-1)*100,
                baseline_wall_drift_percent=(old[1]['wall']['median_ms']/old[0]['wall']['median_ms']-1)*100,
                new_internal_live_mib=p['default']['runtime_status']['owned_memory']['totals']['live_allocation_info_bytes']/2**20,
                old_internal_live_mib=old[0]['runtime_status']['owned_memory']['totals']['live_allocation_info_bytes']/2**20,
                combined_internal_live_mib=p['combined']['runtime_status']['owned_memory']['totals']['live_allocation_info_bytes']/2**20,
                combined_internal_peak_mib=p['combined']['runtime_status']['owned_memory']['totals']['peak_allocation_info_bytes']/2**20,
                default_after_features_internal_live_mib=p['default_after_features']['runtime_status']['owned_memory']['totals']['live_allocation_info_bytes']/2**20,
                new_samples=p['default']['wall']['samples'],combined_samples=p['combined']['wall']['samples'])
            row['default_gpu_change_percent']=(row['new_gpu_ms']/row['old_gpu_ms']-1)*100
            row['default_wall_change_percent']=(row['new_wall_ms']/row['old_wall_ms']-1)*100
            row['combined_gpu_added_ms']=row['combined_gpu_ms']-row['new_gpu_ms']
            row['combined_wall_added_ms']=row['combined_wall_ms']-row['new_wall_ms']
            row['timing_variable']=any(abs(row[k]/row['initial_'+k]-1)>.10 for k in ['new_gpu_ms','new_wall_ms','combined_gpu_ms','combined_wall_ms']) or abs((initial_old[1]['gpu']['median_ms']/initial_old[0]['gpu']['median_ms']-1)*100)>10
            row['profiles']={name:dict(gpu_ms=v['gpu']['median_ms'],wall_ms=v['wall']['median_ms'],gpu_p95_ms=v['gpu']['p95_ms'],
                wall_p95_ms=v['wall']['p95_ms'],first_call_ms=v['first_call_ms'],samples=v['wall']['samples'],
                timing_source='recheck' if name in replaced_profiles else 'original sweep') for name,v in p.items()}
        else:
            row['passed']=False
            row['errors']={name:meta.get(name,{}).get('error','not completed') for name in ['before','candidate','after'] if not meta.get(name,{}).get('passed')}
            errors.append(dict(route='native',resolution=key,errors=row['errors']))
        native.append(row)
        for route,details in run['portable'].get(key,{}).items():
            if route.startswith('gpu_'):continue
            case=load_case(details)
            if not case:
                portable.append(dict(resolution=key,route=route,passed=False,error=details.get('error')))
                errors.append(dict(route=route,resolution=key,error=details.get('error')));continue
            rechecked=checks.get('portable',{}).get(route) if key=='5120x2160' else None
            replacement=load_case(rechecked)
            if replacement:case=replacement
            portable.append(dict(resolution=key,route=route,passed=True,timing_source='5 s warmup / 5 s measurement recheck' if replacement else 'original sweep',
                gpu_ms=case['gpu']['median_ms'],
                wall_ms=case['wall']['median_ms'],gpu_p95_ms=case['gpu']['p95_ms'],wall_p95_ms=case['wall']['p95_ms'],
                samples=case['wall']['samples'],low_sample_count=case['wall']['samples']<12,first_call_ms=case['first_call_ms'],
                backend_resident_mib=case['runtime_status']['backend_allocation_bytes']/2**20,
                host_buffers_sampled_mib=case['runtime_status'].get('host_frame_buffers',{}).get('temporary_capacity_sampled_high_water_bytes',0)/2**20,
                stages_mean_ms=case['stages_mean_ms']))
    summary=dict(schema='phoenix-neural-performance-summary/1',gpu=run['gpu'],native=native,portable=portable,errors=errors,
        gpu_scope_native='model + output/presentation; excludes input canonicalization/preparation',
        gpu_scope_portable='neural backend operator events; excludes host processing',
        wall_scope_native='synchronous GPU-resident API including input preparation; no host image transfers',
        wall_scope_portable='host-plane API including transfers and host boundary/presentation',
        all_graph_blocks=True,passes=1,game_fps=False,
        requested_resolution_count=len(run['resolutions']),native_completed=sum(r['passed'] for r in native),
        portable_completed=sum(r['passed'] for r in portable),
        limitations=['One serial sweep, 0.75 s warmup and >=2 s measured target, >=12 samples unless 15 s cap reached.',
            'Flagged timing repeats use 5 s warmup and >=5 s measured; selected recheck timings are labelled in the CSV.',
            'Long portable cases with fewer than 12 samples are short-run estimates; their p95 is not a robust percentile.',
            'Synthetic structured SDR image, zero motion with active temporal history; input=inference=output size.',
            'Native and portable use different structured patterns and sampling sequences; before/after comparisons match within each route. Cross-route figures are not an exact-pixel matched benchmark.',
            'Per-profile session state accumulates lazy native allocations; default_after_features measures the return path.',
            'Native allocation ledger excludes application input/output textures and driver caches; portable resident bytes exclude host buffers.',
            'GPU scopes differ by route; use complete call for route-specific cost and do not call inverse GPU time game FPS.',
            'No games or NVIDIA model DLL execution; standard GPU drivers/NVAPI were used. Finite output and native baseline reset hashes checked.',
            'A reset hash difference is a correctness flag; its pixel-error magnitude is not measured by this timing harness.'])
    if (DEST/'repeatability.json').exists():summary['repeatability']=read(DEST/'repeatability.json')
    summary['portable_feature_cost']={backend:read(DEST/f'portable-feature-cost-{backend}.json') for backend in ['cuda','vulkan'] if (DEST/f'portable-feature-cost-{backend}.json').exists()}
    summary['low_sample_cases']=[{'resolution':r['resolution'],'route':r['route'],'samples':r['samples']} for r in portable if r.get('low_sample_count')]
    summary['correctness_flags']=[r['resolution'] for r in native if r['passed'] and not r['native_compatibility_exact']]
    summary['timing_variation_flags']=[r['resolution'] for r in native if r.get('timing_variable')]
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    def csv_file(name,rows):
        keys=list(dict.fromkeys(k for row in rows for k in row if k not in ('profiles','errors','stages_mean_ms')))
        with (DEST/name).open('w',newline='') as file:
            writer=csv.DictWriter(file,keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    csv_file('native-resolutions.csv',native);csv_file('portable-resolutions.csv',portable)
    bg='#101721';fg='#e5eef9';muted='#adbbce';colors=['#8c9bac','#71d9cd','#ffb86b']
    font=lambda size:ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',size)
    canvas=Image.new('RGB',(1700,1090),bg);draw=ImageDraw.Draw(canvas)
    draw.text((24,15),'Phoenix after the changes: native RTX resolution sweep',font=font(30),fill=fg)
    draw.text((24,59),'RTX 5080 · full graph · one pass · same input / inference / output size · medians in milliseconds',font=font(19),fill=muted)
    for i,(label,color) in enumerate(zip(['Unchanged SDK','Candidate default','Candidate: style + map + UI + field'],colors)):
        x=24+i*510;draw.rectangle((x,99,x+18,117),fill=color);draw.text((x+27,95),label,font=font(18),fill=fg)
    valid=[r for r in native if r['passed']]
    panels=[(208,845,'GPU processing', ['old_gpu_ms','new_gpu_ms','combined_gpu_ms']),
        (1005,1642,'Complete GPU-texture API call',['old_wall_ms','new_wall_ms','combined_wall_ms'])]
    for left,right,title,keys in panels:
        draw.text((left,135),title+' (logarithmic axis)',font=font(21),fill=fg)
        maximum=max((r[k] for r in valid for k in keys),default=10)*1.1
        low=.5;high=2**math.ceil(math.log2(maximum));axis=lambda value:left+(math.log2(max(value,low))-math.log2(low))/(math.log2(high)-math.log2(low))*(right-left)
        tick=low
        while tick<=high:
            x=axis(tick);draw.line((x,190,x,936),fill='#2c3a49');draw.text((x-12,164),f'{tick:g}',font=font(16),fill=muted);tick*=2
        for i,row in enumerate(native):
            y=205+i*29
            suffix=(' *' if row.get('timing_variable') else '')+(' !' if row['passed'] and not row.get('native_compatibility_exact') else '')
            if left==208:draw.text((24,y-11),row['resolution']+suffix,font=font(17),fill=fg)
            if not row['passed']:
                draw.text((left,y-10),'First-frame error (old + new)',font=font(15),fill=muted);continue
            for j,key in enumerate(keys):
                yy=y+(j-1)*7;draw.line((left,yy,axis(row[key]),yy),fill=colors[j],width=4)
                initial=row.get('initial_'+key)
                if initial and row.get('timing_variable'):
                    draw.line((axis(row[key]),yy,axis(initial),yy),fill=colors[j],width=1)
                    x=axis(initial);draw.ellipse((x-3,yy-3,x+3,yy+3),outline=colors[j],width=1)
    draw.text((24,958),'GPU timing excludes input preparation; the complete API call includes it. Producer uploads/readback are outside these native calls.',font=font(17),fill=muted)
    draw.text((24,989),'Standalone processing costs, not game FPS. Desktop/GPU activity was not locked down. Full timing and memory values are in the report.',font=font(17),fill=muted)
    draw.text((24,1018),'* Timing varied by >10% between initial and longer-warmup repeats; markers show the observed spread, not confidence intervals.',font=font(17),fill=muted)
    draw.text((24,1047),'! Native reset-output repeatability issue. 1024x768 failed its first native frame in both old and new versions.',font=font(17),fill='#ffb86b')
    canvas.save(DEST/'native-performance.png')
    canvas=Image.new('RGB',(1700,1035),bg);draw=ImageDraw.Draw(canvas)
    draw.text((24,15),'Portable routes: complete host-frame call cost',font=font(30),fill=fg)
    draw.text((24,59),'RTX 5080 · includes CPU preparation, transfers, resolve and optional field/style/UI composition · medians in milliseconds',font=font(18),fill=muted)
    series=[('candidate_cuda_default','CUDA default','#71d9cd'),('candidate_vulkan_default','Vulkan default','#81a9ff'),
        ('candidate_cuda_combined','CUDA full effects','#ffb86b'),('candidate_vulkan_combined','Vulkan full effects','#c397f5')]
    for i,(_,title,color) in enumerate(series):
        x=24+i*410;draw.rectangle((x,99,x+18,117),fill=color);draw.text((x+27,95),title,font=font(18),fill=fg)
    left,right=208,1642;low=8
    maximum=max((r['wall_ms'] for r in portable if r['passed'] and r['route'].startswith('candidate_')),default=64)*1.1
    high=2**math.ceil(math.log2(maximum));axis=lambda value:left+(math.log2(max(value,low))-math.log2(low))/(math.log2(high)-math.log2(low))*(right-left)
    draw.text((left,135),'Complete API call (logarithmic time axis)',font=font(21),fill=fg)
    tick=low
    while tick<=high:
        x=axis(tick);draw.line((x,190,x,936),fill='#2c3a49');draw.text((x-17,164),f'{tick:g}',font=font(16),fill=muted);tick*=2
    indexed={(r['resolution'],r['route']):r for r in portable}
    for i,(w,h) in enumerate(run['resolutions']):
        key=f'{w}x{h}';y=205+i*29;draw.text((24,y-11),key,font=font(17),fill=fg)
        for j,(route,_,color) in enumerate(series):
            row=indexed.get((key,route),{});yy=y+(j-1.5)*6
            if row.get('passed'):draw.line((left,yy,axis(row['wall_ms']),yy),fill=color,width=4)
    draw.text((left,921),'8K: Vulkan allocation failed; CUDA default/full-effects estimates use 2/1 measured calls.',font=font(15),fill='#ffb86b')
    draw.text((24,958),'Full effects = style + control map + UI + translation field. These steps are CPU boundary operations in the portable API.',font=font(17),fill=muted)
    draw.text((24,989),'Different interface and timing scope from the native GPU-resident API. These are processing costs, not game FPS.',font=font(17),fill=muted)
    canvas.save(DEST/'portable-performance.png')
    lines=['# Phoenix performance after neural-control changes','',
        'Measured on RTX 5080. Each resolution is the actual inference and output size; no upscaling or reduced graph. Full graph, one pass.',
        '', '## Native GPU-resident calls','',
        'Median milliseconds. Old SDK baseline pools the before/after runs. Complete API calls include preparation and synchronization; GPU timestamps exclude input preparation.',
        '', '| Resolution | Old GPU | New GPU | Full features GPU | Old API | New API | Full features API | Internal live MiB, default / full |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in native:
        if r['passed']:lines.append('| '+r['resolution']+' | '+' | '.join(n(r[k]) for k in ['old_gpu_ms','new_gpu_ms','combined_gpu_ms','old_wall_ms','new_wall_ms','combined_wall_ms'])+' | '+n(r['new_internal_live_mib'])+' / '+n(r['combined_internal_live_mib'])+' |')
        else:lines.append('| '+r['resolution']+' | not completed | | | | | | |')
    lines+=['','## Portable host-plane calls','',
        'The portable GPU timer covers neural operators only. Complete calls include uploads/readback, CPU boundary work and optional CPU field/style/UI composition; they have a different scope from GPU-resident native calls.',
        '', '| Resolution | Route | GPU median | API median | API p95 | Samples | Backend resident MiB |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in portable:
        lines.append('| '+r['resolution']+' | '+r['route']+' | '+(' | '.join(n(r[k] if k!='wall_p95_ms' or not r['low_sample_count'] else None) for k in ['gpu_ms','wall_ms','wall_p95_ms','samples','backend_resident_mib']) if r['passed'] else 'not completed | | | |')+' |')
    lines+=['','## 1080p feature isolation','',
        'Median complete-call milliseconds. Native uses GPU-resident textures; portable interfaces include host transfers and CPU boundary processing.',
        '', '| Profile | Native API | Portable CUDA API | Portable Vulkan API |', '| --- | ---: | ---: | ---: |']
    hd=next(r for r in native if r['resolution']=='1920x1080')
    costs={b:{r['profile']:r for r in v['profiles']} for b,v in summary['portable_feature_cost'].items()}
    for profile in ['default','advanced_default','natural','cinematic','control_map','ui_overlay','distortion','combined','default_after_features']:
        lines.append('| '+profile+' | '+n(hd.get('profiles',{}).get(profile,{}).get('wall_ms'))+' | '+n(costs.get('cuda',{}).get(profile,{}).get('wall',{}).get('median_ms'))+' | '+n(costs.get('vulkan',{}).get(profile,{}).get('wall',{}).get('median_ms'))+' |')
    lines+=['','## Correctness and timing flags','',
        '- 1024x768: native first-frame failure after successful session creation in both old and candidate DLLs.',
        '- 3840x1600 and 3840x2160: reset-output differences occur in both old and candidate native runs; numerical repeats quantify them in repeatability.json. Timing availability does not establish image correctness.',
        '- 5120-wide timing results vary substantially across the initial and longer-warmup runs. Do not interpret their difference as a reliable regression/speedup percentage.',
        '- Desktop/GPU activity was not locked down. Other screen-validation entries appeared in NVML near the rechecks but were absent from a subsequent live-process check; overlap and causation are unproven.',
        '- 8K Vulkan: old and new allocation failure. CUDA backend resident allocation alone is about 16.04 GiB; default/combined timings have only two/one measured samples and indicate memory pressure.',
        '- Native advanced buffers remain cached when effects are disabled; default_after_features_internal_live_mib reports retained session allocations.']
    lines+=['','## Scope and evidence','']+['- '+s for s in summary['limitations']]
    if errors:lines+=['','## Incomplete cases','', '```json',json.dumps(errors,indent=2),'```']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(native_completed=summary['native_completed'],portable_completed=summary['portable_completed'],errors=len(errors))))

if __name__=='__main__':main()
