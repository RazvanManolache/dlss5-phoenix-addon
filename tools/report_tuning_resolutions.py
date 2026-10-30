"""Publishable tables/plot from the explicit resolution sweep; no GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from compare_four_models import ROOT, sha, write

STYLES=('auto','rtx40','rtx30','rtx20')
LABELS={'auto':'Native 50','rtx40':'40-style','rtx30':'30-style','rtx20':'20-style'}
MODELS={'original':'Original','small-fp8':'Lightweight FP8'}


def fixture_sha(width,height):
    """Hash the exact raw FP16 input independently of any GPU output."""
    import numpy as np
    digest=hashlib.sha256()
    x=np.arange(width,dtype=np.float32)[None,:]
    for first in range(0,height,128):
        last=min(first+128,height)
        y=np.arange(first,last,dtype=np.float32)[:,None]
        pixels=np.ones((last-first,width,4),np.float16)
        pixels[:,:,0]=x/width
        pixels[:,:,1]=y/height
        pixels[:,:,2]=.5+.25*np.sin(x*.17)*np.cos(y*.11)
        digest.update(pixels.tobytes())
    return digest.hexdigest()


def cells(report):
    """Only completed, numerically valid cells get performance summaries."""
    result=[]
    for decision in report['decisions']:
        row={key:decision[key] for key in ('extent','style','model','passed')}
        if not decision['passed']:
            row['errors']=decision.get('errors',['Output mismatch'])
            result.append(row)
            continue
        matched=[case for case in report['cases'] if all(case[k]==decision[k] for k in ('extent','style','model'))]
        if len(matched)!=report['rounds']*2 or not all(case['passed'] for case in matched):
            raise ValueError('Incomplete paired cases')
        row.update(control_ms=decision['control_ms'],candidate_ms=decision['candidate_ms'],
                   reduction_percent=-decision['paired_median_change_percent'],
                   paired_reductions_percent=[-value for value in decision['paired_changes_percent']],
                   exact=decision['exact'])
        candidate=[case for case in matched if case['label']=='candidate']
        # These are absolute readings for the identical winning bundle across
        # sizes. Reconstructed single-target control containers are different;
        # their host RAM deltas are not promoted into optimization claims.
        fields={'device_local':'device_local_bytes','device_nonlocal':'device_nonlocal_bytes',
                'host_private_commit':'host_private_bytes','host_working_set':'host_working_set_bytes'}
        row['candidate_process_memory_mib']={label:statistics.median(case['memory'][field] for case in candidate)/1048576
                                             for label,field in fields.items()}
        row['candidate_owned_device_allocation_mib']=statistics.median(
            case['owned_memory']['heap_allocations']['device_default']['live_allocation_info_bytes']
            for case in candidate)/1048576
        result.append(row)
    return result


def mixed(row):
    values=row['paired_reductions_percent']
    return min(values)<0<max(values)


def plot(rows,extents,destination,confirmation=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'auto':'#2464bd','rtx40':'#8852b4','rtx30':'#c47708','rtx20':'#bd3944'}
    figure,axes=plt.subplots(1,2,figsize=(13.2,5.1),sharey=True)
    lookup={(r['extent'],r['style'],r['model']):r for r in rows}
    for axis,(model,title) in zip(axes,MODELS.items()):
        for offset,style in enumerate(STYLES):
            x=np.arange(len(extents))+(offset-1.5)*.055
            selected=[lookup.get((extent,style,model)) for extent in extents]
            y=[r['reduction_percent'] if r and r['passed'] else np.nan for r in selected]
            low=[r['reduction_percent']-min(r['paired_reductions_percent']) if r and r['passed'] else 0 for r in selected]
            high=[max(r['paired_reductions_percent'])-r['reduction_percent'] if r and r['passed'] else 0 for r in selected]
            axis.errorbar(x,y,yerr=[low,high],color=colors[style],label=LABELS[style],
                          marker='o',markersize=4,linewidth=1.5,capsize=3)
        if confirmation and '7680x4320' in extents:
            decision=next(r for r in confirmation['decisions'] if r['model']==model)
            gains=[-v for v in decision['paired_changes_percent']]
            median=-decision['paired_median_change_percent']
            axis.errorbar([extents.index('7680x4320')+.20],[median],
                          yerr=[[median-min(gains)],[max(gains)-median]],fmt='D',
                          markersize=5,markerfacecolor='white',color=colors['auto'],capsize=3)
        axis.axhline(0,color='#555',linewidth=.9)
        axis.set_title('Phoenix DLSS5 — '+title,fontsize=12,pad=12)
        axis.set_xticks(range(len(extents)),[extent.replace('x','×') for extent in extents],rotation=25,ha='right',fontsize=9)
        axis.grid(axis='y',alpha=.22)
        axis.set_axisbelow(True)
        axis.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('GPU time reduction versus reconstructed control (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,ncol=4,loc='upper center',bbox_to_anchor=(.5,.91),frameon=False)
    figure.suptitle('Do the retained kernel changes scale with resolution?',fontsize=15,y=.98)
    figure.text(.5,.033,'RTX 5080 · three paired process runs · bars show observed pair ranges, not confidence intervals',ha='center',fontsize=9,color='#444')
    if confirmation:
        figure.text(.5,.008,'Hollow diamonds: separate 8K five-pair native check with 3 s warmup and 3 s measurement',ha='center',fontsize=8,color='#444')
    figure.tight_layout(rect=(0,.075,1,.855))
    figure.savefig(destination,dpi=160,facecolor='white')
    plt.close(figure)


def markdown(report,rows,image_name,confirmation=None):
    lines=['# Phoenix DLSS5: resolution sweep','',
           'The retained kernel changes are tested at 720p, 2560×1080 ultrawide, 1440p, 4K, '
           '6144×3456 (6K) and 7680×4320 (8K). Both Original and Lightweight run through '
           'Phoenix DLSS5 Native RTX on an RTX 5080. Automatic/native remains the recommended style.','',
           f'![Paired GPU-time reductions across resolutions]({image_name})','',
           'Positive percentages mean less GPU time. Each value is the median of three '
           'matched before/after changes; † means the three pairs include both a gain '
           'and a regression. Error bars show the observed range, not a confidence interval.','']
    lines += ['## What held up','']
    for style in ('rtx20','rtx30','rtx40','auto'):
        selected=[r for r in rows if r['style']==style and r['passed']]
        if selected:
            reductions=[r['reduction_percent'] for r in selected]
            wins=sum(value>0 for value in reductions)
            lines.append(f"- **{LABELS[style]}:** lower median GPU time in {wins}/{len(selected)} cells; "
                         f"observed median reductions range from {min(reductions):+.2f}% to {max(reductions):+.2f}%.")
    if confirmation:
        values={r['model']:-r['paired_median_change_percent'] for r in confirmation['decisions']}
        lines += ['',f"The additional 8K native-50 check measured {values['original']:+.2f}% for Original "
                  f"and {values['small-fp8']:+.2f}% for Lightweight. The initial three-pair values remain "
                  'in the matrix; the longer-warmup results are shown separately below. '
                  'Small native-50 effects deserve more caution than the larger compatibility-path gains.']
    lines.append('')
    lookup={(r['extent'],r['style'],r['model']):r for r in rows}
    for model,title in MODELS.items():
        lines += ['## '+title,'','| Resolution | Native 50 | 40-style | 30-style | 20-style |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for extent in report['extents']:
            values=[]
            for style in STYLES:
                row=lookup.get((extent,style,model))
                values.append('Failed' if not row or not row['passed'] else
                              f"{row['reduction_percent']:+.2f}%"+(' †' if mixed(row) else ''))
            lines.append('| '+extent.replace('x','×')+' | '+' | '.join(values)+' |')
        lines.append('')
    lines += ['## Tuned native 50 processing time','',
              '| Resolution | Original GPU ms | Lightweight GPU ms |',
              '| --- | ---: | ---: |']
    for extent in report['extents']:
        values=[]
        for model in MODELS:
            row=lookup.get((extent,'auto',model))
            values.append(f"{row['candidate_ms']:.3f}" if row and row['passed'] else 'Failed')
        lines.append('| '+extent.replace('x','×')+' | '+' | '.join(values)+' |')
    lines += ['','## Memory and validation','',
              '| Resolution | Original process device-local MiB | Lightweight process device-local MiB |',
              '| --- | ---: | ---: |']
    for extent in report['extents']:
        values=[]
        for model in MODELS:
            row=lookup.get((extent,'auto',model))
            values.append(f"{row['candidate_process_memory_mib']['device_local']:.1f}" if row and row['passed'] else 'Failed')
        lines.append('| '+extent.replace('x','×')+' | '+' | '.join(values)+' |')
    passed=sum(r['passed'] for r in rows)
    lines += ['',f"**{passed}/{len(rows)} model/style/size cells passed**, across {len(report['cases'])} paired-process cases. "
              'Successful cells have finite, repeatable outputs and exact reset/history hashes '
              'against their matching control in every process. Failed cells retain their reason '
              'without a successful timing summary.','',
              'Memory figures are medians of process readings for the winning native-50 bundle '
              'after measurement and capture checks. They include harness and driver allocations; '
              'they are not pure model-buffer sizes, cold startup peaks or vendor-internal accounting. '
              'The control uses reconstructed single-target modules, so host-RAM differences against '
              'its different containers are not used as optimization claims.','',
              '## Method and limits','',
              '- Three alternating before/after process pairs per cell; one second of warmup, '
              'two seconds of measurement and at least 30 measured frames. The explicit sampling '
              'cap is extended to 120 seconds for slow high-resolution paths. GPU clocks use '
              'their normal dynamic settings.','- GPU milliseconds are separate medians of process '
              'medians. Percentages are medians of matched-round changes, so they need not equal '
              'the ratio of those separate time medians.','- Full graph, '
              'eight blocks, one pass, default controls, deterministic FP16 detail/gradient input '
              'and zero motion. Matching output copies and GPU barriers are timed; loading, '
              'producer upload, CPU gaps and readback are excluded. These are processing timings, '
              'not gameplay FPS.','- The saved native-50 control images are reused byte for byte. '
              'Compatibility controls are reconstructed by removing the winning changes in a '
              'temporary source copy. All eight model/style combinations first pass the earlier '
              '1080p reset/history hashes. Recompiled compatibility image hashes are recorded '
              'rather than claimed identical to the earlier binaries.','- Older instruction '
              'styles are compiled for the physical RTX 5080. This does not predict performance '
              'on physical RTX 20/30/40 cards. No game, installed files or published package is changed.','',
              'During this sweep, a CPU-only capture-check overhead was removed: successful '
              'literal checks no longer allocate an error-message string for every FP16 component. '
              'Completed cells were retained; unfinished cells use the corrected probe for all '
              'of their pairs. Both probe hashes are recorded per case, and all eight control '
              'combinations were checked again. GPU recording and the timed rendering code are unchanged.','',
              'The compact public data is [native-tuning-resolutions.json](native-tuning-resolutions.json). '
              'Raw local receipts, source/image pins and failure details are under '
              '`artifacts/current/kernel-tuning/resolutions.json` (ignored by Git). '
              'The maintained runner is `tools/benchmark_tuning_resolutions.py`.','']
    if confirmation:
        lines += ['## Native 50 at 8K: longer-warmup confirmation','',
                  'The initial three pairs straddled zero for both models. A separate five-pair '
                  'confirmation therefore uses three seconds of warmup and three seconds of '
                  'measurement per process. The main matrix above keeps its original three-pair '
                  'values rather than replacing them with a more favorable repeat.','',
                  '| Model | Control GPU ms | Tuned GPU ms | Median time reduction | Observed pair range |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for decision in confirmation['decisions']:
            gains=[-v for v in decision['paired_changes_percent']]
            lines.append(f"| {MODELS[decision['model']]} | {decision['control_ms']:.3f} | "
                         f"{decision['candidate_ms']:.3f} | {-decision['paired_median_change_percent']:+.2f}% | "
                         f"{min(gains):+.2f}% to {max(gains):+.2f}% |")
        lines += ['', 'All 20 confirmation processes retain exact outputs against the main '
                  'sweep. Pair ranges are descriptive, not confidence intervals. The full local '
                  'receipt is `artifacts/current/kernel-tuning/native-8k-confirmation.json`.','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'artifacts/current/kernel-tuning/resolutions.json')
    parser.add_argument('--output',type=Path,default=ROOT/'docs')
    parser.add_argument('--confirmation',type=Path,default=ROOT/'artifacts/current/kernel-tuning/native-8k-confirmation.json')
    args=parser.parse_args()
    report=json.loads(args.input.read_text())
    if report['state']!='completed':
        parser.error('Wait for the complete sweep, including recorded failures')
    expected=len(report['extents'])*len(STYLES)*len(MODELS)
    if len(report['decisions'])!=expected:
        raise ValueError('Incomplete resolution coverage')
    rows=cells(report)
    confirmation=None
    if args.confirmation.exists():
        confirmation=json.loads(args.confirmation.read_text())
        if confirmation['state']!='completed' or not confirmation['passed']:
            parser.error('Wait for successful completion of the native 8K confirmation')
        if confirmation['candidate_manifest_sha256']!=sha(ROOT/'artifacts/current/native-kernel-styles/native_manifest.json'):
            raise ValueError('Native confirmation tested a different candidate')
    fixture_pins={extent:fixture_sha(*map(int,extent.split('x'))) for extent in report['extents']}
    for row in rows:
        if row['passed']:
            matching=[case for case in report['cases'] if all(case[k]==row[k] for k in ('extent','style','model'))]
            row['reset_output_differs_from_input']=all(
                case['captures']['reset_a']['sha256']!=fixture_pins[row['extent']] for case in matching)
    args.output.mkdir(parents=True,exist_ok=True)
    image=args.output/'native-tuning-resolutions.png'
    plot(rows,report['extents'],image,confirmation)
    public=dict(schema='phoenix-resolution-tuning-summary/1',receipt_sha256=sha(args.input),
                candidate_manifest_sha256=sha(ROOT/'artifacts/current/native-kernel-styles/native_manifest.json'),
                gpu=report['gpu_before'],extents=report['extents'],rounds=report['rounds'],
                passed=report['passed'],input_fixture_sha256=fixture_pins,cells=rows,
                harness_revisions=report.get('harness_revisions',[]),
                probe_sha256=sorted({case['probe_sha256'] for case in report['cases']}),
                native_8k_confirmation=confirmation['decisions'] if confirmation else [],
                scope='Synthetic full-model GPU timing; single RTX 5080; reconstructed control; no game tests')
    write(args.output/'native-tuning-resolutions.json',public)
    (args.output/'NATIVE_TUNING_RESOLUTIONS.md').write_text(markdown(report,rows,image.name,confirmation),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],cells=len(rows),output=str(args.output))))


if __name__=='__main__':
    main()
