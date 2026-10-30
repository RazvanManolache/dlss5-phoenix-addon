"""Refresh native README numbers/charts, preserving dated portable/feature evidence."""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import statistics

from compare_four_models import ROOT, sha, write
from readme_summary import relative_links, replace_block, update_front_page, performance_summary

MODELS=('original','small-fp8')
LABELS={'original':'Original','small-fp8':'Lightweight FP8'}
ENGINES={'phoenix':'Phoenix DLSS5','nvidia':'NVIDIA DLSS5'}


def value(row,metric='gpu'):
    return row[metric]['median_process_median_ms']


def pixel_axis(sizes):
    """Resolution charts must preserve work ratios, not space every size equally."""
    return [width * height / 1_000_000 for width, height in sizes]


def require_complete(report,styles):
    """Reject partial/invalid results before replacing the public snapshot."""
    if not report.get('passed') or report.get('state')!='completed':
        raise ValueError('Native/NVIDIA sweep must finish successfully')
    expected=len(report['extents'])*len(MODELS)
    if len(report['comparisons'])!=expected or len(report['records'])!=expected*2:
        raise ValueError('Incomplete native resolution matrix')
    if not all(row['exact'] for row in report['comparisons']):
        raise ValueError('Native/NVIDIA output parity failed')
    pairs={(extent,model) for extent in report['extents'] for model in MODELS}
    actual={(row['extent'],row['model']) for row in report['comparisons']}
    records={(row['route'],row['model'],'x'.join(map(str,row['extent']))) for row in report['records']}
    if actual!=pairs or records!={(engine,model,extent) for extent,model in pairs for engine in ENGINES}:
        raise ValueError('Duplicate or missing native cells')
    if len(report['cases'])!=expected*2*report['rounds']:
        raise ValueError('Incomplete native process pairs')
    for row in report['cases']:
        if not row['passed'] or any(not math.isfinite(row[k]['median_ms']) or row[k]['median_ms']<=0 for k in ('gpu','wall')):
            raise ValueError('Invalid native timing')
    if not styles.get('passed') or styles.get('state')!='completed' or len(styles['records'])!=6:
        raise ValueError('Style context must finish successfully')
    if {(row['model'],row['style']) for row in styles['records']}!={(model,style) for model in MODELS for style in ('rtx20','rtx30','rtx40')}:
        raise ValueError('Incomplete instruction-style coverage')


def charts(records,comparisons,style_rows,out,date):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
    colors={'phoenix':'#21856c','nvidia':'#3269b5'}
    lookup={(r['route'],r['model'],tuple(r['extent'])):r for r in records}
    def get(engine,model,size=(1920,1080)):return lookup[engine,model,size]
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for axis,metric,title in zip(axes,('gpu','memory'),('GPU processing time (ms)','Process device-local residency (MiB)')):
        for offset,engine in enumerate(ENGINES):
            rows=[get(engine,model) for model in MODELS]
            values=[value(r) if metric=='gpu' else r['memory']['device_local_bytes']/1048576 for r in rows]
            bars=axis.bar(np.arange(2)+(offset-.5)*.34,values,width=.32,color=colors[engine],label=ENGINES[engine])
            if metric=='gpu':
                low=[value(r)-r['gpu']['min_process_median_ms'] for r in rows]
                high=[r['gpu']['max_process_median_ms']-value(r) for r in rows]
                axis.errorbar(np.arange(2)+(offset-.5)*.34,values,yerr=[low,high],fmt='none',color='#333',capsize=3)
            axis.bar_label(bars,fmt='%.3f' if metric=='gpu' else '%.1f',padding=4,fontsize=9)
        axis.set_xticks(range(2),['Original','Lightweight FP8']);axis.set_title(title,fontsize=11)
        axis.set_ylim(0,max(axis.get_ylim()[1]*1.16,1));axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',alpha=.18);axis.set_axisbelow(True)
    fig.suptitle('1080p: Phoenix DLSS5 and NVIDIA DLSS5',fontsize=15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
    fig.text(.5,.015,f'RTX 5080 · {date} · median of three process medians · error bars: GPU process-median ranges',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.84));fig.savefig(out/'native-1080p.png',dpi=160);plt.close(fig)

    sizes=sorted({tuple(r['extent']) for r in records},key=lambda size:size[0]*size[1])
    compare={(r['model'],r['extent']):r for r in comparisons}
    megapixels=pixel_axis(sizes)
    fig,axes=plt.subplots(2,2,figsize=(14,7.4),sharex='col')
    for axis in axes.flat:
        axis.set_xscale('log')
    for column,model in enumerate(MODELS):
        for engine in ENGINES:
            axes[0,column].plot(megapixels,[value(get(engine,model,size)) for size in sizes],
                               marker='o',markersize=4,label=ENGINES[engine],color=colors[engine])
        axes[0,column].set_yscale('log');axes[0,column].set_title(LABELS[model]);axes[0,column].set_ylabel('GPU ms (log scale)')
        axes[0,column].set_ylim(.8,80)
        axes[0,column].set_yticks([1,2,5,10,20,50])
        axes[0,column].yaxis.set_major_formatter(ScalarFormatter())
        axes[0,column].minorticks_off()
        selected=[compare[model,'x'.join(map(str,size))] for size in sizes]
        middle=[r['median_phoenix_extra_gpu_percent'] for r in selected]
        low=[m-min(r['paired_phoenix_extra_gpu_percent']) for r,m in zip(selected,middle)]
        high=[max(r['paired_phoenix_extra_gpu_percent'])-m for r,m in zip(selected,middle)]
        axes[1,column].errorbar(megapixels,middle,yerr=[low,high],fmt='o-',color=colors['phoenix'],capsize=3)
        axes[1,column].axhline(0,color='#555',linewidth=.9)
        axes[1,column].set_ylabel('Phoenix DLSS5 extra GPU time (%)')
        ticks=[size for size in sizes if size in [(320,180),(640,360),(1280,720),(1920,1080),(3840,2160),(7680,4320)]]
        axes[1,column].set_xticks(pixel_axis(ticks),['×'.join(map(str,s)) for s in ticks],rotation=35,ha='right',fontsize=8)
        axes[1,column].set_xlabel('Resolution positioned by pixel count (log scale)')
        for axis in axes[:,column]:
            axis.minorticks_off()
            axis.grid(axis='y',alpha=.2);axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Matched native models across resolution',fontsize=15)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,frameon=False)
    fig.text(.5,.012,'Positive extra time means Phoenix DLSS5 is slower; negative means faster. Pair ranges are descriptive, not confidence intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.91));fig.savefig(out/'native-scaling.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    fields=('device_local_bytes','host_private_bytes')
    for axis,field,title in zip(axes,fields,('Process device-local residency','Process private commitment')):
        for engine in ENGINES:
            for model in MODELS:
                axis.plot(range(len(sizes)),[get(engine,model,size)['memory'][field]/1048576 for size in sizes],
                          color=colors[engine],linestyle='-' if model=='original' else '--',marker='o',markersize=3,
                          label=ENGINES[engine]+' / '+LABELS[model])
        axis.set_title(title);axis.set_ylabel('MiB');axis.grid(axis='y',alpha=.2)
        axis.set_xticks(range(len(sizes)),['×'.join(map(str,s)) for s in sizes],rotation=45,ha='right',fontsize=8)
        axis.spines[['top','right']].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.01,'Current-process readings include the harness and driver. Private commitment is not resident RAM or model-only storage.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.84));fig.savefig(out/'native-current-memory.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for axis,model in zip(axes,MODELS):
        rows=[get('phoenix',model),get('nvidia',model)]+[next(r for r in style_rows if r['model']==model and r['style']==style) for style in ('rtx40','rtx30','rtx20')]
        names=['Phoenix DLSS5 / native 50','NVIDIA DLSS5','Phoenix DLSS5 / 40-style','Phoenix DLSS5 / 30-style','Phoenix DLSS5 / 20-style']
        values=[value(r) for r in rows]
        axis.barh(range(5),values,color=[colors['phoenix'],colors['nvidia'],'#8852b4','#c47708','#bd3944'])
        axis.errorbar(values,range(5),xerr=[[value(r)-r['gpu']['min_process_median_ms'] for r in rows],
                                          [r['gpu']['max_process_median_ms']-value(r) for r in rows]],
                      fmt='none',color='#333',capsize=3)
        for i,v in enumerate(values):axis.text(v+max(values)*.015,i,f'{v:.3f} ms',va='center',fontsize=9)
        axis.set_yticks(range(5),names,fontsize=9);axis.invert_yaxis();axis.set_xlim(0,max(values)*1.26)
        axis.set_title(LABELS[model]);axis.set_xlabel('GPU processing time (ms), lower is better')
        axis.grid(axis='x',alpha=.17);axis.set_axisbelow(True);axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Synthetic instruction strategies: every bar runs on RTX 5080',fontsize=14)
    fig.text(.5,.015,'1080p · identical structured input and model per panel · older styles do not simulate older GPU hardware',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.92));fig.savefig(out/'instruction-styles-1080p.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/current/native-vendor-refresh/run.json')
    parser.add_argument('--styles',type=Path,default=ROOT/'artifacts/current/native-vendor-refresh/styles-1080p.json')
    args=parser.parse_args()
    report=json.loads(args.source.read_text(encoding='utf-8'));styles=json.loads(args.styles.read_text(encoding='utf-8'))
    require_complete(report,styles)
    if styles['reference_receipt_sha256']!=sha(args.source):raise ValueError('Style context uses another reference')
    out=ROOT/'docs/benchmarks/current'
    performance_readme=ROOT/'docs/PERFORMANCE.md'
    performance_text=performance_readme.read_text(encoding='utf-8')
    previous=json.loads((out/'measurements.json').read_text(encoding='utf-8'))
    historical=previous.get('historical_broad_refresh',previous)
    old_readme=(out/'README.md').read_text(encoding='utf-8')
    marker='<!-- historical-broad-refresh -->'
    legacy=old_readme.split(marker,1)[1] if marker in old_readme else old_readme[old_readme.index('## Portable host-plane runtime'):]
    records=report['records'];get=lambda engine,model,size=(1920,1080):next(r for r in records if r['route']==engine and r['model']==model and r['extent']==list(size))
    derived={}
    for engine in ENGINES:
        original,light=[get(engine,model) for model in MODELS]
        derived[engine]=dict(lightweight_gpu_time_reduction_percent=100*(1-value(light)/value(original)),
                             lightweight_device_local_reduction_percent=100*(1-light['memory']['device_local_bytes']/original['memory']['device_local_bytes']))
    public=dict(schema='phoenix-public-benchmarks/4',date=report['measured_date'],
                hardware=dict(gpu=report['gpu_before'],older_rtx_cards_owned=False,physical_gpu='RTX 5080'),
                fixture=report['fixture'],aggregate=report['aggregate'],timing_scopes=report['timing_scope'],
                model_dll_sha256=report['model_dll_sha256'],model_dll_bytes=report['model_dll_bytes'],
                native_parity=report['comparisons'],records=records,derived_1080p=derived,
                instruction_styles_1080p=styles['records'],native_refresh_receipt_sha256=sha(args.source),
                style_receipt_sha256=sha(args.styles),native_manifest=report['native_manifest'],
                probe_sha256=report['probe_sha256'],probe_source_sha256=report['probe_source_sha256'],
                forwarder_sha256=report['forwarder_sha256'],ngx_core_sha256=report['ngx_core_sha256'],
                historical_broad_refresh=historical)
    inventory=args.source.parent/'hardware.json'
    if inventory.exists():public['hardware'].update(json.loads(inventory.read_text(encoding='utf-8')))
    if 'cpu' in public['hardware']:public['hardware']['cpu']=public['hardware']['cpu'].strip()
    charts(records,report['comparisons'],styles['records'],out,report['measured_date'])
    write(out/'measurements.json',public)
    summary=performance_summary(get, 'median_process_median_ms')
    gaps=[next(c['median_phoenix_extra_gpu_percent'] for c in report['comparisons'] if c['extent']=='1920x1080' and c['model']==model) for model in MODELS]
    if any(abs(gap)>5 for gap in gaps):summary='**Fresh native RTX measurements for both models on RTX 5080.**'
    lines=['# Current native two-model benchmark','',
           f"Native measurements refreshed **{report['measured_date']}** from the tuned development bundle. "
           f"The portable/feature study retained below is dated **{historical['date']}** and was not rerun. "
           'One RTX 5080, native x64, full trained graphs, one pass, fixed structured FP16 input and zero motion. '
           'Both engines use the same authenticated 310.9.16 model DLL. No games or installed addons were changed.','',
           '[Measurements JSON](measurements.json) · [Synthetic architecture study](../../NATIVE_TUNING_RESOLUTIONS.md)','',
           '## Native RTX and NVIDIA DLSS5','',summary,'',
           '![Matched 1080p GPU time and process residency](native-1080p.png)','',
           '| Input | Phoenix DLSS5 Original GPU ms | Phoenix DLSS5 Lightweight GPU ms | NVIDIA DLSS5 Original GPU ms | NVIDIA DLSS5 Lightweight GPU ms |',
           '| --- | ---: | ---: | ---: | ---: |']
    sizes=sorted({tuple(r['extent']) for r in records},key=lambda size:size[0]*size[1])
    for size in sizes:
        lines.append('| '+'×'.join(map(str,size))+' | '+' | '.join(f'{value(get(engine,model,size)):.3f}' for engine in ENGINES for model in MODELS)+' |')
    lines+=['','![Native GPU time and paired relative differences across resolution](native-scaling.png)','',
            'Points are positioned by actual pixel count on a logarithmic horizontal axis. The [resolution profiling investigation](../../NATIVE_SCALING_PROFILE.md) separates pixel growth from the padding/token step just below 1080p and attributes extra GPU work to specific kernel families.','',
            '## Performance conclusions','',
            'Compare the same model and size. Positive extra GPU time below means Phoenix DLSS5 is slower; negative means faster. '
            'Percentages are medians of matched-pair changes, not ratios of the separate renderer medians. '
            'All three process medians, per-process sample counts and timing summaries remain in the local receipt. '
            'Per-frame sample arrays are not retained by this compact refresh.','',
            '| Model | Input | Phoenix DLSS5 extra GPU time | Observed pair range |','| --- | --- | ---: | ---: |']
    for comparison in report['comparisons']:
        values=comparison['paired_phoenix_extra_gpu_percent']
        lines.append(f"| {LABELS[comparison['model']]} | {comparison['extent']} | {comparison['median_phoenix_extra_gpu_percent']:+.2f}% | {min(values):+.2f}% to {max(values):+.2f}% |")
    lines+=['','Both models match their corresponding NVIDIA DLSS5 preset byte for byte in reset A, history and reset B '
            f"at all {len(sizes)} extents, across {len(report['cases'])} fresh processes. This applies to the native-50/default path. "
            'The two trained models produce different images; this is implementation parity, not a moving-scene quality ranking.','',
            '## Instruction styles: synthetic context','',
            "**I don't own RTX 20/30/40 hardware.** Their native kernels are built, but physical-card inference is pending. "
            'Every bar below is measured on RTX 5080 with the same structured 1080p fixture and sampling settings. '
            'Older strategies still compile for the real target GPU. They do not emulate older-card throughput, caches or bandwidth.','',
            '![Synthetic native and older instruction strategies compared with NVIDIA DLSS5 at 1080p](instruction-styles-1080p.png)','',
            'The 20/30 paths use FP16 decomposition and can round differently from native FP8. Exact NVIDIA parity above '
            'is for the default native path. The separate [before/after study](../../NATIVE_TUNING_RESOLUTIONS.md) '
            'checks each style against its own control over 720p–8K.','',
            '## Memory at 1080p','',
            '| Engine / model | Device-local MiB | Nonlocal MiB | Private commitment MiB | Working set MiB |',
            '| --- | ---: | ---: | ---: | ---: |']
    for engine in ENGINES:
        for model in MODELS:
            memory=get(engine,model)['memory']
            lines.append('| '+ENGINES[engine]+' / '+LABELS[model]+' | '+' | '.join(f"{memory[key]/1048576:.1f}" for key in ('device_local_bytes','device_nonlocal_bytes','host_private_bytes','host_working_set_bytes'))+' |')
    lines+=['','DXGI residency includes the process, harness and driver. Private commitment is not resident RAM; working set '
            'is shown separately. These are snapshots after measurement and output validation, not cold-start peaks or vendor-internal buffer accounting.','',
            '## Memory scaling and host RAM','',
            '![Current native process residency and private commitment across resolution](native-current-memory.png)','',
            '## Sampling and reproduction','',
            '- Three fresh-process pairs per model/size, with renderer order reversed on alternate rounds. Three seconds of warmup and three seconds of measurement, at least 30 measured frames. GPU clocks remain dynamic.',
            '- Headline times are medians of process medians. Observed ranges are descriptive, not confidence intervals. GPU time includes recorded engine work, barriers and matching output copy. Call time includes recording/submission/fence wait; neither includes producer upload or output readback. Do not add GPU time to call time.',
            '- The current native bundle is used unchanged. Model, NGX core, probe and module hashes are retained in the JSON. Successful finite, reset-repeatable outputs and exact matching-renderer captures are required before publication.',
            '- Reproduce with `tools/benchmark_native_vendor.py`, then `tools/benchmark_style_vendor_context.py`, then `tools/report_native_vendor.py`. These scripts perform GPU work except the final report step, and use disposable build/input folders. Proprietary model DLLs and SDKs remain external.','',
            f"## Earlier portable and feature study ({historical['date']})",'',
            'Everything below retains that study’s measurements and build identities. Its native reference rows are historical '
            'feature baselines, separate from the freshly updated default native results above. The JSON keeps that complete '
            'snapshot under `historical_broad_refresh`; it does not label older portable or feature timings as newly measured.','',marker,legacy]
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    text=performance_text
    block=['<!-- current-two-model-benchmark -->','## Native RTX: the main event','',summary,
           'Both Original and Lightweight have comparable isolated processing latency at 1080p; the full report shows where higher-resolution differences remain.','',
           f"Measured **{report['measured_date']}** using the tuned development kernels, RTX 5080, native x64 and one full pass. These are not game FPS or measurements of every download/route.",'',
           '![Phoenix DLSS5 and NVIDIA DLSS5: matched 1080p processing time and GPU-local process residency](docs/benchmarks/current/native-1080p.png)','',
           '| Renderer / model at 1080p | GPU ms | Full call ms |','| --- | ---: | ---: |']
    for engine in ENGINES:
        for model in MODELS:
            row=get(engine,model)
            block.append(f"| Native RTX {ENGINES[engine]} / {LABELS[model]} | {value(row):.3f} | {value(row,'wall'):.3f} |")
    block+=['',f"**A lighter model, not a longer coffee break.** Lightweight uses **{derived['phoenix']['lightweight_gpu_time_reduction_percent']:.1f}% less GPU time** than Original within Phoenix DLSS5 at 1080p. It produces a different image: pick the look you prefer, not just the smallest number.",'',
            f"**Same model, matching pixels.** The native-50/default Phoenix DLSS5 path matches its corresponding NVIDIA DLSS5 preset byte for byte in reset/history fixtures at all **{len(sizes)} extents**, from 320×180 through 8K. The 20/30 instruction strategies can use different floating-point rounding.",'',
            f"**Memory gets a vote too.** At 1080p, Lightweight reduces measured process GPU-local residency by **{derived['phoenix']['lightweight_device_local_reduction_percent']:.1f}% within Phoenix DLSS5** and **{derived['nvidia']['lightweight_device_local_reduction_percent']:.1f}% within NVIDIA DLSS5**. These include harness and driver allocations. [Current RAM/VRAM readings](docs/benchmarks/current/README.md#memory-at-1080p).",'',
            '**More knobs, fewer mysteries.** Both models support tone/structure/skin tuning, compatible external weights, block controls, repeats, passes and adaptive budgets where the selected route supports them. [Controls and limits](docs/PORTABLE_MODEL_CONTROLS.md) · [Appearance controls](docs/NEURAL_CONTROLS.md).','',
            '**The portability side quest is still alive.** CUDA, Vulkan, CPU, HIP, SYCL and Metal have their own limits. Their retained timings have their own measurement dates. [Backend matrix](docs/SUPPORT.md) · [Dated portable study](docs/benchmarks/current/README.md#portable-host-plane-runtime).','',
            '[Full current native report and resolution charts](docs/benchmarks/current/README.md) · [Machine-readable results](docs/benchmarks/current/measurements.json) · [Historical released/current comparison](docs/benchmarks/current/release-ab.md)',
            '<!-- /current-two-model-benchmark -->']
    block[-1:-1]=['', '**Less repeated work, same model inputs.** The [portable reuse study](docs/PORTABLE_REUSE_STUDY.md) tests faster CPU operand loading and shared host input preparation against the previous code, with exact output checks for both models. It also separates GPU fallback experiments from default CUDA/Vulkan execution.', '',
        '**The 1080p bend, investigated.** Pixel count explains the broad curve; a real padding/token step just below 1080p also appears in NVIDIA DLSS5. The [profiling report](docs/NATIVE_SCALING_PROFILE.md) identifies the expensive stages and documents register experiments rejected after measurement.']
    detailed=relative_links('\n'.join(block), 'README.md', 'docs/PERFORMANCE.md')
    performance_readme.write_text(replace_block(text, detailed), encoding='utf-8')
    update_front_page(ROOT, get, report['measured_date'], 'median_process_median_ms')
    print(json.dumps(dict(passed=True,records=len(records),native_processes=len(report['cases']),style_processes=len(styles['cases']))))


if __name__=='__main__':
    main()
