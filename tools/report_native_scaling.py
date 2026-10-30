"""Explain native resolution scaling with geometry and measured kernel attribution."""
import collections
import json
from pathlib import Path
import re
import statistics
from compare_four_models import ROOT, sha

MODELS = ['original', 'small-fp8']
LABELS = {'original': 'Original', 'small-fp8': 'Lightweight FP8'}
GROUPS = ['local_32_64', 'local_128_256', 'local_512', 'boundary',
          'global_ffn', 'global_attention', 'global_misc']
GROUP_LABELS = ['Local 32/64 channels', 'Local 128/256 channels', 'Local 512 channels',
                'Input + reconstruction', 'Global feed-forward', 'Global attention', 'Repack/projection/bridge']


def group_kernel(name):
    if 'pre_block' in name or 'post_block' in name: return 'boundary'
    if name.startswith('cc_split_swin_'): return 'local_512'
    if name.startswith('cc_vit_1d_attention'): return 'global_attention'
    if name.startswith('cc_vit_1d_ffn'): return 'global_ffn'
    if name.startswith('cc_vit_') or name.startswith('cc_dec_input_'): return 'global_misc'
    channels = re.search(r'_swin_\d+h_(32|64|128|256)_', name)
    if channels: return 'local_32_64' if int(channels[1]) <= 64 else 'local_128_256'
    raise ValueError('Unclassified native kernel: ' + name)


def excess_ms(before_ms, after_ms, before_pixels, after_pixels):
    """Positive means more time than a zero-intercept pixel-proportional baseline."""
    if before_pixels <= 0 or after_pixels <= 0: raise ValueError('Positive pixel counts required')
    return after_ms - before_ms * after_pixels / before_pixels


def aggregate(cases, model, extent, engine='phoenix', instrumented=False):
    rows = [r for r in cases if r['model'] == model and r['extent'] == extent
            and r['engine'] == engine and r['instrumented'] == instrumented]
    if len(rows) != 2: raise ValueError('Expected two independently warmed process medians')
    geometry = rows[0]['phoenix_plan_geometry']
    if any(r['phoenix_plan_geometry'] != geometry for r in rows): raise ValueError('Geometry changed')
    result = dict(model=model, extent=extent, engine=engine, instrumented=instrumented,
        gpu_ms=statistics.median(r['gpu']['median_ms'] for r in rows),
        process_medians_ms=[r['gpu']['median_ms'] for r in rows],
        geometry=geometry, native_kernel_style=rows[0].get('native_kernel_style'),
        owned_bytes=statistics.median(r.get('owned_memory', {}).get('totals', {}).get('live_allocation_info_bytes', 0) for r in rows),
        process_device_local_bytes=statistics.median(r['memory']['device_local_bytes'] for r in rows))
    result['gpu_ms_per_megapixel'] = result['gpu_ms'] / (extent[0] * extent[1] / 1e6)
    if instrumented:
        kernels = []
        for index, launch in enumerate(rows[0]['individual_kernel_launches']):
            if any(row['individual_kernel_launches'][index] != launch for row in rows): raise ValueError('Kernel schedule changed')
            kernels.append(dict(**launch, group=group_kernel(launch['name']),
                                gpu_ms=statistics.median(row['individual_kernels'][index]['median_ms'] for row in rows)))
        result['kernels'] = kernels
        result['groups'] = {group: sum(row['gpu_ms'] for row in kernels if row['group'] == group) for group in GROUPS}
        result['profile_gpu_stages'] = {key: statistics.median(row['gpu_stages'][key]['median_ms'] for row in rows)
                                       for key in rows[0]['gpu_stages']}
    return result


def build_report(source, deep, registers, resources):
    if not all(r.get('passed') and r.get('state') == 'completed' for r in [source, deep, registers]):
        raise ValueError('Incomplete or failed GPU study')
    if len(source['cases']) != 80 or len(deep['cases']) != 8 or len(registers['cases']) != 24:
        raise ValueError('Incomplete process matrix')
    cases, summaries, attribution, parity = source['cases'], [], [], []
    extents = [list(map(int, extent.split('x'))) for extent in source['extents']]
    for model in MODELS:
        baseline = next(row for row in cases if row['model'] == model and row['extent'] == [1920, 1080]
                        and row['engine'] == 'phoenix' and not row['instrumented'])
        baseline_hashes = {key: value['sha256'] for key, value in baseline['captures'].items()}
        if any(row['captures'] != baseline_hashes for row in registers['cases'] if row['model'] == model):
            raise ValueError('Register experiment differs from matched profile fixture')
        for extent in extents:
            summaries.extend([aggregate(cases, model, extent), aggregate(cases, model, extent, instrumented=True)])
            native = [r for r in cases if r['model'] == model and r['extent'] == extent and r['engine'] == 'phoenix']
            hashes = [{k: c['sha256'] for k, c in row['captures'].items()} for row in native]
            if any(value != hashes[0] for value in hashes): raise ValueError('Instrumented/plain temporal mismatch')
            vendor = [r for r in cases if r['model'] == model and r['extent'] == extent and r['engine'] == 'nvidia']
            if vendor:
                exact = all({k: c['sha256'] for k, c in row['captures'].items()} == hashes[0] for row in vendor)
                if not exact: raise ValueError('NVIDIA/native temporal mismatch')
                parity.append(dict(model=model, extent=extent, exact_reset_history=True))
                summaries.append(aggregate(cases, model, extent, engine='nvidia'))
        before = next(r for r in summaries if r['model'] == model and r['extent'] == [1920, 1024] and r['instrumented'])
        after = next(r for r in summaries if r['model'] == model and r['extent'] == [1920, 1032] and r['instrumented'])
        ratio = 1032 / 1024
        groups = [dict(group=group, before_ms=before['groups'][group], after_ms=after['groups'][group],
                       proportional_ms=before['groups'][group] * ratio,
                       excess_ms=excess_ms(before['groups'][group], after['groups'][group], 1024, 1032)) for group in GROUPS]
        kernels = [dict(index=a['index'], name=a['name'], group=a['group'], before_ms=a['gpu_ms'], after_ms=b['gpu_ms'],
                        grid_before=a['grid'], grid_after=b['grid'],
                        excess_ms=excess_ms(a['gpu_ms'], b['gpu_ms'], 1024, 1032)) for a, b in zip(before['kernels'], after['kernels'])]
        stages = []
        for extent in [[1920, 1024], [1920, 1032]]:
            rows = [r for r in deep['cases'] if r['model'] == model and r['extent'] == extent]
            reference = next(r for r in cases if r['model'] == model and r['extent'] == extent
                             and r['engine'] == 'phoenix' and not r['instrumented'])
            expected = {key: value['sha256'] for key, value in reference['captures'].items()}
            if any({key: value['sha256'] for key, value in row['captures'].items()} != expected for row in rows):
                raise ValueError('Stage-only profiling changed temporal outputs')
            selection = rows[0]['kernel_selections']
            if extent == [1920, 1024]: before_selection = selection
            elif selection != before_selection: raise ValueError('Entry selector changed at the boundary')
            stages.append(dict(extent=extent, gpu_ms=statistics.median(r['gpu']['median_ms'] for r in rows),
                stages={key: statistics.median(r['gpu_stages'][key]['median_ms'] for r in rows) for key in rows[0]['gpu_stages']}))
        attribution.append(dict(model=model, pixel_ratio=ratio,
            groups=groups, kernels=sorted(kernels, key=lambda r: r['excess_ms'], reverse=True),
            deep_stages=stages, kernel_entries_unchanged=True))
    return dict(schema='phoenix-native-scaling-analysis/1', date=source['measured_date'],
        passed=True, models=MODELS, hardware='RTX 5080, driver 616.64', full_graph=True,
        rounds=source['rounds'], plain_and_kernel_processes=80, stage_processes=8, register_processes=24,
        model_sha256=source['model_sha256'], manifest_sha256=source['native_manifest_sha256'],
        source_sha256=source['source_sha256'], summaries=summaries, attribution=attribution, reference_parity=parity,
        register_experiment=dict(retained=False, reason='No tested register cap beat the current 128-register production choice',
            resources=registers['resources'], control_proof=registers['control_proof'], summary=registers['summary']),
        driver_resources=resources)


def charts(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.8))
    for column, model in enumerate(MODELS):
        rows = [r for r in report['summaries'] if r['model'] == model and not r['instrumented'] and r['engine'] == 'phoenix']
        pixels = [r['extent'][0] * r['extent'][1] / 1e6 for r in rows]
        values = [r['gpu_ms'] for r in rows]
        axis = axes[0, column]
        axis.plot(pixels, values, 'o-', color='#178870', label='Phoenix DLSS5 (uninstrumented)')
        axis.plot(pixels, [values[0] * p / pixels[0] for p in pixels], '--', color='#777', label='Pixel-proportional from 720p')
        axis.set_xscale('log'); axis.set_yscale('log')
        axis.set_xticks([1, 2, 4, 8], ['1', '2', '4', '8'])
        axis.set_yticks([1, 2, 5, 10, 20]); axis.yaxis.set_major_formatter(ScalarFormatter()); axis.minorticks_off()
        axis.set_title(LABELS[model]); axis.set_xlabel('Actual input megapixels (log scale)'); axis.set_ylabel('GPU milliseconds (log scale)')
        axis.grid(alpha=.18)
        groups = report['attribution'][column]['groups']
        axis = axes[1, column]
        values = [row['excess_ms'] * 1000 for row in groups]
        axis.barh(range(len(GROUPS)), values, color=['#268c79' if value > 0 else '#888' for value in values])
        axis.set_yticks(range(len(GROUPS)), GROUP_LABELS, fontsize=9); axis.invert_yaxis()
        axis.axvline(0, color='#555', linewidth=.8); axis.grid(axis='x', alpha=.18); axis.set_axisbelow(True)
        axis.set_xlabel('Diagnostic microseconds beyond pixel-proportional growth')
        axis.set_title('1024 to 1032 image rows: where excess time goes', fontsize=11)
        for row, value in enumerate(values):
            axis.text(value + (1 if value >= 0 else -1), row, f'{value:+.1f}', va='center',
                      ha='left' if value >= 0 else 'right', fontsize=8)
        axis.set_xlim(min(-8, min(values) * 1.4), max(values) * 1.25)
    for axis in axes.flat: axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('1080p scaling: pixel growth, padding steps, and measured kernel costs', fontsize=15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .955), ncol=2, frameon=False)
    fig.text(.5, .018, 'Top: two independently warmed process medians per point. Bottom: single-launch timestamp attribution can perturb overlap.\nKernel-median sums are diagnostic; use uninstrumented totals for latency. Full networks, one pass, both models on RTX 5080.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .91)); fig.subplots_adjust(hspace=.39)
    fig.savefig(ROOT / 'docs/native-scaling-profile.png', dpi=150); plt.close(fig)


def main():
    folder = ROOT / 'artifacts/current/native-scaling-profile'
    read = lambda name: json.loads((folder / name).read_text(encoding='utf-8'))
    report = build_report(read('run.json'), read('deep-stages.json'), read('boundary-registers.json'), read('resources.json'))
    (ROOT / 'docs/native-scaling-profile.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    charts(report)
    lookup = lambda model, extent, engine='phoenix': next(r for r in report['summaries'] if r['model'] == model
        and r['extent'] == extent and r['engine'] == engine and not r['instrumented'])
    lines = ['# Native resolution scaling: is there a 1080p cliff?', '',
        '**There is no general superlinear jump at 1080p. There is a real padding/token boundary just below it.** This investigation separates those two effects using full-model Phoenix DLSS5/NVIDIA DLSS5 comparisons and individual Phoenix kernel timestamps.', '',
        '![Pixel scaling and diagnostic attribution](native-scaling-profile.png)', '',
        '## What the original chart hid', '',
        'Its horizontal axis placed every resolution the same distance apart. The move from 720p to 1080p contains 125% more pixels; it is not the same amount of work as moving between adjacent small resolutions. The native comparison chart now places points by actual pixel count, using a logarithmic horizontal axis.', '',
        'The original 12-resolution sweep measured Original at 2.475 → 3.841 ms and Lightweight at 1.450 → 2.250 ms from 720p to 1080p: about 55% more GPU time for 125% more pixels. Thus cost per pixel **decreases**. The fresh profiling sweep below is separate; its absolute times should not be pooled with the earlier sweep.', '',
        '## A genuine boundary: 1920×1024 to 1920×1032', '',
        '| Input | Padded spatial extent | Global token grid | Tokens | Full-resolution FP8 stem skip |',
        '| --- | --- | --- | ---: | ---: |']
    for extent in [[1920, 1024], [1920, 1032], [1920, 1080]]:
        row = lookup('original', extent); g = row['geometry']
        lines.append(f"| {'×'.join(map(str, extent))} | {g['W0']}×{g['H0']} | {g['LW']}×{g['LH']} | {g['T']} | {g['W0']*g['H0']*32/1048576:.1f} MiB |")
    lines += ['', 'Eight extra image rows mean only **0.781%** more displayed pixels, but **12.5%** more padded spatial pixels and **25%** more latent tokens. Global attention can also grow with token-pair count; its launch grid changes from two to three query tiles. There is no renderer/model/transport switch here.', '',
        '| Engine / model | 1024 rows (ms) | 1032 rows (ms) | Growth | 1080 rows (ms) |', '| --- | ---: | ---: | ---: | ---: |']
    for model in MODELS:
        for engine, label in [('phoenix', 'Phoenix DLSS5'), ('nvidia', 'NVIDIA DLSS5')]:
            a, b, c = [lookup(model, [1920, h], engine) for h in [1024, 1032, 1080]]
            lines.append(f"| {label} / {LABELS[model]} | {a['gpu_ms']:.3f} | {b['gpu_ms']:.3f} | {100*(b['gpu_ms']/a['gpu_ms']-1):.1f}% | {c['gpu_ms']:.3f} |")
    lines += ['', 'The step also appears in NVIDIA DLSS5. Its internal stages were not instrumented; Phoenix source and launch metadata explain Phoenix’s own geometry change. Both implementations match byte-for-byte in reset/history outputs at all four reference-checked extents, including both sides of this boundary.', '',
        '## Which parts exceed proportional growth?', '',
        'For each profiled kernel, the pixel-only expectation is its 1024-row time multiplied by `1032 / 1024`. Positive excess is actual time minus that expectation. Medians are first collected per independently warmed process, then summarized. Single-launch timestamps perturb overlap; their sum is not an exact accounting identity for total GPU time.', '',
        '| Kernel family | Original excess (µs) | Lightweight excess (µs) |', '| --- | ---: | ---: |']
    for i, label in enumerate(GROUP_LABELS):
        lines.append(f"| {label} | {report['attribution'][0]['groups'][i]['excess_ms']*1000:+.1f} | {report['attribution'][1]['groups'][i]['excess_ms']*1000:+.1f} |")
    lines += ['', 'Local image processing is the main source of this boundary’s excess. The 128/256-channel stages lead for Original; the 32/64-channel stages and input/reconstruction matter more for Lightweight. Global attention is a contributor, **not the dominant cause** of the measured step. The full per-launch ranking and grids are in the [data](native-scaling-profile.json).', '',
        'Stage-only timestamps retain the original dispatch batches. They put almost all of the increase inside the neural dispatch region; initial preparation remains below 3 µs, and the matching output copy changes by only a few microseconds. These separate diagnostic runs are not pooled with the plain timing sweep.', '',
        '| Model / stage | 1024 rows (ms) | 1032 rows (ms) |', '| --- | ---: | ---: |']
    for attribution in report['attribution']:
        for stage in ['engine_prepare', 'dispatch_region', 'output_copy']:
            a, b = attribution['deep_stages']
            lines.append(f"| {LABELS[attribution['model']]} / {stage.replace('_',' ')} | {a['stages'][stage]:.4f} | {b['stages'][stage]:.4f} |")
    lines += ['', '## Code and memory review', '',
        '- `src/native/model/generated_plan.h`: `Geometry` rounds the spatial image to 64/128-pixel boundaries, with a shifted-window margin, then rounds the latent axes to multiples of four. Those are real compute dimensions, not merely spare allocation bytes.',
        '- `src/native/model/native_layout.h` and the model registry: the full schedule stays at 154 launches for Original or 78 for Lightweight, with all eight global blocks and one pass. The Lightweight model removes local blocks, not global attention.',
        '- Production kernel entry selections are identical on both sides of the boundary. The bundle’s separate large-image entry threshold is 18,662,400 pixels, far above 1080p.',
        '- Driver queries show fixed register/local-storage sizes for the same compiled entries. There is no per-thread spill-size switch triggered by 1080p.',
        '- The GPU reports a 64 MiB L2 cache. The full stem skip grows from 60 to 67.5 MiB here, so cache behavior is a reasonable hypothesis, but this study has **no hardware cache-miss/bandwidth counters**. Crossing the nominal capacity is not proof that cache misses caused the step; other live tensors and streaming reuse also matter.',
        '- Measured per-process device-local residency near this boundary is roughly 500–575 MiB. This does not suggest VRAM exhaustion on this 16 GiB card; residency totals alone do not measure paging.', '',
        '## Focused register experiment: rejected', '',
        'Input preparation and reconstruction together cost about 0.5 ms around 1080p. Their current 128-register entries have 64/104 bytes of driver-reported local storage per thread. Local storage is not itself a measurement of spill traffic. We rebuilt only these two entries from the same PTX, verified the 128-register control has identical machine instructions, and tried 96, 160 and 192 registers.', '',
        '| Register limit for both boundary kernels | Original GPU ms | Lightweight GPU ms |', '| ---: | ---: | ---: |']
    for budget in [128, 96, 160, 192]:
        values = [next(row['gpu_ms'] for row in report['register_experiment']['summary'] if row['model'] == model and row['budget'] == budget) for model in MODELS]
        lines.append(f"| {budget}{' (production)' if budget==128 else ''} | {values[0]:.3f} | {values[1]:.3f} |")
    lines += ['', 'At 160 registers the input kernel’s local storage falls to zero and output storage to 24 bytes; at 192 both reach zero. However, active blocks per SM fall from 16 to 12 or 8, and whole-model latency gets worse. All variants preserve reset/history pixels, but **none were promoted**. Eliminating local storage is not automatically a performance improvement.', '',
        '## What is worth optimizing next', '',
        'The profile points toward local-window work (128/256 channels for Original; also 32/64 for Lightweight) and input/reconstruction: reduce redundant memory movement or improve tile reuse while preserving arithmetic and padded outputs. Blindly reducing the model’s padding, tokens, blocks or precision would change the computation and needs reference-equivalence evidence first. The current 128-register boundary budget remains the better measured choice.', '',
        '## Method and limits', '',
        'RTX 5080, driver 616.64; full Original and Lightweight models; fixed structured FP16 SDR input, zero motion, automatic masks and one pass. Eight extents × two models × two native measurement modes × two fresh processes, plus 16 NVIDIA reference processes: **80 processes**. A further **8** stage-only processes preserve dispatch batching and verify preparation/network/copy attribution; **24** processes test register alternatives. Each profiling process warms for one second and measures at least 40 frames for at least one second; register experiments use two-second warmup/measurement intervals and three process pairs. No games were launched or modified.', '',
        'This is a static-frame performance investigation, not a game-motion/image-quality study. Instrumented timings are diagnostic; plain timings determine total latency. [Machine-readable results](native-scaling-profile.json) · Runners: `tools/profile_native_scaling.py`, `tools/experiment_boundary_registers.py`, `tools/report_native_scaling.py`.', '']
    (ROOT / 'docs/NATIVE_SCALING_PROFILE.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Native scaling explanation and compact evidence written')


if __name__ == '__main__': main()
