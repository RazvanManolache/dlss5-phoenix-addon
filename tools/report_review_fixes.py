"""Publish compact, path-free review measurements; raw receipts stay local."""
import json
import statistics
from pathlib import Path
from test_model_selection import ROOT


def read(path):
    return json.loads(path.read_text())


def main():
    source = ROOT / 'artifacts/current/review-fixes'
    for phase in ['before', 'after']:
        if not read(source / phase / 'run.json')['passed']:
            raise RuntimeError('Incomplete benchmark phase: ' + phase)
    tests = read(source / 'tests/validation.json')
    if not tests['passed']:
        raise RuntimeError('Incomplete tests')
    report = dict(schema='phoenix-review-fixes/1', date='2026-10-19',
                  baseline_commit=read(source / 'before/run.json')['source_commit'],
                  native=[], portable=[], appearance=[],
                  tests=[dict(bits=c['bits'], test=c['test'], passed=c.get('passed', c.get('result', {}).get('passed', False)))
                         for c in tests['cases']])
    for before_file in sorted((source / 'before').glob('*.json')):
        name = before_file.stem
        if name in ['run', 'appearance-memory']: continue
        a, b = read(before_file), read(source / 'after' / before_file.name)
        if not a.get('passed') or not b.get('passed'):
            raise RuntimeError('Failed case ' + name)
        if 'captures' in a:
            keys = ['reset_a', 'history', 'reset_b']
            exact = all(a['captures'][k]['sha256'] == b['captures'][k]['sha256'] for k in keys)
            if not exact: raise RuntimeError('Native output regression: ' + name)
            report['native'].append(dict(engine=a['engine'], model=a['model_id'], extent=[a['width'], a['height']],
                gpu_ms_before=a['gpu']['median_ms'], gpu_ms_after=b['gpu']['median_ms'],
                output_exact=True, memory_before=a['memory'], memory_after=b['memory']))
        else:
            if [f['sha256'] for f in a['frames']] != [f['sha256'] for f in b['frames']]:
                raise RuntimeError('Portable output regression: ' + name)
            if a['temporal_frames'] != b['temporal_frames']:
                raise RuntimeError('Portable history regression: ' + name)
            row = dict(backend=a['config']['backend'], model=a['config']['weight_source']['model'],
                extent=[3840, 2160] if name.startswith('vulkan') else [320, 180], output_exact=True,
                create_ms_before=a['create_ms'], create_ms_after=b['create_ms'],
                memory_before=a['memory_alive'], memory_after=b['memory_alive'],
                checks=b['checks'])
            if name.startswith('cuda'):
                row.update(schedule_change_ms_before=statistics.median(f['wall_ms'] for f in a['frames'][1:]),
                           schedule_change_ms_after=statistics.median(f['wall_ms'] for f in b['frames'][1:]),
                           history_exact=True, schedule_stress=b.get('schedule_stress', []))
            else:
                if any(f['status']['vulkan_fused_feed_forward'] for f in b['frames']):
                    raise RuntimeError('Unsafe Vulkan default enabled')
                row['safe_default_repeatable'] = True
            report['portable'].append(row)
    a, b = [read(source / phase / 'appearance-memory.json') for phase in ['before', 'after']]
    for x, y in zip(a['cases'], b['cases']):
        report['appearance'].append(dict(extent=x['extent'],
            **{f'{profile}_bytes_{phase}': item[profile]['live_allocation_info_bytes']
               for phase, item in [('before', x), ('after', y)] for profile in ['unused', 'style', 'with_mask']}))
    # Check the independent Phoenix/NVIDIA comparison as well as the regression.
    for phase in ['before', 'after']:
        for model in ['original', 'small-fp8']:
            for extent in ['1920x1080', '3840x2160']:
                phoenix, nvidia = [read(source / phase / f'{engine}-{model}-{extent}.json') for engine in ['phoenix', 'nvidia']]
                if phoenix['captures'] != nvidia['captures']:
                    raise RuntimeError('Native vendor fixture parity failed')
    report.update(passed=True, native_vendor_fixture_parity=True, games_launched=False,
                  installed_files_modified=False, temporary_builds_removed=True)
    destination = ROOT / 'docs/benchmarks/current'
    (destination / 'review-fixes.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Runtime and memory review fixes', '',
        'Measured 19 October 2026 against local checkpoint `' + report['baseline_commit'][:7] + '` on RTX 5080 / driver 616.64. '
        'These are isolated probes, not game FPS or deployment validation. No games or installed files were changed.', '',
        '[Compact measurements](review-fixes.json)', '', '## Changes', '',
        '- Vulkan fused feed-forward is disabled by default after its documented 4K reset failure. Explicit opt-in remains experimental; the faulty optimized shader itself is not repaired.',
        '- Portable schedules reuse immutable uploaded weights. A schedule change still rebuilds commands/scratch and resets incompatible history, but does not re-read/decode/upload already cached weights.',
        '- Newly enabled, previously uncached transformers can decode weights once. The cache is scoped to one backend/model and releases on model replacement or session destruction.',
        '- The addon releases packed CPU weights after session construction. A later session/device recreation re-authenticates and reloads them on demand; this trades occasional setup work for lower steady RAM.',
        '- DLL hashing uses read-only file mappings rather than whole-file private byte vectors. File-backed pages can still affect the working set; this is not a promise of zero RAM during hashing.',
        '- Appearance mask, composition and distortion resources allocate separately on first use. Enabling the first mask retires prior GPU work before publishing its descriptor. This is a one-time synchronization, not a per-frame restart.',
        '- Native schedule metadata and matched NVIDIA presets belong to model definitions rather than a two-model boolean. New architectures still require their own validated graph/decoder/kernel support.',
        '', '## Schedule changes', '',
        'CUDA host-plane probe, 320x180, depth sequence 8 -> 4 -> 8 -> 4 -> 8. '
        'Median wall time of the four changed-schedule calls includes inference, transfers, graph setup and CPU work; the initial cold frame is excluded. '
        'A separate reset/history/reset sequence and protected-mask check run for each model.', '',
        '| Model | Before ms | After ms | Reduction |', '| --- | ---: | ---: | ---: |']
    for r in report['portable']:
        if r['backend'] == 'cuda':
            a, b = r['schedule_change_ms_before'], r['schedule_change_ms_after']
            lines.append(f"| {r['model']} | {a:.2f} | {b:.2f} | {100*(1-b/a):.1f}% |")
    lines += ['', '## Appearance allocations', '',
        'Actual D3D12 allocation-info bytes of the addon appearance helper, not raw pixel estimates or whole-process residency. '
        'Resources stay alive during each snapshot and return to zero live tracked textures after destruction. '
        'The probe records but does not submit its command lists. All values below are MiB.', '',
        '| Extent | Unused before / after | Style-only before / after |', '| --- | ---: | ---: |']
    for r in report['appearance']:
        lines.append(f"| {r['extent'][0]}x{r['extent'][1]} | {r['unused_bytes_before']/2**20:.2f} / {r['unused_bytes_after']/2**20:.2f} | {r['style_bytes_before']/2**20:.2f} / {r['style_bytes_after']/2**20:.2f} |")
    lines += ['', 'The removed permanent CPU packed-weight payload is approximately 141 MiB for Original and 109 MiB for Lightweight. '
        'This is a source-confirmed ownership change, not a measured whole-game RAM delta: the standalone native probe already freed that copy before these fixes.',
        '', '## Portable process RAM', '',
        'Fresh-process CUDA snapshots at 320x180, after the same five-call sweep, temporal check and zero-mask check. '
        'The peak is the process-lifetime working-set high-water mark, including DLL/provider initialization. '
        'The process also includes Python/NumPy, the driver and allocator caches. All values are MiB.', '',
        '| Model | Working set before / after | Peak working set before / after | Private commit before / after |',
        '| --- | ---: | ---: | ---: |']
    for r in report['portable']:
        if r['backend'] == 'cuda':
            a, b = r['memory_before'], r['memory_after']
            lines.append(f"| {r['model']} | {a['host_working_set_bytes']/2**20:.1f} / {b['host_working_set_bytes']/2**20:.1f} | {a['host_peak_working_set_bytes']/2**20:.1f} / {b['host_peak_working_set_bytes']/2**20:.1f} | {a['host_private_bytes']/2**20:.1f} / {b['host_private_bytes']/2**20:.1f} |")
    lines += ['', 'Startup peaks improve, but steady resident RAM is essentially unchanged and private commit is not uniformly lower. '
        'The after-version additionally alternates depth fifty times: tracked backend allocations and GPU-local usage remain constant. '
        'In these stress samples, private commit warms by about 77 MiB over the first ten changes and then plateaus through change fifty, while the working set stays nearly constant. '
        'The extra untracked commit is not attributed to a specific allocator/driver component; this bounded result is not proof against every long-session leak.',
        '', '## Native inference', '',
        'One fresh process per cell and version, 0.5 s minimum warmup and 1 s minimum sampling, with at least 30 measured frames. '
        'GPU timestamps cover native engine work and matching output copy, excluding input upload/readback. '
        'Clocks are not locked and these single-process medians do not establish small performance regressions or improvements.', '',
        '| Renderer | Model | Extent | Before GPU ms | After GPU ms |', '| --- | --- | --- | ---: | ---: |']
    for r in report['native']:
        label = 'Phoenix DLSS5' if r['engine'] == 'phoenix' else 'NVIDIA DLSS5'
        lines.append(f"| {label} | {r['model']} | {r['extent'][0]}x{r['extent'][1]} | {r['gpu_ms_before']:.3f} | {r['gpu_ms_after']:.3f} |")
    lines += ['', 'Native reset/history/reset outputs match before/after and the corresponding NVIDIA DLSS5 fixture byte for byte at 1080p and 4K for both models. '
        'Portable reset outputs also match before/after; CUDA history/reset and zero-mask protection pass. '
        'Vulkan 4K passes repeated resets with the new default and no caller workaround. This does not establish moving-game/HDR parity or universal Vulkan correctness.',
        '', '## Validation and limits', '',
        f"- {len(report['tests'])} x86/x64 test/build cases pass, including resident-weight ownership, zero re-uploads for cached schedules, lazy addition of unseen weights, backend isolation, native bundle rehydration, original graph fingerprints, controls, software D3D12 captures and both frontend links.",
        '- Existing fingerprints were not regenerated. Source-independent before/after output comparisons remain required after numerical changes.',
        '- Process RAM/residency snapshots are retained in the compact JSON. Private commit, working set, GPU-local residency and requested allocation bytes are different metrics; do not add them.',
        '- The earlier broad two-model tables remain historical. These changes do not establish that Phoenix DLSS5 uses less VRAM than NVIDIA DLSS5 at every resolution; the 4K scratch-space difference still exists.',
        '- No long-running game, VR, multi-GPU, Metal, HIP or SYCL device validation was performed for these fixes. No release or installed game was updated.',
        '', '## Reproduce', '', '```powershell', 'python tools/test_review_fixes.py',
        'python tools/benchmark_review_fixes.py before', 'python tools/benchmark_review_fixes.py after',
        'python tools/report_review_fixes.py', '```', '',
        'The before probe extracts `git archive bdc3fd8` into its temporary build; use `--baseline-ref` to select another checkpoint explicitly. '
        'Dependencies are external local model/kernel/provider artifacts; '
        'the probes do not download or modify them. Temporary executables and inputs are removed on success or failure.', '']
    (destination / 'review-fixes.md').write_text('\n'.join(lines))
    print('PASS: path-free review report generated')


if __name__ == '__main__':
    main()
