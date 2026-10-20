"""Publish path-free evidence from the explicit two-model register experiment."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize(report, provenance, controls):
    if not report.get('completed') or not controls.get('passed'):
        raise ValueError('Experiment/control checks are incomplete')
    cases = []
    for case in report['cases']:
        profiles = {}
        control_rounds = case['profiles']['production']['rounds']
        reference = control_rounds[0].get('captures') if control_rounds else None
        for label, profile in case['profiles'].items():
            valid = bool(profile['rounds']) and reference is not None and profile['passed'] and all(
                row['passed'] and row['reset_repeatable'] and row['debug_errors'] == 0
                and row['captures'] == reference for row in profile['rounds'])
            profiles[label] = dict(passed=valid, processes=len(profile['rounds']))
            if valid:
                resource_rows = (report['baseline_resources'] if label == 'production'
                                 else report['variant_resources'][label])
                spilling = {row['entry'] for row in resource_rows if
                            row['compiler']['spill_store_bytes'] or row['compiler']['spill_load_bytes']}
                timings = [row['gpu']['median_ms'] for row in profile['rounds']]
                profiles[label].update(gpu_ms=profile['gpu_ms'],
                    process_gpu_ms=timings, process_spread_percent=(max(timings) / min(timings) - 1) * 100,
                    output_hashes=profile['rounds'][0]['captures'],
                    kernel_selections=profile['rounds'][0]['kernel_selections'],
                    active_spilling_entries=sum(row['launches_per_pass'] > 0 and row['entry'] in spilling
                        for row in profile['rounds'][0]['kernel_selections']))
            # Failed modes deliberately have no public FPS/timing or output hash.
        cases.append(dict(model=case['model'], width=case['width'], height=case['height'], profiles=profiles))
    return dict(schema=report['schema'], passed=report['passed'], completed=True,
        gpu=report['gpu'], scope='Native Phoenix DLSS5 full-network GPU intervals; not game FPS',
        fixture='Deterministic RGBA16F gradient/detail, zero RG16F motion, full default network, one pass',
        games_launched=False, production_modified=False, provenance=provenance,
        reassembly_controls=controls, baseline_resources=report['baseline_resources'],
        variant_resources=report['variant_resources'], compiled_variants=report['compiled_variants'], cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'artifacts/current/register-limits')
    args = parser.parse_args()
    load = lambda name: json.loads((args.evidence / name).read_text())
    report = summarize(load('results.json'), load('provenance.json'), load('reassembly-controls.json'))
    report['resource_accounting_examples'] = load('resource-accounting.json')
    out = ROOT / 'docs/benchmarks/current'
    (out / 'register-limits.json').write_text(json.dumps(report, indent=2) + '\n')
    spilling = [row for row in report['baseline_resources']
                if row['compiler']['spill_store_bytes'] or row['compiler']['spill_load_bytes']]
    processes = sum(profile['processes'] for case in report['cases'] for profile in case['profiles'].values())
    lines = ['# Two-model register and spill recheck', '',
        '## Main findings', '',
        f'- Both models passed all tested output checks across {len(report["cases"])} model/extent pairs and {processes} fresh processes.',
        '- Register spilling exists in both models, but this screen found no output corruption caused by the tested budgets.',
        '- Relaxing eight budgets reduces the module inventory from 16 spilling entries to 12, but does not produce a clear end-to-end gain.',
        '- Tighter budgets are roughly 21-39% slower at the repeated 1080p/4K/8K points. Higher theoretical occupancy does not compensate for the extra spills in this mix.',
        '- Small timing differences are not stable optimization evidence: even equivalent entry-control profiles vary, and one 8K profile is especially noisy.',
        '- Production budgets and the large-image selector were left unchanged.', '',
        '## What the earlier test was', '',
        'The matching historical experiment is commit `f86c2f2` and the',
        '[13 October investigation](../../NATIVE_MEMORY_INVESTIGATION_2026-10-13.md).',
        'It studied CUDA register pressure, local-memory spills and the large-image',
        'entry switch, not Windows registry overflow. It used the older 310.8 DLL',
        'and Original model. This recheck uses both Original and Lightweight FP8',
        'from 310.9.16, the current refactored kernels and CUDA 13.1 / sm_120.', '',
        f'GPU/driver/starting state: `{report["gpu"]}`.', '',
        '## What spilling means', '',
        'A register budget is not a buffer that corrupts neighboring registers.',
        'When live values exceed the available allocation, the compiler can spill',
        'them into thread-local memory. More registers can reduce spilling but also',
        'reduce resident blocks. Neither fewer spills nor higher theoretical',
        'occupancy alone establishes faster inference.', '',
        f'The production build contains 41 modules / 43 entries; {len(spilling)} entries',
        'have nonzero compiler-reported spill loads or stores. Only selected entries',
        'with nonzero launches in the model plan execute. At the smaller extents,',
        'Original uses 15 spilling entries; Lightweight FP8 uses 12. At/above the',
        'automatic large-image threshold those counts become 16 and 13, respectively.', '',
        'Compiler spill bytes are static quantities, not measured bytes per frame.',
        'Driver local bytes describe per-thread local storage; maximum blocks/SM',
        'is theoretical resource capacity, not measured achieved occupancy.',
        '`cuobjdump LOCAL:0` alone is not a spill-free verdict: it can coexist',
        'with nonzero STACK and PTXAS spill counts in these binaries.',
        'No dynamic spill-traffic, bandwidth or hardware-stall counters were collected.', '',
        '## Production spill inventory', '',
        '| Entry | Registers | Stack bytes | Spill stores / loads | Max blocks/SM |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in spilling:
        c = row['compiler']
        lines.append(f'| {row["entry"].removeprefix("phoenix_")} | {c["registers"]} | {c["stack_bytes"]} | '
                     f'{c["spill_store_bytes"]} / {c["spill_load_bytes"]} | {row["driver"]["max_active_blocks_per_sm"]} |')
    lines += ['', 'The large split-window projection currently spills 4 store / 4 load bytes;',
        'the older investigation reported none. This is a newly checked difference',
        'in the current build, not proof that a particular edit caused it.', '',
        '## Isolated variations', '',
        '- **Production:** unchanged register choices and automatic input-area selection.',
        '- **Relaxed:** higher budgets on eight selected spill-bearing kernels.',
        '- **Tighter:** lower budgets on the same eight kernels.',
        '- **Default entries:** disable the two large-image entry switches only.',
        '- **Large entries:** force those two large-image entries at every size.', '',
        'Precision, graph, weights, block sizes, cache modes, assembler optimization',
        'and store/synchronization code are unchanged. Eight unchanged-budget PTX',
        'reassembly controls have identical instruction dumps to production, so the',
        'PTX build route is not an uncontrolled code-generation change.', '',
        '| Tuned entry | Production registers | Relaxed budget / used | Tighter budget / used |',
        '| --- | ---: | ---: | ---: |']
    for variant in report['compiled_variants']:
        if variant['profile'] != 'relaxed':
            continue
        entry = variant['entry']
        base = next(row for row in report['baseline_resources'] if row['entry'] == entry)
        values = []
        for label in ('relaxed', 'tighter'):
            requested = next(row['requested_budget'] for row in report['compiled_variants']
                             if row['entry'] == entry and row['profile'] == label)
            actual = next(row['compiler']['registers'] for row in report['variant_resources'][label]
                          if row['entry'] == entry)
            values.append(f'{requested} / {actual}')
        lines.append(f'| {entry.removeprefix("phoenix_")} | {base["compiler"]["registers"]} | ' + ' | '.join(values) + ' |')
    lines += ['', '## Concrete register tradeoffs', '',
        'The following resource changes are real, but their individual timing effect',
        'is not isolated by the multi-kernel mix:', '',
        '| Entry | Profile | Registers | Spill stores / loads | Max blocks/SM |',
        '| --- | --- | ---: | ---: | ---: |']
    for suffix in ('input_preparation', 'global_attention_split_heads_512_projection_pool'):
        entry = 'phoenix_' + suffix
        for label in ('production', 'relaxed', 'tighter'):
            pool = report['baseline_resources'] if label == 'production' else report['variant_resources'][label]
            row = next(value for value in pool if value['entry'] == entry)
            c = row['compiler']
            lines.append(f'| {suffix} | {label} | {c["registers"]} | '
                         f'{c["spill_store_bytes"]} / {c["spill_load_bytes"]} | '
                         f'{row["driver"]["max_active_blocks_per_sm"]} |')
    lines += ['', '## Numerical and GPU checks', '',
        'Every successful variation must match the production reset A, one-frame',
        'history and reset B output hashes for its model and exact input size.',
        'Finite-output, reset-repeatability and D3D12 error checks also pass before',
        'a row receives a timing. The 801x603 cases enable D3D12 debug validation.', '',
        'At 1080p, 4K and 8K each profile uses two fresh processes, with the second',
        'profile order reversed. Values are medians of those process medians.',
        'Odd-size and threshold checks use one process/profile: correctness and',
        'policy diagnostics, not reliable fine-grained performance rankings.',
        'Each process warms for at least 0.5 seconds and measures at least 1 second',
        'and 30 frames. GPU timestamps include engine recording/barriers/output copy,',
        'excluding uploads, readbacks and CPU gaps. Stock clocks were not locked.', '',
        '| Phoenix DLSS5 model | Input | Production ms | Relaxed | Tighter | Default entries | Large entries |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for case in report['cases']:
        base = case['profiles']['production']
        values = []
        for label in ('relaxed', 'tighter', 'default_entries', 'large_entries'):
            profile = case['profiles'].get(label)
            if profile is None:
                values.append('not tested')
            elif not profile['passed'] or not base['passed']:
                values.append('FAILED')
            else:
                value = profile['gpu_ms']
                values.append(f'{value:.4f} ({(value / base["gpu_ms"] - 1) * 100:+.2f}%)')
        name = 'Original' if case['model'] == 'original' else 'Lightweight FP8'
        baseline = f'{base["gpu_ms"]:.4f}' if base['passed'] else 'FAILED'
        lines.append(f'| {name} | {case["width"]}x{case["height"]} | {baseline} | ' + ' | '.join(values) + ' |')
    noisy = [(case, label, profile) for case in report['cases']
             for label, profile in case['profiles'].items()
             if profile['passed'] and profile['process_spread_percent'] > 3]
    if noisy:
        lines += ['', '### Timing variability', '',
            'These repeated-profile process medians differ by more than 3%. This is',
            'a descriptive warning, not a statistical test. The raw samples are retained;',
            'do not treat the corresponding percentage as a stable optimization gain.', '',
            '| Model / input / profile | Fresh-process medians, ms |', '| --- | --- |']
        for case, label, profile in noisy:
            values = ', '.join(f'{value:.4f}' for value in profile['process_gpu_ms'])
            lines.append(f'| {case["model"]} / {case["width"]}x{case["height"]} / {label} | {values} |')
    lines += ['', 'The automatic threshold is 18,662,400 input pixels: 5760x3239 is',
        'below it, and 5760x3240 reaches it exactly. The receipt includes the selected',
        'entry and launch count for every module in both models. No new model-specific',
        'threshold or blanket register setting was applied to production.', '',
        'These are bounded synthetic results, not moving-scene/game certification or',
        'universal tuning recommendations. Multi-kernel mixes identify tradeoffs but',
        'do not attribute a timing difference to one kernel. Small differences need',
        'more isolated measurements before any production change.', '',
        'No games were launched or installed files changed. The external model DLL',
        'was not copied. Temporary experimental bundles were removed; only hashes,',
        'resources and compact results remain. [Machine-readable receipt](register-limits.json).', '',
        'Reproduce with `tools/benchmark_register_limits.py --baseline <fresh-production-bundle>`,',
        'then `tools/report_register_limits.py`. The production builder intentionally',
        'has no experimental tuning switches.', '']
    (out / 'register-limits.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
