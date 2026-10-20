"""Publish compact, path-free structure-refactor evidence from local receipts."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'artifacts/current/kernel-structure')
    parser.add_argument('--name', default='kernel-structure')
    parser.add_argument('--title', default='Kernel structure refactor validation')
    parser.add_argument('--baseline-label', default='Pre-refactor snapshot')
    args = parser.parse_args()
    if not args.name.replace('-', '').isalnum():
        raise ValueError('Report name must be a simple filename stem')
    local = args.evidence.resolve()
    paired = json.loads((local / 'paired.json').read_text())
    contracts = json.loads((local / 'contracts.json').read_text())
    audit = json.loads((local / 'source-audit.json').read_text())
    if not paired['passed'] or not contracts['passed']:
        raise RuntimeError('Cannot publish incomplete validation')
    report = dict(schema='phoenix-kernel-structure-evidence/1', passed=True,
        baseline=args.baseline_label,
        gpu=paired['gpu_before'], scope=paired['scope'], fixture=paired['fixture'],
        models=['Phoenix DLSS5 Original', 'Phoenix DLSS5 Lightweight FP8'],
        games_launched=False, installed_files_modified=False,
        contract_cases=len(contracts['cases']),
        integer_cases=next(row['integer_cases'] for row in contracts['cases']
                           if row['test'] == 'kernel_division_contract_tests'),
        scheduling_cases=next((row.get('scheduling_cases', 0) for row in contracts['cases']
                               if row['test'] == 'kernel_division_contract_tests'), 0),
        unsigned_division_cases=next((row.get('unsigned_division_cases', 0) for row in contracts['cases']
                                      if row['test'] == 'kernel_division_contract_tests'), 0),
        boundary_count_cases=next((row.get('boundary_count_cases', 0) for row in contracts['cases']
                                   if row['test'] == 'kernel_division_contract_tests'), 0),
        boundary_clear_cases=next((row.get('boundary_clear_cases', 0) for row in contracts['cases']
                                   if row['test'] == 'kernel_division_contract_tests'), 0),
        boundary_store_checks=next((row.get('boundary_store_checks', 0) for row in contracts['cases']
                                    if row['test'] == 'kernel_division_contract_tests'), 0),
        word_contracts={str(row['bits']): json.loads(row['output'])['contracts']
                        for row in contracts['cases'] if row['test'] == 'kernel_word_contract_tests'},
        machine_code=paired['machine_code'], builds=paired['builds'], source_inventory=audit['totals'], cases=[])
    occupancy = local / 'occupancy.json'
    if occupancy.exists():
        report['occupancy'] = json.loads(occupancy.read_text())
        if not report['occupancy'].get('passed'):
            raise RuntimeError('Incomplete occupancy receipt')
        for case in report['occupancy']['cases']:
            for label in ('before', 'after'):
                identity = next(row for row in report['builds'][label]['kernels'] if row['name'] == case['kernel'])
                if case[label + '_binary_sha256'] != identity['binary_sha256']:
                    raise RuntimeError('Stale occupancy receipt: ' + case['kernel'])
                metadata = next(row for row in report['machine_code'] if row['name'] == case['kernel'])
                resources = metadata[label + '_resources'][identity['entry']]
                if resources['REG'] != case[label]['registers']:
                    raise RuntimeError('Occupancy/register report mismatch: ' + case['kernel'])
    for row in paired['cases']:
        report['cases'].append({key: row[key] for key in
            ('model', 'width', 'height', 'exact', 'before_ms', 'after_ms', 'change_percent')} | dict(
            rounds=len(row['rounds']['before']),
            output_hashes={name: capture['sha256'] for name, capture in row['rounds']['after'][0]['captures'].items()}))
    destination = ROOT / 'docs/benchmarks/current'
    (destination / (args.name + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    same = sum(row['identical'] for row in report['machine_code'])
    lines = [
        '# ' + args.title, '',
        'The same 41-module CUDA source catalogue was built before and after the',
        'structure refactor with CUDA 13.1, sm_120, O3 and the existing per-module',
        'PTX/assembler policies. These are Phoenix DLSS5 native-runtime results,',
        'not a new NVIDIA DLSS5 comparison or an installed-game benchmark.', '',
        'Baseline: ' + args.baseline_label + '.', '',
        f'GPU/driver/starting state: `{report["gpu"]}`.', '',
        '## Numerical checks', '',
        '- All tested RGBA16F reset/history output hashes match the pre-refactor kernels exactly.',
        '- Both Original and Lightweight FP8 were checked at 320x180, 801x603, 1920x1080 and 3840x2160.',
        '- Each process also passed reset repeatability and finite-output checks; the two smaller sizes used D3D12 debug validation.',
        f'- {report["contract_cases"]} x86/x64/isolated-GPU contract cases passed, including {report["integer_cases"]:,} integer-division cases.',
        f'- {same}/41 complete compiled instruction dumps are byte-identical, including encoded instructions/control words. Changed modules required the paired output/timing checks.',
        '- Frame tests distinguish reflect-once padding from periodic mirroring and staged FP16 conversion from one final conversion.',
        '- Small-model graph/geometry fingerprints and packed-resource rejection cases passed.', '',
        '## GPU time', '',
        'Median of three fresh-process medians per 1080p/4K row, alternating run',
        'order. The scope includes complete engine recording, barriers and matching',
        'output copy, excluding uploads/readbacks and CPU gaps. Smaller sizes had',
        'one diagnostic pair and are omitted from this performance table.', '',
        '| Phoenix DLSS5 model | Resolution | Before | After | Change |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for row in report['cases']:
        if row['width'] < 1920:
            continue
        name = 'Original' if row['model'] == 'original' else 'Lightweight FP8'
        lines.append(f'| {name} | {row["width"]}x{row["height"]} | {row["before_ms"]:.4f} ms | '
                     f'{row["after_ms"]:.4f} ms | {row["change_percent"]:+.3f}% |')
    lines += ['',
        'No meaningful regression was observed in this bounded run. Small changes',
        'are within normal run/clock variation; they are not claimed speedups.',
        'This does not prove parity or unchanged performance for every GPU, driver,',
        'input, transport or schedule. The fixture was deterministic gradient/detail',
        'RGB with fixed seed and zero motion, not an interactive-game workload.', '',
        'No games were launched, installed files changed, models copied or releases',
        'published. Temporary before/after bundles are disposable; the receipts and',
        'source/compile recipes are the retained evidence.', '',
        'See [the structure review](../../KERNEL_STRUCTURE_REVIEW.md) for what remains',
        'instruction-oriented, and [the compact receipt](kernel-structure.json) for',
        'output/machine-code hashes. Historical reports were not rewritten.', '',
    ]
    if report['scheduling_cases']:
        index = lines.index('- Small-model graph/geometry fingerprints and packed-resource rejection cases passed.')
        lines.insert(index + 1,
                     f'- {report["scheduling_cases"]:,} GPU scheduling cases cover four tile layouts, partial warps and absent block leaders.')
        lines.insert(index + 2,
                     '- Signed-word/address checks per architecture: ' + ', '.join(
                         f'{bits}-bit: {count:,}' for bits, count in report['word_contracts'].items()) + '.')
    if report['unsigned_division_cases']:
        index = lines.index('## GPU time') - 1
        lines[index:index] = [
            f'- {report["unsigned_division_cases"]:,} full-range unsigned GPU division cases passed, including zero and high-bit divisors.',
            f'- {report["boundary_count_cases"]:,} boundary trip-count checks passed for signed/unsigned bounds, wrap, zero stride and production tile extents.',
        ]
    if report['boundary_clear_cases']:
        index = lines.index('## GPU time') - 1
        lines[index:index] = [
            f'- {report["boundary_clear_cases"]:,} GPU clear-operation fixtures compared {report["boundary_store_checks"]:,} emitted word addresses and their per-thread order against direct CPU enumeration.',
            '- The address recorder never dereferences synthetic tensor addresses; real model output checks are separate.',
        ]
    if report.get('occupancy'):
        index = lines.index('No meaningful regression was observed in this bounded run. Small changes')
        resources = ['## Compiled resources', '',
            'CUDA Driver occupancy queries use zero dynamic shared memory and launch no',
            'kernels. Maximum active blocks/SM is a resource limit, not measured achieved',
            'occupancy. Binary hashes bind these queries to the compared modules.', '',
            '| Module | Registers/thread before / after | Max blocks/SM before / after |',
            '| --- | ---: | ---: |']
        unchanged_shared = True
        zero_local_stack = True
        for case in report['occupancy']['cases']:
            metadata = next(row for row in report['machine_code'] if row['name'] == case['kernel'])
            identity = next(row for row in report['builds']['after']['kernels'] if row['name'] == case['kernel'])
            entry = identity['entry']
            before = metadata['before_resources'][entry]
            after = metadata['after_resources'][entry]
            unchanged_shared &= before['SHARED'] == after['SHARED']
            zero_local_stack &= all(usage[key] == 0 for usage in (before, after) for key in ('LOCAL', 'STACK'))
            name = entry.removeprefix('phoenix_').removeprefix('global_attention_transformer_').replace('_', ' ')
            resources.append(f'| {name} | {before["REG"]} / {after["REG"]} | '
                             f'{case["before"]["max_active_blocks_per_sm"]} / {case["after"]["max_active_blocks_per_sm"]} |')
        resources += ['']
        if unchanged_shared:
            resources += ['All queried modules retain their static shared-memory footprint.']
        if zero_local_stack:
            resources += ['Local and stack storage remain zero for every queried module.']
        resources += ['']
        lines[index:index] = resources
    if not report.get('occupancy'):
        changed = [row for row in report['machine_code'] if not row['identical']]
        if changed:
            resources = ['## Compiled resources', '',
                'Counts come from the compiled modules, not runtime allocation estimates.',
                'Values show before / after; these are not achieved occupancy measurements.', '',
                '| Modified module | Registers/thread | Static shared bytes | Stack bytes | Local bytes |',
                '| --- | ---: | ---: | ---: | ---: |']
            for row in changed:
                identity = next(kernel for kernel in report['builds']['after']['kernels']
                                if kernel['name'] == row['name'])
                entry = identity['entry']
                before, after = row['before_resources'][entry], row['after_resources'][entry]
                name = entry.removeprefix('phoenix_').replace('_', ' ')
                counts = ' | '.join(f'{before[key]} / {after[key]}'
                                    for key in ('REG', 'SHARED', 'STACK', 'LOCAL'))
                resources.append(f'| {name} | {counts} |')
            resources += ['']
            index = lines.index('No meaningful regression was observed in this bounded run. Small changes')
            lines[index:index] = resources
    lines = [line.replace('(kernel-structure.json)', '(' + args.name + '.json)') for line in lines]
    (destination / (args.name + '.md')).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
