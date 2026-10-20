"""Inventory instruction-oriented source debt without inventing tensor semantics."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ('check_vector_address', 'initialize_copy_barrier', 'copy_global_to_shared',
           'arrive_copy_barrier', 'wait_copy_barrier', 'broadcast_warp_word',
           'vote_warp_predicate', 'broadcast_active_lane', 'signed_tile_quotient',
           'unsigned_word_quotient', 'boundary_tail_steps',
           'prepare_unsigned_word_divisor', 'prepared_unsigned_word_quotient',
           'boundary_item_address', 'clear_boundary_vector', 'clear_boundary_rows',
           'token_tile_work', 'copy_warp_schedule', 'tile_copy_schedule', 'load_four_words_cached',
           'load_four_words_global_cached', 'store_four_words_streaming')


def audit(root=ROOT):
    production = root / 'src/native/kernels/production'
    rows = []
    definitions = {name: [] for name in HELPERS}
    scheduling_sources = []
    for path in sorted(production.rglob('*')):
        if path.suffix not in ('.cu', '.cuh', '.h'):
            continue
        source = path.read_text(encoding='utf-8')
        relative = path.relative_to(root).as_posix()
        if path.suffix == '.cu' and 'tile_copy_schedule<shared_bytes,' in source:
            scheduling_sources.append(relative)
        rows.append(dict(file=relative, lines=len(source.splitlines()),
            indexed_word_declarations=len(re.findall(r'uint32_t\s+\w*word_\d+\s*=', source)),
            indexed_matrix_declarations=len(re.findall(r'uint32_t\s+matrix_(?:accumulator|fragment)_\d+\s*=', source)),
            inline_division_biases=(len(re.findall(r'\b0x0?ffffffeu\b', source))
                                    if path.suffix == '.cu' else 0)))
        for name in HELPERS:
            if re.search(r'__device__\s+__forceinline__\s+(?:void|bool|uint32_t|uint64_t|TokenTileWork|CopyWarpSchedule|UnsignedWordDivisor)\s*' + name + r'\(', source):
                definitions[name].append(relative)
    return dict(schema='phoenix-kernel-structure-audit/1',
        note='Counts locate review work, not bugs or proprietary-source claims. Indexed matrix fragments can be appropriate.',
        kernel_modules=sum(path.suffix == '.cu' for path in production.rglob('*.cu')),
        totals={key: sum(row[key] for row in rows) for key in
                ('lines', 'indexed_word_declarations', 'indexed_matrix_declarations', 'inline_division_biases')},
        helper_definitions=definitions, scheduling_sources=scheduling_sources, files=rows)


def check_contracts(result, root=ROOT):
    for name, definitions in result['helper_definitions'].items():
        # Four scalar loads/stores deliberately remain distinct from uint4.
        expected = 2 if name in HELPERS[-3:] else 1
        if len(definitions) != expected:
            raise RuntimeError(f'{name}: expected {expected} deliberate definitions, found {definitions}')
    catalogue = json.loads((root / 'src/native/model/production_kernel_contracts.json').read_text())
    blocks = {row['source']: row['block'] for row in catalogue['kernels']}
    for source in result['scheduling_sources']:
        if source not in blocks or len(blocks[source]) != 3:
            raise RuntimeError(f'{source}: copy schedule has no valid catalogue block')
        if blocks[source][2] != 1:
            raise RuntimeError(f'{source}: copy schedule requires blockDim.z=1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = audit()
    if args.check:
        check_contracts(result)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    else:
        if args.check:
            print(json.dumps(dict(passed=True, kernel_modules=result['kernel_modules'], totals=result['totals'])))
        else:
            print(text, end='')


if __name__ == '__main__':
    main()
