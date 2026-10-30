"""Publish compact, path-free portable optimization evidence and its limits."""
import hashlib
import json
from pathlib import Path
import re
import statistics
from test_model_selection import ROOT


def main():
    source = ROOT / 'artifacts/current/portable-tuning'
    read = lambda name: json.loads((source / name).read_text(encoding='utf-8'))
    operators = read('cpu-operators.json')
    operator_rows = []
    for line in operators['output'].splitlines()[1:]:
        values = dict(re.findall(r'(\w+)=([\d.]+)', line))
        operator_rows.append({key: float(value) for key, value in values.items()})
    conditioning = read('frame-conditioning.json')
    frame_rows = []
    for index, case in enumerate(conditioning['results']['control'][0]['cases']):
        row = {key: case[key] for key in ['extent', 'spatial_mask', 'auto_mask', 'sha256']}
        for label in ['control', 'candidate']:
            row[label + '_ms'] = statistics.median(run['cases'][index]['median_ms']
                for run in conditioning['results'][label])
        frame_rows.append(row)
    files = ['gpu-1080p.json', 'cpu-amd-320.json', 'cpu-amd-1080p.json', 'amd-fallback-320.json']
    full_rows = []
    for name in files:
        result = read(name)
        if not result['passed']:
            raise RuntimeError('Incomplete benchmark: ' + name)
        for row in result['summary']:
            samples = [sample for sample in result['measurements']
                       if sample['route'] == row['route'] and sample['model'] == row['model']]
            fields = ['resident_weight_bytes', 'resident_scratch_bytes', 'backend_allocation_bytes']
            allocations = {tuple(sample['initial_status'].get(key) for key in fields) for sample in samples}
            if len(allocations) != 1:
                raise RuntimeError('Graph-owned allocation sizes changed')
            full_rows.append(dict(extent=result['extent'], paired_processes=result['paired_processes'],
                                  scalar_matrices=result.get('scalar_matrices', False),
                                  graph_owned_memory_unchanged=True,
                                  graph_owned_bytes=dict(zip(fields, allocations.pop())), **row))
    maintained = ['native_runtime/cpu_avx2.cpp', 'native_runtime/cpu_avx512.cpp',
                  'native_runtime/cpu_left_tile.h', 'native_runtime/frame.h',
                  'native_runtime/cuda_narrow_math.cuh', 'native_runtime/backend_gpu.cpp',
                  'src/backends/vulkan/gemm_portable_tiled.comp']
    report = dict(schema='phoenix-portable-reuse/1', date='2026-10-30',
        hardware=dict(cpu='Ryzen 9 9950X3D, 32 worker threads for whole-model CPU runs',
                      nvidia='RTX 5080, driver 616.64', amd='integrated AMD Radeon Graphics'),
        scope='Phoenix DLSS5 portable backends; not the native RTX instruction kernels or NVIDIA DLSS5',
        decisions=dict(cpu_operand_reuse='retained', host_conditioning_reuse='retained',
                       vulkan_coalesced_word_loads='retained for scalar fallback',
                       cuda_weight_decoder='discarded: no robust improvement; prototype remains test-only'),
        model_dll_sha256=read(files[0])['model_sha256'],
        exact_cpu_layout_cases=240, exact_packed_input_cases=conditioning['exact_cases'],
        cpu_operators=operator_rows, packed_input_preparation=frame_rows, full_model=full_rows,
        unavailable=['Metal GPU execution/Apple compiler', 'HIP SDK/device execution', 'SYCL SDK/device execution'],
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in maintained})
    matrix_path = source / 'resident-matrices.json'
    if matrix_path.exists():
        matrices = read(matrix_path.name)
        report['resident_matrices'] = {key: matrices[key] for key in ['passed', 'scope', 'pairs', 'summary']}
        report['resident_matrices']['minimum_measure_ms'] = 80
        report['resident_matrices']['processes'] = [dict(route=process['route'], variant=process['variant'],
            pair=process['pair'], rows=[dict(shape=[row[key] for key in ['m', 'n', 'k']],
                median_ms=row['median_ms'], measured_calls=len(row['samples_ms']),
                output_sha256=row['output_sha256'], owned_bytes=row['owned_bytes']) for row in process['rows']])
            for process in matrices['measurements']]
    (ROOT / 'docs/portable-reuse.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = ['# Portable backend reuse study', '',
        'These changes apply to **Phoenix DLSS5** portable execution. The native RTX instruction kernels and NVIDIA DLSS5 runtime are separate paths; their published timings are unchanged.', '',
        'Measured on an RTX 5080, Ryzen 9 9950X3D and its integrated Radeon GPU. No older discrete AMD GPU, Mac, HIP SDK or SYCL SDK was available for this study.', '',
        '## What changed', '',
        '- AVX2 loads eight consecutive half values per row and transposes an 8×8 tile in registers. The matrix loop retains its ascending reduction/FMA order.',
        '- AVX-512 retains each gathered pair of halves for the next reduction step instead of gathering it twice.',
        '- Shared host frame preparation encodes uniform conditioning values once per frame. Spatial maps still override tone and structure per pixel; signed zero, padding and history inputs remain exact.',
        '- A register-only CUDA FP8-to-half decoder was tested, then **discarded from production**: the measured GPU benefit was negligible. Its legacy ±480 codes deliberately differ from E4M3FN NaN codes. The verified prototype remains isolated in the CUDA math test.',
        '- Vulkan tests aligned four-byte weight loads in the coalesced scalar fallback. Cooperative-matrix execution uses a different shader and already groups its weight reads.', '',
        'These are bounded register/tile reuse changes. They add no expanded per-model weight cache, change no model parameters, and retain precision/publication boundaries.', '',
        '## Whole-model measurements', '',
        'Full Original and Lightweight networks, one pass, all eight transformer blocks, identical structured SDR input, fixed seed and zero motion. Each process warms independently, then checks reset, history and reset-again output hashes. Every paired control/candidate hash matched.', '',
        'CUDA and NVIDIA Vulkan at 1080p and CPU/AMD at 320×180 use three alternating process pairs. The CPU/AMD 1080p scaling check uses **one pair only**: useful validation, not a stable performance estimate. Numbers below are full host-call medians, including preparation, transfers and resolve, rather than isolated GPU inference.', '',
        '| Backend | Input | Model | Before (ms) | After (ms) | Host-call reduction | Process pairs |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    labels = dict(cpu32='CPU (32 workers)', cuda='CUDA', vulkan_nvidia='NVIDIA Vulkan', vulkan_amd='AMD Vulkan')
    for row in full_rows:
        backend = labels[row['route']] + (' (scalar fallback)' if row['scalar_matrices'] else '')
        lines.append(f"| {backend} | {'×'.join(map(str,row['extent']))} | {'Original' if row['model']=='original' else 'Lightweight'} | {row['control']['wall_ms']:.2f} | {row['candidate']['wall_ms']:.2f} | {row['wall_reduction_percent']:+.1f}% | {row['paired_processes']} |")
    lines += ['', 'A positive reduction means less time; small differences should be read against run-to-run variation. GPU event times and preparation times are retained in the [compact data](portable-reuse.json). Do not interpret noisy host-call differences as a GPU-kernel speedup. The CUDA candidate measurements include the decoder experiment that was subsequently discarded; production CUDA GPU execution remains unchanged.', '',
        '## Isolated CPU work', '',
        'Seven alternating timing pairs per resident matrix shape, with at least 40 ms per measurement. The operator test covers transposed/non-transposed FP8 and half weights, batch offsets, padded strides and incomplete row/column/reduction tiles. All 240 before/after layout checks are bit-exact.', '',
        '| Loader | Reduction width | Before (ms) | After (ms) | Speedup |', '| --- | ---: | ---: | ---: | ---: |']
    for row in operator_rows:
        lines.append(f"| {'AVX2' if row['rows_per_vector']==8 else 'AVX-512'} | {int(row['k'])} | {row['control_ms']:.4f} | {row['candidate_ms']:.4f} | {row['speedup']:.2f}× |")
    lines += ['', 'The shared host input-packing test uses three process pairs and nine measured calls per process/case after warmup. All 12 packed-input cases match exactly, including odd extents, automatic masks, spatial maps and signed-zero controls.', '',
        '| Input packing (auto mask on) | Spatial map | Before (ms) | After (ms) |', '| --- | --- | ---: | ---: |']
    for row in frame_rows:
        if row['extent'][0] > 100 and row['auto_mask']:
            lines.append(f"| {'×'.join(map(str,row['extent']))} | {'Yes' if row['spatial_mask'] else 'No'} | {row['control_ms']:.2f} | {row['candidate_ms']:.2f} |")
    lines += ['', 'This is the CPU host feature packer. Backends that advertise device feature packing bypass it; it does not make their GPU network faster. Metal, HIP and SYCL can benefit when they use this shared host fallback, but no device performance claim is made for them here.', '',
        '## GPU fallback experiments', '',
        'Three alternating process pairs per shape, two warmups and at least 80 ms of measured resident matrix calls per process/shape. All outputs match exactly. These are isolated operations, not whole-frame latency.', '',
        'The longer sampling window removed large outliers seen with the old three-call microbenchmark. CUDA was effectively flat (about 1.00–1.03×), so its decoder change was not promoted. The direct FP8 CUDA path already loads packed bytes.', '',
        '| AMD scalar fallback matrix (M×N×K) | Before (ms) | After (ms) | Speedup |',
        '| --- | ---: | ---: | ---: |']
    if matrix_path.exists():
        for row in report['resident_matrices']['summary']:
            if row['route'] == 'amd_scalar':
                lines.append(f"| {'×'.join(map(str,row['shape']))} | {row['control_ms']:.3f} | {row['candidate_ms']:.3f} | {row['speedup']:.2f}× |")
    lines += ['', 'The six larger eligible shapes improve by roughly 12–14% in elapsed time; the four other tested shapes remain close to unchanged. Default AMD/NVIDIA cooperative-matrix execution already groups weight loads and does not use this scalar fallback shader.', '',
        '## Validation and limits', '',
        '- CPU, CUDA, NVIDIA Vulkan and AMD Vulkan operator contracts passed, including 1,354,743 finite conversion comparisons per backend.',
        '- CUDA narrow-math tests passed 6,205,420 comparisons with no differences, including all 256 legacy FP8 encodings.',
        '- No model DLLs, SDKs, game files or frame dumps are included in the data. Games and installed configurations were untouched.',
        '- The known-broken fused Vulkan feed-forward optimization remains disabled by default.',
        '- Graph-owned weight, scratch and total backend allocation sizes matched before/after in every whole-model case. This is an ownership check, not a measurement of driver-private caches or system-wide RAM.',
        '- Generated Metal arithmetic passed host simulation, including 10,000 FP8 oracle cases. This is distinct from an Apple compiler/device validation. HIP and SYCL device compilation/execution remain unverified on this machine.', '',
        '## Reproduce', '',
        'Use independently compiled, model-free control/candidate runtimes with an external model DLL. The maintained runners are `tools/test_cpu_operand_reuse.py`, `tools/test_frame_conditioning_reuse.py`, `tools/benchmark_portable_reuse.py` and `tools/benchmark_reuse_matrices.py`. CPU loaders and the frame header before this study are available in base commit `a151512`; pass a read-only copy of those files to the differential runners. Use `--scalar-matrices` for the forced Vulkan fallback comparison. Device UUIDs are discovered locally and omitted from this public report.', '']
    (ROOT / 'docs/PORTABLE_REUSE_STUDY.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Portable report and compact measurements updated')


if __name__ == '__main__':
    main()
