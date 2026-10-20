"""Export allowlisted, path-free benchmark measurements; never launch inference."""
import hashlib
import json
from pathlib import Path

from compare_native_methods import ROOT, build_comparison

NAMES = {1: 'Native x64', 2: 'x64 library inside x86',
         3: 'x86 Phoenix DLSS5 + native driver', 4: 'External x64 helper'}
RENDERERS = {'phoenix': 'Phoenix DLSS5', 'nvidia': 'NVIDIA DLSS5'}


def main():
    names = ['wow64-method-benchmark.json', 'wow64-x86-host-benchmark.json',
             'dx12-method-benchmark.json']
    raw = [(ROOT/'artifacts/current'/name).read_bytes() for name in names]
    result = build_comparison(*(json.loads(value) for value in raw))
    result['source_receipt_sha256'] = {
        name: hashlib.sha256(value).hexdigest() for name, value in zip(names, raw)}
    result['schema'] = 'phoenix-public-execution-benchmarks/1'
    result['release'] = '0.81'
    result['measurement_context'] = 'Pre-release execution experiments; not a rerun of release binaries.'
    for row in result['rows']:
        method = row.pop('method')
        row['execution'] = NAMES[method]
        if method == 2:
            row['wall_scope'] = 'Resident-input synchronized API call; x64 worker entered once, not per-frame x86 handoff.'
        elif method == 1:
            row['wall_scope'] = 'Resident-input synchronized API call.'
        elif method == 3:
            row['wall_scope'] = 'Resident processing plus per-frame x86/native dispatch handoff; no host image transfer.'
        if not row['available']:
            row['reason'] = 'No x86 NVIDIA DLSS5 graph host source; use the complete x64 library inside x86 or the helper.'
    result['limitations'] = [
        'Wall-time scopes differ; do not treat the helper/resident ratio as architectural overhead.',
        'Resident tests use synthetic pixels; the helper uses a different DX12 fixture image.',
        'Resident calls are collected individually; helper latency uses 41 status-poll samples.',
        'Single serial run per case; no confidence interval or statistical overhead claim.',
        'Completed output throughput is not game FPS.',
        'New add-on native x86 paths also stage CPU pixels; their end-to-end latency is not measured here.']
    output = ROOT/'docs/benchmarks/0.81'
    output.mkdir(parents=True, exist_ok=True)
    (output/'measurements.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    lines = ['# 0.81 Execution Measurements', '',
        'RTX 5080, driver 616.64; 1920 x 1080; 1:1; full network; reset frames.',
        'Each case: 5 seconds warmup, then 20 seconds measurement. These retained',
        'pre-release measurements were not rerun on the release binaries.', '',
        '**Not game FPS. GPU time excludes CPU capture/readback/IPC. Wall-time scopes differ.**', '',
        '| Renderer | Execution | GPU mean / p95 (ms) | Wall mean / p95 (ms) | Timing samples | Completed outputs/s |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in result['rows']:
        if row['available']:
            lines.append(f"| {RENDERERS[row['backend']]} | {row['execution']} | "
                f"{row['gpu_mean_ms']:.3f} / {row['gpu_p95_ms']:.3f} | "
                f"{row['wall_mean_ms']:.3f} / {row['wall_p95_ms']:.3f} | "
                f"{row['samples']} | {row['completed_outputs_per_second']:.2f} |")
        else:
            lines.append(f"| {RENDERERS[row['backend']]} | {row['execution']} | Not available | Not available | - | - |")
    lines += ['', '## Read the Scopes', '',
        '- **Native x64 / x64 library inside x86:** GPU-resident synchronized API calls.',
        '  The x86 experiment enters its x64 worker once, not once per frame.',
        '- **x86 Phoenix DLSS5 + native driver:** resident images with a real per-frame',
        '  x86 dispatch-packet handoff; no host-image transfer.',
        '- **External x64 helper:** actual x86 DX12 fixture capture/readback, IPC,',
        '  model processing and return. Latencies are status-poll sampled; output',
        '  throughput is counted separately, not the inverse of sampled latency.', '',
        *['- '+value for value in result['limitations']], '',
        'The GPU column includes route-specific model/output work; Phoenix DLSS5 native',
        'probes include output copy. Do not combine the historical October 14 VRAM',
        'study with these timings as if they were one matched measurement.', '',
        '![GPU processing time only](execution-gpu-time.png)', '',
        '[Machine-readable measurements and receipt hashes](measurements.json).',
        'Raw receipts remain local because they include host paths and diagnostics.',
        'Regenerate this export with `python tools/export_release_benchmarks.py`',
        'when the three validated local receipts are available. No GPU is used.', '']
    (output/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    draw_chart(result, output/'execution-gpu-time.png')
    print(json.dumps(dict(passed=True, rows=len(result['rows']), gpu_used=False)))


def draw_chart(result, path):
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new('RGB', (1200, 650), '#15191c')
    draw = ImageDraw.Draw(canvas)
    def font(size):
        for candidate in ('C:/Windows/Fonts/segoeui.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
            if Path(candidate).is_file():
                return ImageFont.truetype(candidate, size)
        return ImageFont.load_default(size=size)
    draw.text((34, 22), 'Phoenix DLSS5 0.81 | GPU processing by execution path', font=font(28), fill='white')
    draw.text((34, 66), 'RTX 5080 | 1920 x 1080 | 5s warmup + 20s measurement | reset frames', font=font(19), fill='#c8cfd3')
    colors = {'phoenix': '#52c6ad', 'nvidia': '#e0b962'}
    for index, label in enumerate(NAMES.values()):
        y = 128 + index*105
        draw.text((34, y+12), label, font=font(17), fill='white')
        for backend, delta in (('phoenix', 0), ('nvidia', 34)):
            row = next(row for row in result['rows'] if row['execution'] == label and row['backend'] == backend)
            if row['available']:
                end = 365+row['gpu_mean_ms']*140
                draw.rectangle((365, y+delta, end, y+delta+25), fill=colors[backend])
                draw.text((end+12, y+delta-2), f"{row['gpu_mean_ms']:.3f} ms", font=font(18), fill='white')
            else:
                draw.text((365, y+delta), 'NVIDIA DLSS5: no x86 graph-host implementation', font=font(17), fill='#a7b1b8')
    draw.text((34, 562), 'Phoenix DLSS5', font=font(20), fill=colors['phoenix'])
    draw.text((220, 562), 'NVIDIA DLSS5', font=font(20), fill=colors['nvidia'])
    draw.text((34, 600), 'GPU-only bars exclude capture/IPC/return. Different input fixtures; not game FPS.', font=font(19), fill='#c8cfd3')
    canvas.save(path)


if __name__ == '__main__':
    main()
