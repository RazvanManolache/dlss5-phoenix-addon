"""Keep generated benchmark headlines short; detailed evidence lives in docs."""
from pathlib import Path
import posixpath
import re
from urllib.parse import urlsplit, urlunsplit

START = '<!-- current-two-model-benchmark -->'
END = '<!-- /current-two-model-benchmark -->'
PERFORMANCE_SUMMARY = (
    '**Performance on par with NVIDIA DLSS5 at 1080p on the tested RTX 5080.** '
    'Both Original and Lightweight have comparable isolated inference latency; '
    'this is not end-to-end game FPS or portable-backend performance.')


def relative_links(text, source, destination):
    """Rebase Markdown/HTML links when moving prose; keep shell examples intact."""
    def target(value):
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or value.startswith(('#', '/')):
            return value
        path = posixpath.normpath(posixpath.join(posixpath.dirname(source), parsed.path))
        rebased = posixpath.relpath(path, posixpath.dirname(destination) or '.')
        return urlunsplit(('', '', rebased, parsed.query, parsed.fragment))
    def prose(value):
        value = re.sub(r'(\]\()([^\s)]+)(\))', lambda m: m[1] + target(m[2]) + m[3], value)
        return re.sub(r'((?:href|src)=["\'])([^"\']+)', lambda m: m[1] + target(m[2]), value)
    chunks = re.split(r'(^```[^\n]*\n.*?^```[^\n]*(?:\n|$))', text, flags=re.M | re.S)
    return ''.join(chunk if index % 2 else prose(chunk) for index, chunk in enumerate(chunks))


def benchmark_block(get, date, metric):
    """Keep latency and memory together, with full accounting in the linked report."""
    lines = [START, '## Native RTX: the main event', '', performance_summary(get, metric), '',
        '| Model at 1080p | Phoenix DLSS5 GPU ms | NVIDIA DLSS5 GPU ms | Phoenix DLSS5 VRAM MiB | NVIDIA DLSS5 VRAM MiB |',
        '| --- | ---: | ---: | ---: | ---: |']
    for model, label in [('original', 'Original'), ('small-fp8', 'Lightweight FP8')]:
        phoenix, nvidia = [get(route, model)['gpu'][metric] for route in ['phoenix', 'nvidia']]
        local = [get(route, model)['memory']['device_local_bytes'] / 2**20
                 for route in ['phoenix', 'nvidia']]
        lines.append(f'| {label} | {phoenix:.3f} | {nvidia:.3f} | {local[0]:.1f} | {local[1]:.1f} |')
    # Working set measures resident RAM; private commitment is a different metric.
    ram = {model: ' / '.join(f"{get(route, model)['memory']['host_working_set_bytes'] / 2**20:.1f}"
                            for route in ['phoenix', 'nvidia'])
           for model in ['original', 'small-fp8']}
    lines += ['', '**RAM gets a seat at the table too.** Resident RAM (Phoenix DLSS5 / NVIDIA DLSS5): '
              f"**{ram['original']} MiB for Original**, **{ram['small-fp8']} MiB for Lightweight**. "
              'VRAM is GPU-local residency; both are process snapshots including harness/driver overhead, '
              'not whole-game totals or peak requirements. '
              '[Memory details](docs/benchmarks/current/README.md#memory-at-1080p).']
    lines += ['', f'Measured {date}; full graphs, one pass. Small margins vary between runs; '
              'higher-resolution differences, including Original at 4K, are covered in the full report.', '',
              '[Performance guide](docs/PERFORMANCE.md) · '
              '[Measurements and methodology](docs/benchmarks/current/README.md) · '
              '[1080p scaling investigation](docs/NATIVE_SCALING_PROFILE.md)', END]
    return '\n'.join(lines)


def performance_summary(get, metric):
    """Do not preserve a parity headline when a later measurement changes it."""
    if all(abs(get('phoenix', model)['gpu'][metric] / get('nvidia', model)['gpu'][metric] - 1) <= .05
           for model in ['original', 'small-fp8']):
        return PERFORMANCE_SUMMARY
    return ('**Native RTX measurements for both models on RTX 5080.** '
            'Compare the measured GPU times below; these are isolated processing '
            'times, not end-to-end game FPS or portable-backend performance.')


def replace_block(text, block):
    if text.count(START) != 1 or text.count(END) != 1:
        raise RuntimeError('Benchmark markers must each occur once')
    before, rest = text.split(START)
    _, after = rest.split(END)
    return before + block + after


def update_front_page(root, get, date, metric):
    path = Path(root) / 'README.md'
    path.write_text(replace_block(path.read_text(encoding='utf-8'), benchmark_block(get, date, metric)), encoding='utf-8')
