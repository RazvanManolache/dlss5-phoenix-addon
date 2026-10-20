"""Explicit isolated register/spill experiments; never edit production kernel choices.

Build a fresh production bundle first. Variants live in one TemporaryDirectory;
only compact resource, numerical and timing receipts survive the experiment.
Static spill sizes are compiler quantities, not measured dynamic memory traffic.
"""
import argparse
import ctypes as ct
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from compare_four_models import ROOT, build, sha, telemetry, write
from production_build_native_kernels import configure_boundary_ptx
from benchmark_kernel_structure import compiled_resources

CUDA = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin')
TARGETS = ('input_preparation', 'output_reconstruction',
           'window_attention_channels_32_input_view', 'window_attention_channels_32_output_view',
           'window_attention_channels_64_attention', 'window_attention_channels_128_input_view',
           'window_attention_channels_256_upsample', 'global_attention_split_heads_512_projection_pool')


def spill_properties(text):
    """Parse each entry separately: a module can contain default and large entries."""
    pattern = (r'Function properties for (\S+)\s*\n\s*(\d+) bytes stack frame, '
               r'(\d+) bytes spill stores, (\d+) bytes spill loads\s*\n[^\n]*Used (\d+) registers')
    return {name: dict(zip(('stack_bytes', 'spill_store_bytes', 'spill_load_bytes', 'registers'),
                          map(int, values)))
            for name, *values in re.findall(pattern, text)}


def set_register_budget(ptx, entries, budget):
    """Override only named entry directives; never alter a kernel's instructions."""
    if not 32 <= budget <= 255:
        raise ValueError('Register budget outside bounded experiment range')
    for entry in entries:
        pattern = r'(\.visible\s+\.entry\s+' + re.escape(entry) + r'\s*\([^)]*\))([^\{]*)(\{)'
        def replace(match):
            directives, count = re.subn(r'(?m)^\s*\.maxnreg\s+\d+\s*$', '', match[2])
            if count > 1:
                raise ValueError('Duplicate register directive: ' + entry)
            return match[1] + directives.rstrip() + '\n.maxnreg ' + str(budget) + '\n' + match[3]
        ptx, count = re.subn(pattern, replace, ptx)
        if count != 1:
            raise ValueError('Missing/duplicate PTX entry: ' + entry)
    return ptx


class ResourceQuery:
    """CUDA Driver resource/occupancy queries only; this class launches no kernels."""
    def __init__(self):
        self.driver = ct.WinDLL('nvcuda.dll')
        self.functions = {}
        self.context = ct.c_void_p()
        self.device = ct.c_int()
        self.call('cuInit', [ct.c_uint], 0)
        self.call('cuDeviceGet', [ct.POINTER(ct.c_int), ct.c_int], ct.byref(self.device), 0)
        self.call('cuDevicePrimaryCtxRetain', [ct.POINTER(ct.c_void_p), ct.c_int],
                  ct.byref(self.context), self.device)
        self.call('cuCtxSetCurrent', [ct.c_void_p], self.context)

    def call(self, name, types, *args):
        function = self.functions.get(name)
        if function is None:
            function = getattr(self.driver, name)
            function.argtypes, function.restype = types, ct.c_int
            self.functions[name] = function
        result = function(*args)
        if result:
            raise RuntimeError(f'{name}: CUDA error {result}')

    def query(self, binary, entry, threads):
        module, function = ct.c_void_p(), ct.c_void_p()
        self.call('cuModuleLoad', [ct.POINTER(ct.c_void_p), ct.c_char_p],
                  ct.byref(module), str(binary).encode())
        try:
            self.call('cuModuleGetFunction', [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p],
                      ct.byref(function), module, entry.encode())
            result = {}
            # CUfunction_attribute values from the installed CUDA driver header.
            for name, attribute in (('registers', 4), ('static_shared_bytes', 1), ('local_bytes', 3)):
                value = ct.c_int()
                self.call('cuFuncGetAttribute', [ct.POINTER(ct.c_int), ct.c_int, ct.c_void_p],
                          ct.byref(value), attribute, function)
                result[name] = value.value
            blocks = ct.c_int()
            self.call('cuOccupancyMaxActiveBlocksPerMultiprocessor',
                      [ct.POINTER(ct.c_int), ct.c_void_p, ct.c_int, ct.c_size_t],
                      ct.byref(blocks), function, threads, 0)
            return result | dict(max_active_blocks_per_sm=blocks.value, dynamic_shared_bytes=0)
        finally:
            self.call('cuModuleUnload', [ct.c_void_p], module)

    def close(self):
        self.call('cuCtxSetCurrent', [ct.c_void_p], None)
        self.call('cuDevicePrimaryCtxRelease_v2', [ct.c_int], self.device)


def inventory(bundle, query):
    manifest = json.loads((bundle / 'native_manifest.json').read_text())
    rows = []
    for kernel in manifest['kernels']:
        binary = bundle / kernel['binary']
        properties = spill_properties((binary.parent / 'build.log').read_text())
        entries = [kernel['entry']] + ([kernel['large_input_entry']] if 'large_input_entry' in kernel else [])
        for entry in entries:
            if entry not in properties:
                raise RuntimeError('No compiler properties for ' + entry)
            runtime = query.query(binary, entry, math.prod(kernel['block']))
            if runtime['registers'] != properties[entry]['registers']:
                raise RuntimeError('Compiled/loaded register count mismatch: ' + entry)
            rows.append(dict(module=kernel['name'], entry=entry, binary_sha256=sha(binary),
                             block=kernel['block'], compiler=properties[entry], driver=runtime))
    return rows


def compile_variants(baseline, work, controls):
    catalogue = json.loads((ROOT / 'src/native/model/production_kernel_contracts.json').read_text())['kernels']
    manifest = json.loads((baseline / 'native_manifest.json').read_text())
    variants = {}
    for label in ('relaxed', 'tighter', 'default_entries', 'large_entries'):
        folder = work / label
        shutil.copytree(baseline, folder)
        variants[label] = folder
    compiled = []
    for suffix in TARGETS:
        entry = 'phoenix_' + suffix
        source = next(row for row in catalogue if row['entry'] == entry)
        kernel = next(row for row in manifest['kernels'] if row['entry'] == entry)
        current = next(row['compiler']['registers'] for row in controls if row['entry'] == entry)
        folder = work / ('ptx_' + suffix)
        folder.mkdir()
        ptx = folder / 'kernel.ptx'
        # Reuse the production recipe and MSVC environment, changing only its
        # output kind. Keep source, O3, architecture and assembler policy intact.
        recipe = (baseline / kernel['binary']).parent / 'build.cmd'
        command = recipe.read_text().splitlines()[-1]
        command = command.replace('-cubin ', '-ptx ')
        command = re.sub(r'-Xptxas=\S+\s+', '', command)
        command = re.sub(r' -o .+$', lambda _: ' -o ' + subprocess.list2cmdline([str(ptx)]), command)
        script = folder / 'build.cmd'
        script.write_text('\n'.join(recipe.read_text().splitlines()[:-1] + [command]) + '\n')
        result = subprocess.run(['cmd', '/d', '/c', str(script)], cwd=ROOT,
                                capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        options = source.get('ptx_options')
        if options:
            configure_boundary_ptx(ptx, 'sm_120', entry, options)
        original = ptx.read_text()
        budgets = dict(relaxed=min(224, math.ceil((current + 32) / 8) * 8),
                       tighter=max(96, math.ceil((current - 32) / 8) * 8))
        for label, budget in budgets.items():
            variant = variants[label]
            binary = variant / kernel['binary']
            altered = folder / (label + '.ptx')
            altered.write_text(set_register_budget(original, [entry], budget))
            result = subprocess.run([str(CUDA / 'ptxas.exe'), '-arch=sm_120',
                                     '-O' + str(source.get('assembler_optimization', 3)), '-v',
                                     '--suppress-debug-info', str(altered), '-o', str(binary)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            (binary.parent / 'build.log').write_text(result.stdout + result.stderr)
            compiled.append(dict(profile=label, entry=entry, requested_budget=budget,
                                 assembler_optimization=source.get('assembler_optimization', 3)))
        print('Compiled register experiments:', suffix, budgets, flush=True)
    for label, folder in variants.items():
        config = json.loads((folder / 'native_manifest.json').read_text())
        for row in config['kernels']:
            row['binary_sha256'] = sha(folder / row['binary'])
            if label in ('default_entries', 'large_entries') and 'large_input_entry' in row:
                large = row.pop('large_input_entry')
                row.pop('large_input_min_pixels')
                if label == 'large_entries':
                    row['entry'] = large
        write(folder / 'native_manifest.json', config)
    return variants, compiled


def verify_reassembly_controls(baseline, work):
    """Prove that the PTX route itself did not change the instruction bodies."""
    catalogue = json.loads((ROOT / 'src/native/model/production_kernel_contracts.json').read_text())['kernels']
    manifest = json.loads((baseline / 'native_manifest.json').read_text())
    rows = []
    for suffix in TARGETS:
        entry = 'phoenix_' + suffix
        source = next(row for row in catalogue if row['entry'] == entry)
        kernel = next(row for row in manifest['kernels'] if row['entry'] == entry)
        folder = work / ('ptx_' + suffix)
        binary = folder / 'control.cubin'
        result = subprocess.run([str(CUDA / 'ptxas.exe'), '-arch=sm_120',
            '-O' + str(source.get('assembler_optimization', 3)), '-v', '--suppress-debug-info',
            str(folder / 'kernel.ptx'), '-o', str(binary)], capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        dumps = [subprocess.check_output([str(CUDA / 'cuobjdump.exe'), '--dump-sass', str(path)])
                 for path in (baseline / kernel['binary'], binary)]
        rows.append(dict(entry=entry, identical=dumps[0] == dumps[1],
                         baseline_binary_sha256=sha(baseline / kernel['binary']),
                         control_binary_sha256=sha(binary),
                         sass_sha256=hashlib.sha256(dumps[1]).hexdigest()))
        binary.unlink()
    if not all(row['identical'] for row in rows):
        raise RuntimeError('PTX controls differ; register budgets are not the only changed variable')
    return dict(passed=True, kernels_launched=False, cases=rows)


def run_case(executable, forwarder, model, bundle, width, height, work):
    model_dll = ROOT.parent / 'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
    config = dict(width=width, height=height, engine='phoenix',
        weight_source=dict(kind='nvidia_dll', path=str(model_dll), sha256=sha(model_dll), model=model),
        bundle_path=str(bundle), forwarder=str(forwarder), cache=str(work / 'cache'),
        color_file=str(work / 'color.rgba16f'), motion_file=str(work / 'motion.rg16f'),
        capture=False, debug=width < 1920, warmup_seconds=.5, measure_seconds=1., report_kernel_selection=True)
    job, result_file = work / 'job.json', work / 'result.json'
    write(job, config)
    result_file.unlink(missing_ok=True)
    result = subprocess.run([str(executable), str(job), str(result_file)], cwd=work,
                            capture_output=True, text=True, timeout=180)
    if result.returncode or not result_file.exists():
        return dict(passed=False, error=result.stderr[-1500:], exit_code=result.returncode)
    report = json.loads(result_file.read_text())
    return {key: report[key] for key in ('passed', 'gpu', 'wall', 'captures', 'debug_errors',
            'reset_repeatable', 'kernel_selections', 'memory', 'neural_launches')}


def fixture(work, width, height):
    # Build one row at a time, avoiding several full-size float32 grids at 8K.
    pixels = np.memmap(work / 'color.rgba16f', mode='w+', dtype=np.float16, shape=(height, width, 4))
    x = np.arange(width, dtype=np.float32)
    for y in range(height):
        pixels[y, :, 0] = x / width
        pixels[y, :, 1] = np.float32(y) / height
        pixels[y, :, 2] = .5 + .25 * np.sin(x * .17) * np.cos(np.float32(y) * .11)
        pixels[y, :, 3] = 1
    pixels.flush()
    del pixels
    with (work / 'motion.rg16f').open('wb') as stream:
        row = bytes(width * 4)
        for _ in range(height):
            stream.write(row)


def provenance(work, baseline):
    """Bind receipts to actual probe/graph/model bytes without copying payloads."""
    files = [path for folder in (ROOT / 'src/native', ROOT / 'native_runtime')
             for path in folder.rglob('*') if path.suffix in ('.h', '.hpp', '.cpp', '.cuh', '.cu')]
    files += [ROOT / 'tests/native/four_model_probe.cpp', ROOT / 'src/integration/reshade/nr_forwarder.cpp']
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update((path.relative_to(ROOT).as_posix() + '\0' + sha(path) + '\n').encode())
    return dict(historical_commit='f86c2f20c211dc384a7755941e2a3a53541f6e96',
        source_snapshot_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_tree_is_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
        source_dependency_digest_sha256=digest.hexdigest(), source_files_hashed=len(files),
        probe_binary_sha256=sha(work / 'four_model_probe.exe'), forwarder_sha256=sha(work / 'phoenix-nvngx.dll'),
        model_dll_sha256=sha(ROOT.parent / 'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'),
        native_graph_sha256=sha(baseline / 'native_graph.json'), gpu_kernels_instrumented=False)


def resource_accounting_examples(baseline):
    manifest = json.loads((baseline / 'native_manifest.json').read_text())
    return {kernel['entry']: compiled_resources(baseline, kernel)[kernel['entry']]
            for kernel in manifest['kernels'] if kernel['entry'] in (
                'phoenix_input_preparation', 'phoenix_global_attention_split_heads_512_projection_pool')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/current/register-limits')
    args = parser.parse_args()
    baseline, destination = args.baseline.resolve(), args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / 'results.json'
    report = dict(schema='phoenix-register-experiments/1', passed=False, gpu=telemetry(),
                  games_launched=False, production_modified=False, cases=[])
    write(output, report)
    query = ResourceQuery()
    try:
        report['baseline_resources'] = inventory(baseline, query)
        write(destination / 'resource-accounting.json', resource_accounting_examples(baseline))
        write(output, report)
        with tempfile.TemporaryDirectory(prefix='register-experiment-', dir=ROOT / 'build') as folder:
            work = Path(folder)
            variants, report['compiled_variants'] = compile_variants(baseline, work, report['baseline_resources'])
            write(destination / 'reassembly-controls.json', verify_reassembly_controls(baseline, work))
            report['variant_resources'] = {label: inventory(path, query) for label, path in variants.items()}
            write(output, report)
            query.close()
            query = None
            executable, forwarder = build(work)
            write(destination / 'provenance.json', provenance(work, baseline))
            profiles = dict(production=baseline, **variants)
            for width, height in ((801, 603), (1920, 1080), (3840, 2160),
                                  (5760, 3239), (5760, 3240), (7680, 4320)):
                fixture(work, width, height)
                for model in ('original', 'small-fp8'):
                    # Use all policies at 1080p/4K/8K. Near the switch, isolate
                    # entry selection instead of repeating unrelated spill mixes.
                    labels = list(profiles) if width in (1920, 3840, 7680) else (
                        ['production', 'default_entries', 'large_entries'] if width == 5760
                        else ['production', 'relaxed', 'tighter'])
                    runs = {label: [] for label in labels}
                    rounds = 2 if width in (1920, 3840, 7680) else 1
                    reference = None
                    for repeat in range(rounds):
                        order = labels if repeat == 0 else list(reversed(labels))
                        for label in order:
                            result = run_case(executable, forwarder, model, profiles[label], width, height, work)
                            if label == 'production' and result['passed']:
                                reference = result['captures'] if reference is None else reference
                            if not result['passed']:
                                runs[label].append(result)
                                continue
                            runs[label].append(result)
                            print(model, width, height, label, repeat, result['gpu']['median_ms'], flush=True)
                    if reference is None:
                        raise RuntimeError('Production control failed')
                    case = dict(model=model, width=width, height=height, profiles={})
                    for label, values in runs.items():
                        good = all(row['passed'] and row['captures'] == reference for row in values)
                        case['profiles'][label] = dict(passed=good, rounds=values)
                        if good:
                            case['profiles'][label]['gpu_ms'] = statistics.median(row['gpu']['median_ms'] for row in values)
                    report['cases'].append(case)
                    write(output, report)
            report['passed'] = all(profile['passed'] for case in report['cases'] for profile in case['profiles'].values())
            report['completed'] = True
        report['temporary_variants_removed'] = True
        write(output, report)
    finally:
        if query is not None:
            query.close()


if __name__ == '__main__':
    main()
