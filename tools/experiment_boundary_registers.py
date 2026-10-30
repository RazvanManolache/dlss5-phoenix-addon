"""Profile-led input/output register study; production sources stay unchanged."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
import numpy as np
from benchmark_readme_case import fixture
from benchmark_register_limits import ResourceQuery
from compare_four_models import ROOT, build, sha, telemetry, write
from experiment_native_tuning import MODEL, CUDA
from production_build_native_kernels import configure_boundary_ptx
from test_model_selection import run, toolchain


def sass_instructions(binary):
    text = subprocess.check_output([str(CUDA / 'cuobjdump.exe'), '-arch', 'sm_120', '--dump-sass', str(binary)], text=True)
    return re.findall(r'/\*[0-9a-f]+\*/\s+(.*)', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/current/native-scaling-profile/boundary-registers.json')
    args = parser.parse_args()
    base = ROOT / 'artifacts/current/native-kernel-styles'
    manifest = json.loads((base / 'native_manifest.json').read_text(encoding='utf-8'))
    catalogue = json.loads((ROOT / 'src/native/model/production_kernel_contracts.json').read_text(encoding='utf-8'))['kernels']
    targets = [row for row in catalogue if row.get('ptx_options')]
    if len(targets) != 2:
        raise RuntimeError('Expected exactly the two frame-boundary kernels')
    report = dict(passed=False, state='building', extent=[1920, 1080], rounds=3, source_modified=False,
                  model_sha256=sha(MODEL), gpu_before=telemetry(), resources=[], control_proof=[], cases=[], summary=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save(): write(args.output, report)
    save()
    _, env = toolchain(64)
    with tempfile.TemporaryDirectory(prefix='boundary-registers-', dir=ROOT / 'build') as temporary:
        work = Path(temporary).resolve()
        if work.parent != (ROOT / 'build').resolve(): raise RuntimeError('Unsafe cleanup root')
        bundles = {}
        for budget in [96, 160, 192]:
            folder = work / str(budget); folder.mkdir()
            config = copy.deepcopy(manifest)
            files = {'native_graph.json'} | {row['binary'] for row in config['kernels']}
            files |= {row['binary'] for value in config.get('kernel_styles', {}).values() for row in value['kernels']}
            for name in files:
                target = folder / name; target.parent.mkdir(parents=True, exist_ok=True)
                os.link(base / name, target)
            bundles[budget] = (folder, config)
        for target in targets:
            raw_ptx = work / (target['entry'] + '.ptx')
            run([CUDA / 'nvcc.exe', '--entries', target['entry'], '-allow-unsupported-compiler',
                 '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', '-diag-suppress', '177', '-diag-suppress', '550',
                 '-ptx', '-arch=sm_120', '-O3', '-std=c++17', '-lineinfo', target['source'], '-o', raw_ptx],
                cwd=ROOT, env=env, timeout=300)
            original = raw_ptx.read_text(encoding='utf-8')
            spec = next(row for row in manifest['kernels'] if row['name'] == target['name'])
            for budget in [128, 96, 160, 192]:
                ptx, binary = work / f'{target["entry"]}-{budget}.ptx', work / f'{target["entry"]}-{budget}.cubin'
                ptx.write_text(original, encoding='utf-8')
                configure_boundary_ptx(ptx, 'sm_120', target['entry'], dict(texture_mode='independent', max_registers=budget))
                run([CUDA / 'ptxas.exe', '-arch=sm_120', '--entry', target['entry'],
                     '-O' + str(target.get('assembler_optimization', 3)), '-v', '--suppress-debug-info',
                     ptx, '-o', binary], cwd=work, env=env, timeout=300)
                query = ResourceQuery()
                try: properties = query.query(binary, target['entry'], math.prod(spec['block']))
                finally: query.close()
                report['resources'].append(dict(entry=target['entry'], budget=budget, **properties))
                if budget == 128:
                    baseline, candidate = sass_instructions(base / spec['binary']), sass_instructions(binary)
                    if not baseline or baseline != candidate:
                        raise RuntimeError('128-register reassembly differs from production instructions')
                    report['control_proof'].append(dict(entry=target['entry'], same_sass=True,
                        instruction_sha256=hashlib.sha256('\n'.join(candidate).encode()).hexdigest()))
                else:
                    folder, config = bundles[budget]
                    row = next(row for row in config['kernels'] if row['name'] == target['name'])
                    output = folder / row['binary']; output.unlink(); shutil.copy2(binary, output)
                    row['binary_sha256'] = sha(output)
                    row['architecture_images'] = [dict(architecture='sm_120', sha256=sha(output), bytes=output.stat().st_size)]
            print('Compiled boundary register variants:', target['entry'], flush=True)
            save()
        for folder, config in bundles.values(): write(folder / 'native_manifest.json', config)
        executable, forwarder = build(work)
        width, height = report['extent']
        rgba = np.ones((height, width, 4), np.float16); rgba[..., :3] = fixture(width, height)
        (work / 'color.rgba16f').write_bytes(rgba.tobytes())
        (work / 'motion.rg16f').write_bytes(np.zeros((height, width, 2), np.float16).tobytes())
        report['state'] = 'running'; save()
        for model in ['original', 'small-fp8']:
            expected = None
            for repeat in range(3):
                order = [128, 96, 160, 192] if repeat % 2 == 0 else [192, 160, 96, 128]
                for budget in order:
                    config = dict(width=width, height=height, engine='phoenix',
                        weight_source=dict(kind='nvidia_dll', path=str(MODEL), sha256=report['model_sha256'], model=model),
                        bundle_path=str(base if budget == 128 else bundles[budget][0]), native_kernel_style='auto',
                        forwarder=str(forwarder), color_file=str(work / 'color.rgba16f'), motion_file=str(work / 'motion.rg16f'),
                        capture=False, debug=False, warmup_seconds=2., measure_seconds=2., minimum_measured_frames=40)
                    write(work / 'job.json', config)
                    output = work / 'result.json'; output.unlink(missing_ok=True)
                    process = subprocess.run([str(executable), str(work / 'job.json'), str(output)],
                                             cwd=work, capture_output=True, text=True, timeout=120)
                    result = json.loads(output.read_text(encoding='utf-8')) if output.exists() else {}
                    if process.returncode or not result.get('passed'):
                        raise RuntimeError(result.get('error', process.stderr))
                    hashes = {k: v['sha256'] for k, v in result['captures'].items()}
                    if expected is None: expected = hashes
                    if hashes != expected: raise RuntimeError('Register change altered temporal outputs')
                    if not all(value['finite'] for value in result['captures'].values()): raise RuntimeError('Nonfinite output')
                    report['cases'].append(dict(model=model, budget=budget, round=repeat,
                        gpu=result['gpu'], wall=result['wall'], captures=hashes, memory=result['memory']))
                    print(model, budget, f"{result['gpu']['median_ms']:.5f} ms", flush=True); save()
            for budget in [128, 96, 160, 192]:
                rows = [r for r in report['cases'] if r['model'] == model and r['budget'] == budget]
                report['summary'].append(dict(model=model, budget=budget,
                    gpu_ms=statistics.median(r['gpu']['median_ms'] for r in rows),
                    process_medians_ms=[r['gpu']['median_ms'] for r in rows]))
        report.update(passed=True, state='completed', gpu_after=telemetry()); save()
    report['temporary_builds_removed'] = True; save()


if __name__ == '__main__': main()
