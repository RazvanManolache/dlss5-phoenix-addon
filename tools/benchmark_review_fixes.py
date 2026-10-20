"""Isolated, sequential before/after review probes; never installs or launches games."""
import argparse
import ctypes as C
import hashlib
import json
import subprocess
import tempfile
import io
import tarfile
import shutil
import time
from pathlib import Path

import numpy as np
from benchmark_readme_case import fixture
from compare_four_models import build
from compare_neural_controls import configure, status, advanced_request, Frame3
from test_model_selection import ROOT, toolchain, run


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['before', 'after'])
    parser.add_argument('--portable-only', action='store_true')
    parser.add_argument('--baseline-ref', default='bdc3fd8', help='Immutable pre-fix checkpoint')
    args = parser.parse_args()
    output = ROOT / 'artifacts/current/review-fixes' / args.phase
    output.mkdir(parents=True, exist_ok=True)
    dependencies = ROOT / 'build/four_model_comparison/readme_runtime'
    model = ROOT.parent / 'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
    source = dict(kind='nvidia_dll', path=str(model), sha256=sha(model))
    receipt = dict(phase=args.phase, passed=False, games_launched=False, cases=[],
                   source_commit=subprocess.check_output(['git', 'rev-parse', args.baseline_ref if args.phase == 'before' else 'HEAD'], cwd=ROOT, text=True).strip())
    (output / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (ROOT / 'build').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='review-fixes-', dir=ROOT / 'build') as temporary:
        work = Path(temporary)
        compiler, env = toolchain(64)
        runtime = work / 'phoenix_runtime_cpp.dll'
        source_root = ROOT
        if args.phase == 'before':
            source_root = work / 'checkpoint'
            source_root.mkdir()
            archive = subprocess.check_output(['git', 'archive', args.baseline_ref], cwd=ROOT)
            with tarfile.open(fileobj=io.BytesIO(archive)) as source_archive:
                source_archive.extractall(source_root, filter='data')
        run([compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/O2', '/EHsc', '/MD', '/DNOMINMAX',
             '/LD', source_root / 'native_runtime/runtime.cpp', '/Fo:' + str(work / 'runtime.obj'),
             '/Fe:' + str(runtime), 'bcrypt.lib'], cwd=work, env=env)
        receipt['runtime_sha256'] = sha(runtime)
        # Compare the real addon appearance helper too, not only Network's
        # smaller standalone allocation ledger. The list is never submitted.
        memory_source = source_root / 'tests/native/appearance_memory_probe.cpp'
        if args.phase == 'before':
            shutil.copy2(ROOT / 'tests/native/appearance_memory_probe.cpp', memory_source)
        executable = work / 'appearance_memory_probe.exe'
        run([compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/O2', '/EHsc', '/MD', '/DNOMINMAX',
             '/I' + str(ROOT.parent / 'external/DLSS/include'), '/I' + str(ROOT / 'third_party/nvapi'),
             memory_source, '/Fo:' + str(work / 'appearance.obj'), '/Fe:' + str(executable),
             'd3d12.lib', 'dxgi.lib', 'd3dcompiler.lib', 'bcrypt.lib'], cwd=work, env=env)
        memory = json.loads(run([executable, args.phase], cwd=work, env=env))
        (output / 'appearance-memory.json').write_text(json.dumps(memory, indent=2) + '\n')
        if not args.portable_only:
            executable, forwarder = build(work, source_root)
            receipt['native_probe_sha256'] = sha(executable)
            cores = list(Path('C:/Windows/System32/DriverStore/FileRepository').glob('nv_*.inf_amd64_*/_nvngx.dll'))
            if len(cores) != 1:
                raise RuntimeError('An unambiguous installed NGX core is required')
            for w, h in [(1920, 1080), (3840, 2160)]:
                rgba = np.ones((h, w, 4), np.float16)
                rgba[..., :3] = fixture(w, h)
                color, motion = work / 'input.rgba16f', work / 'motion.rg16f'
                color.write_bytes(rgba.tobytes())
                motion.write_bytes(np.zeros((h, w, 2), np.float16).tobytes())
                for engine in ['phoenix', 'nvidia']:
                    for model_id in ['original', 'small-fp8']:
                        name = f'{engine}-{model_id}-{w}x{h}'
                        config = dict(width=w, height=h, engine=engine, weight_source={**source, 'model': model_id},
                            bundle_path=str(ROOT / 'artifacts/current/native-kernels'), ngx_core=str(cores[0]),
                            forwarder=str(forwarder), cache=str(work / 'ngx-cache'),
                            color_file=str(color), motion_file=str(motion), capture=False,
                            debug=False, warmup_seconds=.5, measure_seconds=1.)
                        job = work / 'job.json'
                        job.write_text(json.dumps(config))
                        result = subprocess.run([str(executable), str(job), str(output / (name + '.json'))],
                                                capture_output=True, text=True, timeout=180, cwd=work)
                        if result.returncode:
                            raise RuntimeError(result.stderr + result.stdout)
                        receipt['cases'].append(name)
                        print(result.stdout.strip(), flush=True)
        # Run the portable half in a child: loaded Windows DLLs must be released
        # before TemporaryDirectory removes the build, including on test failure.
        for backend in ['cuda', 'vulkan']:
            for model_id in ['original', 'small-fp8']:
                job = work / 'portable-job.json'
                job.write_text(json.dumps(dict(runtime=str(runtime), dependencies=str(dependencies),
                    source=source, output=str(output), phase=args.phase, backend=backend, model=model_id)))
                result = subprocess.run([__import__('sys').executable, __file__, '--portable-child', str(job)],
                                        capture_output=True, text=True, timeout=300, cwd=ROOT)
                print(result.stdout, flush=True)
                if result.returncode:
                    raise RuntimeError(result.stderr)
    receipt.update(passed=True, temporary_build_removed=True)
    (output / 'run.json').write_text(json.dumps(receipt, indent=2) + '\n')


def portable(job):
    api = configure(Path(job['runtime']))
    memory_api = C.CDLL(str(Path(job['dependencies']) / 'readme_memory.dll'))
    memory_api.phoenix_benchmark_memory.argtypes = [C.c_uint]
    memory_api.phoenix_benchmark_memory.restype = C.c_char_p
    def memory(): return json.loads(memory_api.phoenix_benchmark_memory(0x10de))
    for backend, extent in [('cuda', (320, 180)), ('vulkan', (3840, 2160))]:
        w, h = extent
        for model_id in ['original', 'small-fp8']:
            if (backend, model_id) != (job['backend'], job['model']): continue
            route = 'vulkan_nvidia' if backend == 'vulkan' else backend
            previous = json.loads((ROOT / f'artifacts/current/readme-benchmarks/cases/{route}-{model_id}-1920x1080-default-p0.json').read_text())
            config = dict(backend=backend, device=previous['runtime_status']['device'], runtime_root=job['dependencies'],
                          allow_unvalidated=True,
                          weight_source={**job['source'], 'model': model_id})
            # The before baseline uses the documented correctness workaround.
            # After intentionally omits it to check the new safe default.
            if job['phase'] == 'before':
                config['vulkan_fused_feed_forward'] = False
            before = memory()
            started = time.perf_counter()
            session = api.phoenix_runtime_create(json.dumps(config).encode(), w, h)
            create_ms = (time.perf_counter() - started) * 1000
            if not session:
                raise RuntimeError(api.phoenix_runtime_error().decode())
            try:
                image, output = fixture(w, h), np.empty((h, w, 3), np.float32)
                rows = []
                depths = [8, 4, 8, 4, 8] if backend == 'cuda' else [8, 8, 8]
                for depth in depths:
                    start = time.perf_counter()
                    code = api.phoenix_runtime_frame(session, image.ctypes.data, None, None, None,
                        json.dumps(dict(reset=True, transformer_blocks=depth)).encode(), output.ctypes.data)
                    elapsed = (time.perf_counter() - start) * 1000
                    if code:
                        raise RuntimeError(api.phoenix_runtime_error().decode())
                    rows.append(dict(depth=depth, wall_ms=elapsed, sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                                     status=status(api, session)))
                repeatable = all(len({r['sha256'] for r in rows if r['depth'] == d}) == 1 for d in set(depths))
                if not repeatable or not np.isfinite(output).all():
                    raise RuntimeError('Portable reset output is not repeatable/finite')
                temporal = []
                checks = {}
                stress = []
                alive = None
                if backend == 'cuda':
                    motion = np.zeros((h, w, 2), np.float32)
                    for reset in [True, False, True]:
                        code = api.phoenix_runtime_frame(session, image.ctypes.data, motion.ctypes.data, None, None,
                            json.dumps(dict(reset=reset, has_motion=True)).encode(), output.ctypes.data)
                        if code: raise RuntimeError(api.phoenix_runtime_error().decode())
                        temporal.append(dict(sha256=hashlib.sha256(output.tobytes()).hexdigest(), history_used=status(api,session)['history_used']))
                    if temporal[0]['sha256'] != temporal[2]['sha256'] or not temporal[1]['history_used']:
                        raise RuntimeError('History/reset regression')
                    api.phoenix_runtime_frame_v3.argtypes = [C.c_void_p, C.POINTER(Frame3)]
                    mask = np.zeros((h, w, 4), np.float32)
                    mask[..., 3] = 1
                    request = advanced_request(image, output, dict(reset=True), mask=mask)
                    if api.phoenix_runtime_frame_v3(session, C.byref(request)):
                        raise RuntimeError(api.phoenix_runtime_error().decode())
                    checks['zero_mask_max_error'] = float(np.max(np.abs(image - output)))
                    if checks['zero_mask_max_error'] > 1e-6:
                        raise RuntimeError('Control mask no longer protects the input')
                    alive = memory()
                    if job['phase'] == 'after':
                        for index in range(51):
                            depth = 8 if index % 2 == 0 else 4
                            code = api.phoenix_runtime_frame(session, image.ctypes.data, None, None, None,
                                json.dumps(dict(reset=True, transformer_blocks=depth)).encode(), output.ctypes.data)
                            if code: raise RuntimeError(api.phoenix_runtime_error().decode())
                            if index % 10 == 0:
                                state = status(api, session)
                                stress.append(dict(schedule_changes=index, memory=memory(),
                                    backend_allocation_bytes=state['backend_allocation_bytes'],
                                    resident_weight_bytes=state['resident_weight_bytes']))
                        if len({r['backend_allocation_bytes'] for r in stress}) != 1:
                            raise RuntimeError('Tracked buffers grew across repeated schedule changes')
                name = f'{backend}-{model_id}-{w}x{h}'
                (Path(job['output']) / (name + '.json')).write_text(json.dumps(dict(passed=True, config=config, frames=rows,
                    temporal_frames=temporal, checks=checks, create_ms=create_ms, memory_before=before,
                    memory_alive=alive or memory(), schedule_stress=stress), indent=2))
                print(name, [round(r['wall_ms'], 2) for r in rows], flush=True)
            finally:
                api.phoenix_runtime_destroy(session)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--portable-child':
        portable(json.loads(Path(sys.argv[2]).read_text()))
    else:
        main()
