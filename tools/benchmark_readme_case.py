"""One isolated portable model benchmark, with real timing and deterministic-output checks."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import psutil
from compare_neural_controls import configure, advanced_request, Frame3, status
from benchmark_neural_portable import stats


def fixture(width, height):
    y, x = np.mgrid[:height, :width]
    image = np.empty((height, width, 3), np.float32)
    image[..., 0] = .125 + (x % 128) / 256.
    image[..., 1] = .25 + (y % 128) / 256.
    image[..., 2] = .2 + ((x // 17 + y // 13) % 2) * .35
    image[height//4:3*height//4, width//3:2*width//3] = [.7, .47, .36]
    return np.ascontiguousarray(image.astype(np.float16).astype(np.float32))


def digest(array): return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    w, h = job['extent']
    report = {'passed': False, 'state': 'initializing', 'model': job['model'], 'route': job['route'],
              'profile': job.get('profile', 'default'), 'extent': [w, h], 'process_index': job['process_index']}
    def save(): args.output.write_text(json.dumps(report, indent=2) + '\n')
    save()
    api = configure(job['runtime'])
    memory_api = C.CDLL(str(Path(job['config']['runtime_root']) / 'readme_memory.dll'))
    memory_api.phoenix_benchmark_memory.argtypes = [C.c_uint]
    memory_api.phoenix_benchmark_memory.restype = C.c_char_p
    def memory(): return json.loads(memory_api.phoenix_benchmark_memory(job.get('vendor_id', 0)))
    report['memory_before'] = memory()
    api.phoenix_runtime_frame_v3.argtypes = [C.c_void_p, C.POINTER(Frame3)]
    session = None
    try:
        started = time.perf_counter()
        session = api.phoenix_runtime_create(json.dumps(job['config']).encode(), w, h)
        if not session: raise RuntimeError(api.phoenix_runtime_error().decode())
        report['create_ms'] = (time.perf_counter() - started) * 1000
        image = fixture(w, h)
        motion = np.zeros((h, w, 2), np.float32)
        output = np.empty_like(image)
        metadata = {'reset': True, 'has_motion': True, 'passes': 1, 'repeats': [1]*8,
                    'controls': {'auto_mask': True, 'style': 0, 'intensity': 1., 'local_tone': 1.,
                                 'local_structure': 1., 'skin_structure': -1., 'seed': 0, 'sampling_mode': 'fixed'}}
        patch = job.get('metadata', {})
        metadata['controls'].update(patch.get('controls', {}))
        metadata.update({k:v for k,v in patch.items() if k != 'controls'})
        advanced = job.get('advanced', False)
        planes = {'motion': motion}
        if advanced:
            mask = np.ones((h, w, 4), np.float32); mask[:, :w//3, 0] = 0
            mask[:, w//2:, 1:3] = .4
            ui = np.zeros((h, w, 4), np.float32); ui[:max(1,h//10), w//16:w//3] = 1
            field = np.zeros((h, w, 4), np.float32); field[..., 0] = -1/w; field[..., 2] = 1/w
            if report['profile'] in ('combined', 'control_map'): planes['mask'] = mask
            if report['profile'] in ('combined', 'ui_overlay'): planes['ui'] = ui
            if report['profile'] in ('combined', 'distortion'): planes['distortion'] = field
        def call(reset=None):
            if reset is not None: metadata['reset'] = reset
            start = time.perf_counter()
            if advanced:
                request = advanced_request(image, output, metadata, **planes)
                code = api.phoenix_runtime_frame_v3(session, C.byref(request))
            else:
                code = api.phoenix_runtime_frame(session, image.ctypes.data, motion.ctypes.data, None, None,
                                                json.dumps(metadata).encode(), output.ctypes.data)
            elapsed = (time.perf_counter() - start) * 1000
            if code: raise RuntimeError(api.phoenix_runtime_error().decode())
            return elapsed
        cold = call(True)
        report.update(state='warming', first_call_ms=cold, initial_status=status(api, session))
        save()
        metadata['reset'] = False
        slow = job['config']['backend'] == 'cpu' or job.get('slow', False)
        warm_count = 0
        started = time.perf_counter()
        minimum_warm = 240 if report['profile'] == 'adaptive_gpu' else (1 if slow else 4)
        while warm_count < minimum_warm or (not slow and time.perf_counter()-started < .75):
            call(); warm_count += 1
        report.update(state='measuring', warmup_frames=warm_count, warmup_duration_s=time.perf_counter()-started)
        save()
        walls, operators, gpu, stages = [], [], [], []
        schedules = []
        started = time.perf_counter()
        minimum, seconds = (3, 0) if slow else (12, 2.)
        while len(walls) < minimum or time.perf_counter()-started < seconds:
            walls.append(call())
            row = status(api, session)
            operators.append(row['operator_ms'])
            if row['network_schedule']['gpu_sample']: gpu.append(row['network_schedule']['gpu_ms'])
            stages.append({key: row.get(key, 0) for key in ['prepare_ms','execute_with_transfers_ms','resolve_ms','canonicalize_ms','presentation_ms']})
            schedules.append(row['network_schedule']['transformer_blocks'])
            if len(walls) >= minimum and time.perf_counter()-started > 30: break
        report.update(measure_duration_s=time.perf_counter()-started, measured_samples=len(walls), minimum_samples=minimum)
        # Output checks always use the same reset/history sequence, independently
        # of warmup duration or the preceding temporal measurement window.
        call(True); reset_a = output.copy()
        call(False); history = output.copy()
        call(True); reset_b = output.copy()
        finite = bool(np.isfinite(reset_a).all() and np.isfinite(history).all() and np.isfinite(reset_b).all())
        repeatable = bool(np.array_equal(reset_a, reset_b))
        report.update(wall=stats(walls), operator=stats(operators), gpu=stats(gpu) if gpu else None,
                      wall_samples_ms=walls, operator_samples_ms=operators, gpu_samples_ms=gpu,
                      measured_depths=sorted(set(schedules)), stages_mean_ms={k:float(np.mean([s[k] for s in stages])) for k in stages[0]},
                      finite_output=finite, reset_repeatable=repeatable,
                      output_hashes={'reset':digest(reset_a),'history':digest(history),'reset_again':digest(reset_b)},
                      output_mean_abs_change=float(np.mean(np.abs(reset_a-image))),
                      input_sha256=digest(image), runtime_status=status(api,session),
                      timing_scope={'gpu':'Backend GPU events across all network passes; excludes host work/transfers outside those events',
                                    'wall':'Synchronous RGB32F host-plane frame call including preparation, transfers and resolve; not game frame time'})
        process = psutil.Process().memory_info()
        report['process_memory'] = {'working_set_bytes':process.rss, 'private_bytes':getattr(process,'private',process.vms),
                                    'peak_working_set_bytes':getattr(process,'peak_wset',process.rss),
                                    'scope':'Entire isolated Python/NumPy/runtime/provider process; not model-only RAM'}
        report['memory'] = memory()
        if not repeatable:
            difference=np.abs(reset_a-reset_b)
            changed=np.any(reset_a!=reset_b,axis=2)
            coords=np.argwhere(changed)
            worst=np.unravel_index(np.argmax(difference),difference.shape)
            report['reset_difference']={'changed_elements':int(np.count_nonzero(difference)),
                'changed_pixels':int(np.count_nonzero(changed)),'max_abs':float(difference.max()),
                'mean_abs':float(difference.mean()),'half_changed_elements':int(np.count_nonzero(reset_a.astype(np.float16)!=reset_b.astype(np.float16))),
                'pixel_bbox_yx':[coords.min(axis=0).tolist(),coords.max(axis=0).tolist()],
                'worst_yxc':list(map(int,worst)),'worst_reset':float(reset_a[worst]),'worst_reset_again':float(reset_b[worst])}
            if job.get('diagnostic_outputs'):
                np.savez_compressed(job['diagnostic_outputs'],reset=reset_a,reset_again=reset_b,history=history)
        if job.get('golden') and report['profile'] == 'default':
            golden_path = Path(job['golden'])
            if job['config']['backend'] == 'cpu' and job['process_index'] == 0:
                np.savez(golden_path, reset=reset_a, history=history)
            if golden_path.exists():
                golden = np.load(golden_path)
                report['cpu_comparison'] = {}
                for key, actual in [('reset',reset_a),('history',history)]:
                    difference = np.abs(actual.astype(np.float64)-golden[key].astype(np.float64))
                    report['cpu_comparison'][key] = {'exact_float32':bool(np.array_equal(actual,golden[key])),
                        'max_abs':float(difference.max()),'mean_abs':float(difference.mean()),
                        'exact_half_pixels':float(np.mean(actual.astype(np.float16)==golden[key].astype(np.float16)))}
        report.update(passed=finite and repeatable, state='completed')
        save()
        print(f"{job['route']} {job['model']} {w}x{h} {report['profile']}: call {report['wall']['median_ms']:.3f} ms", flush=True)
    except Exception as error:
        report.update(passed=False, state='failed', error=str(error)); save()
        raise
    finally:
        if session: api.phoenix_runtime_destroy(session)


if __name__ == '__main__': main()
