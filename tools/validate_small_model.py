"""Exercise the smaller model through the public CPU runtime API, without a GPU."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--backend-root', type=Path, help='Optional existing provider folder; avoids duplicating backend DLLs')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime_root, model_path = args.runtime.resolve(), args.model.resolve()
    backend_root = args.backend_root.resolve() if args.backend_root else runtime_root
    library = C.CDLL(str(runtime_root / 'phoenix_runtime_cpp.dll'))
    library.phoenix_runtime_create.argtypes = [C.c_char_p, C.c_uint, C.c_uint]
    library.phoenix_runtime_create.restype = C.c_void_p
    library.phoenix_runtime_destroy.argtypes = [C.c_void_p]
    library.phoenix_runtime_error.restype = C.c_char_p
    pointer = C.POINTER(C.c_float)
    library.phoenix_runtime_frame.argtypes = [C.c_void_p, pointer, pointer, pointer, pointer, C.c_char_p, pointer]
    library.phoenix_runtime_frame.restype = C.c_int
    library.phoenix_runtime_status.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    library.phoenix_runtime_status.restype = C.c_size_t
    library.phoenix_runtime_replace_model.argtypes = [C.c_void_p, C.c_char_p]
    library.phoenix_runtime_replace_model.restype = C.c_int
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    source = dict(kind='nvidia_dll', path=str(model_path), sha256=sha, model='small-fp8')
    config = dict(backend='cpu', device='cpu', cpu_threads=8, runtime_root=str(backend_root),
                  allow_unvalidated=True, validation_stage_hashes=True, weight_source=source)
    width, height = 16, 12
    color = (C.c_float * (width * height * 3))(*[
        .15 + .6 * ((x / (width - 1)) if c == 0 else (y / (height - 1)) if c == 1 else ((x + y) % 7) / 6)
        for y in range(height) for x in range(width) for c in range(3)])
    motion = (C.c_float * (width * height * 2))()
    session = library.phoenix_runtime_create(json.dumps(config).encode(), width, height)
    if not session:
        raise RuntimeError(library.phoenix_runtime_error().decode())
    def status():
        data = C.create_string_buffer(library.phoenix_runtime_status(session, None, 0))
        library.phoenix_runtime_status(session, data, len(data))
        return json.loads(data.value)
    cases = []
    try:
        initial = status()
        assert initial['model_id'] == 'small-fp8'
        assert initial['model_resource'] == 'WEIGHTS_CC_SMALL_FP8'
        assert initial['model_history_gain'] == .76708984375
        for name, reset, controls in [('reset_a', True, {}), ('history', False, {}),
                                      ('reset_b', True, {}), ('controls', True, {'style': 2, 'local_tone': .4, 'local_structure': .6, 'auto_mask': False})]:
            output = (C.c_float * len(color))(*([float('nan')] * len(color)))
            start = time.monotonic()
            result = library.phoenix_runtime_frame(session, color, motion, None, None,
                json.dumps(dict(reset=reset, has_motion=True, controls=controls)).encode(), output)
            if result:
                raise RuntimeError(library.phoenix_runtime_error().decode())
            import math
            assert all(math.isfinite(v) for v in output)
            state = status()
            assert state['model_id'] == 'small-fp8' and state['state'] == 'completed'
            blocks = {r['block'] for r in state['layers'] if r['block'] >= 0}
            assert blocks == set(range(31)), blocks
            digest = hashlib.sha256(bytes(output)).hexdigest()
            assert digest == state['presented_output_sha256']
            assert bytes(output) != bytes(color), 'No neural output effect'
            case = dict(name=name, output_sha256=digest, seconds=time.monotonic() - start,
                        history_used=state['history_used'], command_count=state['command_count'],
                        resident_weight_bytes=state['resident_weight_bytes'], blocks=sorted(blocks),
                        validation_stages=state['validation_stages'])
            cases.append(case)
            print(json.dumps({k: v for k, v in case.items() if k not in ['blocks', 'validation_stages']}), flush=True)
        assert cases[0]['output_sha256'] == cases[2]['output_sha256'], 'Reset output is not deterministic'
        assert not cases[0]['history_used'] and cases[1]['history_used']
        assert cases[0]['output_sha256'] != cases[3]['output_sha256'], 'Neural controls did not affect the new model'
        before = status()
        invalid = {**source, 'sha256': '0' * 64}
        assert library.phoenix_runtime_replace_model(session, json.dumps(invalid).encode()) == -1
        assert status() == before, 'Rejected replacement changed the live model/history'
        original = {**source, 'model': 'original'}
        assert library.phoenix_runtime_replace_model(session, json.dumps(original).encode()) == 0
        assert status()['model_id'] == 'original' and not status()['history_used']
        assert library.phoenix_runtime_replace_model(session, json.dumps(source).encode()) == 0
        assert status()['model_id'] == 'small-fp8' and status()['sampling_next_index'] == 0
    finally:
        library.phoenix_runtime_destroy(session)
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == sha
    report = dict(passed=True, gpu_used=False, model='small-fp8', width=width, height=height,
                  cpu_runtime_api_executed=True, vendor_parity_verified=False,
                  reset_repeatable=True, temporal_history_used=True, controls_affect_output=True,
                  invalid_replacement_preserves_session=True, original_and_small_replacement_passed=True,
                  supplied_model_unchanged=True, cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
