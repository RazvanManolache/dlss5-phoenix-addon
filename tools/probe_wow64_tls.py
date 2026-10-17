"""Isolated cross-mode TLS/loader investigation; optional GPU module probe, never games."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time
from probe_nvapi_module import execute
from probe_wow64_native import WOW64EXT_COMMIT

ROOT = Path(__file__).resolve().parents[1]


class ProbeDirectory(tempfile.TemporaryDirectory):
    def cleanup(self):
        # The debugger can release its last image mapping just after it exits.
        for attempt in range(8):
            try:
                return super().cleanup()
            except PermissionError:
                if attempt == 7: raise
                time.sleep(.25)


def tls_success(row):
    return row.get('exit_code') == 0 and not row.get('timed_out', False) and any(
        event.get('event') == 'tls_result' and event.get('stage') == 5 and
        event.get('template_ok') is True and event.get('zero_fill_ok') is True
        for event in row.get('events', []))


def d3d12_conclusions(rows, frames):
    def valid(name, model):
        process, result = rows.get(name, {}), frames.get(name, {})
        if process.get('exit_code') != 0 or process.get('timed_out') or not result.get('passed') or not result.get('gpu_copy_verified'):
            return False
        if not model:
            return result.get('cuda_status') == 0
        return any(frame.get('changed_rgb_samples', 0) > 0 and
            (model != 'nvidia' or frame.get('evaluations', 0) >= 3)
            for frame in result.get('frames', []))
    result = {}
    for backend in ('cuda', 'phoenix', 'nvidia'):
        native, cross = 'native_d3d12_'+backend, 'thread_d3d12_'+backend
        if native not in rows and cross not in rows:
            result[backend] = None
            continue
        model = '' if backend == 'cuda' else backend
        result[backend] = dict(native_control=valid(native, model),
            same_process_x86_success=valid(native, model) and valid(cross, model),
            cross_stage=frames.get(cross, {}).get('failed_stage', frames.get(cross, {}).get('stage')))
    return result


def portable_conclusions(rows, control, cross):
    def valid(name, result):
        process = rows.get(name, {})
        frames = result.get('frames', [])
        return (process.get('exit_code') == 0 and not process.get('timed_out') and result.get('passed') is True and
            len(frames) >= 3 and all(frame.get('changed', 0) > 0 and frame.get('commands', 0) > 0 for frame in frames))
    native_ok = valid('native_frame', control)
    return dict(native_control=native_ok, same_process_x86_success=native_ok and valid('thread_frame', cross),
        cross_stage=cross.get('failed_stage', cross.get('stage')))


def model_parity_conclusions(rows, frames):
    evaluated=d3d12_conclusions(rows,frames)
    result={}
    for backend in ('phoenix','nvidia'):
        names=('native_d3d12_'+backend,'thread_d3d12_'+backend)
        if evaluated[backend] is None:
            result[backend]=None
            continue
        identity=True; hashes=[]
        for name,bits in zip(names,(64,32)):
            process,frame=rows.get(name,{}),frames.get(name,{})
            identity &= bool(process.get('process_pid')) and frame.get('process_pid')==process.get('process_pid') and \
                frame.get('process_bits')==bits and frame.get('pointer_bits')==64 and frame.get('helper_pid')==0
            outputs=frame.get('frames',[])
            valid=bool(outputs) and all(re.fullmatch('[0-9a-f]{64}',item.get('output_sha256','')) for item in outputs)
            valid &= len(outputs)>=3 if backend=='phoenix' else any(item.get('evaluations',0)>=3 for item in outputs)
            hashes.append([item['output_sha256'] for item in outputs] if valid else [])
        passed=evaluated[backend]['same_process_x86_success'] and identity and all(hashes)
        result[backend]=dict(same_process_inference=bool(passed),
            bit_exact_against_x64=bool(passed and hashes[0]==hashes[1]),
            game_presentation_validated=False)
    return result


def benchmark_conclusions(rows, frames):
    """Never assign a timing to a crashed, wrong-process or unvalidated output."""
    result={}
    for backend in ('phoenix','nvidia'):
        cases={}
        for prefix,bits in (('native',64),('thread',32)):
            name=prefix+'_d3d12_'+backend
            process,frame=rows.get(name,{}),frames.get(name,{})
            timing=frame.get('benchmark',{})
            valid=(process.get('exit_code')==0 and not process.get('timed_out') and frame.get('passed') is True and
                bool(process.get('process_pid')) and frame.get('process_pid')==process.get('process_pid') and
                frame.get('process_bits')==bits and frame.get('pointer_bits')==64 and frame.get('helper_pid')==0 and
                frame.get('gpu_copy_verified') is True and timing.get('width')==1920 and timing.get('height')==1080 and
                timing.get('scale')==1 and timing.get('warmup_seconds')==5 and timing.get('requested_seconds')==20 and
                timing.get('changed_rgb_samples',0)>0 and bool(re.fullmatch('[0-9a-f]{64}',timing.get('output_sha256',''))))
            count=timing.get('completed_frames',0)
            elapsed,fps=timing.get('measured_seconds',0),timing.get('completed_output_fps',0)
            valid &= isinstance(count,int) and count>=16 and isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and 20<=elapsed<=30
            valid &= isinstance(fps,(int,float)) and math.isfinite(fps) and fps>0
            if valid: valid &= math.isclose(fps,count/elapsed,rel_tol=1e-6)
            for field in ('gpu_processing','synchronized_call'):
                stats=timing.get(field) or {}
                valid &= stats.get('samples')==count
                valid &= all(isinstance(stats.get(key),(int,float)) and math.isfinite(stats[key]) and stats[key]>0
                    for key in ('mean_ms','median_ms','p95_ms'))
            cases[prefix]=dict(passed=bool(valid),**({'timing':timing} if valid else {'error':'Incomplete or invalid timed native model run'}))
        result[backend]=dict(passed=all(case['passed'] for case in cases.values()),cases=cases,
            game_presentation_benchmarked=False,per_frame_x86_handoff_benchmarked=False)
    return result


def x86_host_conclusions(rows, frames, native_frames):
    def valid(name,bits):
        process,result=rows.get(name,{}),frames.get(name,{})
        outputs=result.get('frames',[])
        benchmark=result.get('benchmark')
        expected_frames=7+benchmark.get('total_iterations',0) if benchmark else 6
        hashes=[item.get('output_sha256','') for item in outputs]
        sequence=(len(hashes)==6 and hashes[0]==hashes[1]==hashes[2]==hashes[5] and hashes[0]!=hashes[3] and hashes[3]!=hashes[4])
        return (process.get('exit_code')==0 and not process.get('timed_out') and result.get('passed') is True and
            bool(process.get('process_pid')) and result.get('process_pid')==process['process_pid']==result.get('driver_pid') and
            result.get('host_bits')==bits and result.get('driver_bits')==64 and result.get('helper_pid')==0 and
            result.get('phoenix64_loaded') is False and result.get('host_owns_graph') is True and result.get('host_owns_controls') is True and
            result.get('controls_and_input_verified') is True and result.get('authenticated_kernels')==41 and
            result.get('completed_frames')==expected_frames and result.get('dispatched_kernels')==expected_frames*154 and
            result.get('functions_destroyed')==41 and result.get('modules_destroyed')==41 and
            result.get('cleanup_failures')==0 and result.get('native_compat_restored') is True and
            sequence and all(item.get('index')==i and item.get('changed_rgb_samples',0)>0 and
                isinstance(item.get('gpu_ms'),(int,float)) and math.isfinite(item['gpu_ms']) and item['gpu_ms']>0 and
                re.fullmatch('[0-9a-f]{64}',item.get('output_sha256','')) for i,item in enumerate(outputs)))
    control=valid('native_driver_host',64)
    cross=valid('thread_driver_host',32)
    parity=control and cross and [item['output_sha256'] for item in frames['native_driver_host']['frames']]==[
        item['output_sha256'] for item in frames['thread_driver_host']['frames']]
    native=native_frames.get('native_d3d12_phoenix',{})
    native_process=rows.get('native_d3d12_phoenix',{})
    reference=(native.get('passed') is True and native_process.get('exit_code')==0 and not native_process.get('timed_out') and
        bool(native_process.get('process_pid')) and native.get('process_pid')==native_process['process_pid'] and
        native.get('process_bits')==64 and native.get('pointer_bits')==64 and native.get('helper_pid')==0 and
        len(native.get('frames',[]))>=3 and all(re.fullmatch('[0-9a-f]{64}',item.get('output_sha256','')) for item in native['frames']))
    reference_parity=reference and parity and all(
        frames['thread_driver_host']['frames'][i]['output_sha256']==native['frames'][0].get('output_sha256') for i in (0,1,2,5))
    return dict(scope='x86_host_generated_native_graph',native_host_control=control,
        same_process_x86_inference=bool(control and cross),bit_exact_host_architectures=bool(parity),
        bit_exact_against_native_api=bool(reference_parity),game_presentation_validated=False,
        production_route_integrated=False)


def x86_host_benchmark_conclusions(rows,frames,native_frames):
    proof=x86_host_conclusions(rows,frames,native_frames)
    cases={}
    for name in ('native_driver_host','thread_driver_host'):
        timing=frames.get(name,{}).get('benchmark',{})
        count=timing.get('completed_frames',0)
        seconds=timing.get('measured_seconds',0)
        valid=(proof['bit_exact_against_native_api'] and timing.get('width')==1920 and timing.get('height')==1080 and
            timing.get('scale')==1 and timing.get('warmup_seconds')==5 and timing.get('requested_seconds')==20 and
            timing.get('reset_each_frame') is True and isinstance(count,int) and count>=16 and
            isinstance(seconds,(int,float)) and math.isfinite(seconds) and 20<=seconds<=30 and
            isinstance(timing.get('total_iterations'),int) and timing['total_iterations']>=count and
            timing.get('changed_rgb_samples',0)>0 and bool(re.fullmatch('[0-9a-f]{64}',timing.get('output_sha256',''))))
        if valid: valid=math.isclose(timing.get('completed_output_fps',0),count/seconds,rel_tol=1e-6)
        for field in ('gpu_processing','synchronized_call'):
            stats=timing.get(field,{})
            valid &= stats.get('samples')==count and all(isinstance(stats.get(key),(int,float)) and math.isfinite(stats[key]) and stats[key]>0
                for key in ('mean_ms','median_ms','p95_ms'))
        cases[name]=dict(passed=bool(valid),**({'timing':timing} if valid else {'error':'Incomplete or invalid x86-host benchmark'}))
    return dict(passed=all(case['passed'] for case in cases.values()),cases=cases,
        game_presentation_benchmarked=False,per_frame_x86_handoff_benchmarked=True)


def legacy_conclusions(rows, frames):
    def valid(name):
        process, result = rows.get(name, {}), frames.get(name, {})
        return (process.get('exit_code') == 0 and not process.get('timed_out') and result.get('passed') is True and
            result.get('gpu_copy_verified') is True and result.get('shader_create_status') == 0 and
            result.get('shader_created') is True)
    control, cross = 'native_d3d12_legacy', 'thread_d3d12_legacy'
    if control not in rows and cross not in rows:
        return None
    native_ok = valid(control)
    return dict(scope='shader_creation_only', native_control=native_ok,
        same_process_x86_success=native_ok and valid(cross), inference_run=False)


def driver_bridge_conclusions(rows):
    row = rows.get('thread_gpu_native_driver', {})
    events = row.get('events', [])
    pid = row.get('process_pid')
    driver = next((event for event in events if event.get('event') == 'native_driver_bridge'), {})
    client = next((event for event in events if event.get('event') == 'native_thread_task'), {})
    worker = next((event for event in events if event.get('event') == 'gpu_worker_result'), {})
    control = rows.get('native_gpu', {})
    native_ok = control.get('exit_code') == 0 and not control.get('timed_out', False) and any(
        event.get('stage', event.get('event')) == 'worker_result' and event.get('worker_stage') == 12 and
        event.get('status') == 0 and event.get('module_created') == 1 for event in control.get('events', []))
    observed = (bool(pid) and not row.get('timed_out', False) and row.get('exit_code') in (0, 2) and
        client.get('completed') is True and client.get('caller_pointer_bits') == 32 and
        client.get('pid') == pid and driver.get('pid') == pid and driver.get('driver_pointer_bits') == 64 and
        driver.get('phoenix_runtime_loaded') is False and driver.get('bridged_escape_count', 0) > 0)
    return dict(scope='driver_module_creation_only', native_control=native_ok,
        same_process_driver_calls_observed=observed,
        module_creation_succeeded=native_ok and observed and row.get('exit_code') == 0 and
            worker.get('worker_stage') == 12 and worker.get('status') == 0 and worker.get('module_created') == 1,
        worker_stage=worker.get('worker_stage'), nvapi_status=worker.get('status'),
        phoenix_inference_succeeded=False)


def appcompat_conclusions(rows, frames):
    fault_seen=any(row.get('exit_code')==0xc0000005 and any(fault.get('module','').lower()=='apphelp.dll' and
        fault.get('machine')==0x14c for fault in row.get('first_chance_faults',[]))
        for name,row in rows.items() if name.startswith('thread_frame_unseparated'))
    def clean_error(name):
        process,result=rows.get(name,{}),frames.get(name,{})
        return (process.get('exit_code')==2 and not process.get('timed_out') and not process.get('first_chance_faults') and
            result.get('error')=='no physical device (0)' and result.get('frames')==[] and
            result.get('native_appcompat_view') is True and result.get('native_appcompat_restored') is True)
    separated_ok=all(clean_error(name) for name in ('thread_frame','thread_frame_repeat'))
    return dict(x86_apphelp_fault_reproduced=fault_seen, separated_error_repeated=separated_ok,
        crash_guard_validated=fault_seen and separated_ok, inference_run=False)


def conclusions(rows):
    def event_success(name, event, predicate):
        return rows.get(name, {}).get('exit_code') == 0 and not rows.get(name, {}).get('timed_out', False) and any(
            row.get('event') == event and predicate(row) for row in rows.get(name, {}).get('events', []))
    def gpu_success(name):
        probe = rows.get(name, {})
        return probe.get('exit_code') == 0 and not probe.get('timed_out', False) and any(
            row.get('event', row.get('stage')) in ('gpu_worker_result', 'worker_result') and
            row.get('worker_stage') == 12 and row.get('status') == 0 and row.get('module_created') == 1
            for row in probe.get('events', []))
    def gpu_stage(name, stage):
        probe = rows.get(name, {})
        return probe.get('exit_code') in (0, 2) and not probe.get('timed_out', False) and any(
            row.get('event', row.get('stage')) in ('gpu_worker_result', 'worker_result') and
            row.get('worker_stage', 0) >= stage for row in probe.get('events', []))
    native_error_ok = event_success('native_phoenix_api', 'native_phoenix_error_control',
        lambda row: row.get('status') == -1 and row.get('message') == 'Null native session')
    thread_error_ok = event_success('thread_native', 'native_thread_task', lambda row:
        row.get('completed') is True and row.get('defaults_ok') is True and row.get('status') == -1 and
        row.get('message') == 'Null native session' and row.get('stack_limit', 0) < row.get('stack_address', 0) < row.get('stack_base', 0))
    return dict(native_control=tls_success(rows.get('native_tls', {})),
        cross_tls=tls_success(rows.get('cross_tls', {})),
        cross_repaired_tls=tls_success(rows.get('cross_repair', {})),
        native_thread_tls=tls_success(rows.get('thread_tls', {})),
        native_thread_repaired_tls=tls_success(rows.get('thread_repair', {})),
        initialized_cross_tls=tls_success(rows.get('cross_initialize', {})),
        initialized_native_thread_tls=tls_success(rows.get('thread_initialize', {})),
        native_phoenix_error_control=native_error_ok,
        phoenix_library_loaded_before_fault=any(row.get('event') == 'phoenix_load_result' and row.get('loaded') is True
            for row in rows.get('initialized_native', {}).get('events', [])),
        phoenix_defaults_matched_before_fault=any(row.get('event') == 'phoenix_defaults' and row.get('matched') is True
            for row in rows.get('initialized_native', {}).get('events', [])),
        initialized_dxgi_loader=rows.get('initialized_dxgi', {}).get('exit_code') == 0 and any(
            event.get('event') == 'initialized_dxgi_load_result' and event.get('loaded') is True and event.get('status') == 0
            for event in rows.get('initialized_dxgi', {}).get('events', [])),
        borrowed_stack_phoenix_error=rows.get('initialized_native', {}).get('exit_code') == 0 and any(
            event.get('event') == 'phoenix_error_contract' and event.get('frame_status') == -1 and
            event.get('message') == 'Null native session' and event.get('x86_tls_preserved') is True
            for event in rows.get('initialized_native', {}).get('events', [])),
        native_gpu_control=gpu_success('native_gpu'),
        initialized_gpu_module=gpu_success('native_gpu') and gpu_success('initialized_gpu'),
        native_thread_gpu_module=gpu_success('native_gpu') and gpu_success('thread_gpu'),
        redirected_native_thread_gpu_module=gpu_success('native_gpu') and gpu_success('thread_gpu_native_umd'),
        native_thread_phoenix_error=native_error_ok and thread_error_ok,
        phoenix_x64_cpu_api=native_error_ok and thread_error_ok,
        phoenix_route_implemented=False, gpu_device_created=gpu_success('native_gpu') and any(
            gpu_stage(name, 5) for name in ('initialized_gpu', 'thread_gpu', 'thread_gpu_native_umd', 'thread_gpu_native_driver')),
        gpu_queue_created=gpu_success('native_gpu') and any(
            gpu_stage(name, 6) for name in ('initialized_gpu', 'thread_gpu', 'thread_gpu_native_umd', 'thread_gpu_native_driver')), inference_run=False)


def symbol_rva(text, marker='PHX_TLS_RVA'):
    found = re.search(re.escape(marker)+r'=([0-9a-f`]+)', text, re.I)
    if not found:
        return None
    value = int(found[1].replace('`', ''), 16)
    return value if 0 < value < 0x1000000 else None


def driver_images(output):
    images = {}
    for line in output.splitlines():
        match = re.search(r'ModLoad:.*?([A-Za-z]:\\.*\\nvldumd(?:x)?\.dll)\s*$', line, re.I)
        if not match:
            continue
        selected = Path(match[1])
        for path in (selected, selected.with_name('nvldumdx.dll')):
            if path.name.lower() in images or not path.is_file():
                continue
            data = path.read_bytes()
            if len(data) < 64 or data[:2] != b'MZ': continue
            pe = struct.unpack_from('<I', data, 60)[0]
            if pe+6 > len(data) or data[pe:pe+4] != b'PE\0\0': continue
            images[path.name.lower()] = dict(machine=hex(struct.unpack_from('<H', data, pe+4)[0]),
                sha256=hashlib.sha256(data).hexdigest(), observed_loaded=path == selected)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wow64ext', type=Path, required=True)
    parser.add_argument('--trace-dxgi', action='store_true')
    parser.add_argument('--run-gpu-probes', action='store_true')
    parser.add_argument('--trace-gpu', action='store_true')
    parser.add_argument('--trace-cuda-init', action='store_true', help='Compare native CUDA initializer disassembly/call checkpoints in fresh debugger children')
    parser.add_argument('--cuda-break-rva', action='append', type=lambda value:int(value,16), default=[], help='Observed initializer checkpoint RVA in the selected CUDA driver; hexadecimal')
    parser.add_argument('--cuda-indirect-slot-rva', type=lambda value:int(value,16), help='Observed cuInit dispatch slot RVA; disassemble its current target without patching it')
    parser.add_argument('--preload-driver-object-loader', action='store_true', help='Preload the real matching native driver object-loader DLL before CUDA initialization')
    parser.add_argument('--retry-native-reservation', action='store_true', help='Retry a failed unconstrained reserve-only VirtualAlloc2 call with a valid explicit high address')
    parser.add_argument('--try-native-umd', action='store_true')
    parser.add_argument('--try-native-driver-bridge', action='store_true', help='x86 coordinator calls a driver-only x64 thunk, without loading x64 Phoenix')
    parser.add_argument('--try-native-driver-arithmetic', action='store_true', help='Try an independent tiny cubin through the driver-only legacy shader thunk')
    parser.add_argument('--try-x86-host', action='store_true', help='Run the full x86-compiled graph/control host against the same-process native driver executor')
    parser.add_argument('--try-portable-frame', action='store_true')
    parser.add_argument('--try-d3d12-frame', action='store_true', help='Attempt real Phoenix and NVIDIA inference, plus CUDA diagnostics')
    parser.add_argument('--benchmark', action='store_true', help='Time GPU-resident native model calls at 1080p: 5 seconds warm-up, then 20 seconds; isolated workers only')
    parser.add_argument('--benchmark-x86-host', action='store_true', help='Time route 3 at 1080p, including per-frame same-process x86/native handoff')
    parser.add_argument('--benchmark-reset', action='store_true', help='Reset each frame for comparison with reset-only backbuffer/route-3 inputs')
    parser.add_argument('--ngx-sdk', type=Path, default=ROOT.parent/'external/DLSS/include')
    parser.add_argument('--ngx-core', type=Path)
    parser.add_argument('--large-address-aware', action='store_true', help='Allow disposable x86 probe to use the full WOW64 address range')
    parser.add_argument('--try-native-identity', action='store_true', help='Process-local query shim only in the disposable CUDA diagnostic')
    parser.add_argument('--cuda-no-preload', action='store_true', help='Use documented lazy CUDA JIT-library loading in disposable children')
    parser.add_argument('--cuda-no-jit', action='store_true', help='Disable CUDA JIT only in disposable children')
    parser.add_argument('--inspect-threads', action='store_true', help='Inspect native driver-thread context before its first instruction')
    parser.add_argument('--bridge-create-dc', action='store_true', help='Use the real x86 GDI display-context creation through the same-process coordinator')
    parser.add_argument('--bridge-display-gdi', action='store_true', help='Use real x86 display enumeration and device-capability APIs in this process')
    parser.add_argument('--bridge-kmt-escape', action='store_true', help='Marshal unchanged graphics-driver requests through the real x86 Windows entry point')
    parser.add_argument('--native-main-image', action='store_true', help='Map the real x64 control image as driver-only native executable metadata')
    parser.add_argument('--native-peb-image', action='store_true', help='Temporarily select a real x64 main-image mapping in the native PEB only')
    parser.add_argument('--native-appcompat-view', action='store_true', help='Temporarily separate native compatibility metadata from inherited x86 shim state')
    parser.add_argument('--compare-appcompat', action='store_true', help='A/B native compatibility state in fresh portable children, then repeat the separated case')
    parser.add_argument('--native-registry', action='store_true', help='Select the 64-bit view for native driver registry reads only')
    parser.add_argument('--wrap-driver-threads', action='store_true', help='Observe actual native callback entry and TLS on driver-owned threads')
    parser.add_argument('--repair-driver-tls', action='store_true', help='Allocate real native per-thread TLS before driver callbacks; research only')
    parser.add_argument('--cuda-first', action='store_true', help='Initialize native CUDA before any D3D12 device is created')
    parser.add_argument('--early-native-identity', action='store_true', help='Process-local native export view before driver DLL initialization')
    parser.add_argument('--vulkan-icd', type=Path)
    parser.add_argument('--direct-vulkan-icd', action='store_true', help='Diagnose native ICD enumeration without the Vulkan loader')
    parser.add_argument('--vulkan-headers', type=Path, default=ROOT.parent/'external/OptiScaler/external/vulkan/include')
    parser.add_argument('--case', action='append', help='Selected cases plus mandatory positive controls')
    parser.add_argument('--bundle', type=Path, default=ROOT/'artifacts/current/native-kernels')
    parser.add_argument('--native-library', type=Path, default=ROOT/'artifacts/current/sdk/windows-x64/bin/phoenix_native.dll')
    args = parser.parse_args()
    if args.try_native_driver_arithmetic:
        args.try_native_driver_bridge=True
    if args.trace_gpu and not args.run_gpu_probes: parser.error('--trace-gpu requires --run-gpu-probes')
    if args.trace_cuda_init and not args.try_d3d12_frame:
        parser.error('--trace-cuda-init requires --try-d3d12-frame')
    if (args.cuda_break_rva or args.cuda_indirect_slot_rva is not None) and not args.trace_cuda_init:
        parser.error('CUDA instruction checkpoints require --trace-cuda-init')
    if args.try_native_driver_bridge and not args.run_gpu_probes:
        parser.error('--try-native-driver-bridge requires --run-gpu-probes')
    if args.try_native_umd and not args.run_gpu_probes: parser.error('--try-native-umd requires --run-gpu-probes')
    if args.try_portable_frame and not args.run_gpu_probes: parser.error('--try-portable-frame requires --run-gpu-probes')
    if args.direct_vulkan_icd and (not args.try_portable_frame or not args.vulkan_icd):
        parser.error('--direct-vulkan-icd requires --try-portable-frame and --vulkan-icd')
    if args.compare_appcompat and (not args.native_appcompat_view or not args.try_portable_frame):
        parser.error('--compare-appcompat requires --native-appcompat-view and --try-portable-frame')
    if args.try_d3d12_frame and (not args.run_gpu_probes or not args.ngx_core or not args.ngx_core.is_file()):
        parser.error('--try-d3d12-frame requires --run-gpu-probes and --ngx-core FILE')
    if args.benchmark and (not args.try_d3d12_frame or args.trace_gpu or args.trace_cuda_init):
        parser.error('--benchmark requires --try-d3d12-frame without a debugger trace')
    if args.try_x86_host and (not args.try_d3d12_frame or args.benchmark):
        parser.error('--try-x86-host requires --try-d3d12-frame without --benchmark')
    if args.benchmark_x86_host and not args.try_x86_host:
        parser.error('--benchmark-x86-host requires --try-x86-host')
    if args.benchmark_reset and not args.benchmark:
        parser.error('--benchmark-reset requires --benchmark')
    blob = None
    if args.run_gpu_probes:
        kernel = json.loads((args.bundle/'native_manifest.json').read_text())['kernels'][0]
        blob = (args.bundle/kernel['binary']).resolve()
        if not blob.is_relative_to(args.bundle.resolve()) or hashlib.sha256(blob.read_bytes()).hexdigest() != kernel['binary_sha256']:
            raise ValueError('Kernel path or SHA-256 mismatch')
    extension = args.wow64ext.resolve()
    if subprocess.check_output(['git', '-C', str(extension), 'rev-parse', 'HEAD'], text=True).strip() != WOW64EXT_COMMIT or subprocess.check_output(
        ['git', '-C', str(extension), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Use the pinned, unmodified external WOW64Ext checkout')
    vswhere = Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    debugger = Path(os.environ['ProgramFiles(x86)'])/'Windows Kits/10/Debuggers/x64/cdb.exe'
    ntdll = Path(os.environ['SystemRoot'])/'System32/ntdll.dll'
    data = ntdll.read_bytes()
    pe = struct.unpack_from('<I', data, 60)[0]
    stamp = struct.unpack_from('<I', data, pe+8)[0]
    report = dict(schema='phoenix-wow64-tls-probe/1', production_package_modified=False,
        games_launched=False, gpu_probes_requested=args.run_gpu_probes, gpu_used=False, model_files_loaded=False,
        large_address_aware=args.large_address_aware,
        native_identity_requested=args.try_native_identity,
        driver_only_bridge_requested=args.try_native_driver_bridge,
        probe_options={key: getattr(args, key) for key in ('native_registry', 'bridge_create_dc', 'bridge_display_gdi', 'bridge_kmt_escape',
            'native_main_image', 'native_peb_image', 'native_appcompat_view', 'early_native_identity', 'cuda_first', 'cuda_no_preload',
            'cuda_no_jit', 'inspect_threads', 'wrap_driver_threads', 'repair_driver_tls', 'direct_vulkan_icd', 'compare_appcompat')},
        system_image_sha256=hashlib.sha256(data).hexdigest(), image_timestamp=stamp,
        native_library_sha256=hashlib.sha256(args.native_library.read_bytes()).hexdigest() if args.native_library.is_file() else None,
        sources={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'tests/native/wow64_tls_worker.cpp', 'tests/native/wow64_tls_probe.cpp',
            'tests/native/wow64_tls_contract.h', 'tools/probe_wow64_tls.py',
            'tests/native/wow64_gpu_worker.cpp', 'tests/native/wow64_cross_probe.cpp', 'tests/native/wow64_probe_contract.h',
            'tests/native/wow64_umd_probe.h', 'tests/native/wow64_frame_worker.cpp',
            'tests/native/wow64_runtime_diagnostics.h', 'tests/native/wow64_probe_contract_tests.cpp')},
        builds={}, probes={})
    # Research entry points include the maintained addon adapters, so bind the
    # actual implementation as well as the small probe wrappers.
    report['sources'].update({path.relative_to(ROOT).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT/'src/integration/compatibility/wow64').rglob('*') if path.is_file()})
    receipt = ROOT/'artifacts/current'/('wow64-d3d12-investigation.json' if args.try_d3d12_frame else 'wow64-tls-investigation.json')
    if args.bridge_kmt_escape:
        if not args.run_gpu_probes or not args.try_d3d12_frame:
            parser.error('--bridge-kmt-escape requires --run-gpu-probes --try-d3d12-frame')
        receipt = ROOT/'artifacts/current/wow64-kmt-investigation.json'
    if args.try_native_driver_bridge:
        receipt = ROOT/'artifacts/current/x86-driver-bridge-investigation.json'
    if args.try_native_driver_arithmetic:
        receipt = ROOT/'artifacts/current/x86-driver-arithmetic-investigation.json'
    if args.trace_cuda_init:
        receipt = ROOT/'artifacts/current/wow64-cuda-init-investigation.json'
        cuda_image=(args.ngx_core.parent/'nvcuda64.dll').read_bytes()
        cuda_pe=struct.unpack_from('<I',cuda_image,60)[0]
        cuda_size=struct.unpack_from('<I',cuda_image,cuda_pe+24+56)[0]
        points=args.cuda_break_rva+([] if args.cuda_indirect_slot_rva is None else [args.cuda_indirect_slot_rva])
        if struct.unpack_from('<H',cuda_image,cuda_pe+4)[0]!=0x8664 or any(not 0<rva<cuda_size-8 for rva in points):
            parser.error('CUDA checkpoints must fit the selected x64 driver image')
        report['cuda_driver_sha256']=hashlib.sha256(cuda_image).hexdigest()
        import pefile
        parsed=pefile.PE(data=cuda_image,fast_load=True)
        parsed.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']])
        anchor=next(symbol.address for symbol in parsed.DIRECTORY_ENTRY_EXPORT.symbols if symbol.name==b'cuInit')
        report['cuda_init_rva']=anchor
        report['cuda_break_rvas']=args.cuda_break_rva
        report['cuda_indirect_slot_rva']=args.cuda_indirect_slot_rva
        report['preload_driver_object_loader']=args.preload_driver_object_loader
    if args.benchmark:
        receipt=ROOT/'artifacts/current/wow64-method-benchmark.json'
        report['benchmark_profile']=dict(width=1920,height=1080,scale=1,warmup_seconds=5,measured_seconds=20,
            vsync=False,reset_each_frame=args.benchmark_reset,scope='Isolated GPU-resident model calls, not addon/game-presentation throughput')
        try:
            report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
        except (OSError,subprocess.CalledProcessError): report['gpu']='Unavailable'
    if args.try_x86_host:
        receipt=ROOT/'artifacts/current/wow64-x86-host-investigation.json'
        report['x86_host_frames']={}
        for name in ('tests/native/wow64_x86_host.h','tests/native/wow64_driver_contract.h','tests/native/wow64_driver_worker.cpp'):
            report['sources'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    if args.benchmark_x86_host:
        receipt=ROOT/'artifacts/current/wow64-x86-host-benchmark.json'
        report['benchmark_profile']=dict(width=1920,height=1080,scale=1,warmup_seconds=5,measured_seconds=20,reset_each_frame=True)
        try:
            report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
        except (OSError,subprocess.CalledProcessError): report['gpu']='Unavailable'
    if args.try_d3d12_frame:
        report['schema'] = 'phoenix-wow64-d3d12-probe/1'
        report['sources']['tests/native/wow64_d3d12_frame_worker.cpp'] = hashlib.sha256(
            (ROOT/'tests/native/wow64_d3d12_frame_worker.cpp').read_bytes()).hexdigest()
        report['sources']['src/integration/reshade/nr_forwarder.cpp'] = hashlib.sha256(
            (ROOT/'src/integration/reshade/nr_forwarder.cpp').read_bytes()).hexdigest()
        report['d3d12_frames'] = {}
    def save():
        receipt.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with ProbeDirectory(prefix='wow64-tls-', dir=ROOT/'artifacts') as temporary:
        build = Path(temporary)
        if args.try_native_driver_arithmetic:
            import shutil
            compiler=shutil.which('ptxas')
            if not compiler: raise RuntimeError('CUDA assembler required for arithmetic shader')
            blob=build/'arithmetic.cubin'
            subprocess.run([compiler,'--gpu-name=sm_120',str(ROOT/'tests/native/nvapi_arithmetic.ptx'),'-o',str(blob)],check=True,timeout=60)
            report['arithmetic_blob_sha256']=hashlib.sha256(blob.read_bytes()).hexdigest()
            report['arithmetic_scope']='shader_creation_only'
        for bits in (64, 32):
            out = build/str(bits)
            out.mkdir()
            common = '/nologo /std:c++20 /O2 /EHsc /MT /DNOMINMAX'
            if args.try_x86_host: common+=' /DPHX_X86_HOST_PROBE /bigobj'
            commands = []
            if bits == 64:
                commands.append(f'cl /nologo /std:c++20 /O2 /Oi /GS- /GR- /LD '
                    f'"{ROOT}/tests/native/wow64_tls_worker.cpp" /link /NODEFAULTLIB /ENTRY:DllMain /OUT:tls_worker64.dll kernel32.lib')
                if args.run_gpu_probes:
                    commands += [f'cl /nologo /std:c++20 /O2 /Oi /GS- /GR- /LD "{ROOT}/tests/native/wow64_gpu_worker.cpp" '
                        '/link /NODEFAULTLIB /ENTRY:DllMain /OUT:gpu_worker64.dll kernel32.lib d3d12.lib dxgi.lib',
                        f'cl {common} "{ROOT}/tests/native/wow64_cross_probe.cpp" /Fe:gpu_control.exe']
                    if args.try_native_umd:
                        commands += [f'cl /nologo /std:c++20 /O2 /Oi /MT /GS- /GR- /LD /DPHX_WOW64_UMD_PROBE '
                            f'"{ROOT}/tests/native/wow64_gpu_worker.cpp" /link '
                            '/OUT:gpu_native_umd64.dll kernel32.lib d3d12.lib dxgi.lib gdi32.lib advapi32.lib']
                    if args.try_native_driver_bridge:
                        commands += [f'cl /nologo /std:c++20 /O2 /Oi /MT /GS- /GR- /LD /DPHX_WOW64_UMD_PROBE /DPHX_NATIVE_DRIVER_BRIDGE '
                            f'"{ROOT}/tests/native/wow64_gpu_worker.cpp" /link '
                            '/OUT:gpu_driver_bridge64.dll kernel32.lib d3d12.lib dxgi.lib gdi32.lib user32.lib advapi32.lib']
                    if args.try_native_driver_arithmetic:
                        for library,options in (('gpu_arithmetic_control64',''),('gpu_driver_arithmetic64','/DPHX_WOW64_UMD_PROBE /DPHX_NATIVE_DRIVER_BRIDGE')):
                            commands += [f'cl /nologo /std:c++20 /O2 /Oi /MT /GS- /GR- /LD /DPHX_NATIVE_DRIVER_ARITHMETIC {options} '
                                f'"{ROOT}/tests/native/wow64_gpu_worker.cpp" /link '
                                f'/OUT:{library}.dll kernel32.lib d3d12.lib dxgi.lib gdi32.lib user32.lib advapi32.lib']
                    if args.try_portable_frame:
                        commands += [f'cl {common} /LD /I"{args.vulkan_headers.resolve()}" "{ROOT}/tests/native/wow64_frame_worker.cpp" /Fe:frame_worker64.dll ole32.lib runtimeobject.lib user32.lib gdi32.lib advapi32.lib dxgi.lib']
                    if args.try_d3d12_frame:
                        commands += [f'cl {common} /LD /I"{args.ngx_sdk.resolve()}" '
                            f'"{ROOT}/tests/native/wow64_d3d12_frame_worker.cpp" "{ROOT}/src/integration/reshade/nr_forwarder.cpp" '
                            '/Fe:probe-nvngx.dll /link d3d12.lib dxgi.lib user32.lib gdi32.lib advapi32.lib']
                    if args.try_x86_host:
                        commands += [f'cl {common} /LD "{ROOT}/tests/native/wow64_driver_worker.cpp" '
                            '/Fe:driver_worker64.dll /link d3d12.lib dxgi.lib user32.lib gdi32.lib advapi32.lib']
            else:
                commands.append('cl /nologo /O2 /Oi /GS- /GR- /LD /DWOW64EXT_EXPORTS /DUNICODE /D_UNICODE /DWIN32 '
                    f'"{extension}/src/wow64ext.cpp" /link /NODEFAULTLIB /ENTRY:DllMain /OUT:wow64ext.dll kernel32.lib')
            commands.append(f'cl {common} "{ROOT}/tests/native/wow64_tls_probe.cpp" /Fe:tls_probe.exe gdi32.lib user32.lib')
            commands.append(f'cl {common} "{ROOT}/tests/native/wow64_probe_contract_tests.cpp" /Fe:contract_tests.exe')
            if bits == 32:
                commands.append(f'cl {common} "{ROOT}/tests/native/wow64_tls_probe.cpp" /Fe:tls_gui.exe /link gdi32.lib user32.lib /SUBSYSTEM:WINDOWS /ENTRY:wmainCRTStartup')
                if args.large_address_aware:
                    commands += ['editbin /LARGEADDRESSAWARE tls_gui.exe', 'editbin /LARGEADDRESSAWARE tls_probe.exe']
            recipe = out/'build.cmd'
            lines = ['@echo off', f'call "{studio}/VC/Auxiliary/Build/vcvars{bits}.bat" >nul 2>&1',
                     'if errorlevel 1 exit /b %errorlevel%']
            for command in commands:
                lines += [command, 'if errorlevel 1 exit /b %errorlevel%']
            recipe.write_text('\n'.join(lines)+'\n')
            result = subprocess.run(['cmd', '/d', '/c', str(recipe)], cwd=out, capture_output=True, text=True, timeout=120)
            report['builds'][str(bits)] = dict(passed=result.returncode == 0,
                diagnostics=(result.stdout+result.stderr)[-6000:] if result.returncode else '')
            save()
            if result.returncode: raise RuntimeError(report['builds'][str(bits)]['diagnostics'])
            print(f'{bits}-bit probes compiled', flush=True)
            contract=execute([str(out/'contract_tests.exe')],out,timeout=10)
            report.setdefault('contract_controls',{})[str(bits)]=contract
            save()
            if contract.get('exit_code') != 0 or not any(event.get('event')=='gdi_contract' and
                    event.get('passed') is True and event.get('pointer_bits')==bits for event in contract.get('events',[])):
                raise RuntimeError(f'{bits}-bit coordinator contract failed')
        layouts=[next(event for event in report['contract_controls'][str(bits)]['events'] if event.get('event')=='driver_packet_contract') for bits in (64,32)]
        if any(layouts[0][key]!=layouts[1][key] for key in ('channel_bytes','launch_offset')):
            raise RuntimeError('Driver packet architecture layouts differ')
        native, cross, gui, worker = build/'64/tls_probe.exe', build/'32/tls_probe.exe', build/'32/tls_gui.exe', build/'64/tls_worker64.dll'
        worker_data = worker.read_bytes()
        worker_pe = struct.unpack_from('<I', worker_data, 60)[0]
        directory_rva, directory_size = struct.unpack_from('<II', worker_data, worker_pe+24+112+9*8)
        report['tls_fixture'] = dict(directory_rva=directory_rva, directory_size=directory_size)
        section_count = struct.unpack_from('<H', worker_data, worker_pe+6)[0]
        section_start = worker_pe+24+struct.unpack_from('<H', worker_data, worker_pe+20)[0]
        for i in range(section_count):
            virtual_size, start, raw_size, raw = struct.unpack_from('<IIII', worker_data, section_start+i*40+8)
            if start <= directory_rva < start+max(virtual_size, raw_size):
                report['tls_fixture']['directory_fields'] = list(struct.unpack_from('<QQQQII', worker_data, raw+directory_rva-start))
        save()
        if not directory_rva or directory_size < 40:
            raise ValueError('TLS control has no valid PE TLS directory')
        rva = initializer = allocator = free_tls = None
        if debugger.is_file():
            script = build/'symbols.cdb'
            script.write_text('.symfix\n.reload /f ntdll.dll\n.printf "PHX_TLS_RVA=%p\\n", (ntdll!LdrpHandleTlsData-ntdll)\n'
                '.printf "PHX_TLS_INIT_RVA=%p\\n", (ntdll!LdrpInitializeTls-ntdll)\n'
                '.printf "PHX_TLS_ALLOC_RVA=%p\\n", (ntdll!LdrpAllocateTls-ntdll)\n'
                '.printf "PHX_TLS_FREE_RVA=%p\\n", (ntdll!LdrpFreeTls-ntdll)\n'
                'dt ntdll!_PEB ImageBaseAddress\ndt ntdll!_PEB AppCompatFlags\ndt ntdll!_PEB AppCompatFlagsUser\n'
                'dt ntdll!_PEB pShimData\ndt ntdll!_PEB AppCompatInfo\nq\n', encoding='ascii')
            result = subprocess.run([str(debugger), '-cf', str(script), str(native), '--tls', str(worker)],
                cwd=build/'64', capture_output=True, text=True, timeout=90)
            output = result.stdout+result.stderr
            rva = symbol_rva(output)
            initializer = symbol_rva(output, 'PHX_TLS_INIT_RVA')
            allocator = symbol_rva(output, 'PHX_TLS_ALLOC_RVA')
            free_tls = symbol_rva(output, 'PHX_TLS_FREE_RVA')
            report['symbols'] = dict(handler_rva=rva, initializer_rva=initializer, debugger_exit=result.returncode,
                allocator_rva=allocator, free_rva=free_tls,
                peb_image_offset=0x10 if re.search(r'\+0x010\s+ImageBaseAddress',output) else None,
                appcompat_offsets=[int(match[1],16) for name in ('AppCompatFlags','AppCompatFlagsUser','pShimData','AppCompatInfo')
                    if (match:=re.search(r'\+0x([0-9a-f]+)\s+'+name+r'\s+:',output,re.I))],
                diagnostics=[line for line in output.splitlines() if 'PHX_TLS_' in line or 'Symbol' in line or 'error' in line.lower()][-12:])
            save()
        cases = [('native_tls', native, '--tls'), ('cross_tls', gui, '--tls'), ('thread_tls', gui, '--thread')]
        if args.native_library.is_file(): cases.insert(1, ('native_phoenix_api', native, '--native-api'))
        if rva:
            cases += [('cross_repair', gui, '--repair'), ('thread_repair', gui, '--thread-repair')]
        if initializer:
            cases += [('cross_initialize', gui, '--initialize'), ('thread_initialize', gui, '--thread-initialize'),
                      ('initialized_dxgi', gui, '--initialize-dxgi')]
            if args.native_library.is_file(): cases += [('initialized_native', gui, '--initialize-native'), ('thread_native', gui, '--thread-native')]
        if args.run_gpu_probes:
            cases += [('native_gpu', build/'64/gpu_control.exe', '--worker')]
            if initializer: cases += [('initialized_gpu', gui, '--initialize-gpu'), ('thread_gpu', gui, '--thread-gpu')]
            if initializer and args.try_native_umd: cases += [('thread_gpu_native_umd', gui, '--thread-gpu')]
            if initializer and args.try_native_driver_bridge: cases += [('thread_gpu_native_driver', gui, '--thread-gpu')]
            if initializer and args.try_native_driver_arithmetic:
                cases += [('native_gpu_arithmetic',build/'64/gpu_control.exe','--worker'),('thread_gpu_native_driver_arithmetic',gui,'--thread-gpu')]
            if initializer and args.try_portable_frame:
                runtime_root = ROOT/'artifacts/current/sdk/windows-x64/bin'
                devices = json.loads(subprocess.check_output([str(runtime_root/'phoenix_frame.exe'),
                    '--vulkan-devices', str(runtime_root)], text=True, timeout=20))
                selected = next(device for device in devices if device.get('vendor') == 0x10de and device.get('supported'))
                job = dict(width=64, height=64, frames=3, output=str(build/'frame-output.json'), config=dict(
                    runtime_root=str(runtime_root), backend='vulkan', device=selected['id'],
                    allow_unvalidated=True, cpu_threads=2, weight_source=json.loads((ROOT/'artifacts/current/weight-source.json').read_text())))
                job.update(native_registry=args.native_registry,bridge_create_dc=args.bridge_create_dc,
                    bridge_display_gdi=args.bridge_display_gdi,
                    native_appcompat_view=args.native_appcompat_view,ntdll_timestamp=stamp,
                    appcompat_offsets=report.get('symbols',{}).get('appcompat_offsets'),
                    force_native_identity=args.try_native_identity,early_native_identity=args.early_native_identity,
                    direct_vulkan_icd=args.direct_vulkan_icd)
                if args.vulkan_icd: job['vulkan_icd'] = str(args.vulkan_icd.resolve())
                (build/'frame-job.json').write_text(json.dumps(job))
                cases += [('native_frame', native, '--frame-library')]
                if args.compare_appcompat:
                    cases += [(name,gui,'--thread-frame') for name in
                        ('thread_frame_unseparated','thread_frame_unseparated_2','thread_frame_unseparated_3')]
                cases += [('thread_frame', gui, '--thread-frame')]
                if args.compare_appcompat: cases += [('thread_frame_repeat',gui,'--thread-frame')]
            if initializer and args.try_d3d12_frame:
                weights = json.loads((ROOT/'artifacts/current/weight-source.json').read_text())
                for backend in ('cuda', 'legacy', 'phoenix', 'nvidia'):
                    width,height=(1920,1080) if args.benchmark or args.benchmark_x86_host else (640,384)
                    job = dict(width=width, height=height, backend=backend, benchmark=args.benchmark, benchmark_reset=args.benchmark_reset,
                        output=str(build/'d3d12-output.json'), cuda_diagnostic=backend=='cuda',
                        force_native_identity=args.try_native_identity,
                        cuda_no_preload=args.cuda_no_preload, cuda_no_jit=args.cuda_no_jit,
                        inspect_threads=args.inspect_threads,
                        bridge_create_dc=args.bridge_create_dc,
                        bridge_display_gdi=args.bridge_display_gdi,
                        bridge_kmt_escape=args.bridge_kmt_escape,
                        native_main_image=args.native_main_image, native_executable=str(native),
                        native_peb_image=args.native_peb_image, peb_image_offset=report.get('symbols',{}).get('peb_image_offset'),
                        native_appcompat_view=args.native_appcompat_view,
                        appcompat_offsets=report.get('symbols',{}).get('appcompat_offsets'),
                        native_registry=args.native_registry,
                        wrap_driver_threads=args.wrap_driver_threads,
                        repair_driver_tls=args.repair_driver_tls, tls_allocate_rva=allocator,
                        tls_free_rva=free_tls, ntdll_timestamp=stamp,
                        cuda_first=args.cuda_first,
                        preload_driver_object_loader=args.preload_driver_object_loader,
                        retry_native_reservation=args.retry_native_reservation,
                        early_native_identity=args.early_native_identity,
                        cuda_library=str(args.ngx_core.parent/'nvcuda64.dll'), ngx_core=str(args.ngx_core.resolve()),
                        kernel_library=str(blob), kernel_entry=kernel['entry'], kernel_block=kernel['block'],
                        model_library=weights['path'], native_library=str(args.native_library.resolve()),
                        config=dict(width=width, height=height, bundle_path=str(args.bundle.resolve()), weight_source=weights))
                    (build/f'd3d12-{backend}.json').write_text(json.dumps(job))
                    cases += [(f'native_d3d12_{backend}', native, '--frame-library'),
                              (f'thread_d3d12_{backend}', gui, '--thread-frame')]
                if args.try_x86_host:
                    job=json.loads((build/'d3d12-phoenix.json').read_text())
                    job['output']=str(build/'x86-host-output.json')
                    job['benchmark_x86_host']=args.benchmark_x86_host
                    (build/'x86-host-job.json').write_text(json.dumps(job))
                    cases += [('native_driver_host',native,'--x86-host-control'),('thread_driver_host',gui,'--thread-x86-host')]
        if args.case:
            selected = set(args.case)
            if selected.difference(name for name, _, _ in cases): raise ValueError('Unknown or disabled case')
            selected.update(('native_tls', 'native_phoenix_api', 'native_gpu'))
            if args.try_native_driver_arithmetic: selected.add('native_gpu_arithmetic')
            if args.try_portable_frame: selected.add('native_frame')
            if args.compare_appcompat: selected.update(('thread_frame_unseparated','thread_frame_unseparated_2',
                'thread_frame_unseparated_3','thread_frame','thread_frame_repeat'))
            if args.try_d3d12_frame:
                for name in tuple(selected):
                    if name.startswith('thread_d3d12_'): selected.add(name.replace('thread_', 'native_', 1))
            if args.try_x86_host:
                selected.update(('native_driver_host','thread_driver_host','native_d3d12_phoenix'))
            cases = [row for row in cases if row[0] in selected]
        for name, executable, mode in cases:
            command = [str(executable), mode, str(worker)]
            if name == 'native_frame': command = [str(executable), mode, str(build/'64/frame_worker64.dll'), str(build/'frame-job.json')]
            if name == 'native_gpu': command = [str(executable), mode, str(build/'64/gpu_worker64.dll'), str(blob)]
            if name == 'native_gpu_arithmetic': command = [str(executable),mode,str(build/'64/gpu_arithmetic_control64.dll'),str(blob)]
            if name == 'native_phoenix_api': command[-1] = str(args.native_library.resolve())
            if name=='native_driver_host': command=[str(executable),mode,str(build/'64/driver_worker64.dll'),str(build/'x86-host-job.json')]
            if name=='thread_driver_host': command += [f'{initializer:x}',f'{stamp:x}',str(build/'64/driver_worker64.dll'),str(build/'x86-host-job.json')]
            if 'repair' in name: command += [f'{rva:x}', f'{stamp:x}']
            if 'initialize' in name or name in ('thread_native', 'thread_gpu', 'thread_gpu_native_umd') or name.startswith('thread_gpu_native_driver') or name.startswith('thread_frame'): command += [f'{initializer:x}', f'{stamp:x}']
            if name in ('initialized_native', 'thread_native'): command.append(str(args.native_library.resolve()))
            if name in ('initialized_gpu', 'thread_gpu'): command += [str(build/'64/gpu_worker64.dll'), str(blob)]
            if name == 'thread_gpu_native_umd': command += [str(build/'64/gpu_native_umd64.dll'), str(blob)]
            if name == 'thread_gpu_native_driver': command += [str(build/'64/gpu_driver_bridge64.dll'), str(blob)]
            if name == 'thread_gpu_native_driver_arithmetic': command += [str(build/'64/gpu_driver_arithmetic64.dll'),str(blob)]
            if name.startswith('thread_gpu_native_driver') and args.cuda_first:
                if not args.ngx_core or not args.ngx_core.is_file(): raise ValueError('--cuda-first driver thunk needs --ngx-core FILE to locate the matching CUDA driver')
                command.append(str(args.ngx_core.parent/'nvcuda64.dll'))
            if name.startswith('thread_frame'):
                command += [str(build/'64/frame_worker64.dll'), str(build/'frame-job.json')]
                if args.compare_appcompat:
                    frame_job=json.loads((build/'frame-job.json').read_text())
                    frame_job['native_appcompat_view']=not name.startswith('thread_frame_unseparated')
                    (build/'frame-job.json').write_text(json.dumps(frame_job))
            if '_d3d12_' in name:
                backend = name.rsplit('_', 1)[1]
                if name.startswith('native_'):
                    command = [str(executable), mode, str(build/'64/probe-nvngx.dll'), str(build/f'd3d12-{backend}.json')]
                else:
                    command += [f'{initializer:x}', f'{stamp:x}', str(build/'64/probe-nvngx.dll'), str(build/f'd3d12-{backend}.json')]
                report['gpu_used'] = True
                if backend in ('phoenix','nvidia'):
                    report['model_files_requested'] = True
                    report['model_files_loaded'] = None
            if name in ('native_gpu', 'initialized_gpu', 'thread_gpu', 'thread_gpu_native_umd', 'native_frame', 'thread_frame') or 'gpu_native_driver' in name or name=='native_gpu_arithmetic': report['gpu_used'] = True
            if name=='native_frame' or name.startswith('thread_frame'):
                report['model_files_requested'] = True
                report['model_files_loaded'] = None
            report['probes'][name] = execute(command, executable.parent, timeout=75 if name.endswith('native_umd') or name.endswith('frame') or '_d3d12_' in name or name.endswith('driver_host') else 20)
            if name.endswith('driver_host') and (build/'x86-host-output.json').is_file():
                report['x86_host_frames'][name]=json.loads((build/'x86-host-output.json').read_text())
                (build/'x86-host-output.json').unlink()
                report['x86_host_conclusions']=x86_host_conclusions(report['probes'],report['x86_host_frames'],report['d3d12_frames'])
                if report['x86_host_frames'][name].get('passed'): report['model_files_loaded']=True
            faults=build/'frame-faults.jsonl'
            if faults.is_file():
                report['probes'][name]['first_chance_faults']=[json.loads(line) for line in faults.read_text().splitlines()]
                faults.unlink()
            if '_d3d12_' in name and (build/'d3d12-output.json').is_file():
                report['d3d12_frames'][name] = json.loads((build/'d3d12-output.json').read_text())
                (build/'d3d12-output.json').unlink()
                report['d3d12_conclusions'] = d3d12_conclusions(report['probes'], report['d3d12_frames'])
                report['model_parity_conclusions'] = model_parity_conclusions(report['probes'], report['d3d12_frames'])
                if args.benchmark:
                    report['benchmark_conclusions']=benchmark_conclusions(report['probes'],report['d3d12_frames'])
                if report['d3d12_frames'][name].get('passed') and report['d3d12_frames'][name].get('frames'):
                    report['model_files_loaded']=True
                report['legacy_conclusions'] = legacy_conclusions(report['probes'], report['d3d12_frames'])
            if (name=='native_frame' or name.startswith('thread_frame')) and (build/'frame-output.json').is_file():
                result=json.loads((build/'frame-output.json').read_text())
                report.setdefault('portable_variants',{})[name]=result
                if name in ('native_frame','thread_frame'):
                    report['portable_frame_control' if name=='native_frame' else 'portable_frame']=result
                (build/'frame-output.json').unlink()
                report['portable_conclusions'] = portable_conclusions(report['probes'],
                    report.get('portable_frame_control',{}), report.get('portable_frame',{}))
                if args.compare_appcompat: report['appcompat_comparison']=appcompat_conclusions(report['probes'],report['portable_variants'])
            save()
            print(json.dumps(dict(case=name, **report['probes'][name])), flush=True)
            if args.trace_cuda_init and name in ('native_d3d12_cuda','thread_d3d12_cuda'):
                script=build/'cuda-init.cdb'
                lines=['sxd 0x4000001f','sxd bpe',
                    'sxe -c ".echo PHX_CUDA_FAULT;.exr -1;kb 12;q" av',
                    'bu nvcuda64!cuInit ".echo PHX_CUDA_INIT;uf @rip;gc"']
                if args.cuda_indirect_slot_rva is not None:
                    slot=args.cuda_indirect_slot_rva
                    lines[-1]=f'bu /1 nvcuda64!cuInit ".echo PHX_CUDA_INIT;bl;uf @rip;.echo PHX_CUDA_TARGET;dq nvcuda64+0x{slot:x} L1;u poi(nvcuda64+0x{slot:x}) L20;gc"'
                for checkpoint in args.cuda_break_rva:
                    delta=checkpoint-report['cuda_init_rva']
                    lines.append(f'bu /1 nvcuda64!cuInit{delta:+#x} ".echo PHX_CUDA_POINT_{checkpoint:x};r;kb 6;'
                        '.if (wo(@rip-6)==0x15ff) {.echo PHX_CUDA_INDIRECT;dq @rip+dwo(@rip-4) L1;u poi(@rip+dwo(@rip-4)) L10;};u @rip L8;gc"')
                lines.append('g')
                script.write_text('\n'.join(lines)+'\n',encoding='ascii')
                trace=execute([str(debugger),'-G','-cf',str(script),*command],executable.parent,timeout=60,include_stdout=True,stdout_limit=45000)
                trace['syntax_valid']=not any(error in trace['stdout'] for error in ('Numeric expression missing','Extra character error','Syntax error','Unable to resolve unqualified'))
                trace['checkpoint_returns']={marker:int(value,16) for marker,value in re.findall(
                    r'^PHX_CUDA_POINT_([0-9a-f]+)\r?\nrax=([0-9a-f]+)',trace['stdout'],re.MULTILINE)}
                if (build/'d3d12-output.json').is_file():
                    trace['result']=json.loads((build/'d3d12-output.json').read_text())
                report.setdefault('cuda_init_traces',{})[name]=trace
                (build/'d3d12-output.json').unlink(missing_ok=True)
                save()
            if name == 'native_tls' and not tls_success(report['probes'][name]):
                script = build/'native-control.cdb'
                script.write_text('.symfix\n.reload /f ntdll.dll\n'
                    'bu ntdll!LdrpHandleTlsData ".printf \\"PHX_TLS_CONTROL module=%p count=%x reason=%x\\\\n\\", poi(@rcx+0x30), dwo(ntdll!LdrpActiveThreadCount), dwo(@rcx+0x10c); gc"\n'
                    'g\n', encoding='ascii')
                trace = subprocess.run([str(debugger), '-G', '-cf', str(script), str(native), '--tls', str(worker)],
                    cwd=native.parent, capture_output=True, text=True, timeout=30)
                report['native_control_trace'] = [line for line in (trace.stdout+trace.stderr).splitlines()
                    if 'PHX_' in line or 'tls_' in line or 'Exception' in line or 'error' in line.lower()][-30:]
                report['conclusions'] = conclusions(report['probes'])
                save()
                raise RuntimeError('Native TLS control failed; cross-mode claims refused. See native_control_trace.')
            if name == 'native_phoenix_api' and report['probes'][name]['exit_code'] != 0:
                raise RuntimeError('Normal x64 Phoenix API control failed; cross-mode comparison refused')
            if name == 'native_gpu' and report['probes'][name]['exit_code'] != 0:
                raise RuntimeError('Normal x64 GPU control failed; cross-mode comparison refused')
            if name == 'initialized_native' and report['probes'][name]['exit_code'] != 0:
                script = build/'phoenix-error.cdb'
                script.write_text('.symfix\nsxe eh\ng\ng\n.exr -1\n.ecxr\nkb 18\ng\n.exr -1\nkb 18\nq\n', encoding='ascii')
                trace = subprocess.run([str(debugger), '-G', '-cf', str(script), *command],
                    cwd=executable.parent, capture_output=True, text=True, timeout=45)
                output = trace.stdout+trace.stderr
                report['phoenix_error_trace'] = dict(debugger_exit=trace.returncode,
                    cpp_exception_observed='e06d7363' in output.lower(),
                    lines=[line for line in output.splitlines() if any(word in line.lower() for word in (
                        'exception', 'cxx', 'raise', 'dispatch', 'wow64', 'phoenix_native', 'phoenix_', 'framehandler'))][-45:])
                save()
            if (name.startswith('thread_gpu') or name == 'thread_frame' or name.startswith('thread_d3d12_')) and args.trace_gpu and debugger.is_file():
                script = build/'gpu.cdb'
                script.write_text('.symfix\nsxe -c ".echo PHX_GPU_FAILFAST;.exr -1;kb 20;q" c0000409\n'
                    'sxe -c ".echo PHX_GPU_AV;.exr -1;r;ub @rip L10;dq @$peb+0xf8 L2;kb 20;q" av\n'
                    'sxe -c ".echo PHX_GPU_EXCEPTION;.exr -1;kb 12;gn" eh\n'
                    'bu d3d12!D3D12CreateDevice ".echo PHX_CREATE_DEVICE;kb 5;gc"\ng\ng\n', encoding='ascii')
                trace = subprocess.run([str(debugger), '-G', '-cf', str(script), *command],
                    cwd=executable.parent, capture_output=True, text=True, timeout=60)
                output = trace.stdout+trace.stderr
                report['gpu_trace'] = dict(debugger_exit=trace.returncode, lines=[line for line in output.splitlines()
                    if any(word in line.lower() for word in ('modload:', 'exception', 'parameter[', 'phx_',
                        'd3d12', 'dxgi', 'nvwg', 'wow64', 'thread_task', 'worker_result',
                        'rax=', 'rdx=', 'rsp=', 'rip=', 'mov ', 'lea ', 'gs:', 'rcx=', 'cs='))][-120:])
                report['gpu_trace']['driver_images'] = driver_images(output)
                report['gpu_trace']['tail'] = output.splitlines()[-85:]
                if '_d3d12_' in name:
                    report.setdefault('d3d12_traces', {})[name] = report['gpu_trace']
                    (build/'d3d12-output.json').unlink(missing_ok=True)
                elif name=='thread_frame' and (build/'frame-output.json').is_file():
                    report['debugger_portable_frame']=json.loads((build/'frame-output.json').read_text())
                    (build/'frame-output.json').unlink()
                faults.unlink(missing_ok=True)
                save()
        if args.trace_dxgi and debugger.is_file():
            script = build/'dxgi.cdb'
            script.write_text('.symfix\nsxe av\n.reload /f ntdll.dll\n'
                'bu ntdll!LdrpHandleTlsData ".printf \\"PHX_TLS_HANDLER teb=%p vector=%p\\\\n\\", @$teb, poi(@$teb+0x58); gc"\n'
                'g\ng\n.exr -1\nkb 15\nq\n', encoding='ascii')
            result = subprocess.run([str(debugger), '-G', '-cf', str(script), str(gui), '--dxgi'],
                cwd=gui.parent, capture_output=True, text=True, timeout=60)
            output = result.stdout+result.stderr
            report['dxgi_trace'] = dict(debugger_exit=result.returncode,
                fatal_access_violation='ExceptionCode: c0000005' in output,
                lines=[line for line in output.splitlines() if any(word in line.lower() for word in (
                    'phx_', 'tls_', 'exceptioncode:', 'exceptionaddress:', 'dyn_tls', 'gdi32', 'user32', 'dxgi!',
                    'syntax error', 'couldn\'t resolve', 'could not resolve'))][-45:])
            save()
        if args.try_d3d12_frame:
            report['conclusions'] = d3d12_conclusions(report['probes'], report['d3d12_frames'])
            report['d3d12_conclusions'] = report['conclusions']
        else:
            report['conclusions'] = conclusions(report['probes'])
        if args.try_native_driver_bridge:
            report['driver_bridge_conclusions'] = driver_bridge_conclusions(report['probes'])
        if args.try_native_driver_arithmetic:
            mapped=dict(native_gpu=report['probes'].get('native_gpu_arithmetic',{}),
                thread_gpu_native_driver=report['probes'].get('thread_gpu_native_driver_arithmetic',{}))
            result=driver_bridge_conclusions(mapped)
            result['scope']='arithmetic_shader_creation_only'
            result['shader_creation_succeeded']=result.pop('module_creation_succeeded')
            report['driver_arithmetic_conclusions']=result
        if args.try_x86_host:
            report['schema']='phoenix-wow64-x86-host-probe/1'
            report['x86_host_conclusions']=x86_host_conclusions(report['probes'],report['x86_host_frames'],report['d3d12_frames'])
            report['conclusions']=report['x86_host_conclusions']
        if args.benchmark_x86_host:
            report['x86_host_benchmark_conclusions']=x86_host_benchmark_conclusions(report['probes'],report['x86_host_frames'],report['d3d12_frames'])
            report['conclusions']=report['x86_host_benchmark_conclusions']
        save()
        print(json.dumps(report['conclusions']), flush=True)
    if args.benchmark_x86_host:
        return 0 if report.get('x86_host_benchmark_conclusions',{}).get('passed') else 1
    if args.benchmark:
        return 0 if all(row['passed'] for row in report.get('benchmark_conclusions',{}).values()) and len(report.get('benchmark_conclusions',{}))==2 else 1
    if args.try_x86_host:
        return 0 if report.get('x86_host_conclusions',{}).get('bit_exact_against_native_api') else 1
    return 0 if tls_success(report['probes'].get('native_tls', {})) else 1


if __name__ == '__main__':
    raise SystemExit(main())
