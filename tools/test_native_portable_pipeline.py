"""Development-only portable engines in the shared native D3D12 stage pipeline."""
import ctypes as C
import hashlib
import os
import json
from pathlib import Path
import shutil
import subprocess
import _ctypes
import numpy as np
from test_model_selection import ROOT
from build_experimental_rtx_addon import extract_runtime
from compare_neural_controls import NativeControls, configure


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def checks(library, work, model, gpu=False):
    handles=[]
    try:
        return _checks(library,work,model,gpu,handles)
    finally:
        for api in reversed(handles):_ctypes.FreeLibrary(api._handle)


def _checks(library, work, model, gpu, handles):
    root=work/'portable-provider';scratch=work/'portable-resources'
    root.mkdir();scratch.mkdir()
    extract_runtime(ROOT/'artifacts/current/addon/Phoenix.addon64',root,scratch)
    shutil.copy2(ROOT/'artifacts/current/portable-pipeline/phoenix_runtime_cpp.dll',root/'phoenix_runtime_cpp.dll')
    devices=[('cpu','cpu')]
    if gpu:
        found=json.loads(subprocess.check_output([str(root/'phoenix_cpp_devices.exe'),str(root)],text=True))
        for item in found['devices']:
            if item['id'].startswith(('cuda:','vulkan:')):
                devices.append((item['id'].split(':')[0],item['id']))
    sdk=C.CDLL(str(library));sdk.phoenix_native_create.argtypes=[C.c_char_p,C.c_void_p,C.c_void_p];sdk.phoenix_native_create.restype=C.c_void_p
    sdk.phoenix_native_destroy.argtypes=[C.c_void_p];sdk.phoenix_native_error.restype=C.c_char_p
    sdk.phoenix_native_status.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];sdk.phoenix_native_status.restype=C.c_size_t
    sdk.phoenix_native_frame_host.argtypes=[C.c_void_p]+[t for _ in range(3) for t in (C.c_void_p,C.c_size_t,C.c_size_t)]+[C.POINTER(NativeControls)]
    # status() in older helpers has a fixed runtime symbol; read the SDK directly.
    def report(session):
        size=sdk.phoenix_native_status(session,None,0);text=C.create_string_buffer(size)
        if not size or sdk.phoenix_native_status(session,text,size)!=size:raise RuntimeError('SDK status failed')
        return json.loads(text.value)
    handles.append(sdk)
    runtime=configure(root/'phoenix_runtime_cpp.dll')
    handles.append(runtime)
    w,h=32,24;cases=[]
    y,x=np.mgrid[:h,:w];rgba=np.empty((h,w,4),np.float16)
    rgba[...,0]=.1+(x%9)*.07;rgba[...,1]=.2+(y%7)*.08;rgba[...,2]=.15+((x+y)%5)*.1;rgba[...,3]=.25+(x%4)*.125
    rgb=rgba[...,:3].astype(np.float32).copy();motion=np.zeros((h,w,2),np.float32)
    for backend,device in devices:
        for selected in ('original','small-fp8'):
            blocks=1 if backend=='cpu' else 8
            weights={'kind':'nvidia_dll','path':str(model),'sha256':sha(model),'model':selected}
            runner={'runtime_root':str(root),'runtime_sha256':sha(root/'phoenix_runtime_cpp.dll'),
                'backend':backend,'device':device,'allow_unvalidated':True,'cpu_threads':2,'weight_source':weights}
            config={'debug':True,'width':w,'height':h,'weight_source':weights,'portable_runtime':runner,
                'pipeline':{'PipelineMonitoring':0},'tuning':{'central_blocks':blocks}}
            session=sdk.phoenix_native_create(json.dumps(config).encode(),None,None)
            reference=runtime.phoenix_runtime_create(json.dumps(runner).encode(),w,h)
            if not session or not reference:
                message=str(sdk.phoenix_native_error())+str(runtime.phoenix_runtime_error())
                if session:sdk.phoenix_native_destroy(session)
                if reference:runtime.phoenix_runtime_destroy(reference)
                raise RuntimeError(message)
            errors=[]
            try:
                for reset in (1,0,1):
                    controls=NativeControls(C.sizeof(NativeControls),reset,1,1,1,1,-1,0,1)
                    actual=np.empty_like(rgba);expected=np.empty_like(rgb)
                    meta={'reset':bool(reset),'has_motion':True,'has_depth':False,'passes':1,'transformer_blocks':blocks,
                        'controls':{'auto_mask':False},'adaptive_budget':{'enabled':False}}
                    if runtime.phoenix_runtime_frame(reference,rgb.ctypes.data,motion.ctypes.data,None,None,json.dumps(meta).encode(),expected.ctypes.data):
                        raise RuntimeError(runtime.phoenix_runtime_error().decode())
                    mv=np.zeros((h,w,2),np.float16)
                    if sdk.phoenix_native_frame_host(session,rgba.ctypes.data,rgba.nbytes,w*8,mv.ctypes.data,mv.nbytes,w*4,
                        actual.ctypes.data,actual.nbytes,w*8,C.byref(controls)):
                        raise RuntimeError(sdk.phoenix_native_error().decode())
                    error=float(np.max(np.abs(actual[...,:3].astype(np.float32)-expected)))
                    if error>.001 or not np.array_equal(actual[...,3],rgba[...,3]):raise RuntimeError(f'Portable bridge oracle differs: {error}')
                    errors.append(error)
                state=report(session)
                if state['backend']!='portable_pipeline_d3d12' or len(state['portable_neural_stages'])!=1:raise RuntimeError('Wrong execution backend')
                if state['processing_host_pid']!=os.getpid() or len(state['inference_pids'])!=1 or state['inference_pid'] in (0,os.getpid()):
                    raise RuntimeError('Portable worker was misidentified as its parent host')
                cases.append({'passed':True,'backend':backend,'device':device,'model':selected,'oracle_max_errors':errors,
                    'alpha_preserved':True,'reset_history_reset_checked':True,'blocks':blocks})
                print(f'{selected} {device}: SDK portable neural pixels checked.',flush=True)
            finally:
                sdk.phoenix_native_destroy(session);runtime.phoenix_runtime_destroy(reference)
            if backend=='cuda':
                for clock,offline in (('source_call',False),('backend_gpu',False),('backend_gpu',True)):
                    budget_config=json.loads(json.dumps(config))
                    budget_config['offline']=offline
                    budget_config['neural_budget_clock']=clock
                    budget_config['tuning']={'central_blocks':4,'minimum_central_blocks':3,'max_processing_ms':.01}
                    trial=sdk.phoenix_native_create(json.dumps(budget_config).encode(),None,None)
                    if not trial:raise RuntimeError(sdk.phoenix_native_error().decode())
                    try:
                        controls.reset=1;out=np.empty_like(rgba)
                        for frame in range(32):
                            if sdk.phoenix_native_frame_host(trial,rgba.ctypes.data,rgba.nbytes,w*8,None,0,0,out.ctypes.data,out.nbytes,w*8,C.byref(controls)):
                                raise RuntimeError(sdk.phoenix_native_error().decode())
                        trial_status=report(trial)
                        if trial_status['tuning']['central_blocks']!=(4 if offline else 3) or trial_status['gpu_monitoring_enabled']:
                            raise RuntimeError('Independent budget/offline quality contract failed')
                        if not offline and trial_status['pipeline_telemetry']['neural_budget'].get('clock')!=clock:
                            raise RuntimeError('Budget used a different timing domain')
                        if not offline and clock=='backend_gpu' and not trial_status['gpu_sampled_last_frame']:
                            raise RuntimeError('GPU budget did not receive backend GPU timings')
                    finally:sdk.phoenix_native_destroy(trial)
                cases[-1]['monitoring_off_budget_and_offline_fixed']=True
            # Exercise geometry, NIS, ordering, fusion and independent passes in
            # the actual adapter, with all temporal services enabled. Their
            # individual native shaders have independent oracles in the suite.
            config['output_width']=w*2;config['output_height']=h*2
            config['pipeline']={'ProcessingChain':True,'ChainScale':True,'AdditionalScaler':1,
                'ChainColor':True,'ChainBrightness':.05,'MultiscaleFusion':True,'ChainFusion':True,
                'MotionEstimation':True,'SceneDetection':True,'PipelineMonitoring':1}
            config['tuning']['network_passes']=2
            session=sdk.phoenix_native_create(json.dumps(config).encode(),None,None)
            if not session:raise RuntimeError(sdk.phoenix_native_error().decode())
            try:
                output=np.empty((h*2,w*2,4),np.float16);controls.reset=1
                if sdk.phoenix_native_frame_host(session,rgba.ctypes.data,rgba.nbytes,w*8,None,0,0,output.ctypes.data,output.nbytes,w*16,C.byref(controls)):
                    raise RuntimeError(sdk.phoenix_native_error().decode())
                state=report(session)
                if not np.isfinite(output).all() or state['independent_temporal_stages']!=4:raise RuntimeError('Portable stage graph did not execute all branches/passes')
                if len(set(state['inference_pids']))!=4 or any(pid in (0,os.getpid()) for pid in state['inference_pids']):
                    raise RuntimeError('Independent stages did not report distinct native workers')
                cases[-1]['geometry_color_nis_fusion_passes']=True
                cases[-1]['distinct_native_worker_identities']=True
                cases[-1]['stage_histories']=state['independent_temporal_stages']
            finally:sdk.phoenix_native_destroy(session)
    shutil.copy2(library,root/'phoenix_native.dll')
    from test_model_selection import toolchain,run
    compiler,env=toolchain(64)
    executable=work/'portable-frame-pipeline.exe'
    flags=[compiler,'/nologo','/std:c++20','/Zc:__cplusplus','/EHsc','/O2','/MT','/DNOMINMAX',
        '/I'+str(ROOT/'native_runtime'),'/I'+str(ROOT.parent/'external/DLSS/include')]
    run(flags+[ROOT/'tests/native/portable_frame_pipeline_tests.cpp','/Fo:'+str(work/'portable-frame-pipeline.obj'),
        '/Fe:'+str(executable),'d3d11.lib','d3d12.lib','dxgi.lib','d3dcompiler.lib','user32.lib','bcrypt.lib',
        'psapi.lib','cabinet.lib','shell32.lib','ole32.lib','uuid.lib','windowscodecs.lib'],cwd=work,env=env)
    host_bridge=json.loads(run([executable,root,model],cwd=work,env=env))
    if not host_bridge.get('passed'):raise RuntimeError('Shared asynchronous capture pipeline failed')
    included={'phoenix_cpp_devices.exe','phoenix_native_broker.exe','phoenix_native_cpp.json','phoenix_runtime_cpp.dll',
        'phoenix_cpu_cpp.dll','phoenix_cuda_cpp.dll','phoenix_vulkan_cpp.dll','phoenix_frame.exe'}
    files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and
        (p.relative_to(root).as_posix() in included or p.relative_to(root).as_posix().startswith('vulkan/'))}
    return {'passed':True,'cases':cases,'python_required_at_runtime':False,'backend_execution':'native workers and D3D12 shared stages',
        'asynchronous_capture_bridge':host_bridge,'source_addon_sha256':sha(ROOT/'artifacts/current/addon/Phoenix.addon64'),'provider_files':files}

