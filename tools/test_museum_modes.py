"""GPU regression for a local DX12 museum; --background isolates camera input.

Use --helper to check the isolated renderer instead of automatic routing.
Restores the INI and closes its own fixture/helper processes on completion.
Disabled saved processing fails before any mode is changed, unless the explicit
diagnostic flag --allow-disabled-startup is supplied.
"""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import traceback
import shutil
import sys
import configparser
import io
import argparse
import psutil
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bits', type=int, choices=(32,64), default=64)
parser.add_argument('--helper', action='store_true')
parser.add_argument('--background', action='store_true')
parser.add_argument('--allow-disabled-startup', action='store_true')
parser.add_argument('--live-tuning', action='store_true')
args = parser.parse_args()
architecture = 'x86' if args.bits == 32 else 'x64'
expect_helper = args.helper or args.bits == 32
suffix = '-x86' if args.bits == 32 else ''
GAME = ROOT.parent / ('dlss5-everything-testing/games/dx12-' + architecture)
MAIL = GAME / 'PhoenixOutput/bridge'
kernel = ctypes.WinDLL('kernel32')
kernel.GetTickCount64.restype = ctypes.c_ulonglong
process = None
endpoint = {}
created = []
report = {'bits': args.bits, 'cases': [], 'usability': [], 'live_tuning': []}
images = {}
ini = (GAME / 'ReShade.ini').read_bytes()
output = Path(tempfile.mkdtemp(prefix='modes-test-', dir=ROOT / 'build'))
helpers_before = {p.pid for p in psutil.process_iter(['name']) if p.info['name'] == 'phoenix_graphics_host.exe'}

def emit(value):
    print(json.dumps(value), flush=True)

def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return {}

def wait(predicate, timeout=35):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Fixture exited: {process.returncode}')
        result = predicate()
        if result:
            return result
        time.sleep(.1)
    raise TimeoutError('Runtime check timed out')

def command(op='status', **fields):
    token = uuid.uuid4().hex
    request = dict(schema='phoenix-control/1', session=endpoint['session'], pid=process.pid,
                   id=token, op=op, expires_tick_ms=kernel.GetTickCount64()+29000, **fields)
    temporary = MAIL / 'request-test.tmp'
    temporary.write_text(json.dumps(request))
    os.replace(temporary, MAIL / 'request.json')
    value = wait(lambda: r if (r := read(MAIL / 'response.json')).get('id') == token else None)
    assert value.get('ok'), value
    return value

def status():
    return command()['snapshot']

def summary(s):
    i = s.get('in_process') or s.get('helper') or {}
    return {k: s.get(k) for k in ('phase','status','execution_mode','fallback_reason','helper_pid','delivered','presented')} | {
        'nvidia': i.get('nvidia'), 'nvidia_status': i.get('nvidia_status'), 'error': i.get('error'),
        'settings': {k:s.get('applied_settings',{}).get(k) for k in ('Backend','Comparison','SourceA','SourceB')},
        'gpu_rows': (i.get('gpu_timing') or {}).get('rows',[])}

def check_route(snapshot):
    expected = 'helper' if expect_helper else 'in_process'
    assert snapshot.get('execution_mode') == expected, summary(snapshot)
    assert bool(snapshot.get('helper_pid')) == (expected == 'helper'), summary(snapshot)

def check_startup(snapshot, allow_disabled=False):
    result = {'enabled': snapshot.get('enabled'),
              'backend': snapshot.get('applied_settings', {}).get('Backend'),
              'disabled_override': allow_disabled and snapshot.get('enabled') is False}
    if snapshot.get('enabled') is not True and not allow_disabled:
        raise RuntimeError('Saved processing is disabled or unavailable. Enable Phoenix DLSS5 before testing; '
                           '--allow-disabled-startup is diagnostic only and restores the disabled setting.')
    return result

def capture(name, keep=True):
    r = command('screenshot')
    path = Path(r['screenshot_path'])
    created.append(path)
    wait(lambda: (s := status())['screenshot']['path'] == str(path) and not s['screenshot']['pending'])
    with Image.open(path) as image:
        result = np.asarray(image.convert('RGB')).copy()
        if keep:
            image.save(ROOT / 'build' / ('temp-mode' + suffix + '-' + name + '.png'))
    return result

try:
    if args.helper:
        config=configparser.ConfigParser(interpolation=None)
        config.optionxform=str
        config.read_string(ini.decode('utf-8-sig'))
        config['Phoenix.Compatibility']['ExecutionMode']='helper'
        text=io.StringIO();config.write(text)
        (GAME/'ReShade.ini').write_text(text.getvalue())
    extra=['--background'] if args.background else []
    process = subprocess.Popen([str(GAME/('Museum_dx12_' + architecture + '.exe')),'--windowed','--width','960','--height','540',
                                *extra,'--fixed-time','--seconds','360','--output',str(output)],cwd=GAME)
    emit({'launched': process.pid})
    endpoint = wait(lambda: e if (e:=read(MAIL/'endpoint.json')).get('pid') == process.pid else None,60)
    startup = status()
    report['startup'] = {'enabled': startup.get('enabled'),
                         'backend': startup.get('applied_settings', {}).get('Backend')}
    report['startup'] = check_startup(startup, args.allow_disabled_startup)
    emit({'startup': report['startup']})
    if report['startup']['disabled_override']:
        emit({'warning': 'Processing is temporarily enabled for diagnostics; saved disabled startup is restored.'})
        command('set', enabled=True)
    for name, settings in [
        ('input', {'Backend':0,'Comparison':0,'SourceA':0,'SourceB':2}),
        ('phoenix', {'Backend':3,'Comparison':0,'SourceA':0,'SourceB':2}),
        ('input-phoenix', {'Backend':2,'Comparison':1,'SourceA':0,'SourceB':2}),
        ('nvidia', {'Backend':1,'Comparison':0}),
        ('nvidia-phoenix', {'Backend':2,'Comparison':1,'SourceA':1,'SourceB':2}),
        ('phoenix-nvidia', {'Backend':2,'Comparison':1,'SourceA':2,'SourceB':1}),
        ('input-nvidia', {'Backend':2,'Comparison':1,'SourceA':0,'SourceB':1}),
        ('triple', {'Backend':2,'Comparison':0}),
        ('input-difference', {'Backend':2,'Comparison':2,'SourceA':0,'SourceB':2}),
    ]:
        item = {'name':name,'settings':settings,'passed':False}
        report['cases'].append(item)
        emit({'testing':name})
        started = time.monotonic()
        # Never hide a mode-change bug by re-enabling processing on every case.
        command('set',settings=settings | {'ForceReset':False,'Divider':.37,'ComparisonLabels':False,'ComparisonLines':False})
        deadline = time.monotonic()+27
        previous = None
        while time.monotonic() < deadline:
            s = status()
            item['last_status'] = summary(s)
            if settings['Backend']==0:
                item['passed']=True
                break
            if s['phase'] == 5 or not s['enabled']:
                break
            if s['delivered'] >= 3 and s['presented'] >= 3 and s.get('presented_revision') == s.get('selection_revision') and s.get('applied_settings',{}).get('Backend') == settings['Backend']:
                check_route(s)
                item.setdefault('first_presented_seconds', round(time.monotonic()-started,3))
                need_nv = settings['Backend']==1 or settings['Backend']==2 and (settings.get('SourceA',1)==1 or settings.get('SourceB',2)==1)
                info=s.get('helper') or s.get('in_process') or {}
                have_nv=info.get('nvidia',{}).get('evaluates',0)>=20 and info.get('nvidia',{}).get('ready',False)
                if previous is not None and s['delivered'] > previous and (not need_nv or have_nv):
                    item['passed']=True
                    break
                previous=s['delivered']
            time.sleep(.25)
        if item['passed']:
            pixels=capture(name)
            item['fixture_status']=read(output/'status.json')
            images[name]=pixels
            item['pixel_mean']=float(pixels.mean())
            if name == 'phoenix':
                item['unchanged_mode_temporal_mae'] = [
                    float(np.abs(capture('temporal',keep=False).astype(float)-pixels.astype(float)).mean())
                    for _ in range(4)]
                emit({'phoenix_temporal_mae':item['unchanged_mode_temporal_mae']})
            if name in ('phoenix','nvidia') and 'input' in images:
                h,w=pixels.shape[:2]
                center=(slice(h//4,3*h//4),slice(w//4,3*w//4))
                delta=np.abs(pixels[center].astype(float)-images['input'][center].astype(float))
                item['central_mean_difference_from_input']=float(delta.mean())
                item['central_changed_fraction_gt2']=float((delta.max(axis=2)>2).mean())
                item['passed']=delta.mean()>.25 and item['central_changed_fraction_gt2']>.01
                emit({'pixel_checks':name,'difference':item['central_mean_difference_from_input'],'passed':item['passed']})
            expected={'input-phoenix':('input','phoenix'),'nvidia-phoenix':('nvidia','phoenix'),
                      'phoenix-nvidia':('phoenix','nvidia'),'input-nvidia':('input','nvidia')}.get(name)
            if expected and all(key in images for key in expected):
                h,w=pixels.shape[:2]
                split=int(w*.37)
                bands=[(slice(20,h-20),slice(20,split-12)),(slice(20,h-20),slice(split+12,w-20))]
                item['half_errors']=[float(np.abs(pixels[band].astype(float)-images[key][band].astype(float)).mean()) for band,key in zip(bands,expected)]
                item['passed']=all(error<1.5 for error in item['half_errors'])
                emit({'pixel_checks':name,'half_errors':item['half_errors'],'passed':item['passed']})
        else:
            helper=s.get('helper_pid')
            if helper and psutil.pid_exists(helper):
                log=Path(psutil.Process(helper).exe()).parent/'ReShade.log'
                if log.exists():
                    item['helper_log_tail']=log.read_text(errors='replace').splitlines()[-45:]
        emit({'case':name,'passed':item['passed'],'status':{k:v for k,v in item['last_status'].items() if k!='gpu_rows'}})
    def selection_ready(mode):
        current = status()
        if current['phase'] == 5:
            raise RuntimeError(current['status'])
        ready = (current['enabled'] and
            current['applied_settings']['Backend'] == mode and
            current['presented_revision'] == current['selection_revision'] and
            current['presented'] >= 5)
        if ready:
            check_route(current)
        return current if ready else None

    if not expect_helper:
        initial_creates = next(c for c in report['cases'] if c['name']=='nvidia')['last_status']['nvidia']['creates']
        assert initial_creates == 1, 'NVIDIA DLSS5 feature was not initialized exactly once at fixed geometry'
        returns = []
        for mode in (1, 3, 1):
            started = time.monotonic()
            command('set', settings={'Backend':mode})
            ready = wait(lambda: selection_ready(mode),60)
            info = ready['in_process']['nvidia']
            assert info['creates'] == initial_creates, 'Backend switch recreated the NVIDIA DLSS5 feature'
            if mode == 1:
                pixels = capture('nvidia-return',keep=False)
                error = float(np.abs(pixels.astype(float)-images['nvidia'].astype(float)).mean())
                assert error < 1.5, ('NVIDIA DLSS5 return image mismatch', error)
                returns.append({'first_presented_seconds':round(time.monotonic()-started,3),
                                'creates':info['creates'],'pixel_mae':error})
        report['nvidia_reuse'] = {'passed':True,'helper_pid':0,'returns':returns}

    if args.live_tuning:
        defaults = {'OverrideStrengths':False,'Tone':1.0,'Structure':1.0,'Advanced':False,
                    'AutoMask':False,'Skin':-1.0,'NativeOptions':False,'Intensity':1.0,'Style':0}
        for mode, name in ((3,'phoenix'),(1,'nvidia')):
            command('set', settings=defaults | {'Backend':mode})
            baseline = wait(lambda: selection_ready(mode),60)
            pid = baseline['helper_pid']
            captures = []
            revisions = []
            for level in (.1,1.9):
                before = baseline['selection_revision']
                command('set',settings={'OverrideStrengths':True,'Tone':level,'Structure':level})
                baseline = wait(lambda: selection_ready(mode),60)
                assert baseline['helper_pid'] == pid and not baseline['restart_pending'], 'Tuning restarted the renderer'
                assert baseline['selection_revision'] > before
                info = baseline.get('helper') or baseline.get('in_process') or {}
                revision = info.get('live_tuning',{})
                if expect_helper:
                    assert revision['requested_revision'] == revision['presented_revision'] > 0, revision
                revisions.append(revision)
                captures.append(capture(name+'-strength-'+str(level),keep=False).astype(float))
            strength_delta = float(np.abs(captures[0]-captures[1]).mean())
            assert strength_delta > .1, ('Strength controls had no visible effect', name, strength_delta)
            skin_images = []
            for skin in (0.0,2.0):
                command('set', settings={'OverrideStrengths':False,'Tone':1.0,'Structure':1.0,
                                        'Advanced':True,'AutoMask':True,'Skin':skin})
                current = wait(lambda: selection_ready(mode),60)
                assert current['helper_pid'] == pid and not current['restart_pending'], 'Skin tuning restarted the renderer'
                skin_images.append(capture(name+'-skin-'+str(skin),keep=False).astype(float))
            skin_delta = float(np.abs(skin_images[0]-skin_images[1]).mean())
            assert skin_delta > .01, ('Skin control had no visible effect', name, skin_delta)
            command('set',settings=defaults)
            restored = wait(lambda: selection_ready(mode),60)
            assert restored['helper_pid'] == pid
            restored_pixels = capture(name+'-restored',keep=False).astype(float)
            restore_error = float(np.abs(restored_pixels-images[name].astype(float)).mean())
            assert restore_error < 1.5, ('Tuning did not restore the baseline image', restore_error)
            item = {'backend':name,'passed':True,'helper_pid':pid,'strength_delta_mae':strength_delta,
                    'skin_delta_mae':skin_delta,'restored_mae':restore_error,'tuning_revisions':revisions}
            if expect_helper:
                def module_names(target):
                    return sorted({Path(m.path).name for m in psutil.Process(target).memory_maps()
                                   if any(key in Path(m.path).name.lower() for key in ('phoenix','nvngx'))})
                item['game_modules'] = module_names(process.pid)
                item['helper_modules'] = module_names(pid)
                assert 'Phoenix.addon32' in item['game_modules'] if args.bits == 32 else True
                assert 'Phoenix.addon64' in item['helper_modules']
                assert 'Phoenix.addon64' not in item['game_modules'] if args.bits == 32 else True
            report['live_tuning'].append(item)
            emit({'live_tuning':item})

    differences = {}
    for view in (2, 3, 4, 5):
        command('set',settings={'Backend':2,'Comparison':view,'SourceA':0,'SourceB':2,'DifferenceGain':1.0})
        wait(lambda: selection_ready(2),60)
        image = capture('difference-' + str(view),keep=False).astype(float)
        assert image.std() > .5 and float(np.abs(image-images['input']).mean()) > 1
        differences[view] = image
    assert all(float(np.abs(differences[a]-differences[b]).mean())>.25 for a,b in ((2,3),(3,4),(4,5)))
    assert np.max(differences[5].max(2)-differences[5].min(2)) <= 1, 'Luminance view is not grayscale'
    command('set',settings={'DifferenceGain':8.0})
    wait(lambda: selection_ready(2),60)
    amplified = capture('difference-gain',keep=False).astype(float)
    assert amplified.mean() > differences[5].mean()*1.1, 'Difference gain has no visible effect'
    report['usability'].append({'scenario':'all-four-difference-views-and-gain','passed':True,
                               'luminance_mean_1x':float(differences[5].mean()),'luminance_mean_8x':float(amplified.mean())})

    for mode in (3, 1):
        command('set', settings={'Backend':mode})
        wait(lambda: selection_ready(mode),60)
        command('set', enabled=False)
        wait(lambda: not status()['enabled'])
        command('set', settings={'Backend':2,'Comparison':1,'SourceA':0,'SourceB':2})
        assert not status()['enabled'], 'Changing selection unexpectedly enabled processing'
        disabled = capture('disabled-' + str(mode))
        assert float(np.abs(disabled.astype(float)-images['input'].astype(float)).mean()) < 1.5
        command('set', enabled=True)
        wait(lambda: selection_ready(2),60)
        report['usability'].append({'scenario':'off-select-comparison-on','from_backend':mode,'passed':True})
    for source_a, source_b, divider in ((0,1,.25),(1,2,.65),(2,0,.37)):
        command('set',settings={'Backend':2,'Comparison':1,'SourceA':source_a,'SourceB':source_b,'Divider':divider})
    final = wait(lambda: selection_ready(2),60)
    assert (final['applied_settings']['SourceA'],final['applied_settings']['SourceB']) == (2,0)
    pixels = capture('rapid-final-selection')
    h,w = pixels.shape[:2]
    split = int(w*.37)
    bands = [(slice(20,h-20),slice(20,split-12)),(slice(20,h-20),slice(split+12,w-20))]
    errors = [float(np.abs(pixels[band].astype(float)-images[key][band].astype(float)).mean())
              for band,key in zip(bands,('phoenix','input'))]
    assert all(error < 1.5 for error in errors), errors
    report['usability'].append({'scenario':'rapid-backend-source-divider-changes','passed':True,'half_errors':errors})
except Exception as error:
    report['error']=str(error)
    emit({'error':str(error),'traceback':traceback.format_exc()})
finally:
    if process is not None and process.poll() is None:
        user=ctypes.WinDLL('user32')
        user.GetWindowThreadProcessId.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        user.PostMessageW.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_size_t,ctypes.c_ssize_t]
        @ctypes.WINFUNCTYPE(ctypes.c_bool,ctypes.c_void_p,ctypes.c_void_p)
        def close(hwnd,_):
            pid=ctypes.c_ulong()
            user.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
            if pid.value==process.pid:
                user.PostMessageW(hwnd,0x10,0,0)
            return True
        user.EnumWindows(close,0)
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(10)
    for p in psutil.process_iter(['name']):
        if p.info['name']=='phoenix_graphics_host.exe' and p.pid not in helpers_before:
            p.terminate()
            try:
                p.wait(10)
            except psutil.TimeoutExpired:
                p.kill()
    (GAME/'ReShade.ini').write_bytes(ini)
    for path in created:
        path.unlink(missing_ok=True)
    assert output.resolve().parent==(ROOT/'build').resolve() and not output.is_symlink()
    shutil.rmtree(output)
    report['closed']=process is None or process.poll() is not None
    report['passed']=len(report['cases']) == 9 and all(c['passed'] for c in report['cases']) and len(report['usability']) == 4 and 'error' not in report and (expect_helper or report.get('nvidia_reuse',{}).get('passed',False)) and (not args.live_tuning or len(report['live_tuning']) == 2)
    (ROOT/('build/temp_modes' + suffix.replace('-','_') + '_result.json')).write_text(json.dumps(report,indent=2))
    emit({'closed':report['closed'],'results':[(c['name'],c['passed']) for c in report['cases']]})

sys.exit(0 if report['passed'] else 1)
