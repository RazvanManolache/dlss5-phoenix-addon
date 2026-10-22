"""Build and exercise the actual native pipeline DLL; GPU execution is opt-in."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import struct
import subprocess
from test_model_selection import ROOT, toolchain, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--tiny-checkpoint', type=Path)
    parser.add_argument('--model-dll',type=Path,help='Optional authenticated dual-model NVIDIA DLL for actual addon chain execution')
    args = parser.parse_args()
    output = ROOT / 'artifacts/current/pipeline-native'
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'validation.json'
    report = {'passed': False, 'gpu_requested': args.gpu, 'cases': []}
    inputs = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ('src', 'native_runtime', 'third_party', 'tests/native')
        for path in (ROOT / directory).rglob('*') if path.is_file() and path.suffix in ('.h', '.hpp', '.cpp', '.inl', '.inc')}
    try:
        compiler, env = toolchain(64)
        with tempfile.TemporaryDirectory(prefix='pipeline-runtime-', dir=ROOT / 'build') as temporary:
            work = Path(temporary)
            flags = [compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/EHsc', '/O2', '/bigobj', '/MT',
                     '/experimental:deterministic', '/pathmap:'+str(ROOT.parent)+'=.',
                     '/DNOMINMAX', '/I' + str(ROOT / 'native_runtime'),
                     '/I' + str(ROOT.parent / 'external/DLSS/include')]
            libraries = ['d3d12.lib', 'dxgi.lib', 'd3dcompiler.lib', 'bcrypt.lib']
            dll = work / 'phoenix_native.dll'
            run(flags + ['/LD', ROOT / 'native_runtime/native_api.cpp', '/Fo:' + str(work / 'native.obj'),
                         '/Fe:' + str(dll), *libraries, '/link', '/IMPLIB:' + str(work / 'phoenix_native.lib')],
                cwd=work, env=env)
            print('Native pipeline DLL compiled.', flush=True)
            suites = [('contracts', 'pipeline_contract_tests.cpp', [str(work)], False),
                      ('spatial', 'native_library_tests.cpp', [], True),
                      ('software-pipeline', 'pipeline_native_tests.cpp', [], True),
                      ('native-chain', 'pipeline_chain_tests.cpp', [], True),
                      ('nis-odd-upscale', 'pipeline_native_tests.cpp', ['--nis'], True)]
            suites.append(('addon-nis-reconstruction', 'pipeline_scaler_bridge_tests.cpp', [], False))
            suites.append(('addon-recorded-chain', 'pipeline_recorded_chain_tests.cpp', [], False))
            if args.gpu:
                suites.append(('hardware-motion', 'pipeline_native_tests.cpp', ['--gpu'], True))
                suites.append(('hardware-motion-chain-view', 'pipeline_native_tests.cpp', ['--chain-motion'], True))
                suites.append(('motion-translation-accuracy', 'pipeline_motion_accuracy_tests.cpp', [], False))
                if args.model_dll:suites.append(('addon-neural-chain-both-models','pipeline_recorded_chain_tests.cpp',
                    [str(ROOT/'artifacts/current/native-kernels'),str(args.model_dll.resolve())],False))
            for name, source, arguments, linked in suites:
                executable = work / (name + '.exe')
                run(flags + [ROOT / 'tests/native' / source, '/Fo:' + str(work / (name + '.obj')),
                             '/Fe:' + str(executable), *([work / 'phoenix_native.lib'] if linked else []),
                             *libraries], cwd=work, env=env)
                result = json.loads(run([executable, *arguments], cwd=work, env=env))
                if not result.get('passed'): raise RuntimeError(name + ' failed')
                report['cases'].append({'suite': name, 'result': result})
                print(name + ' passed', flush=True)
            cli = work / 'phoenix-process.exe'
            run(flags + [ROOT / 'native_runtime/cli.cpp', '/Fo:' + str(work / 'cli.obj'),
                         '/Fe:' + str(cli), work / 'phoenix_native.lib', *libraries], cwd=work, env=env)
            capabilities = json.loads(run([cli, '--capabilities'], cwd=work, env=env))
            if capabilities.get('python_required') is not False: raise RuntimeError('CLI dependency contract')
            config = work / 'recipe.json'
            config.write_text(json.dumps({'width': 31, 'height': 19, 'spatial_only': True,
                'warp': True, 'pipeline': {'PipelineMonitoring': 0}}))
            pixels = struct.pack('<4H', 0x3400, 0x3800, 0x3a00, 0x3c00) * (31 * 19)
            source, target = work / 'input.rgba16f', work / 'output.rgba16f'
            source.write_bytes(pixels)
            run([cli, '--config', config, '--input', source, '--output', target], cwd=work, env=env)
            if target.read_bytes() != pixels: raise RuntimeError('CLI file pixels changed')
            def header(sequence):
                return struct.pack('<4sHHIIQQqQQI', b'PXF1', 1, 60, 31, 19, 7, sequence,
                    sequence * 16666667, len(pixels), 0, 1 if sequence == 0 else 0)
            stream = b''.join(header(i) + pixels for i in range(2))
            piped = subprocess.run([str(cli), '--config', str(config), '--stream'], input=stream,
                cwd=work, env=env, capture_output=True, timeout=60)
            if piped.returncode or piped.stdout != stream:
                raise RuntimeError('CLI streaming failed: ' + piped.stderr.decode(errors='replace'))
            overwritten = subprocess.run([str(cli), '--config', str(config), '--input', str(source),
                '--output', str(target)], cwd=work, env=env, capture_output=True, timeout=60)
            if not overwritten.returncode or target.read_bytes() != pixels:
                raise RuntimeError('CLI overwrote an existing result')
            report['cases'].append({'suite': 'native-cli', 'result': {'passed': True,
                'file_pixels_exact': True, 'stream_pixels_and_timestamps_exact': True,
                'existing_output_preserved': True, 'python_required': False}})
            print('native-cli passed', flush=True)
            if args.tiny_checkpoint:
                from tiny_reference import verify
                result = verify(args.tiny_checkpoint, work, cli, env, run, args.gpu)
                report['cases'].append({'suite': 'tiny-native-reference', 'result': result})
                from convert_tiny_model import convert
                weights, _ = convert(args.tiny_checkpoint)
                model_root=work/'models';model_root.mkdir()
                (model_root/'weights.bin').write_bytes(weights)
                (model_root/'tiny-upscaler.json').write_text(json.dumps({'schema':'phoenix-tiny-2x/1','path':'weights.bin'}))
                executable=work/'addon-tiny-reconstruction.exe'
                run(flags+[ROOT/'tests/native/pipeline_scaler_bridge_tests.cpp','/Fo:'+str(work/'addon-tiny.obj'),
                    '/Fe:'+str(executable),*libraries],cwd=work,env=env)
                result=json.loads(run([executable,model_root],cwd=work,env=env))
                if not result.get('passed'):raise RuntimeError('Addon tiny reconstruction failed')
                report['cases'].append({'suite':'addon-tiny-reconstruction','result':result})
                print('tiny-native-reference passed', flush=True)
            if args.model_dll:
                from test_native_portable_pipeline import checks as portable_checks
                portable=portable_checks(dll,work,args.model_dll.resolve(),args.gpu)
                report['cases'].append({'suite':'portable-shared-pipeline','result':portable})
                (output/'portable-pipeline-validation.json').write_text(json.dumps(portable,indent=2)+'\n')
            if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest for name, digest in inputs.items()):
                raise RuntimeError('Native pipeline source changed during build or validation')
            from prepare_addon_release import safe_payload
            safe_payload('runtime/phoenix_native.dll',dll.read_bytes())
            safe_payload('runtime/phoenix-process.exe',cli.read_bytes())
            report['public_binary_audit_passed']=True
            report.update(passed=True, native_dll_sha256=hashlib.sha256(dll.read_bytes()).hexdigest(),
                          temporary_build_removed=True, source_inputs=inputs)
            for name in ('phoenix_native.dll', 'phoenix_native.lib'):
                shutil.copy2(work / name, output / name)
            shutil.copy2(cli, output / cli.name)
            # Keep the existing small receipts synchronized with the promoted
            # suite result. Older runs left these files describing a prior DLL.
            aliases = {'native-chain': 'chain-validation.json',
                       'addon-recorded-chain': 'addon-chain-validation.json',
                       'addon-neural-chain-both-models': 'addon-chain-gpu-validation.json',
                       'motion-translation-accuracy': 'motion-history-validation.json'}
            for case in report['cases']:
                if case['suite'] in aliases:
                    (output / aliases[case['suite']]).write_text(json.dumps(case['result'], indent=2) + '\n')
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        receipt.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
