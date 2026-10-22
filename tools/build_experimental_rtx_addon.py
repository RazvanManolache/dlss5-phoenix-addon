"""Build/test a model-free experimental x64 addon; never install it or launch games."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import zipfile
from test_model_selection import ROOT, toolchain, run
from validate_portable_bridge import resources
from prepare_addon_release import decompress, safe_payload
from compare_four_models import build as build_probe


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract_runtime(addon, destination, scratch):
    resources(addon, scratch)
    manifest = json.loads((scratch / '1001.bin').read_bytes())
    if (manifest.get('schema') != 'dlss5-phoenix-embedded-runtime/1' or
            manifest.get('model_data_included') is not False or manifest.get('nvidia_dll_included') is not False):
        raise ValueError('Source addon must contain a model-free embedded runtime')
    names = set()
    for row in manifest['files']:
        name = row['path']; relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name or name in names:
            raise ValueError('Invalid or duplicate source resource path')
        names.add(name)
        if not name.startswith('runtime/'):
            continue
        if name == 'runtime/phoenix-nvngx.dll':
            continue  # The addon builder compiles its forwarding shim from current source.
        raw = (scratch / f"{row['resource']}.bin").read_bytes()
        if row['algorithm'] == 'xpress_huff': raw = decompress(raw, row['bytes'])
        elif row['algorithm'] != 'raw': raise ValueError('Unknown source resource compression')
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Source runtime hash mismatch')
        safe_payload(name, raw)
        target = destination / Path(*relative.parts[1:]); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    if not (destination / 'validation_manifest.json').is_file():
        raise ValueError('Source addon has no runtime validation receipt')


def update_pipeline_runtime(runtime,source_addon,pipeline,native_sdk):
    """Overlay only validated ABI-compatible DLLs in disposable extracted staging."""
    evidence=json.loads((pipeline/'validation.json').read_text())
    native=json.loads((native_sdk/'validation.json').read_text())
    legacy=evidence.get('legacy_reference_equivalence',[])
    cli=json.loads((pipeline/'cli-validation.json').read_text())
    if not evidence.get('passed') or not evidence.get('public_binary_audit_passed') or len(evidence.get('cases',[]))!=14 or len(legacy)!=8 or not all(row.get('pixels_exact') for row in legacy):
        raise ValueError('Portable pipeline needs its complete packaged-backend and legacy-reference checks')
    if evidence.get('source_addon_sha256')!=sha(source_addon):raise ValueError('Portable checks describe another source addon')
    if not cli.get('passed') or cli.get('runtime_sha256')!=evidence['runtime_sha256'] or len(cli.get('cases',[]))!=8 or not all(row.get('pixels_exact') and row.get('output_preserved') for row in cli['cases']):
        raise ValueError('Portable native CLI validation differs')
    root=runtime/evidence['packaged_backend_root']
    if not root.resolve().is_relative_to(runtime.resolve()):raise ValueError('Invalid tested backend root')
    for name,digest in evidence['backend_files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:raise ValueError('Packaged backend differs from the tested payload: '+name)
    if sha(pipeline/'phoenix_runtime_cpp.dll')!=evidence['runtime_sha256']:raise ValueError('Portable pipeline DLL changed')
    if not native.get('passed') or not native.get('public_binary_audit_passed') or sha(native_sdk/'phoenix_native.dll')!=native['native_dll_sha256']:raise ValueError('Native SDK validation differs')
    previous=json.loads((runtime/'validation_manifest.json').read_text())
    shutil.copy2(pipeline/'phoenix_runtime_cpp.dll',root/'phoenix_runtime_cpp.dll')
    shutil.copy2(native_sdk/'phoenix_native.dll',root/'phoenix_native.dll')
    # Fresh evidence replaces the historical validation claim. Do not carry its
    # old exact-frame count forward merely because the filenames are unchanged.
    compact=[]
    for row in evidence['cases']:
        compact.append({key:row[key] for key in ('model','backend','output_stage','blocks','passes','independent_temporal_stages','independent_fusion_oracle')})
    validation={'schema':'phoenix-release-runtime-validation/1','passed':True,'inference_validated':True,
        'model_data_included':False,'scope':'24 byte-exact reset/history/reset frames against frozen released portable runtime; 14 portable chain cases with independent fusion oracles; native SDK suite separately validated. CPU checks use one block, GPU checks eight, all use two network passes. No game/older-GPU validation.',
        'exact_frames':sum(row['frames'] for row in legacy),
        'binaries':{p.relative_to(runtime).as_posix():sha(p) for p in runtime.rglob('*') if p.is_file() and p.suffix.lower() in {'.dll','.exe','.spv','.rtl','.cfg','.o'}},
        'portable_pipeline_validation_sha256':sha(pipeline/'validation.json'),
        'portable_cli_validation_sha256':sha(pipeline/'cli-validation.json'),
        'portable_cli_cases':len(cli['cases']),
        'native_sdk_validation_sha256':sha(native_sdk/'validation.json'),
        'portable_pipeline_cases':compact,
        'native_sdk_suites':[row['suite'] for row in native['cases']],
        'previous_reference':{'runtime_sha256':evidence['legacy_reference_runtime_sha256'],
            'validation_scope':previous.get('scope',''),'source_addon_sha256':sha(source_addon)}}
    (runtime/'validation_manifest.json').write_text(json.dumps(validation,indent=2)+'\n')
    return {'portable_runtime_sha256':evidence['runtime_sha256'],'native_sdk_sha256':native['native_dll_sha256'],
        'portable_cases':len(compact),'portable_cli_cases':len(cli['cases']),'legacy_exact_frames':validation['exact_frames']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-addon', type=Path, default=ROOT / 'artifacts/current/addon/Phoenix.addon64')
    parser.add_argument('--native', type=Path, default=ROOT / 'build/native_arch_compat_final')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/current/addon-experimental-rtx')
    parser.add_argument('--pipeline-runtime',type=Path,help='Validated current portable-pipeline folder; also embeds the matching current native SDK')
    parser.add_argument('--native-sdk',type=Path,default=ROOT/'artifacts/current/pipeline-native')
    args = parser.parse_args()
    source, native, destination = [p.resolve() for p in (args.source_addon, args.native, args.output)]
    if not destination.is_relative_to(ROOT / 'artifacts/current'):
        parser.error('Experimental outputs must stay under artifacts/current')
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, release_channel='experimental', model_data_included=False,
                  source_addon_sha256=sha(source), games_launched=False, installed_files_modified=False,
                  older_gpu_inference_validated=False, game_presentation_validated=False)
    receipt = destination / 'VALIDATION.json'
    try:
        with tempfile.TemporaryDirectory(prefix='experimental_rtx_', dir=ROOT / 'build') as temporary:
            work = Path(temporary); core = ROOT / 'build' / (work.name + '_core')
            try:
                scratch = work / 'source'; scratch.mkdir()
                runtime = work / 'runtime'; runtime.mkdir()
                extract_runtime(source, runtime, scratch)
                if args.pipeline_runtime:
                    report['pipeline_runtime_update']=update_pipeline_runtime(runtime,source,args.pipeline_runtime.resolve(),args.native_sdk.resolve())
                if sha(source) != report['source_addon_sha256']:
                    raise RuntimeError('Source addon changed during runtime extraction')
                subprocess.run([sys.executable, ROOT / 'tools/build_addon.py', core.name, '--production-only',
                    '--experimental-native', '--ngx-sdk', ROOT.parent / 'external/DLSS/include',
                    '--embed-runtime', runtime, '--embed-native', native], cwd=ROOT, check=True)
                compiled = json.loads((core / 'report.json').read_text())
                if not compiled.get('passed'): raise RuntimeError('Addon source build did not pass')
                print('Addon compiled; checking real DLL loading and embedded payload.', flush=True)
                addon = core / 'phoenix.addon64'
                compiler, env = toolchain(64)
                flags = [compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/EHsc', '/O2', '/MT', '/DNOMINMAX']
                for name, source_file, extra in (
                    ('load', 'packaged_addon_load_tests.cpp', []),
                    ('embedded', 'embedded_runtime_tests.cpp', ['bcrypt.lib', 'cabinet.lib', 'shell32.lib', 'ole32.lib'])):
                    executable = work / (name + '.exe')
                    run(flags + [ROOT / 'tests/native' / source_file, '/Fo:' + str(work / (name + '.obj')),
                                 '/Fe:' + str(executable), *extra], cwd=work, env=env)
                    if name == 'load': report['dll_load'] = json.loads(run([executable, addon], cwd=work, env=env))
                    else:
                        output = work / 'embedded.json'
                        run([executable, addon, work / 'cache', output], cwd=work, env=env)
                        extracted = json.loads(output.read_text()); root_name = extracted.pop('installed_root')
                        installed_root = Path(bytes(root_name).decode('utf-8') if isinstance(root_name, list) else root_name)
                        report['embedded_payload'] = extracted
                print('DLL attach/detach and payload extraction passed; running both models on the packaged kernels.', flush=True)
                executable, forwarder = build_probe(work)
                subprocess.run([sys.executable, ROOT / 'tools/benchmark_kernel_structure.py',
                    '--before', ROOT / 'artifacts/current/sdk/windows-x64/native', '--after', installed_root / 'native',
                    '--rounds', '1', '--resolutions', '320x180', '1920x1080', '--prepared-probe', executable,
                    '--prepared-forwarder', installed_root / 'runtime/phoenix-nvngx.dll', '--output', work / 'gpu'], cwd=ROOT, check=True)
                gpu = json.loads((work / 'gpu/paired.json').read_text())
                if not gpu.get('passed'): raise RuntimeError('Packaged GPU payload failed')
                report.update(gpu_used=True, gpu_cases=gpu['cases'],
                              sm120_all_images_byte_exact=all(row['identical'] for row in gpu['machine_code']),
                              cpu_tests=compiled['cpu_tests'], software_d3d12_tests=compiled['software_d3d12_tests'],
                              addon_sha256=sha(addon), embedded_manifest_sha256=compiled['embedded_runtime']['manifest_sha256'])
                # Audit the actual bytes being shipped, including every embedded resource.
                from prepare_addon_release import audit_embedded
                report['package_audit'] = audit_embedded(addon, work / 'audit')
                safe_payload('Phoenix.addon64', addon.read_bytes())
                shutil.copy2(addon, destination / 'Phoenix.addon64')
                notes = ('Phoenix DLSS5 - experimental RTX compatibility (Windows x64)\n\n'
                    'Copy Phoenix.addon64 alongside your existing ReShade installation, replacing the existing Phoenix addon.\n'
                    'Do not install multiple Phoenix addons together. Keep your separately obtained model files.\n'
                    'Original and Lightweight FP8 models are supported. No Python, compiler or CUDA SDK is needed to use this addon.\n\n'
                    'RTX 20/30/40: experimental, compiled but not tested on these physical cards.\n'
                    'RTX 5080: actual DLL attach/detach and extraction checked; both models passed GPU reset/history comparisons\n'
                    'at 320x180 and 1920x1080 using the payload extracted from this addon. All 41 sm_120 images are unchanged.\n'
                    'No retail game presentation test was run. This is a local test package, not a stable release.\n')
                (destination / 'README.txt').write_text(notes)
                report.update(passed=True, core_build_suites_passed=True)
                receipt.write_text(json.dumps(report, indent=2) + '\n')
                files = {'Phoenix.addon64': addon.read_bytes(), 'README.txt': notes.encode(),
                         'VALIDATION.json': receipt.read_bytes()}
                for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY.md'): files[name] = (ROOT / name).read_bytes()
                for path in (ROOT / 'third_party').rglob('*'):
                    if path.is_file() and path.name.upper().startswith(('LICENSE', 'NOTICE', 'COPYING', 'MIT_LICENSE')):
                        files['licenses/' + path.relative_to(ROOT / 'third_party').as_posix()] = path.read_bytes()
                for name, raw in files.items(): safe_payload(name, raw)
                archive = destination / 'Phoenix-Experimental-RTX-Windows-x64.zip'
                with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as package:
                    for name, raw in sorted(files.items()): package.writestr(name, raw)
                with zipfile.ZipFile(archive) as package:
                    if package.testzip(): raise RuntimeError('Experimental archive CRC failure')
                    if any(package.read(name) != raw for name, raw in files.items()): raise RuntimeError('Archive content mismatch')
                (destination / 'SHA256SUMS.txt').write_text(sha(archive) + '  ' + archive.name + '\n' + sha(destination / 'Phoenix.addon64') + '  Phoenix.addon64\n')
            except BaseException:
                diagnostics = {}
                if core.exists():
                    for path in core.glob('*.result.json'):
                        try:
                            value = json.loads(path.read_text(encoding='utf-8-sig'))
                            if value.get('passed') is not True: diagnostics[path.name] = value
                        except (ValueError, AttributeError): diagnostics[path.name] = path.read_text(errors='replace')[-6000:]
                    for path in sorted(core.glob('*.build.log'), key=lambda p:p.stat().st_mtime, reverse=True)[:2]:
                        diagnostics[path.name] = path.read_text(errors='replace')[-6000:]
                    if (core/'report.json').exists(): diagnostics['build_report'] = json.loads((core/'report.json').read_text())
                (destination/'build-diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
                raise
            finally:
                if core.exists():
                    if core.is_symlink() or core.is_junction() or core.resolve().parent != (ROOT / 'build').resolve():
                        raise RuntimeError('Unsafe experimental staging cleanup target')
                    shutil.rmtree(core)
        print('Experimental addon, ZIP and test receipt saved to ' + str(destination), flush=True)
    except Exception as error:
        report.update(passed=False, error=str(error)); receipt.write_text(json.dumps(report, indent=2) + '\n'); raise


if __name__ == '__main__': main()
