"""Build and validate an alternate x86 Vulkan runtime without replacing the helper."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import array
import math

ROOT = Path(__file__).resolve().parents[1]


def stdcall_aliases(symbols):
    """Use compiler-emitted stack sizes, never hand-maintained Vulkan signatures."""
    return sorted(set(re.findall(r'\bUNDEF\b[^\n]*\|\s+(_vk\w+@\d+)\s*$', symbols, re.M)))


def frame_comparison(left, right):
    if len(left) != len(right) or not left or len(left) % 4:
        raise ValueError('Architecture output extents differ or are not float32')
    a, b = array.array('f'), array.array('f')
    a.frombytes(left); b.frombytes(right)
    abits, bbits = array.array('I'), array.array('I')
    abits.frombytes(left); bbits.frombytes(right)
    different = 0
    max_ulp = 0
    maximum = total = 0.
    for x, y, ix, iy in zip(a, b, abits, bbits):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Nonfinite architecture output')
        difference = abs(x-y)
        different += x != y
        maximum = max(maximum, difference)
        total += difference
        if x != y:
            ordered_x = (~ix & 0xffffffff) if ix & 0x80000000 else ix | 0x80000000
            ordered_y = (~iy & 0xffffffff) if iy & 0x80000000 else iy | 0x80000000
            max_ulp = max(max_ulp, abs(ordered_x - ordered_y))
    return dict(exact_bytes=left == right, samples=len(a), differing_samples=different,
                max_abs_error=maximum, max_ulp_error=max_ulp, mean_abs_error=total/len(a))


def architecture_acceptance(comparison, x86_rows, x64_rows):
    exact_stages = ('color', 'conditioning_history', 'conditioning_validity', 'resolve_history',
                    'resolve_validity', 'noise', 'network_head', 'next_history')
    if len(x86_rows) != 9 or len(x64_rows) != 9:
        raise ValueError('Missing architecture frame evidence')
    different_gates = []
    for a, b in zip(x86_rows, x64_rows):
        astages, bstages = a.get('validation_stages', []), b.get('validation_stages', [])
        if len(astages) != 1 or len(bstages) != 1:
            raise ValueError('Missing full-model stage hashes')
        left, right = astages[0], bstages[0]
        for key in exact_stages:
            value = left.get(key)
            if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) or value != right.get(key):
                raise ValueError(f'Architecture divergence before/in neural execution or history: {key}')
        for key in ('resolve_gate', 'display_output'):
            if any(not isinstance(stage.get(key), str) or not re.fullmatch('[0-9a-f]{64}', stage[key])
                   for stage in (left, right)):
                raise ValueError('Missing host-resolve evidence')
        if left['resolve_gate'] != right['resolve_gate']:
            different_gates.append(dict(width=a['width'], height=a['height'], frame=a['frame']))
        if left['display_output'] != right['display_output'] and left['resolve_gate'] == right['resolve_gate']:
            raise ValueError('Display divergence not isolated to the host history gate')
    if not comparison['exact_bytes'] and (comparison['max_ulp_error'] > 2 or not different_gates):
        raise ValueError('Host-resolve difference exceeds the demonstrated two-ULP boundary')
    return dict(neural_head_exact=True, next_history_exact=True, preparation_exact=True,
        byte_exact_display=comparison['exact_bytes'], host_gate_difference_frames=different_gates,
        classification='byte_exact' if comparison['exact_bytes'] else 'neural_exact_host_gate_two_ulp')


def validate_frame_rows(rows, bits, byte_count):
    shapes = ((64, 64), (129, 97), (1920, 1080))
    expected = [(width, height, frame) for width, height in shapes for frame in range(3)]
    if [(row.get('width'), row.get('height'), row.get('frame')) for row in rows] != expected:
        raise ValueError('Missing or duplicated full-frame evidence')
    if byte_count != sum(width*height*3*4*3 for width, height in shapes):
        raise ValueError('Full-frame output byte count differs')
    for row in rows:
        if (row.get('pointer_bits') != bits or row.get('backend') != 'vulkan-cpp' or
            row.get('output_changed') is not True or not row.get('command_count', 0) or
            row.get('history_used') is not (row['frame'] != 0)):
            raise ValueError('Frame is not proven native Vulkan inference with the expected history')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deps', type=Path)
    parser.add_argument('--stage', choices=['contracts', 'operators', 'frames'], default='contracts')
    parser.add_argument('--run-gpu', action='store_true')
    parser.add_argument('--weight-source', type=Path, default=ROOT/'artifacts/current/weight-source.json')
    args = parser.parse_args()
    if args.stage != 'contracts' and (not args.run_gpu or not args.deps):
        parser.error('Operator/frame validation requires --run-gpu and --deps')
    vswhere = Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    if not studio:
        raise RuntimeError('No MSVC toolchain')
    receipt = ROOT/'artifacts/current/portable-x86-validation.json'
    report = dict(schema='phoenix-portable-x86-validation/1', stage=args.stage,
        helper_modified=False, production_package_modified=False, tests_run=False,
        gpu_used=False, inference_validated=False, passed=False, architectures={},
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [ROOT/name for name in (
                'src/backends/hip_sycl/operand_address.h', 'src/backends/hip_sycl/compute_ops.h',
                'native_runtime/backend.h', 'native_runtime/backend_vulkan.cpp',
                'native_runtime/scratch_plan.h', 'native_runtime/graph.h', 'native_runtime/runtime.cpp',
                'tests/portable_cpp/address_contract_tests.cpp', 'tests/portable_cpp/x86_frame_probe.cpp',
                'tools/validate_x86_vulkan.py')]})
    def save():
        receipt.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='portable-x86-', dir=ROOT/'artifacts') as temporary:
        build = Path(temporary)
        shader_folder = build/'shaders'
        if args.stage != 'contracts':
            print('Generating the existing production Vulkan shader set', flush=True)
            generated = subprocess.run([os.sys.executable, str(ROOT/'tools/generate_standalone_vulkan.py'),
                '--output', str(shader_folder), '--compiler', str(args.deps.resolve()/'glslang/bin/glslang.exe')],
                capture_output=True, text=True, timeout=240)
            report['shaders'] = dict(passed=generated.returncode == 0,
                diagnostics=(generated.stdout+generated.stderr)[-6000:] if generated.returncode else '')
            save()
            if generated.returncode:
                print(json.dumps(report['shaders']), flush=True)
                return 1
        for arch, vcvars in (('x86', '32'), ('x64', '64')):
            out = build/arch
            out.mkdir()
            row = report['architectures'][arch] = dict(builds={}, tests={})
            def compile(name, command):
                recipe = out/'build.cmd'
                recipe.write_text('\n'.join(['@echo off',
                    f'call "{studio}/VC/Auxiliary/Build/vcvars{vcvars}.bat" >nul 2>&1',
                    'if errorlevel 1 exit /b %errorlevel%', command, 'exit /b %errorlevel%', '']), encoding='utf-8')
                result = subprocess.run(['cmd.exe', '/d', '/c', str(recipe)], cwd=out,
                    capture_output=True, text=True, timeout=180)
                row['builds'][name] = dict(passed=result.returncode == 0,
                    diagnostics=(result.stdout+result.stderr)[-6500:] if result.returncode else '')
                save()
                print(f'{arch} {name}: {"compiled" if not result.returncode else "FAILED"}', flush=True)
                if result.returncode:
                    raise RuntimeError(row['builds'][name]['diagnostics'])
                return result.stdout
            def run(name, command, timeout=120):
                result = subprocess.run(command, cwd=out, capture_output=True, text=True, timeout=timeout)
                row['tests'][name] = dict(passed=result.returncode == 0, exit_code=result.returncode,
                    output=(result.stdout+result.stderr)[-6000:])
                report['tests_run'] = True
                save()
                print(f'{arch} {name}: {"passed" if not result.returncode else "FAILED"}', flush=True)
                if result.returncode:
                    raise RuntimeError(row['tests'][name]['output'])
                return result.stdout
            cpp = 'cl /nologo /std:c++20 /EHsc /O2 /fp:strict /MD /bigobj /DNOMINMAX'
            compile('address_contracts', f'{cpp} "{ROOT}/tests/portable_cpp/address_contract_tests.cpp" /Fe:address_tests.exe')
            run('address_contracts', [str(out/'address_tests.exe')])
            if args.stage == 'contracts':
                continue
            own = ROOT/'src/backends/vulkan'
            compile('provider_object', f'cl /nologo /O2 /std:c11 /TC /MD /D_CRT_SECURE_NO_WARNINGS '
                f'/I"{args.deps.resolve()}/headers/include" /I"{own}" /c "{shader_folder}/phoenix_vulkan.c" /Fo:provider.obj')
            # A dump of the compiled object gives the exact x86 stdcall symbols.
            symbols = compile('provider_symbols', 'dumpbin /nologo /symbols provider.obj')
            vk_functions = sorted(set(re.findall(r'\bUNDEF\b[^\n]*\|\s+_?vk(\w+?)(?:@\d+)?\s*$', symbols, re.M)))
            if not vk_functions:
                raise RuntimeError('No Vulkan imports found in compiled provider')
            loader_def = out/'loader.def'
            loader_def.write_text('LIBRARY vulkan-1.dll\nEXPORTS\n'+''.join('vk'+name+'\n' for name in vk_functions), encoding='ascii')
            compile('loader_import', f'lib /nologo /def:loader.def /machine:{arch} /out:vulkan-1.lib')
            aliases = ' '.join(f'/alternatename:{symbol}={symbol.split("@")[0]}' for symbol in stdcall_aliases(symbols))
            row['stdcall_import_aliases'] = stdcall_aliases(symbols)
            vulkan = out/'vulkan'
            vulkan.mkdir()
            compile('provider', f'link /nologo /DLL provider.obj vulkan-1.lib /DEF:"{shader_folder}/provider.def" '
                f'/OUT:vulkan/phoenix_vulkan.dll {aliases}')
            for shader in shader_folder.glob('*.spv'):
                shutil.copyfile(shader, vulkan/shader.name)
            def dll(name, source, definition, libraries=''):
                compile(name, f'{cpp} /LD "{ROOT}/{source}" /Fe:{name}.dll '
                    f'/link /DEF:"{ROOT}/native_runtime/{definition}" {libraries}')
            dll('phoenix_vulkan_cpp', 'native_runtime/backend_vulkan.cpp', 'vulkan_backend.def')
            dll('phoenix_runtime_cpp', 'native_runtime/runtime.cpp', 'phoenix_runtime.def', 'bcrypt.lib')
            dll('phoenix_cpu_cpp', 'native_runtime/backend_reference.cpp', 'cpu_backend.def')
            compile('c_api_tests', f'cl /nologo /MD /I"{ROOT}/native_runtime" '
                f'/Tc"{ROOT}/tests/portable_cpp/c_api_tests.c" /Fe:c_api_tests.exe /link phoenix_runtime_cpp.lib')
            run('c_api_contracts', [str(out/'c_api_tests.exe')])
            compile('runner', f'{cpp} "{ROOT}/tools/portable_frame.cpp" /Fe:phoenix_frame.exe /link /LARGEADDRESSAWARE')
            devices = json.loads(run('devices', [str(out/'phoenix_frame.exe'), '--vulkan-devices', str(out)]))
            selected = next(device for device in devices if device['vendor'] == 0x10de and device['supported'])
            row['device'] = selected
            compile('operator_tests', f'{cpp} "{ROOT}/tests/portable_cpp/backend_operator_tests.cpp" /Fe:operator_tests.exe /link /LARGEADDRESSAWARE')
            report['gpu_used'] = True
            run('operators', [str(out/'operator_tests.exe'), str(out), 'vulkan', selected['id']], timeout=240)
            if args.stage == 'frames':
                compile('frame_probe', f'{cpp} "{ROOT}/tests/portable_cpp/x86_frame_probe.cpp" /Fe:frame_probe.exe /link /LARGEADDRESSAWARE')
                output = out/'frames.rgb32f'
                frames = run('frames', [str(out/'frame_probe.exe'), str(out), str(args.weight_source.resolve()),
                    selected['id'], str(output)], timeout=240)
                row['frames'] = [json.loads(line) for line in frames.splitlines() if line.startswith('{')]
                row['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
                row['output_bytes'] = output.stat().st_size
                validate_frame_rows(row['frames'], 32 if arch == 'x86' else 64, row['output_bytes'])
                save()
        if args.stage == 'frames':
            left = (build/'x86/frames.rgb32f').read_bytes()
            right = (build/'x64/frames.rgb32f').read_bytes()
            report['frame_comparison'] = frame_comparison(left, right)
            save()
            report['architecture_acceptance'] = architecture_acceptance(report['frame_comparison'],
                report['architectures']['x86']['frames'], report['architectures']['x64']['frames'])
            report['inference_validated'] = True
            # Keep one model-free alternate x86 runtime, never the helper/add-on.
            destination = ROOT/'artifacts/current/portable-x86'
            if destination.resolve().parent != (ROOT/'artifacts/current').resolve():
                raise RuntimeError('Candidate destination escaped the intended artifact directory')
            if destination.exists():
                if destination.is_symlink() or destination.is_junction():
                    raise RuntimeError('Refusing to replace a linked candidate folder')
                for path in destination.rglob('*'):
                    if path.is_symlink() or path.is_junction():
                        raise RuntimeError('Refusing to follow linked candidate contents')
                shutil.rmtree(destination)
            destination.mkdir()
            candidate = build/'x86'
            for name in ('phoenix_vulkan_cpp.dll', 'phoenix_runtime_cpp.dll', 'phoenix_cpu_cpp.dll', 'phoenix_frame.exe'):
                shutil.copyfile(candidate/name, destination/name)
            (destination/'vulkan').mkdir()
            for file in (candidate/'vulkan').iterdir():
                if file.suffix in ('.dll', '.spv'):
                    shutil.copyfile(file, destination/'vulkan'/file.name)
            (destination/'README.txt').write_text(
                'Experimental native x86 Vulkan runtime. The existing x64 helper remains the default.\n'
                'This folder contains no model or NVIDIA DLL. Supply your existing weight-source config.\n'
                'Use phoenix_frame.exe with a documented portable C API job (docs/LIBRARY.md).\n'
                'This candidate has not been integrated into or validated inside a game add-on.\n', encoding='ascii')
            licenses = destination/'licenses'
            licenses.mkdir()
            for folder in ('intel_nr', 'amd_nr'):
                for name in ('LICENSE', 'NOTICE'):
                    shutil.copyfile(ROOT/'third_party'/folder/name, licenses/(folder+'-'+name+'.txt'))
            shutil.copyfile(ROOT/'LICENSE', licenses/'Phoenix-LICENSE.txt')
            report['candidate_sha256'] = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(destination.rglob('*')) if path.is_file()}
        report['passed'] = True
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
