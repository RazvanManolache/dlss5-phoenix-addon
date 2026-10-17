"""Disposable WOW64 cross-mode experiment; never modifies installed games."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from probe_nvapi_module import execute

ROOT = Path(__file__).resolve().parents[1]
WOW64EXT_COMMIT = '82045488f48c8641609d9a5d8bb832937b9b5acd'


def conclusion(probes):
    def completed_event(name, predicate):
        probe = probes.get(name, {})
        return probe.get('exit_code') == 0 and any(predicate(row) for row in probe.get('events', []))
    def completed_module(name):
        return completed_event(name, lambda row:
            row.get('module_created') == 1 and row.get('worker_stage') == 12 and row.get('status', 0) == 0)
    return dict(
        mode_switch_works=completed_event('mode_switch', lambda row: row.get('matched') is True),
        import_free_x64_dll_works=completed_event('cross_bare_worker', lambda row: row.get('worker_stage') == 100),
        native_x64_control_works=completed_module('native_x64_control'),
        cross_bitness_gpu_module_loading_works=any(completed_module(name) for name in
            ('cross_worker', 'native_thread_worker', 'native_thread_noattach_worker',
             'activation_worker', 'activation_thread_worker', 'gui_worker')),
        completed_gpu_inference=False)


def debugger_findings(output, exit_code):
    lines = output.splitlines()
    return dict(debugger_exit_code=exit_code,
        # Quitting the debugger can return zero after a fatal debuggee fault.
        access_violation_observed='ExceptionCode: c0000005' in output,
        lines=[line for line in lines if any(word in line.lower() for word in
            ('phx_', 'exceptionaddress:', 'exceptioncode:', 'attempt to read',
             'attempt to write', 'gs:[58h]', 'tls_init', 'consoleinitialize',
             'call site', ' : gdi', ' : user32', ' : nvapi', ' : dxgi'))][-45:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wow64ext', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=ROOT/'artifacts/current/native-kernels')
    parser.add_argument('--run-gpu-probes', action='store_true')
    parser.add_argument('--loader-debug', action='store_true', help='Process-local initialization tracing; no registry flags')
    parser.add_argument('--case', action='append', help='Run only these named cases; repeat to select several')
    parser.add_argument('--debug-case', action='append', choices=['console-loader', 'gui-worker', 'gui-nvapi'],
                        help='Repeat to trace several cases; requires --loader-debug')
    args = parser.parse_args()
    if args.debug_case and not args.loader_debug:
        parser.error('--debug-case requires --loader-debug')
    if args.debug_case and any(name != 'console-loader' for name in args.debug_case) and not args.run_gpu_probes:
        parser.error('Graphics initialization traces require --run-gpu-probes')
    extension = args.wow64ext.resolve()
    revision = subprocess.check_output(['git', '-C', str(extension), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != WOW64EXT_COMMIT or subprocess.check_output(
            ['git', '-C', str(extension), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Use the pinned, unmodified WOW64Ext source checkout')
    kernel = json.loads((args.bundle/'native_manifest.json').read_text())['kernels'][0]
    blob = (args.bundle/kernel['binary']).resolve()
    if not blob.is_relative_to(args.bundle.resolve()):
        raise ValueError('Kernel escapes the selected bundle')
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    if sha != kernel['binary_sha256']:
        raise ValueError('Kernel SHA-256 mismatch')
    vswhere = Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
                'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    if not studio:
        raise RuntimeError('No MSVC toolchain')
    report = dict(schema='wow64-native-isolation/1', wow64ext_commit=revision,
                  wow64ext_license='LGPL-3.0-or-later; external dynamically loaded probe dependency only',
                  gpu_probes_requested=args.run_gpu_probes, kernel_sha256=sha,
                  production_package_modified=False, builds={}, probes={},
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                      for name in ('tests/native/wow64_bare_worker.cpp', 'tests/native/wow64_cross_probe.cpp',
                                   'tests/native/wow64_gpu_worker.cpp', 'tests/native/wow64_probe_contract.h')})
    receipt = ROOT/'artifacts/current/wow64-native-investigation.json'
    def save():
        receipt.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='wow64-probe-', dir=ROOT/'artifacts') as temporary:
        build = Path(temporary)
        x64, x86 = build/'x64', build/'x86'
        x64.mkdir(); x86.mkdir()
        for arch, out, commands in (
            ('64', x64, [
                f'cl /nologo /std:c++20 /O2 /GS- /GR- /LD "{ROOT}/tests/native/wow64_bare_worker.cpp" '
                '/link /NODEFAULTLIB /ENTRY:DllMain /OUT:probe_bare64.dll',
                f'cl /nologo /std:c++20 /O2 /Oi /GS- /GR- /LD "{ROOT}/tests/native/wow64_gpu_worker.cpp" '
                '/link /NODEFAULTLIB /ENTRY:DllMain /OUT:probe_worker64.dll kernel32.lib d3d12.lib dxgi.lib',
                f'cl /nologo /std:c++20 /EHsc /MT "{ROOT}/tests/native/wow64_cross_probe.cpp" /Fe:native_control.exe']),
            ('32', x86, [
                f'cl /nologo /O2 /Oi /GS- /GR- /LD /DWOW64EXT_EXPORTS /DUNICODE /D_UNICODE /DWIN32 '
                f'"{extension}/src/wow64ext.cpp" /link /NODEFAULTLIB /ENTRY:DllMain /OUT:wow64ext.dll kernel32.lib',
                f'cl /nologo /std:c++20 /EHsc /MT "{ROOT}/tests/native/wow64_cross_probe.cpp" /Fe:cross_probe.exe',
                f'cl /nologo /std:c++20 /EHsc /MT "{ROOT}/tests/native/wow64_cross_probe.cpp" /Fe:cross_gui.exe '
                '/link /SUBSYSTEM:WINDOWS /ENTRY:wmainCRTStartup'])
        ):
            recipe = out/'build.cmd'
            lines = ['@echo off', f'call "{studio}/VC/Auxiliary/Build/vcvars{arch}.bat" >nul 2>&1',
                     'if errorlevel 1 exit /b %errorlevel%']
            for command in commands:
                lines += [command, 'if errorlevel 1 exit /b %errorlevel%']
            recipe.write_text('\n'.join(lines)+'\n', encoding='utf-8')
            compiled = subprocess.run(['cmd.exe', '/d', '/c', str(recipe)], cwd=out,
                                      capture_output=True, text=True, timeout=120)
            report['builds']['x'+arch] = dict(passed=compiled.returncode == 0,
                diagnostics=(compiled.stdout+compiled.stderr)[-5000:] if compiled.returncode else '')
            save()
            if compiled.returncode:
                print(json.dumps(report['builds']), flush=True)
                return 1
        commands = [('mode_switch', [str(x86/'cross_probe.exe'), '--mode-switch'], x86)]
        if args.run_gpu_probes:
            system = Path(os.environ['SystemRoot'])/'System32'
            commands += [
                ('native_x64_control', [str(x64/'native_control.exe'), '--worker', str(x64/'probe_worker64.dll'), str(blob)], x64),
                ('cross_bare_worker', [str(x86/'cross_probe.exe'), '--worker', str(x64/'probe_bare64.dll'), str(blob)], x86),
                ('cross_kernel32', [str(x86/'cross_probe.exe'), '--kernel32'], x86),
                ('cross_kernelbase', [str(x86/'cross_probe.exe'), '--kernelbase'], x86),
                ('gui_kernel32', [str(x86/'cross_gui.exe'), '--kernel32'], x86),
                ('gui_nvapi', [str(x86/'cross_gui.exe'), '--nvapi'], x86),
                ('gui_d3d12', [str(x86/'cross_gui.exe'), '--d3d12'], x86),
                ('gui_worker', [str(x86/'cross_gui.exe'), '--worker', str(x64/'probe_worker64.dll'), str(blob)], x86),
                ('cross_nvapi', [str(x86/'cross_probe.exe'), '--nvapi'], x86),
                ('cross_d3d12', [str(x86/'cross_probe.exe'), '--d3d12'], x86),
                ('cross_worker', [str(x86/'cross_probe.exe'), '--worker', str(x64/'probe_worker64.dll'), str(blob)], x86),
                ('native_thread_kernel32', [str(x86/'cross_probe.exe'), '--thread', str(x64/'probe_bare64.dll'), str(system/'kernel32.dll')], x86),
                ('native_thread_worker', [str(x86/'cross_probe.exe'), '--thread', str(x64/'probe_bare64.dll'), str(x64/'probe_worker64.dll'), str(blob)], x86),
                ('native_thread_noattach_worker', [str(x86/'cross_probe.exe'), '--thread-noattach', str(x64/'probe_bare64.dll'), str(x64/'probe_worker64.dll'), str(blob)], x86),
                ('activation_kernel32', [str(x86/'cross_probe.exe'), '--activation', str(x64/'probe_bare64.dll'), str(system/'kernel32.dll')], x86),
                ('activation_worker', [str(x86/'cross_probe.exe'), '--activation', str(x64/'probe_bare64.dll'), str(x64/'probe_worker64.dll'), str(blob)], x86),
                ('activation_thread_worker', [str(x86/'cross_probe.exe'), '--thread-activation', str(x64/'probe_bare64.dll'), str(x64/'probe_worker64.dll'), str(blob)], x86)]
        if args.case:
            selected = set(args.case)
            unknown = selected.difference(name for name, _, _ in commands)
            if unknown:
                raise ValueError(f'Unknown or disabled cases: {sorted(unknown)}')
            commands = [row for row in commands if row[0] in selected]
        for name, command, out in commands:
            report['probes'][name] = execute(command, out, timeout=30)
            save()
            print(json.dumps(dict(probe=name, **report['probes'][name])), flush=True)
        if args.loader_debug:
            debugger = Path(os.environ['ProgramFiles(x86)'])/'Windows Kits/10/Debuggers/x64/cdb.exe'
            debugger_script = build/'loader.cdb'
            report['loader_debug'] = {}
            for debug_case in args.debug_case or ['console-loader']:
                if debug_case == 'console-loader':
                    script = ('.symfix\ng\ng\nbu ntdll!LdrpCallInitRoutine\ng\n'
                        '.printf "PHX_INIT routine=%p base=%p reason=%x\\n", @rcx, @rdx, @r8\n'
                        'bc *\nbm KERNELBASE_*!ConsoleInitialize\ng\ngu\n'
                        '.printf "PHX_CONSOLE_RETURN=%x\\n", @al\nbc *\ng\n')
                    debug_command = [str(x86/'cross_probe.exe'), '--kernel32']
                else:
                    script = ('.symfix\nsxe av\ng\ng\n.exr -1\nub @rip L8\n'
                              'r rax,rcx\nkb 15\nq\n')
                    debug_command = [str(x86/'cross_gui.exe'), '--nvapi'] if debug_case == 'gui-nvapi' else [
                        str(x86/'cross_gui.exe'), '--worker', str(x64/'probe_worker64.dll'), str(blob)]
                debugger_script.write_text(script, encoding='ascii')
                debug = subprocess.run([str(debugger), '-G', '-cf', str(debugger_script),
                    *debug_command], cwd=x86, capture_output=True, text=True, timeout=45)
                report['loader_debug'][debug_case] = debugger_findings(debug.stdout+debug.stderr, debug.returncode)
                save()
                print(json.dumps(dict(debug_case=debug_case, **report['loader_debug'][debug_case])), flush=True)
        report['conclusion'] = conclusion(report['probes'])
        save()
        print(json.dumps(report['conclusion']), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
