"""Isolate x86/x64 NVAPI module creation; GPU calls require an explicit flag."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def execute(command, cwd, timeout=30, include_stdout=False, stdout_limit=18000):
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    result = dict(process_pid=process.pid, exit_code=process.returncode, timed_out=timed_out, events=events,
                  stderr=stderr[-1500:])
    if include_stdout:
        result['stdout'] = stdout[-stdout_limit:]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-gpu-probes', action='store_true')
    parser.add_argument('--trace-driver', action='store_true', help='Trace the x86 module fault in a disposable debugger process')
    parser.add_argument('--arithmetic', action='store_true', help='Test only an independent arithmetic cubin, not the model kernel')
    parser.add_argument('--bundle', type=Path, default=ROOT/'artifacts/current/native-kernels')
    args = parser.parse_args()
    if args.trace_driver and not args.run_gpu_probes:
        parser.error('--trace-driver requires --run-gpu-probes')
    if args.arithmetic and (not args.run_gpu_probes or args.trace_driver):
        parser.error('--arithmetic requires --run-gpu-probes and is separate from --trace-driver')
    manifest = json.loads((args.bundle/'native_manifest.json').read_text())
    kernel = manifest['kernels'][0]
    blob = (args.bundle/kernel['binary']).resolve()
    if not blob.is_relative_to(args.bundle.resolve()):
        raise ValueError('Kernel path escapes the selected bundle')
    sha = hashlib.sha256(blob.read_bytes()).hexdigest()
    if sha != kernel['binary_sha256']:
        raise ValueError('Kernel SHA-256 differs from the build manifest')
    vswhere = Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
                'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    if not studio:
        raise RuntimeError('No Visual C++ toolchain found')
    report = dict(schema='nvapi-module-isolation/1', gpu_probes_requested=args.run_gpu_probes,
                  blob_sha256=sha, kernel=kernel['name'], architectures={})
    report_path = ROOT/'artifacts/current/x86-native-investigation.json'
    with tempfile.TemporaryDirectory(prefix='nvapi-probe-', dir=ROOT/'artifacts') as temporary:
        build = Path(temporary)
        if args.arithmetic:
            import shutil
            compiler=shutil.which('ptxas')
            if not compiler: raise RuntimeError('CUDA assembler required for arithmetic probe')
            blob=build/'arithmetic.cubin'
            compiled=subprocess.run([compiler,'--gpu-name=sm_120',str(ROOT/'tests/native/nvapi_arithmetic.ptx'),'-o',str(blob)],
                cwd=build,capture_output=True,text=True,timeout=120)
            if compiled.returncode: raise RuntimeError(compiled.stdout+compiled.stderr)
            report.update(schema='nvapi-arithmetic-isolation/1', kernel='phoenix_arithmetic',
                blob_sha256=hashlib.sha256(blob.read_bytes()).hexdigest(), inference_run=False,
                gpu_code_only=True, assembler='ptxas')
            report_path=ROOT/'artifacts/current/x86-native-arithmetic-investigation.json'
        for arch, vcvars in (('x86', '32'), ('x64', '64')):
            out = build/arch
            out.mkdir()
            recipe = out/'build.cmd'
            recipe.write_text(
                '@echo off\n'
                f'call "{studio}/VC/Auxiliary/Build/vcvars{vcvars}.bat" >nul 2>&1\n'
                'if errorlevel 1 exit /b %errorlevel%\n'
                f'cl /nologo /std:c++20 /EHsc /Zi /Od /MT "{ROOT}/tests/native/nvapi_module_probe.cpp" '
                '/Fe:probe.exe /Fd:probe.pdb /link /DEBUG d3d12.lib dxgi.lib\n'
                'if errorlevel 1 exit /b %errorlevel%\n'
                f'cl /nologo /std:c++20 /EHsc /Od /MT "{ROOT}/tests/native/module_capability_tests.cpp" '
                '/Fe:capability_tests.exe\n', encoding='utf-8')
            compiled = subprocess.run(['cmd.exe', '/d', '/c', str(recipe)], cwd=out,
                                      capture_output=True, text=True, timeout=120)
            row = dict(compiled=compiled.returncode == 0)
            report['architectures'][arch] = row
            if compiled.returncode:
                row['compiler_error'] = (compiled.stdout+compiled.stderr)[-3500:]
            else:
                row['describe'] = execute([str(out/'probe.exe'), '--describe'], out)
                row['cpu_tests'] = execute([str(out/'capability_tests.exe')], out)
                if args.run_gpu_probes:
                    # Separate short-lived processes distinguish capability-query
                    # faults from module-creation faults, without a game process.
                    for stage in (('arithmetic',) if args.arithmetic else ('cuda', 'capability', 'module', 'shader', 'safe-module')):
                        command = [str(out/'probe.exe'), '--'+stage, str(blob)]
                        if stage == 'shader':
                            command += [kernel['entry'], *map(str, kernel['block'])]
                        row[stage] = execute(command, out)
                    if arch == 'x86' and args.trace_driver:
                        debugger = Path(os.environ['ProgramFiles(x86)'])/'Windows Kits/10/Debuggers/x64/cdb.exe'
                        if not debugger.is_file():
                            raise RuntimeError('The x64 Windows debugger is required for --trace-driver')
                        script = out/'driver.cdb'
                        script.write_text('sxd 0x4000001f\nsxd bpe\nsxe av\ng\n.effmach x86\nr\nkv 16\nub @eip L12\nu @eip L12\n'
                                          'dps @esp L24\nlm m nv*\nq\n', encoding='ascii')
                        row['module_debugger'] = execute([str(debugger), '-G', '-cf', str(script),
                            str(out/'probe.exe'), '--module', str(blob)], out, timeout=45, include_stdout=True)
            report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(dict(architecture=arch, **row)), flush=True)
    return 0 if all(row.get('compiled') and row.get('describe', {}).get('exit_code') == 0
                    and row.get('cpu_tests', {}).get('exit_code') == 0
                    for row in report['architectures'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
