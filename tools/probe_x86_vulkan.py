"""Bounded x86/x64 Vulkan compute and optional CUDA-launch-extension probe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from probe_nvapi_module import execute

ROOT = Path(__file__).resolve().parents[1]


def conclusion(probes):
    def passed(arch, stage):
        row = probes.get(arch, {})
        return row.get('exit_code') == 0 and any(
            event.get('stage') == stage and event.get('passed') is True and
            event.get('matched') == event.get('expected') == 4096
            for event in row.get('events', []))
    return dict(x86_spirv_compute_works=passed('x32', 'spirv_result'),
                x64_spirv_compute_works=passed('x64', 'spirv_result'),
                x86_cuda_dispatch_works=passed('x32', 'cuda_result'),
                x64_cuda_dispatch_works=passed('x64', 'cuda_result'),
                completed_gpu_inference=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deps', type=Path, required=True,
                        help='Local headers/include and glslang/bin/glslang.exe tree')
    args = parser.parse_args()
    deps = args.deps.resolve()
    vswhere = Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    if not studio:
        raise RuntimeError('No MSVC toolchain')
    report = dict(schema='x86-vulkan-feasibility/1', production_package_modified=False,
                  completed_gpu_inference=False, builds={}, probes={},
                  source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                      for name in ('tests/native/x86_vulkan_probe.cpp', 'tests/native/x86_vulkan_probe.comp')})
    receipt = ROOT/'artifacts/current/x86-vulkan-investigation.json'
    def save():
        receipt.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='x86-vulkan-probe-', dir=ROOT/'artifacts') as temporary:
        build = Path(temporary)
        shader = build/'probe.spv'
        subprocess.run([str(deps/'glslang/bin/glslang.exe'), '--target-env', 'vulkan1.2',
            '-V', str(ROOT/'tests/native/x86_vulkan_probe.comp'), '-o', str(shader)],
            check=True, capture_output=True, text=True, timeout=30)
        report['shader_sha256'] = hashlib.sha256(shader.read_bytes()).hexdigest()
        for arch in ('32', '64'):
            out = build/arch
            out.mkdir()
            recipe = out/'build.cmd'
            recipe.write_text('\n'.join(['@echo off',
                f'call "{studio}/VC/Auxiliary/Build/vcvars{arch}.bat" >nul 2>&1',
                'if errorlevel 1 exit /b %errorlevel%',
                f'cl /nologo /std:c++20 /EHsc /O2 /MT /I"{deps}/headers/include" '
                f'"{ROOT}/tests/native/x86_vulkan_probe.cpp" /Fe:probe.exe',
                'exit /b %errorlevel%', '']), encoding='ascii')
            compiled = subprocess.run(['cmd.exe', '/d', '/c', str(recipe)], cwd=out,
                capture_output=True, text=True, timeout=120)
            report['builds']['x'+arch] = dict(passed=compiled.returncode == 0,
                diagnostics=(compiled.stdout+compiled.stderr)[-6000:] if compiled.returncode else '')
            save()
            if compiled.returncode:
                print(json.dumps(report['builds']), flush=True)
                return 1
            report['probes']['x'+arch] = execute([str(out/'probe.exe'), str(shader)], out, timeout=30)
            save()
            print(json.dumps(dict(arch='x'+arch, **report['probes']['x'+arch])), flush=True)
        report['conclusion'] = conclusion(report['probes'])
        save()
        print(json.dumps(report['conclusion']), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
