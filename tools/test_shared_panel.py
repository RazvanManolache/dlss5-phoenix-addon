"""Test shared controls on CPU with matching external ImGui v1.92.5-docking sources."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def check_build_gates():
    """Negative packaging/exit-status checks require no compiler, game or GPU."""
    import build_compatibility as packaging
    checks = []
    for passed in (False, True):
        result = subprocess.run([sys.executable, '-c',
            'from test_graphics_apis import report_exit_code; '
            f'raise SystemExit(report_exit_code({{"passed": {passed!r}}}))'],
            cwd=ROOT / 'tools', capture_output=True, text=True)
        if result.returncode != (0 if passed else 1):
            raise RuntimeError('Recorded graphics receipt did not control process exit status')
        checks.append('graphics_process_pass' if passed else 'graphics_process_failure')
    with tempfile.TemporaryDirectory(prefix='phoenix-package-gate-') as temporary:
        core = Path(temporary)
        (core / 'embedded').mkdir()
        (core / 'phoenix.addon64').write_bytes(b'test-only core placeholder')
        (core / 'embedded/manifest.json').write_text(json.dumps({'files': [{'resource': 1100}]}))
        names = ['addon.obj', 'hook.obj', 'buffer.obj', 'trampoline.obj', 'hde64.obj',
                 'embedded/1100.bin']
        for name in names:
            (core / name).write_bytes(name.encode())
        report = {'passed': True, 'binaries': {'phoenix.addon64': packaging.file_sha256(core / 'phoenix.addon64')},
                  'link_inputs': {name: packaging.file_sha256(core / name) for name in
                      [*names, 'embedded/manifest.json']},
                  'source_inputs': {name: packaging.file_sha256(ROOT / name) for name in
                      ['src/integration/reshade/addon.cpp', 'src/integration/reshade/shared_panel.h',
                       'src/integration/reshade/panel_contract.h', 'src/integration/compatibility/controller.h',
                       'tests/native/shared_panel_tests.cpp']}}
        def reject(label, receipt, exports):
            with patch.object(packaging, 'pe_exports', return_value=exports):
                try:
                    packaging.verify_core_inputs(core, receipt)
                except ValueError:
                    checks.append(label)
                    return
            raise RuntimeError('Packaging accepted invalid input: ' + label)
        with patch.object(packaging, 'pe_exports', return_value=packaging.CORE_ABI):
            packaging.verify_core_inputs(core, report)
        checks.append('valid_bound_inputs')
        reject('old_core_without_ui_export', report, packaging.CORE_ABI - {'CnrPHOENIXFeederUiAction'})
        reject('unbound_object_inputs', {**report, 'link_inputs': {}}, packaging.CORE_ABI)
        reject('unbound_source_inputs', {**report, 'source_inputs': {}}, packaging.CORE_ABI)
        original = (core / 'addon.obj').read_bytes()
        (core / 'addon.obj').write_bytes(b'changed after report')
        reject('changed_object', report, packaging.CORE_ABI)
        (core / 'addon.obj').write_bytes(original)
        (core / 'embedded/1100.bin').write_bytes(b'changed after report')
        reject('changed_payload', report, packaging.CORE_ABI)
        (core / 'embedded/1100.bin').write_bytes(b'embedded/1100.bin')
        bad_sources = {**report['source_inputs'], 'tests/native/shared_panel_tests.cpp': '0' * 64}
        reject('changed_source_or_test', {**report, 'source_inputs': bad_sources}, packaging.CORE_ABI)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--imgui-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    vendor = args.imgui_source.resolve()
    def normalized_header(path):
        return '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8-sig').splitlines())
    if normalized_header(vendor / 'imgui.h') != normalized_header(ROOT / 'third_party/imgui_reshade/imgui.h'):
        parser.error('Use matching ImGui v1.92.5-docking sources; equal version numbers alone do not establish matching structure layouts')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'build'):
        parser.error('Test output must be inside build')
    output.mkdir(parents=True, exist_ok=False)
    gate_checks = check_build_gates()
    vswhere = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    results = []
    for bits in (32, 64):
        case = output / str(bits)
        case.mkdir()
        files = [ROOT / 'tests/native/shared_panel_tests.cpp'] + [vendor / (name + '.cpp') for name in
            ('imgui', 'imgui_draw', 'imgui_tables', 'imgui_widgets')]
        sources = ' '.join('"' + str(file) + '"' for file in files)
        recipe = case / 'build.cmd'
        recipe.write_text(f'@echo off\ncall "{studio}/VC/Auxiliary/Build/vcvars{bits}.bat" > "{case}/toolchain.log" 2>&1\n'
            f'cl /nologo /std:c++20 /Zc:__cplusplus /O2 /EHsc /MT /DNOMINMAX /DIMGUI_DISABLE_INCLUDE_IMCONFIG_H /DIMGUI_ENABLE_TEST_ENGINE '
            f'/DImTextureID=ImU64 /I"{ROOT}/third_party/imgui_reshade" /I"{vendor}" {sources} '
            f'/Fe:"{case}/panel_tests.exe" /link d3dcompiler.lib bcrypt.lib cabinet.lib shell32.lib ole32.lib uuid.lib '
            f'> "{case}/build.log" 2>&1\nexit /b %errorlevel%\n')
        subprocess.run(['cmd', '/d', '/c', str(recipe)], cwd=case, check=True)
        result = subprocess.run([str(case / 'panel_tests.exe')], cwd=case, capture_output=True, text=True, timeout=30)
        (case / 'stdout.log').write_text(result.stdout)
        (case / 'stderr.log').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr)
        results.append(json.loads(result.stdout))
        print(result.stdout, flush=True)
    (output / 'report.json').write_text(json.dumps({'passed': all(item['passed'] for item in results),
        'gpu_executed': False, 'imgui_version': '1.92.5-docking', 'header_layout_verified': True,
        'cases': results,
        'negative_gate_checks': gate_checks}, indent=2))


if __name__ == '__main__':
    main()
