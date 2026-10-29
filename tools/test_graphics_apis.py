"""Build and run real ReShade integration checks, with no desktop input.

Supply an external model and addon-capable ReShade runtimes for both bitnesses.
All binaries and GPU pixel evidence remain in the selected build directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def copy_case_configuration(source, directory):
    """Keep each cloned fixture's in-place approval tied to its own folder."""
    destination = directory / source.name
    if source.name == 'Phoenix.json':
        settings = json.loads(source.read_text(encoding='utf-8-sig'))
        settings['sandbox'] = settings['in_place_game_directory'] = str(directory.resolve())
        destination.write_text(json.dumps(settings))
    else:
        destination.write_bytes(source.read_bytes())


def report_exit_code(report):
    """A written failure receipt must also fail shell/CI callers."""
    return 0 if report.get('passed') is True else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--weight-source', type=Path, required=True)
    parser.add_argument('--reshade32', type=Path, required=True)
    parser.add_argument('--reshade64', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vulkan-headers', type=Path)
    parser.add_argument('--vulkan-secondary', action='store_true', help='Use a separate presentation queue for base Vulkan cases')
    parser.add_argument('--vulkan-compute-present', action='store_true', help='Present from a compute-only queue family in base Vulkan cases')
    parser.add_argument('--d3d9-msaa', action='store_true', help='Use the highest supported windowed D3D9 multisampling (8x, 4x or 2x)')
    parser.add_argument('--d3d12-rgb10', action='store_true', help='Exercise packed 10-bit SDR capture and final-frame presentation')
    parser.add_argument('--dxgi-rgb10', action='store_true', help='Exercise packed 10-bit SDR on every selected DXGI API')
    parser.add_argument('--execution-mode', choices=['auto', 'in_process', 'helper'], default='auto',
                        help='Require an in-process route or explicitly exercise the helper fallback')
    parser.add_argument('--control-checks', action='store_true', help='Exercise writable game-control and screenshot commands')
    parser.add_argument('--process-exit-checks', action='store_true', help='Exit with live graphics resources and require bounded process termination')
    parser.add_argument('--thin-d3d12', action='store_true', help='Use the test-only thin frontend instead of testing the shipped x64 addon')
    parser.add_argument('--multiwindow-check', action='store_true', help='Verify two OpenGL windows sharing one device stay isolated')
    parser.add_argument('--resize-checks', action='store_true', help='Cycle DXGI resolutions and verify processed output after each resize')
    parser.add_argument('--motion-checks', action='store_true',
                        help='Change the live image while an older result completes; require the same paired output')
    parser.add_argument('--display', type=int, choices=[2, 3], default=3,
                        help='2: matching original/processed split, 3: processed image')
    parser.add_argument('--bits', nargs='+', type=int, choices=[32, 64], default=[32, 64])
    parser.add_argument('--apis', nargs='+', choices=['d3d9', 'd3d10', 'd3d10_1', 'd3d11', 'd3d12', 'opengl', 'vulkan'],
                        default=['d3d9', 'd3d10', 'd3d10_1', 'd3d11', 'd3d12', 'opengl'])
    args = parser.parse_args()
    if args.vulkan_secondary and args.vulkan_compute_present:
        parser.error('Choose one Vulkan separate-queue scenario per run')
    if 'vulkan' in args.apis and not args.vulkan_headers:
        parser.error('Vulkan test requires --vulkan-headers pointing to an include directory')
    package = args.package.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'build'):
        parser.error('Test outputs must be inside this repository build directory')
    output.mkdir(parents=True, exist_ok=False)
    model = json.loads(args.weight_source.read_text(encoding='utf-8-sig'))
    vswhere = Path(os.environ['ProgramFiles(x86)']) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    studio = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-requires',
                                     'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath'], text=True).strip()
    report = dict(passed=False, cases=[], motion_cases=[], control_cases=[], multiwindow_cases=[], resize_cases=[], layouts=[], model_included=False)
    path = output / 'report.json'

    def build(name, bits, command):
        recipe = output / (name + '.cmd')
        recipe.write_text(f'@echo off\ncall "{studio}/VC/Auxiliary/Build/vcvars{bits}.bat" > "{output}/{name}.toolchain.log" 2>&1\n'
                          f'{command} > "{output}/{name}.build.log" 2>&1\nexit /b %errorlevel%\n')
        subprocess.run(['cmd', '/d', '/c', str(recipe)], check=True, cwd=output)

    try:
        for bits in args.bits:
            common = f'cl /nologo /std:c++20 /Zc:__cplusplus /O2 /EHsc /MT /DNOMINMAX /I"{ROOT}/third_party/reshade"'
            if args.vulkan_headers:
                common += f' /DPHOENIX_TEST_VULKAN /I"{args.vulkan_headers.resolve()}"'
            build(f'api{bits}', bits, f'{common} "{ROOT}/tests/native/compatibility_graphics_matrix.cpp" '
                  f'/Fo:"{output}/api{bits}.obj" /Fe:"{output}/api{bits}.exe" /link user32.lib gdi32.lib bcrypt.lib ole32.lib windowscodecs.lib')
            build(f'layout{bits}', bits, f'{common} "{ROOT}/tests/native/graphics_transport_tests.cpp" '
                  f'/Fo:"{output}/layout{bits}.obj" /Fe:"{output}/layout{bits}.exe" /link bcrypt.lib cabinet.lib shell32.lib ole32.lib')
            report['layouts'].append(json.loads(subprocess.check_output([str(output / f'layout{bits}.exe')], text=True)))
            # The thin frontend always exercises helper transport. Test the
            # actual shipped x64 addon by default, including its route decision.
            if args.thin_d3d12 and bits == 64 and 'd3d12' in args.apis:
                build('thin64', bits, f'{common} /DNOMINMAX /DPHOENIX_COMPATIBILITY_ONLY /DImTextureID=ImU64 '
                      f'/I"{ROOT}/third_party" /I"{ROOT}/third_party/imgui_reshade" /LD '
                      f'"{ROOT}/src/integration/compatibility/addon32.cpp" /Fo:"{output}/thin64.obj" /Fe:"{output}/thin.addon64" '
                      f'/link "{package}/embedded32/runtime.res" d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib '
                      'user32.lib bcrypt.lib cabinet.lib shell32.lib ole32.lib uuid.lib windowscodecs.lib')
            for api in args.apis:
                case = output / f'{bits}_{api}'
                case.mkdir()
                addon = output / 'thin.addon64' if args.thin_d3d12 and bits == 64 and api == 'd3d12' else package / f'Phoenix.addon{bits}'
                for source, name in ((output / f'api{bits}.exe', f'api{bits}.exe'),
                                     (addon, f'Phoenix.addon{bits}'),
                                     ((args.reshade32 if bits == 32 else args.reshade64).resolve(),
                                      'opengl32.dll' if api == 'opengl' else 'dxgi.dll')):
                    os.link(source, case / name)
                (case / 'Phoenix.json').write_text(json.dumps({
                    'schema': 'phoenix-live/1', 'armed': True, 'input_provider': 'feeder_d3d12',
                    'direct_transfer': 'srgb', 'color_domain': 'game_nr_sdr_proxy',
                    'sandbox': str(case), 'in_place_game_directory': str(case),
                    'in_place_approval': 'I approve driving this installed game in place',
                    'frame_limit': 4000000, 'weight_source': model}))
                (case / 'ReShade.ini').write_text('[GENERAL]\nEffectSearchPaths=\nPresetPath=ReShadePreset.ini\n'
                                                f'[Phoenix]\nBackend={args.display}\nComparison=1\nSourceA=0\nSourceB=2\nDivider=0.5\nForceReset=1\n'
                                                f'[Phoenix.Compatibility]\nEnabled=1\nExecutionMode={args.execution_mode}\n'
                                                'StandaloneInput=1\nOffloadAgeFrames=6000\nOffloadAgeMs=120000\n')
                environment = os.environ.copy()
                environment.pop('PHOENIX_TEST_RGB10', None)
                environment.pop('PHOENIX_EXPECT_INPROCESS', None)
                if args.execution_mode == 'in_process':
                    environment['PHOENIX_EXPECT_INPROCESS'] = '1'
                if (api == 'd3d12' and args.d3d12_rgb10) or (args.dxgi_rgb10 and api in ('d3d10', 'd3d10_1', 'd3d11', 'd3d12')):
                    environment['PHOENIX_TEST_RGB10'] = '1'
                if api == 'vulkan':
                    (case / 'VkLayer_phoenix_test.json').write_text(json.dumps({
                        'file_format_version': '1.0.0', 'layer': {
                            'name': 'VK_LAYER_phoenix_integration_test', 'type': 'GLOBAL', 'library_path': './dxgi.dll',
                            'api_version': '1.3.0', 'implementation_version': '1',
                            'description': 'Task-local ReShade integration test',
                            'functions': {'vkGetInstanceProcAddr': 'vkGetInstanceProcAddr', 'vkGetDeviceProcAddr': 'vkGetDeviceProcAddr'}}}))
                    environment['VK_LAYER_PATH'] = str(case)
                evidence = {'bits': bits, 'fixture_api': api, 'passed': False,
                            'addon_sha256': hashlib.sha256(addon.read_bytes()).hexdigest(),
                            'thin_d3d12_frontend': args.thin_d3d12 and bits == 64 and api == 'd3d12'}
                evidence['d3d9_multisampling_requested'] = api == 'd3d9' and args.d3d9_msaa
                try:
                    command = [str(case / f'api{bits}.exe'), api]
                    if args.process_exit_checks: command.append('--process-exit')
                    if api == 'vulkan' and args.vulkan_secondary: command.append('--secondary-queue')
                    if api == 'vulkan' and args.vulkan_compute_present: command.append('--compute-present')
                    if api == 'd3d9' and args.d3d9_msaa: command.append('--msaa-control' if args.control_checks else '--msaa')
                    completed = subprocess.run(command, cwd=case,
                                               capture_output=True, text=True, timeout=110, env=environment)
                    (case / 'stdout.log').write_text(completed.stdout)
                    (case / 'stderr.log').write_text(completed.stderr)
                    evidence.update(json.loads((case / 'result.json').read_text()))
                    evidence['exit_code'] = completed.returncode
                    evidence['passed'] = completed.returncode == 0 and evidence['passed']
                    if args.process_exit_checks:
                        evidence['passed'] = evidence['passed'] and evidence.get('process_exit_requested') is True
                except (OSError, subprocess.TimeoutExpired) as error:
                    evidence['error'] = str(error)
                    # Do not retry, rename or weaken protections after an OS denial.
                report['cases'].append(evidence)
                path.write_text(json.dumps(report, indent=2))
                print(json.dumps(evidence), flush=True)
                if args.motion_checks and evidence['passed']:
                    moving = output / f'{bits}_{api}_motion'
                    moving.mkdir()
                    for file in case.iterdir():
                        if file.suffix in ('.exe', '.dll', '.addon32', '.addon64'):
                            os.link(file, moving / file.name)
                        elif file.name in ('Phoenix.json', 'ReShade.ini', 'VkLayer_phoenix_test.json'):
                            copy_case_configuration(file, moving)
                    moving_env = environment.copy()
                    if api == 'vulkan':
                        moving_env['VK_LAYER_PATH'] = str(moving)
                    result = subprocess.run([str(moving / f'api{bits}.exe'), api, '--dynamic'], cwd=moving,
                                            capture_output=True, text=True, timeout=110, env=moving_env)
                    (moving / 'stdout.log').write_text(result.stdout)
                    (moving / 'stderr.log').write_text(result.stderr)
                    proof = json.loads((moving / 'result.json').read_text())
                    proof['paired_pixels_identical_despite_new_motion'] = (
                        result.returncode == 0 and proof.get('dynamic_current_changed') is True and
                        (moving / 'output.rgba').read_bytes() == (case / 'output.rgba').read_bytes())
                    proof['passed'] = proof.get('passed', False) and proof['paired_pixels_identical_despite_new_motion']
                    report['motion_cases'].append(proof)
                    path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(proof), flush=True)
                extra_modes = []
                if args.control_checks and evidence['passed']:
                    extra_modes.append('control')
                if args.multiwindow_check and api == 'opengl' and evidence['passed']:
                    extra_modes.append('multiwindow')
                if args.resize_checks and api in ('d3d10', 'd3d10_1', 'd3d11', 'd3d12') and evidence['passed']:
                    extra_modes.append('resize')
                for extra in extra_modes:
                    folder = output / f'{bits}_{api}_{extra}'
                    folder.mkdir()
                    for file in case.iterdir():
                        if file.suffix in ('.exe', '.dll', '.addon32', '.addon64'):
                            os.link(file, folder / file.name)
                        elif file.name in ('Phoenix.json', 'ReShade.ini', 'VkLayer_phoenix_test.json'):
                            copy_case_configuration(file, folder)
                    extra_env = environment.copy()
                    if api == 'vulkan': extra_env['VK_LAYER_PATH'] = str(folder)
                    result = subprocess.run([str(folder / f'api{bits}.exe'), api, '--' + extra],
                                            cwd=folder, capture_output=True, text=True, timeout=240 if extra == 'resize' else 150, env=extra_env)
                    (folder / 'stdout.log').write_text(result.stdout)
                    (folder / 'stderr.log').write_text(result.stderr)
                    proof = json.loads((folder / 'result.json').read_text())
                    proof['exit_code'] = result.returncode
                    proof['passed'] = result.returncode == 0 and proof.get('passed') is True
                    report[extra + '_cases'].append(proof)
                    path.write_text(json.dumps(report, indent=2))
                    print(json.dumps(proof), flush=True)
                if isinstance(evidence.get('error'), str) and ('WinError 5' in evidence['error'] or 'WinError 32' in evidence['error']):
                    report['os_blocked_bits'] = bits
                    break
        successful = [case for case in report['cases'] if case['passed']]
        report['api_pixel_outputs_identical'] = bool(successful) and len({case['output_sha256'] for case in successful}) == 1
        report['passed'] = (len(report['cases']) == len(args.bits) * len(args.apis) and
                            all(case['passed'] for case in report['cases']) and report['api_pixel_outputs_identical'])
        if args.motion_checks:
            report['passed'] = (report['passed'] and len(report['motion_cases']) == len(report['cases']) and
                                all(case['passed'] for case in report['motion_cases']))
        if args.motion_checks:
            hashes = [case.get('changed_output_sha256') for case in report['motion_cases']]
            report['changed_input_outputs_identical'] = bool(hashes) and None not in hashes and len(set(hashes)) == 1
            report['passed'] = report['passed'] and report['changed_input_outputs_identical']
        if args.control_checks:
            report['passed'] = (report['passed'] and len(report['control_cases']) == len(report['cases']) and
                                all(case['passed'] for case in report['control_cases']))
        if args.multiwindow_check:
            expected = len(args.bits) if 'opengl' in args.apis else 0
            report['passed'] = (report['passed'] and len(report['multiwindow_cases']) == expected and
                                all(case['passed'] for case in report['multiwindow_cases']))
        if args.resize_checks:
            expected = len(args.bits) * sum(api in ('d3d10', 'd3d10_1', 'd3d11', 'd3d12') for api in args.apis)
            report['passed'] = (report['passed'] and len(report['resize_cases']) == expected and
                               all(case['passed'] for case in report['resize_cases']))
    finally:
        path.write_text(json.dumps(report, indent=2) + '\n')
    return report_exit_code(report)


if __name__ == '__main__':
    raise SystemExit(main())
