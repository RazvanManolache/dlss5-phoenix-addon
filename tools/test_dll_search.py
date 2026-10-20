"""Focused CPU-only DLL search/settings/UI tests and addon compile checks.

No game, driver/model loading, GPU work, installation or redistributable package.
All binaries are temporary; only the compact validation receipt is retained.
"""
import argparse
import json
from pathlib import Path
import tempfile

from test_model_selection import ROOT, run, toolchain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ngx-sdk', type=Path, required=True)
    parser.add_argument('--imgui-source', type=Path, required=True)
    parser.add_argument('--model-library', type=Path)
    parser.add_argument('--model-reference-dll', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/current/dll-search-tests.json')
    args = parser.parse_args()
    imgui = args.imgui_source.resolve()
    normalize = lambda path: '\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8-sig').splitlines())
    if normalize(imgui/'imgui.h') != normalize(ROOT/'third_party/imgui_reshade/imgui.h'):
        raise RuntimeError('UI tests need exactly matching ImGui headers and implementation')
    report = dict(passed=False, gpu_used=False, games_launched=False, installed_files_modified=False, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='dll-search-', dir=ROOT/'build') as directory:
        work = Path(directory)
        for bits in (32, 64):
            compiler, env = toolchain(bits)
            common = [compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/O2', '/EHsc', '/MT', '/bigobj', '/DNOMINMAX']
            libraries = ['d3dcompiler.lib', 'bcrypt.lib', 'cabinet.lib', 'shell32.lib', 'ole32.lib', 'uuid.lib']
            for name in ('model_tools_tests', 'compatibility_policy_tests', 'shared_panel_tests', 'model_library_tests'):
                executable = work/f'{name}-{bits}.exe'
                sources = [ROOT/f'tests/native/{name}.cpp']
                flags = []
                if name == 'shared_panel_tests':
                    flags = ['/DIMGUI_DISABLE_INCLUDE_IMCONFIG_H', '/DIMGUI_ENABLE_TEST_ENGINE', '/DImTextureID=ImU64',
                             f'/I{ROOT}/third_party/imgui_reshade', f'/I{imgui}']
                    sources += [imgui/f'{part}.cpp' for part in ('imgui', 'imgui_draw', 'imgui_tables', 'imgui_widgets')]
                run(common+flags+sources+[f'/Fe:{executable}', '/link', *libraries], cwd=work, env=env)
                arguments = [executable]
                if name == 'model_library_tests' and args.model_library and args.model_reference_dll:
                    arguments += [args.model_library.resolve(), args.model_reference_dll.resolve()]
                result = json.loads(run(arguments, cwd=work, env=env, timeout=60))
                if not result['passed']: raise RuntimeError(f'{name} rejected its contracts')
                report['cases'].append(dict(bits=bits, test=name, result=result))
                print(f'{bits}-bit {name}: passed', flush=True)
            # Compile the actual shared x86/x64 entry point, not just a stand-in
            # controller: this catches missed state fields and platform branches.
            includes = [f'/I{args.ngx_sdk.resolve()}', f'/I{ROOT}/third_party/nvapi',
                        f'/I{ROOT}/third_party/minhook/include', f'/I{ROOT}/third_party/imgui_reshade',
                        f'/I{ROOT}/third_party/reshade']
            run(common+includes+['/DImTextureID=ImU64', '/c', ROOT/'src/integration/reshade/addon.cpp',
                                 f'/Fo:{work}/addon-{bits}.obj'], cwd=work, env=env, timeout=300)
            report['cases'].append(dict(bits=bits, test='production_addon_compile', passed=True))
            print(f'{bits}-bit production addon compile: passed', flush=True)
    report.update(passed=True, temporary_build_removed=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
