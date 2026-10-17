"""Package full x64/x86 addons with an optional x64 helper (no model DLL).

First build a validated embedded x64 core with build_addon.py. ReShade x64 is
an explicit external build input; the resulting addon carries it for the helper.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
from embed_runtime import compress

ROOT = Path(__file__).resolve().parents[1]
LIBRARIES = "d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib bcrypt.lib psapi.lib cabinet.lib shell32.lib ole32.lib uuid.lib windowscodecs.lib"
CORE_ABI = {"AddonInit", "AddonUninit", "CnrPHOENIXCompatibilityStatus",
            "CnrPHOENIXFeederVersion", "CnrPHOENIXFeederEvaluate",
            "CnrPHOENIXFeederProgressJson", "CnrPHOENIXFeederUiJson",
            "CnrPHOENIXFeederUiAction", "CnrPHOENIXFeederLiveSettings"}


def file_sha256(path):
    # The embedded core is large. Identity checks should not allocate its size.
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def verify_core_inputs(core, report):
    """Bind relinked objects/resources and their source identities to the build receipt."""
    core = Path(core).resolve()
    if report.get('passed') is not True:
        raise ValueError('Core source build tests must pass before packaging')
    if report.get('binaries', {}).get('phoenix.addon64') != file_sha256(core / 'phoenix.addon64'):
        raise ValueError('Core binary changed after its source-build report')
    missing = CORE_ABI - pe_exports(core / 'phoenix.addon64')
    if missing:
        raise ValueError('Core lacks required helper ABI: ' + ', '.join(sorted(missing)))
    manifest = json.loads((core / 'embedded/manifest.json').read_text(encoding='utf-8'))
    required = {name + '.obj' for name in ('addon', 'hook', 'buffer', 'trampoline', 'hde64')}
    required.add('embedded/manifest.json')
    required.update(f"embedded/{entry['resource']}.bin" for entry in manifest['files'])
    bindings = report.get('link_inputs', {})
    if not isinstance(bindings, dict) or not required.issubset(bindings):
        raise ValueError('Core report must bind every linked object, manifest and resource in link_inputs')
    verified = {}
    for name in sorted(required):
        path = (core / name).resolve()
        if not path.is_relative_to(core) or bindings[name] != file_sha256(path):
            raise ValueError('Core link input changed after compilation: ' + name)
        verified[name] = bindings[name]
    sources = report.get('source_inputs', {})
    required_sources = set(report.get('project_dependencies', []))
    required_sources.update(('src/integration/reshade/addon.cpp',
                             'src/integration/reshade/shared_panel.h',
                             'src/integration/reshade/panel_contract.h',
                             'src/integration/compatibility/controller.h',
                             'tests/native/shared_panel_tests.cpp'))
    if not isinstance(sources, dict) or not required_sources.issubset(sources):
        raise ValueError('Core report must bind project source and test identities in source_inputs')
    for name, digest in sources.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest != file_sha256(path):
            raise ValueError('Source changed after core compilation/test verification: ' + name)
    return verified, sources


def pe_machine(path):
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError(f"Not a Windows executable: {path}")
    offset = struct.unpack_from("<I", data, 60)[0]
    if data[offset:offset + 4] != b"PE\0\0":
        raise ValueError(f"Not a PE executable: {path}")
    return struct.unpack_from("<H", data, offset + 4)[0]


def pe_exports(path):
    data = path.read_bytes()
    header = struct.unpack_from("<I", data, 60)[0]
    count = struct.unpack_from("<H", data, header + 6)[0]
    optional_size = struct.unpack_from("<H", data, header + 20)[0]
    optional = header + 24
    directory = optional + (112 if struct.unpack_from("<H", data, optional)[0] == 0x20b else 96)
    export_rva = struct.unpack_from("<I", data, directory)[0]

    def offset(rva):
        for index in range(count):
            virtual_bytes, start, raw_bytes, raw = struct.unpack_from("<IIII", data, optional + optional_size + index * 40 + 8)
            if start <= rva < start + max(virtual_bytes, raw_bytes):
                return raw + rva - start
        raise ValueError("Invalid executable export address")

    if not export_rva:
        return set()
    table = offset(export_rva)
    names = struct.unpack_from("<I", data, table + 24)[0]
    name_table = offset(struct.unpack_from("<I", data, table + 32)[0])
    result = set()
    for index in range(names):
        start = offset(struct.unpack_from("<I", data, name_table + index * 4)[0])
        result.add(data[start:data.index(b"\0", start)].decode("ascii"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--core-build", type=Path, required=True)
    parser.add_argument("--reshade64", type=Path, required=True)
    parser.add_argument("--ngx-sdk", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name):
        parser.error("Use a simple output name")
    core = args.core_build.resolve()
    reshade = args.reshade64.resolve()
    if pe_machine(reshade) != 0x8664 or pe_machine(core / "phoenix.addon64") != 0x8664:
        parser.error("The core and helper ReShade must both be x64")
    if not {"ReShadeRegisterAddon", "ReShadeGetImGuiFunctionTable"}.issubset(pe_exports(reshade)):
        parser.error("--reshade64 must be an addon-capable ReShade DLL, not another DXGI proxy")
    report = json.loads((core / "report.json").read_text())
    try:
        linked_inputs, source_inputs = verify_core_inputs(core, report)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    out = ROOT / "build" / args.name
    out.mkdir(exist_ok=False, parents=True)
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    studio = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                                     "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], text=True).strip()

    def run(name, arch, command):
        recipe = out / (name + ".cmd")
        recipe.write_text(f'@echo off\ncall "{studio}/VC/Auxiliary/Build/vcvars{arch}.bat" > "{out}/{name}.toolchain.log" 2>&1\n'
                          f'{command} > "{out}/{name}.build.log" 2>&1\nexit /b %errorlevel%\n')
        subprocess.run(["cmd", "/d", "/c", str(recipe)], cwd=out, check=True)

    includes = f'/I"{ROOT}/third_party" /I"{ROOT}/third_party/reshade"'
    run("host", "64", f'cl /nologo /std:c++20 /Zc:__cplusplus /O2 /EHsc /MT /DNOMINMAX {includes} '
        f'"{ROOT}/src/integration/compatibility/host/dlss5-feed-host64.cpp" /Fo:"{out}/host.obj" /Fe:"{out}/phoenix_graphics_host.exe" '
        '/link d3d12.lib dxgi.lib d3dcompiler.lib version.lib user32.lib gdi32.lib advapi32.lib ole32.lib windowscodecs.lib')
    original = json.loads((core / "embedded/manifest.json").read_text())

    def resources(arch):
        folder = out / ("embedded" + arch)
        folder.mkdir()
        manifest = {**original, "files": []}
        paths = {}
        # Native GPU kernels are architecture-neutral. The x64 portable plugins
        # stay available to the helper, never loaded by the x86 native executor.
        manifest["files"] = list(original["files"])
        paths = {entry["resource"]: core / "embedded" / (str(entry["resource"]) + ".bin") for entry in manifest["files"]}
        additions = {"compatibility/phoenix_graphics_host.exe": out / "phoenix_graphics_host.exe",
                     "compatibility/dxgi.dll": reshade,
                     "compatibility/ReShade-LICENSE.txt": ROOT / "third_party/compatibility_bridge/ReShade-LICENSE.txt",
                     "compatibility/Host-LICENSE.txt": ROOT / "third_party/compatibility_bridge/LICENSE",
                     "compatibility/Host-NOTICE.txt": ROOT / "third_party/compatibility_bridge/NOTICE"}
        if arch == "32":
            additions["compatibility/Phoenix.addon64"] = core / "phoenix.addon64"
        for name, path in additions.items():
            data = path.read_bytes()
            # An x86 game may not have a spare contiguous 200 MiB allocation.
            # The already compressed embedded core gains little from another
            # compression pass; raw resources can be written directly in chunks.
            packed = data if name == "compatibility/Phoenix.addon64" else compress(data)
            algorithm = "xpress_huff" if len(packed) < len(data) else "raw"
            if algorithm == "raw":
                packed = data
            index = 1100 + len(manifest["files"])
            target = folder / (str(index) + ".bin")
            target.write_bytes(packed)
            paths[index] = target
            manifest["files"].append(dict(path=name, resource=index, bytes=len(data),
                                          sha256=hashlib.sha256(data).hexdigest(), algorithm=algorithm, packed_bytes=len(packed)))
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        recipe = folder / "runtime.rc"
        recipe.write_text(f'1001 RCDATA "{(folder / "manifest.json").as_posix()}"\n' +
                          "\n".join(f'{index} RCDATA "{path.as_posix()}"' for index, path in paths.items()) + "\n")
        run("resources" + arch, arch, f'rc /nologo /fo "{folder}/runtime.res" "{recipe}"')
        return folder / "runtime.res"

    resource64 = resources("64")
    # Recheck immediately before linking: packaging/compression may take time.
    verify_core_inputs(core, report)
    objects = " ".join(f'"{core}/{name}.obj"' for name in ("addon", "hook", "buffer", "trampoline", "hde64"))
    run("addon64", "64", f'link /DLL /INCREMENTAL:NO /OUT:"{out}/Phoenix.addon64" {objects} "{resource64}" {LIBRARIES}')
    resource32 = resources("32")
    hooks32 = ' '.join(f'"{ROOT}/third_party/minhook/src/{name}"'
        for name in ('hook.c', 'buffer.c', 'trampoline.c', 'hde/hde32.c'))
    run("addon32", "32", f'cl /nologo /std:c++20 /Zc:__cplusplus /O2 /EHsc /MT /bigobj /DNOMINMAX /DImTextureID=ImU64 '
        f'{includes} /I"{args.ngx_sdk.resolve()}" /I"{ROOT}/third_party/nvapi" '
        f'/I"{ROOT}/third_party/minhook/include" /I"{ROOT}/third_party/imgui_reshade" '
        f'/LD "{ROOT}/src/integration/reshade/addon.cpp" {hooks32} '
        f'/Fo:"{out}\\\\" /Fe:"{out}/Phoenix.addon32" /link /INCREMENTAL:NO "{resource32}" {LIBRARIES}')
    # WINAPI exports are decorated on x86. Preserve the public undecorated ABI
    # used by scanners, the feeder API and compatibility control clients.
    aliases = []
    for name in sorted(pe_exports(out / 'Phoenix.addon32')):
        match = re.fullmatch(r'_(.+)@(\d+)', name)
        if match:
            aliases.append(f'    {match[1]}={name}')
    definition = out / 'addon32.def'
    definition.write_text('EXPORTS\n' + '\n'.join(aliases) + '\n')
    objects32 = ' '.join(f'"{out}/{name}.obj"' for name in ('addon','hook','buffer','trampoline','hde32'))
    run('addon32_exports', '32', f'link /DLL /INCREMENTAL:NO /DEF:"{definition}" '
        f'/OUT:"{out}/Phoenix.addon32" {objects32} "{resource32}" {LIBRARIES}')
    binaries = {}
    for name, machine in (("Phoenix.addon64", 0x8664), ("Phoenix.addon32", 0x14c), ("phoenix_graphics_host.exe", 0x8664)):
        path = out / name
        if pe_machine(path) != machine:
            raise ValueError("Wrong binary architecture: " + name)
        if name.endswith((".addon32", ".addon64")) and not {"AddonInit", "AddonUninit", "CnrPHOENIXCompatibilityStatus"}.issubset(pe_exports(path)):
            raise ValueError("Missing addon/bridge entry points: " + name)
        if name.endswith(('.addon32','.addon64')) and not (CORE_ABI | {'CnrPHOENIXInProcessBackbufferVersion'}).issubset(pe_exports(path)):
            raise ValueError('Packaged addon is missing the required full-runtime ABI: ' + name)
        binaries[name] = dict(bytes=path.stat().st_size, sha256=file_sha256(path))
    run('embedded_test', '64', f'cl /nologo /std:c++20 /EHsc /MT /DNOMINMAX '
        f'"{ROOT}/tests/native/embedded_runtime_tests.cpp" /Fo:"{out}/embedded_test.obj" '
        f'/Fe:"{out}/embedded_test.exe" /link bcrypt.lib cabinet.lib shell32.lib ole32.lib')
    run('verify_embedded', '64', f'"{out}/embedded_test.exe" "{out}/Phoenix.addon64" '
        f'"{out}/payload-cache" "{out}/embedded-check.json"')
    embedded_check = json.loads((out / 'embedded-check.json').read_text())
    if embedded_check.get('passed') is not True:
        raise ValueError('Packaged core and companion extraction failed')
    (out / "report.json").write_text(json.dumps(dict(schema="phoenix-compatibility-build/1", passed=True,
        binaries=binaries, model_data_included=False, nvidia_dll_included=False,
        gpu_inference_validated=False, core_source_build=str(core),
        link_inputs=linked_inputs, source_inputs=source_inputs,
        embedded_payload_verification=embedded_check,
        required_core_exports=sorted(CORE_ABI)), indent=2) + "\n")
    print(json.dumps(dict(passed=True, build=str(out), binaries=list(binaries))))


if __name__ == "__main__":
    main()
