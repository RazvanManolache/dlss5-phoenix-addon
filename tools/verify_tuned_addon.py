"""CPU-check actual tuned addon DLLs and pin their embedded bundle to GPU evidence.

No games or GPU workloads run here. The GPU evidence is a separate explicit
bundle comparison; identical embedded resource hashes tie it to these DLLs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from compare_four_models import ROOT, sha, write
from prepare_addon_release import audit_embedded, safe_payload
from test_model_selection import toolchain, run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    package,bundle=args.package.resolve(),args.bundle.resolve()
    current=(ROOT/'artifacts/current').resolve()
    if not package.is_relative_to(current):
        parser.error('Package must stay under artifacts/current')
    evidence=json.loads(args.results.read_text())
    manifest=json.loads((bundle/'native_manifest.json').read_text())
    if not evidence['passed'] or evidence['manifests']['candidate'] != manifest:
        raise ValueError('Bundle differs from the successful GPU comparison')
    report=dict(passed=False,experimental=True,model_data_included=False,published=False,
                games_launched=False,installed_files_modified=False,older_device_validated=False,
                gpu_bundle_tested_on='RTX 5080',packaged_game_presentation_tested=False,
                native_sm120_byte_identical=False,benchmark_receipt_sha256=sha(args.results),
                benchmark_processes=len(evidence['cases']),comparisons=evidence['decisions'],
                binaries={},dll_load={},embedded={})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(ROOT/'tools/test_addon_probe_lifetime.py'),
                    '--package',str(package),'--output',str(args.output.parent/'probe-lifetime.json')],
                   cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    with tempfile.TemporaryDirectory(prefix='tuned-addon-audit-',dir=ROOT/'build') as temporary:
        work=Path(temporary)
        for bits in (32,64):
            compiler,env=toolchain(bits)
            host=work/f'load{bits}.exe'
            run([compiler,'/nologo','/std:c++20','/EHsc','/O2','/MT',
                 ROOT/'tests/native/packaged_addon_load_tests.cpp',
                 '/Fo:'+str(work/f'load{bits}.obj'),'/Fe:'+str(host)],cwd=work,env=env)
            binary=package/f'Phoenix.addon{bits}'
            report['dll_load'][str(bits)]=json.loads(run([host,binary,'experimental'],cwd=work,env=env))
            if not report['dll_load'][str(bits)]['passed']:
                raise ValueError('Packaged addon load failed')
            directory=work/f'embedded{bits}'
            report['embedded'][str(bits)]=audit_embedded(binary,directory)
            embedded=json.loads((directory/'1001.bin').read_bytes())
            pins={row['path']:row['sha256'] for row in embedded['files']}
            if pins['native/native_manifest.json'] != sha(bundle/'native_manifest.json'):
                raise ValueError('Embedded manifest differs from tested bundle')
            kernels=manifest['kernels']+[row for group in manifest['kernel_styles'].values() for row in group['kernels']]
            for row in kernels:
                if pins['native/'+row['binary']] != row['binary_sha256']:
                    raise ValueError('Embedded kernel differs from tested bundle')
            report['binaries'][binary.name]=sha(binary)
    report['binaries']['phoenix_graphics_host.exe']=sha(package/'phoenix_graphics_host.exe')
    core=json.loads((package/'core-build.json').read_text())
    if not core['passed'] or not json.loads((package/'report.json').read_text())['passed']:
        raise ValueError('Required build checks failed')
    report.update(cpu_suites=len(core['cpu_tests']),software_d3d12_suites=len(core['software_d3d12_tests']))
    for name in ('production-copy-validation.json','production-arithmetic-validation.json',
                 'native-images-validation.json','style-validation.json'):
        receipt=args.results.parent/name
        if not json.loads(receipt.read_text())['passed']:
            raise ValueError('Required receipt failed: '+name)
        report[name]=dict(passed=True,sha256=sha(receipt))
    report.update(passed=True,temporary_builds_removed=True)
    text=json.dumps(report,indent=2)+'\n'
    safe_payload('VALIDATION.json',text.encode())
    args.output.write_text(text)
    (package/'VALIDATION.json').write_text(text)
    (package/'README.txt').write_text('''Phoenix DLSS5 - local experimental RTX compatibility build (x86/x64)

Choose the addon matching your game's ReShade architecture. Replace the previous
Phoenix addon; do not load multiple copies. External model files are required.
Runners, model-free kernels and the graphics helper are embedded.
Original and Lightweight are supported; Automatic/native remains the default.

Weight decoding, Ada copy waits, vector reads and one compatibility register budget
were tuned. Full-model outputs match the previous corresponding instruction style
exactly in deterministic 1080p, 320x180 and 801x603 reset/history fixtures on RTX 5080.
See VALIDATION.json for paired timing and process-memory results.
Physical RTX 20/30/40 inference remains untested. These are not gameplay results.
Both actual addon DLLs passed load/unload, hook lifetime and embedded hash audits.
No game deployment or publication was performed.
''')
    names=['Phoenix.addon32','Phoenix.addon64','phoenix_graphics_host.exe','README.txt','VALIDATION.json']
    (package/'SHA256SUMS.txt').write_text(''.join(sha(package/name)+'  '+name+'\n' for name in names))
    print(json.dumps(dict(passed=True,binaries=report['binaries'],cpu_suites=report['cpu_suites'],
                          software_suites=report['software_d3d12_suites'])))


if __name__=='__main__':
    main()
