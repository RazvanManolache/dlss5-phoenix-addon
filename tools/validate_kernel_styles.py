"""Focused x86/x64 kernel-style contracts; no GPU, model or installed files."""
import json
import argparse
import hashlib
import re
import subprocess
from pathlib import Path
import tempfile
from test_model_selection import ROOT, toolchain, run

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, help='Also check real compiled images offline')
    parser.add_argument('--cuobjdump', type=Path)
    args = parser.parse_args()
    if bool(args.bundle) != bool(args.cuobjdump): parser.error('Supply both --bundle and --cuobjdump')
    report = dict(passed=False, gpu_used=False, cases=[])
    output = ROOT/'artifacts/current/kernel-styles-cpu.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report)+'\n')
    with tempfile.TemporaryDirectory(prefix='kernel-style-tests-', dir=ROOT/'build') as temporary:
        for bits in (32,64):
            folder = Path(temporary)/str(bits); folder.mkdir()
            compiler, env = toolchain(bits)
            for name in ('kernel_style_tests', 'feeder_api_tests', 'compatibility_policy_tests'):
                target = folder/(name+'.exe')
                run([compiler, '/nologo', '/std:c++20', '/Zc:__cplusplus', '/EHsc', '/O2', '/MT', '/DNOMINMAX',
                     '/I'+str(ROOT.parent/'external/DLSS/include'),
                     ROOT/'tests/native'/(name+'.cpp'), '/Fo:'+str(folder/(name+'.obj')), '/Fe:'+str(target),
                     'bcrypt.lib','user32.lib','shell32.lib','ole32.lib','cabinet.lib'], cwd=folder, env=env)
                result = json.loads(run([target], cwd=folder, env=env))
                if not result.get('passed'): raise RuntimeError(name+' failed')
                report['cases'].append(dict(bits=bits, suite=name, result=result))
    if args.bundle:
        from build_kernel_styles import STYLE_TARGETS
        root = args.bundle.resolve()
        manifest = json.loads((root/'native_manifest.json').read_text())
        if not manifest.get('passed') or set(manifest.get('kernel_styles', {})) != set(STYLE_TARGETS):
            raise ValueError('Complete compiled style bundle required')
        images = []
        for style, targets in STYLE_TARGETS.items():
            variant = manifest['kernel_styles'][style]
            if variant['architectures'] != targets or len(variant['kernels']) != 41:
                raise ValueError('Incorrect style/target matrix')
            for row in variant['kernels']:
                path = (root/row['binary']).resolve()
                if not path.is_relative_to(root): raise ValueError('Kernel path escaped bundle')
                with path.open('rb') as stream: actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if actual != row['binary_sha256']: raise ValueError('Kernel digest differs')
                raw = path.read_bytes()
                if raw[:4] == b'\x7fELF':
                    found = ['sm_'+str((int.from_bytes(raw[48:52], 'little') >> 8) & 255)]
                else:
                    result = subprocess.run([str(args.cuobjdump.resolve()), '--list-elf', str(path)],
                        capture_output=True, text=True, timeout=60, check=True)
                    found = re.findall(r'\.(sm_\d+)\.cubin', result.stdout)
                if sorted(found) != sorted(targets): raise ValueError('Real binary targets differ from manifest')
                images.append(dict(style=style, kernel=row['name'], architectures=found, sha256=actual))
        report['compiled_images'] = images
        report['physical_images'] = sum(len(row['architectures']) for row in images)
        report['device_inference_validated'] = False
    report['passed'] = True
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))

if __name__ == '__main__': main()
