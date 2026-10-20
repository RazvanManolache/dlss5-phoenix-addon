"""Rebuild and rerun the complete local inference matrix in disposable workspaces.

External SDKs, model DLLs and imported model libraries are read-only. Games and
deployed files are never touched. Failed cells stay explicit and block publication.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from compare_four_models import sha, write
from test_model_selection import ROOT


def fresh_build_directory(stack, prefix):
    directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix=prefix, dir=ROOT/'build')))
    if directory.parent != (ROOT/'build').resolve() or directory.is_symlink():
        raise RuntimeError('Temporary build escaped workspace')
    # Existing builders require a nonexistent output. Remove only our own empty
    # directory, then let its TemporaryDirectory context own the recreated tree.
    directory.rmdir()
    return directory


def main():
    from contextlib import ExitStack
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vulkan-deps', type=Path, required=True)
    parser.add_argument('--model-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/current/feature-refresh')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'run.json').exists():
        raise RuntimeError('Use a fresh result folder; never mix two build identities')
    prior = ROOT/'docs/benchmarks/current/measurements.json'
    if prior.is_file():
        data = json.loads(prior.read_text())
        write(output/'previous-summary.json', dict(date=data['date'], source_sha256=sha(prior),
            records=[{key: row[key] for key in ('route','model','extent','profile','gpu','wall','memory') if key in row}
                     for row in data['records']]))
    receipt = dict(passed=False, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   games_launched=False, installed_files_modified=False, steps=[])
    def command(name, arguments, timeout=1800):
        receipt['current_step'] = name
        write(output/'refresh.json', receipt)
        print('START '+name, flush=True)
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, timeout=timeout)
        receipt['steps'].append(dict(name=name, exit_code=result.returncode))
        write(output/'refresh.json', receipt)
        if result.returncode:
            raise RuntimeError(name+' failed; compact receipts retained')
        print('DONE '+name, flush=True)
    try:
        with ExitStack() as stack:
            native = fresh_build_directory(stack, 'benchmark_refresh_')
            vulkan = fresh_build_directory(stack, 'benchmark_vulkan_')
            command('numeric_contracts', ['tools/test_numeric_contracts.py','--gpu','--output',output/'numeric-contracts.json'])
            command('native_kernels', ['tools/production_build_native_kernels.py',native.name,
                '--compiler','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe','--jobs','4'])
            command('vulkan_provider', ['tools/build_vulkan.py',vulkan.name,args.vulkan_deps.resolve()])
            command('runtime_and_probes', ['tools/prepare_readme_benchmark.py','--work',native,
                '--native-bundle',native,'--vulkan-bundle',vulkan,'--output',output])
            # The CPU-only shared-weight lifecycle contract uses the same model
            # DLL but does not allocate GPU objects or run inference.
            from test_model_selection import toolchain, run
            compiler, environment = toolchain(64)
            executable = native/'resident-weights.exe'
            run([compiler,'/nologo','/std:c++20','/Zc:__cplusplus','/O2','/EHsc','/fp:strict','/DNOMINMAX',
                 ROOT/'tests/portable_cpp/resident_weight_tests.cpp',f'/Fo:{native}/resident-weights.obj',
                 f'/Fe:{executable}','bcrypt.lib'], cwd=native, env=environment)
            model = ROOT.parent/'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
            write(output/'resident-weights.json', json.loads(run([executable,model],cwd=native,env=environment)))
            history_results = []
            for bits in (32,64):
                compiler, environment = toolchain(bits)
                executable = native/f'history-validity-{bits}.exe'
                run([compiler,'/nologo','/std:c++20','/Zc:__cplusplus','/O2','/EHsc','/fp:strict','/DNOMINMAX',
                    ROOT/'tests/portable_cpp/history_validity_tests.cpp',
                    f'/Fo:{native}/history-validity-{bits}.obj',f'/Fe:{executable}'],cwd=native,env=environment)
                history_results.append(dict(bits=bits,result=json.loads(run([executable],cwd=native,env=environment))))
            write(output/'history-validity.json',dict(passed=True,results=history_results))
            for stage in ('inventory','smoke','native','portable','features','model_sources'):
                command(stage, ['tools/benchmark_readme.py','--stage',stage,'--output',output,
                    '--model-library',args.model_library.resolve(),'--resume','--keep-going'],timeout=7200)
            run_manifest = json.loads((output/'run.json').read_text())
            failures = [name for name, case in run_manifest['cases'].items() if not case.get('passed')]
            receipt['failed_cases'] = failures
            receipt['case_count'] = len(run_manifest['cases'])
            receipt['passed'] = not failures
        receipt['temporary_builds_removed'] = True
        receipt['current_step'] = None
    except Exception as error:
        receipt['error'] = str(error)
        raise
    finally:
        receipt['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(output/'refresh.json', receipt)
    if not receipt['passed']:
        raise RuntimeError('Benchmark completed with failed cells; do not publish their metrics')
    print('All local matrix cases complete; validate and publish the aggregate separately.',flush=True)


if __name__ == '__main__':
    main()
