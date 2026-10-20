"""Check compatibility arithmetic; GPU execution requires explicit --gpu."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from test_model_selection import ROOT, toolchain, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='Execute FP8/FP16 comparison and fault detection on Ada or newer')
    parser.add_argument('--gpu-architecture', choices=('sm_89', 'sm_120'), default='sm_120')
    parser.add_argument('--cuda-bin', type=Path, default=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _, env = toolchain(64)
    compiler = (args.cuda_bin / 'nvcc.exe').resolve()
    flags = [compiler, '-allow-unsupported-compiler', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', '-O2', '-std=c++17']
    sources = [ROOT / 'src/native/kernels/production/common/architecture_math.cuh',
               ROOT / 'tests/native/kernel_architecture_host_tests.cu',
               ROOT / 'tests/native/kernel_architecture_device_tests.cu']
    pins = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    report = dict(passed=False, gpu_used=args.gpu, older_device_validated=False, source_hashes=pins, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='architecture-arithmetic-', dir=ROOT / 'build') as temporary:
            work = Path(temporary)
            executable = work / 'cpu.exe'
            run(flags + ['-arch=sm_75', sources[1], '-o', executable], cwd=work, env=env)
            report['cpu_conversions'] = json.loads(run([executable], cwd=work, env=env))
            if args.gpu:
                for turing in (False, True):
                    executable = work / 'gpu.exe'
                    defines = ['-DPHOENIX_TEST_TURING_MMA'] if turing else []
                    run(flags + ['-arch=' + args.gpu_architecture, *defines, sources[2], '-o', executable], cwd=work, env=env)
                    report['cases'].append(json.loads(run([executable], cwd=work, env=env)))
                run(flags + ['-arch=' + args.gpu_architecture, '-DPHOENIX_TEST_CORRUPT_OPERAND',
                             sources[2], '-o', executable], cwd=work, env=env)
                fault = subprocess.run([str(executable)], cwd=work, env=env, text=True, capture_output=True, timeout=60)
                if fault.returncode != 1 or 'fragment order differs' not in fault.stderr:
                    raise RuntimeError('Deliberately corrupted operand was not detected')
                report['corrupt_operand_mutation_detected'] = True
            if any(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != pin for path, pin in pins.items()):
                raise RuntimeError('Arithmetic sources changed during validation')
        report.update(passed=True, temporary_builds_removed=True)
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
