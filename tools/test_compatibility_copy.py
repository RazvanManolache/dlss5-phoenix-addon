"""Compile compatibility copy/decoder tests; execute GPU cases only with --gpu."""
import argparse
import json
from pathlib import Path
import tempfile

from test_model_selection import ROOT, toolchain, run
from compare_four_models import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--compiler', type=Path,
                        default=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    _, env = toolchain(64)
    sources = [ROOT/'tests/native/compatibility_copy_tests.cu',
               ROOT/'src/native/kernels/production/common/compatibility_copy.cuh',
               ROOT/'src/native/kernels/production/common/pipeline_memory.cuh',
               ROOT/'src/native/kernels/production/common/architecture_math.cuh']
    report = dict(passed=False, gpu_used=args.gpu, older_device_validated=False,
                  source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, report)
    try:
        with tempfile.TemporaryDirectory(prefix='compatibility-copy-', dir=ROOT/'build') as temporary:
            work = Path(temporary)
            for style in (750, 860, 890):
                executable = work/'test.exe'
                run([args.compiler.resolve(), '-allow-unsupported-compiler',
                     '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH', '-arch=sm_120', '-O3', '-std=c++17',
                     f'-DPHOENIX_KERNEL_STYLE={style}', sources[0], '-o', executable], cwd=work, env=env)
                case = dict(style=style, compiled=True, gpu_used=False)
                if args.gpu:
                    case.update(result=json.loads(run([executable], cwd=work, env=env)), gpu_used=True)
                report['cases'].append(case)
                write(args.output, report)
            if any(sha(ROOT/path) != pin for path,pin in report['source_hashes'].items()):
                raise RuntimeError('Test sources changed during validation')
        report.update(passed=True, temporary_builds_removed=True)
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        write(args.output, report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
