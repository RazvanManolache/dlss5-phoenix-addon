"""Build the two optional Windows x64 motion-provider DLLs from pinned SDK builds."""
import argparse
from pathlib import Path

from test_model_selection import ROOT, run, toolchain


def build(output, fidelityfx_sdk, opencv_source, opencv_build):
    output.mkdir(parents=True, exist_ok=True)
    compiler, environment = toolchain(64)
    common = [compiler, '/nologo', '/LD', '/O2', '/EHsc', '/std:c++20', '/DNOMINMAX']
    source = ROOT / 'src/pipeline/motion'
    ffx_lib = fidelityfx_sdk / 'bin/ffx_sdk'
    run([*common, '/MD', '/I' + str(fidelityfx_sdk / 'include'), source / 'fidelityfx.cpp',
         '/Fe:phoenix_motion_fidelityfx.dll', '/link', ffx_lib / 'ffx_opticalflow_x64.lib',
         ffx_lib / 'ffx_backend_dx12_x64.lib', 'd3d12.lib', 'dxgi.lib', 'dxguid.lib', 'd3dcompiler.lib'],
        cwd=output, env=environment)
    modules = ('core', 'imgproc', 'video')
    libraries = [next((opencv_build / 'lib/Release').glob(f'opencv_{module}[0-9]*.lib'))
                 for module in modules]
    run([*common, '/MT', '/I' + str(opencv_build),
         *['/I' + str(opencv_source / 'modules' / module / 'include') for module in modules],
         source / 'dis.cpp', '/Fe:phoenix_motion_dis.dll', '/link', *libraries,
         *sorted((opencv_build / '3rdparty/lib/Release').glob('*.lib'))],
        cwd=output, env=environment)
    for sdk, name in [(fidelityfx_sdk.parent, 'FidelityFX'), (opencv_source, 'OpenCV')]:
        license_path = next(sdk / item for item in ('LICENSE', 'LICENSE.txt') if (sdk / item).is_file())
        (output / f'{name}-LICENSE.txt').write_bytes(license_path.read_bytes())
    # Static OpenCV dependencies built with its SDK ship with these notices.
    third_party = opencv_source / '3rdparty'
    for name, candidate in [('OpenCV-zlib', third_party / 'zlib/README'),
                            ('OpenCV-SoftFloat', opencv_source / 'modules/core/src/softfloat.cpp')]:
        if candidate.is_file():
            (output / f'{name}-NOTICE.txt').write_bytes(
                candidate.read_bytes().split(b'#include', 1)[0] if candidate.suffix == '.cpp'
                else candidate.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'fidelityfx-sdk', 'opencv-source', 'opencv-build'):
        parser.add_argument('--' + name, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    build(args.output, args.fidelityfx_sdk, args.opencv_source, args.opencv_build)


if __name__ == '__main__':
    main()
