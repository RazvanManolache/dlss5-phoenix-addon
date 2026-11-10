#!/usr/bin/env python3
"""Explicit model/GPU validation of native CUDA's bounded reset-RGB contract.

Not registered with CTest. Requires the complete package, external model and Pillow.
"""
import argparse
import array
import ctypes as C
import hashlib
import importlib.util
import json
import math
import sys
import subprocess
import tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("profile_linux", ROOT / "tools/profile_linux.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check-context", action="store_true", help="verify an existing non-primary CUDA context is preserved")
    parser.add_argument("--check-cli", action="store_true", help="also validate the packaged standalone frame client")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.package.resolve()
    if (root / "lib").is_dir(): root = root / "lib"
    config["runtime_root"] = str(root)
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        width, height = image.size
        source = array.array("f", (value / 255 for value in image.tobytes()))
    original = source.tobytes()
    output = array.array("f", [42])*len(source)
    library = root / ("phoenix_native_cuda.dll" if sys.platform == "win32" else "libphoenix_native_cuda.so")
    driver = None
    saved_context = C.c_void_p()
    if args.check_context:
        driver = C.WinDLL("nvcuda.dll") if sys.platform == "win32" else C.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [C.c_uint]
        driver.cuCtxCreate_v2.argtypes = [C.POINTER(C.c_void_p), C.c_uint, C.c_int]
        driver.cuCtxGetCurrent.argtypes = [C.POINTER(C.c_void_p)]
        driver.cuCtxDestroy_v2.argtypes = [C.c_void_p]
        driver.cuCtxSetCurrent.argtypes = [C.c_void_p]
        require(driver.cuInit(0) == 0, "CUDA driver initialization failed")
        require(driver.cuCtxCreate_v2(C.byref(saved_context), 0, 0) == 0, "Cannot create caller context")
    def context_preserved():
        if driver:
            current = C.c_void_p()
            require(driver.cuCtxGetCurrent(C.byref(current)) == 0 and current.value == saved_context.value,
                    "Native API changed the caller CUDA context")
    lib = C.CDLL(str(library))
    lib.phoenix_runtime_create.argtypes = [C.c_char_p, C.c_uint, C.c_uint]
    lib.phoenix_runtime_create.restype = C.c_void_p
    lib.phoenix_runtime_frame_v2.argtypes = [C.c_void_p, C.POINTER(profile.Frame)]
    lib.phoenix_runtime_frame_v2.restype = C.c_int
    lib.phoenix_runtime_error.restype = C.c_char_p
    lib.phoenix_runtime_destroy.argtypes = [C.c_void_p]
    frame = profile.Frame()
    frame.struct_bytes, frame.version = C.sizeof(frame), 2
    frame.metadata = b'{"reset":true}'
    input_plane = profile.Plane((C.c_float*len(source)).from_buffer(source), len(source)*4,
                                0, width, height, 3, 0)
    output_plane = profile.Plane((C.c_float*len(output)).from_buffer(output), len(output)*4,
                                 0, width, height, 3, 0)
    frame.color, frame.output = input_plane, output_plane
    checks = []
    invalid_config = dict(config, native_bundle=str(root/'missing-contract-bundle'))
    require(not lib.phoenix_runtime_create(json.dumps(invalid_config).encode(), width, height),
            "Missing bundle accepted during construction")
    context_preserved()
    checks.append("failed construction preserves caller context")
    session = lib.phoenix_runtime_create(json.dumps(config).encode(), width, height)
    require(session, lib.phoenix_runtime_error().decode())
    context_preserved()
    try:
        def run():
            require(lib.phoenix_runtime_frame_v2(session, C.byref(frame)) == 0,
                    lib.phoenix_runtime_error().decode())
            context_preserved()
        def rejected(message):
            before = output.tobytes()
            require(lib.phoenix_runtime_frame_v2(session, C.byref(frame)) == -1, message)
            context_preserved()
            require(output.tobytes() == before, "Rejected input changed output")
            checks.append(message)
        run()
        expected = output.tobytes()
        require(expected != original, "Native inference retained the input")
        run()
        require(output.tobytes() == expected, "Native reset output is not repeatable")
        checks += ["changed output", "exact repeated reset"]
        if driver:
            original_context=saved_context.value
            alternate=C.c_void_p()
            require(driver.cuCtxCreate_v2(C.byref(alternate),0,0)==0,'Cannot create alternate caller context')
            try:
                saved_context.value=alternate.value
                run()
                require(output.tobytes()==expected,'Caller context switch changed native output')
                checks.append('same-device caller context switch preserves inference and caller identity')
            finally:
                saved_context.value=original_context
                require(driver.cuCtxSetCurrent(saved_context)==0,'Cannot restore caller context')
                require(driver.cuCtxDestroy_v2(alternate)==0,'Cannot retire alternate caller context')
        frame.output.capacity -= 4
        rejected("undersized output rejected without writes")
        frame.output = output_plane
        frame.color.capacity -= 4
        rejected("undersized input rejected without writes")
        frame.color = input_plane
        frame.motion.capacity = 4
        rejected("undeclared optional plane rejected")
        frame.motion = profile.Plane()
        for metadata in (b'{"reset":true,"unknown":1}', b'{"reset":true,"passes":0}', b'{"reset":true,"passes":11}'):
            frame.metadata = metadata
            rejected("unsupported metadata rejected: " + metadata.decode())
        frame.metadata = b'{"reset":true}'
        source[0] = float("nan")
        rejected("nonfinite input rejected without writes")
        C.memmove(C.addressof((C.c_float*len(source)).from_buffer(source)), original, len(original))
        run()
        require(output.tobytes() == expected, "Validation failure damaged session")
        checks.append("valid frame succeeds after validation errors")
        frame.output = input_plane
        run()
        require(source.tobytes() == expected, "Aliased input/output corrupted inference")
        checks.append("aliased input/output exact")
        # Full row padding remains untouched, including the last row's tail.
        stride = width*3+7
        padded = array.array("f", [42])*(stride*height)
        frame.color = output_plane
        C.memmove(C.addressof((C.c_float*len(output)).from_buffer(output)), original, len(original))
        frame.output = profile.Plane((C.c_float*len(padded)).from_buffer(padded), len(padded)*4,
                                     stride*4, width, height, 3, 0)
        run()
        expected_values = array.array("f")
        expected_values.frombytes(expected)
        for row in range(height):
            require(padded[row*stride:row*stride+width*3] == expected_values[row*width*3:(row+1)*width*3],
                    "Strided output differs")
            require(all(value == 42 for value in padded[row*stride+width*3:(row+1)*stride]),
                    "Strided output overwrote padding")
        checks.append("strided output exact with padding preserved")
        class BytesFrame(C.Structure):
            _fields_ = [("struct_bytes",C.c_uint32),("version",C.c_uint32),
                        ("width",C.c_uint32),("height",C.c_uint32),("data",C.c_void_p),
                        ("capacity",C.c_uint64),("stride",C.c_uint64),
                        ("format",C.c_uint32),("flags",C.c_uint32)]
        packed_frame = lib.phoenix_runtime_frame_rgba8
        packed_frame.argtypes = [C.c_void_p,C.POINTER(BytesFrame)]
        packed_frame.restype = C.c_int
        original_values = array.array("f")
        original_values.frombytes(original)
        for bgra in (False,True):
            data = bytearray(width*height*4)
            expected_bytes = bytearray(len(data))
            for pixel in range(width*height):
                data[pixel*4+3] = expected_bytes[pixel*4+3] = (pixel*37)%256
                for channel in range(3):
                    slot = 2-channel if bgra and channel != 1 else channel
                    data[pixel*4+slot] = round(original_values[pixel*3+channel]*255)
                    expected_bytes[pixel*4+slot] = math.floor(max(0,min(1,expected_values[pixel*3+channel]))*255+0.5)
            view = (C.c_uint8*len(data)).from_buffer(data)
            packed = BytesFrame(C.sizeof(BytesFrame),1,width,height,C.addressof(view),len(data),0,int(bgra),1)
            require(packed_frame(session,C.byref(packed)) == 0,
                    lib.phoenix_runtime_error().decode())
            context_preserved()
            require(data == expected_bytes,"GPU byte transport differs from V2 RGB or changed alpha")
            checks.append("GPU " + ("BGRA8" if bgra else "RGBA8") + " exact against V2 with alpha preserved")
            packed.capacity -= 1
            before = bytes(data)
            require(packed_frame(session,C.byref(packed)) == -1 and data == before,
                    "RGBA8 capacity failure changed caller memory")
            context_preserved()
            checks.append("RGBA8 bounded capacity rejection")
            packed.capacity += 1
            for field, value in (("version",2),("flags",0),("format",2),("width",width+1),("stride",width*4-1)):
                previous = getattr(packed,field)
                setattr(packed,field,value)
                require(packed_frame(session,C.byref(packed)) == -1 and data == before,
                        "Invalid packed header changed caller memory: "+field)
                context_preserved()
                setattr(packed,field,previous)
            checks.append("invalid packed headers rejected without writes")
            row = width*4
            padded_bytes = bytearray([173])*((row+13)*height)
            for y in range(height):
                padded_bytes[y*(row+13):y*(row+13)+row] = data[y*row:(y+1)*row]
            # A processed image is a new input here; only transport/padding is asserted.
            padded_view = (C.c_uint8*len(padded_bytes)).from_buffer(padded_bytes)
            packed.data,packed.capacity,packed.stride = C.addressof(padded_view),len(padded_bytes),row+13
            require(packed_frame(session,C.byref(packed)) == 0,lib.phoenix_runtime_error().decode())
            context_preserved()
            require(all(padded_bytes[y*(row+13)+row:(y+1)*(row+13)]==bytes([173])*13 for y in range(height)),
                    "Packed frame changed row padding")
            checks.append("packed row padding preserved")

    finally:
        lib.phoenix_runtime_destroy(session)
        context_preserved()
        if driver:
            require(driver.cuCtxDestroy_v2(saved_context) == 0, "Caller context cleanup failed")
    if args.check_cli:
        client = root/('phoenix_frame.exe' if sys.platform=='win32' else 'phoenix_frame')
        if not client.is_file(): client = args.package.resolve()/'bin/phoenix_frame'
        with tempfile.TemporaryDirectory(prefix='phoenix-native-cli-') as directory:
            temporary = Path(directory)
            (temporary/'input.rgb32f').write_bytes(original)
            job = {'config':config,'width':width,'height':height,'color':str(temporary/'input.rgb32f')}
            job_path = temporary/'job.json'
            job_path.write_text(json.dumps(job))
            result = subprocess.run([str(client),str(job_path),str(temporary/'output.rgb32f')],capture_output=True,text=True)
            require(result.returncode==0,result.stderr)
            require((temporary/'output.rgb32f').read_bytes()==expected,'Standalone client differs from bounded V2')
            checks.append('standalone client exact against V2')
            job['ui_alpha'] = str(temporary/'alpha.r32f')
            (temporary/'alpha.r32f').write_bytes((array.array('f',[2])*width*height).tobytes())
            job_path.write_text(json.dumps(job))
            result = subprocess.run([str(client),str(job_path),str(temporary/'rejected.rgb32f')],capture_output=True,text=True)
            require(result.returncode!=0 and not (temporary/'rejected.rgb32f').exists(),'Standalone client accepted invalid V3 coverage')
            # Coverage outside 0..1 must still fail before creating caller output.
            checks.append('standalone client rejects invalid V3 planes without output')
    args.report.write_text(json.dumps({"schema": "phoenix-native-cuda-contracts/1",
        "result": "passed", "width": width, "height": height, "checks": checks,
        "runtime_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "caller_context_preserved": args.check_context,
        "output_sha256": hashlib.sha256(expected).hexdigest()}, indent=2) + "\n")
    print(f"Native CUDA: {len(checks)} contract checks passed")


if __name__ == "__main__":
    main()
