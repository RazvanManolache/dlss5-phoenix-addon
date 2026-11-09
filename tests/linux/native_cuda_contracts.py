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
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config["runtime_root"] = str(args.package.resolve() / "lib")
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        width, height = image.size
        source = array.array("f", (value / 255 for value in image.tobytes()))
    original = source.tobytes()
    output = array.array("f", [42])*len(source)
    library = args.package.resolve() / "lib/libphoenix_native_cuda.so"
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
    session = lib.phoenix_runtime_create(json.dumps(config).encode(), width, height)
    require(session, lib.phoenix_runtime_error().decode())
    checks = []
    try:
        def run():
            require(lib.phoenix_runtime_frame_v2(session, C.byref(frame)) == 0,
                    lib.phoenix_runtime_error().decode())
        def rejected(message):
            before = output.tobytes()
            require(lib.phoenix_runtime_frame_v2(session, C.byref(frame)) == -1, message)
            require(output.tobytes() == before, "Rejected input changed output")
            checks.append(message)
        run()
        expected = output.tobytes()
        require(expected != original, "Native inference retained the input")
        run()
        require(output.tobytes() == expected, "Native reset output is not repeatable")
        checks += ["changed output", "exact repeated reset"]
        frame.output.capacity -= 4
        rejected("undersized output rejected without writes")
        frame.output = output_plane
        frame.color.capacity -= 4
        rejected("undersized input rejected without writes")
        frame.color = input_plane
        frame.motion.capacity = 4
        rejected("undeclared optional plane rejected")
        frame.motion = profile.Plane()
        for metadata in (b'{}', b'{"reset":false}', b'{"reset":true,"passes":2}'):
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
        lib.phoenix_linux_frame_rgba8.argtypes = [C.c_void_p,C.POINTER(BytesFrame)]
        lib.phoenix_linux_frame_rgba8.restype = C.c_int
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
            require(lib.phoenix_linux_frame_rgba8(session,C.byref(packed)) == 0,
                    lib.phoenix_runtime_error().decode())
            require(data == expected_bytes,"GPU byte transport differs from V2 RGB or changed alpha")
            checks.append("GPU " + ("BGRA8" if bgra else "RGBA8") + " exact against V2 with alpha preserved")
            packed.capacity -= 1
            before = bytes(data)
            require(lib.phoenix_linux_frame_rgba8(session,C.byref(packed)) == -1 and data == before,
                    "RGBA8 capacity failure changed caller memory")
            checks.append("RGBA8 bounded capacity rejection")

    finally:
        lib.phoenix_runtime_destroy(session)
    args.report.write_text(json.dumps({"schema": "phoenix-native-cuda-contracts/1",
        "result": "passed", "width": width, "height": height, "checks": checks,
        "runtime_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(expected).hexdigest()}, indent=2) + "\n")
    print(f"Native CUDA: {len(checks)} contract checks passed")


if __name__ == "__main__":
    main()
