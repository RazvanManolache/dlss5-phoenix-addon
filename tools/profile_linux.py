#!/usr/bin/env python3
"""Explicit fixed-image inference benchmark. Does not launch a game.

Development dependency: Pillow. Models and input images stay external.
"""
import argparse
import array
import ctypes as C
import hashlib
import json
from pathlib import Path
import statistics
import time


class Plane(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_float)), ("capacity", C.c_uint64),
                ("stride", C.c_uint64), ("width", C.c_uint32),
                ("height", C.c_uint32), ("channels", C.c_uint32), ("reserved", C.c_uint32)]


class Frame(C.Structure):
    _fields_ = [("struct_bytes", C.c_uint32), ("version", C.c_uint32),
                ("control_profile", C.c_uint32), ("reserved", C.c_uint32),
                ("color", Plane), ("motion", Plane), ("depth", Plane),
                ("validity", Plane), ("mask", Plane), ("output", Plane), ("metadata", C.c_char_p)]


class BytesFrame(C.Structure):
    _fields_ = [("struct_bytes", C.c_uint32), ("version", C.c_uint32),
                ("width", C.c_uint32), ("height", C.c_uint32), ("data", C.c_void_p),
                ("capacity", C.c_uint64), ("stride", C.c_uint64),
                ("format", C.c_uint32), ("flags", C.c_uint32)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--window-batch-limit", type=int)
    parser.add_argument("--rgba8", action="store_true", help="use the native CUDA packed byte frontend")
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 2:
        parser.error("Use at least one warmup and two measured samples")
    from PIL import Image
    config = json.loads(args.config.read_text())
    root = args.package.resolve() / "lib"
    config["runtime_root"] = str(root)
    if args.window_batch_limit is not None:
        config["window_batch_limit"] = args.window_batch_limit
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        width, height = image.size
        pixels = image.tobytes()
        rgba = image.convert("RGBA").tobytes()
    source = array.array("f", (value / 255 for value in pixels))
    output = array.array("f", [0]) * len(source)
    library = root / ("libphoenix_native_cuda.so" if config["backend"] == "native-cuda" else "libphoenix_runtime_cpp.so")
    lib = C.CDLL(str(library))
    lib.phoenix_runtime_create.argtypes = [C.c_char_p, C.c_uint, C.c_uint]
    lib.phoenix_runtime_create.restype = C.c_void_p
    lib.phoenix_runtime_frame_v2.argtypes = [C.c_void_p, C.POINTER(Frame)]
    lib.phoenix_runtime_frame_v2.restype = C.c_int
    lib.phoenix_runtime_error.restype = C.c_char_p
    lib.phoenix_runtime_status.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
    lib.phoenix_runtime_status.restype = C.c_size_t
    lib.phoenix_runtime_destroy.argtypes = [C.c_void_p]
    frame = Frame()
    frame.struct_bytes, frame.version = C.sizeof(Frame), 2
    frame.color = Plane((C.c_float * len(source)).from_buffer(source), len(source)*4,
                        0, width, height, 3, 0)
    frame.output = Plane((C.c_float * len(output)).from_buffer(output), len(output)*4,
                         0, width, height, 3, 0)
    frame.metadata = b'{"reset":true}'
    byte_data = bytearray(rgba)
    byte_view = (C.c_uint8 * len(byte_data)).from_buffer(byte_data)
    byte_frame = BytesFrame(C.sizeof(BytesFrame), 1, width, height, C.addressof(byte_view), len(byte_data), 0, 0, 1)
    if args.rgba8:
        if config["backend"] != "native-cuda":
            parser.error("--rgba8 requires native-cuda")
        lib.phoenix_linux_frame_rgba8.argtypes = [C.c_void_p, C.POINTER(BytesFrame)]
        lib.phoenix_linux_frame_rgba8.restype = C.c_int
    started = time.perf_counter()
    session = lib.phoenix_runtime_create(json.dumps(config).encode(), width, height)
    if not session:
        raise RuntimeError(lib.phoenix_runtime_error().decode())
    create_ms = (time.perf_counter() - started)*1000
    samples, hashes = [], set()
    try:
        for index in range(args.warmup + args.samples):
            if args.rgba8:
                byte_data[:] = rgba
            started = time.perf_counter()
            result = (lib.phoenix_linux_frame_rgba8(session, C.byref(byte_frame)) if args.rgba8
                      else lib.phoenix_runtime_frame_v2(session, C.byref(frame)))
            call_ms = (time.perf_counter() - started)*1000
            if result:
                raise RuntimeError(lib.phoenix_runtime_error().decode())
            # JSON status and output hashing deliberately stay outside call timing.
            buffer = C.create_string_buffer(lib.phoenix_runtime_status(session, None, 0))
            lib.phoenix_runtime_status(session, buffer, len(buffer))
            status = json.loads(buffer.value)
            hashes.add(hashlib.sha256(byte_data if args.rgba8 else output.tobytes()).hexdigest())
            if index >= args.warmup:
                samples.append({"call_ms": call_ms, **{key: status[key] for key in
                    ("prepare_ms", "execute_with_transfers_ms", "operator_ms", "resolve_ms",
                     "validation_ms", "worker_ms") if key in status}})
    finally:
        lib.phoenix_runtime_destroy(session)
    receipt = {
        "schema": "phoenix-linux-inference-profile/1", "width": width, "height": height,
        "input_rgb8_sha256": hashlib.sha256(pixels).hexdigest(),
        "runtime_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "backend": status.get("backend"), "device": status.get("device"),
        "model": status.get("model"), "model_resource_sha256": status.get("model_resource_sha256"),
        "neural_launches": status.get("neural_launches"),
        "window_batch_limit": status.get("window_batch_limit"),
        "command_count": status.get("command_count"),
        "warmup": args.warmup, "samples": samples, "create_ms": create_ms,
        "median_ms": {key: statistics.median(row[key] for row in samples) for key in samples[0]},
        "output_sha256": sorted(hashes), "output_format": "RGBA8" if args.rgba8 else "RGB32F", "reset_repeatable": len(hashes) == 1,
        "scope": "Synchronous host-plane inference; excludes game capture/presentation. GPU clocks unmanaged.",
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["median_ms"], indent=2))


if __name__ == "__main__":
    main()
