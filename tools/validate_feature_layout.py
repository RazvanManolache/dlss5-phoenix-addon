"""Compare native layout kernels with a preserved bundle, without model weights.

Example: python tools/validate_feature_layout.py --baseline C:/backup/bundle
         --candidate build/native_release --output build/layout_validation.json
Requires Windows, an NVIDIA CUDA driver and NumPy. Neither bundle is modified.
"""

import argparse
import ctypes as ctypes
import json
from pathlib import Path
import struct

import numpy as np


class CudaDriver:
    def __init__(self):
        self.api = ctypes.WinDLL("nvcuda.dll")
        self.call("cuInit", 0)
        device = ctypes.c_int()
        self.call("cuDeviceGet", ctypes.byref(device), 0)
        self.context = ctypes.c_void_p()
        self.call("cuCtxCreate_v2", ctypes.byref(self.context), 0, device)

    def call(self, name, *arguments):
        status = getattr(self.api, name)(*arguments)
        if status:
            raise RuntimeError(f"{name}: CUDA status {status}")

    def close(self):
        self.call("cuCtxDestroy_v2", self.context)

    def execute(self, bundle, kernel, height, width, source):
        guard_words = 256
        sentinel = np.uint32(0xDEADBEEF)
        destination = np.full(source.size + 2 * guard_words, sentinel, dtype=np.uint32)
        guarded_source = np.full_like(destination, 0xA5A5A5A5)
        guarded_source[guard_words:-guard_words] = source
        allocations = []
        module = ctypes.c_void_p()
        try:
            for array in (guarded_source, destination):
                pointer = ctypes.c_uint64()
                self.call("cuMemAlloc_v2", ctypes.byref(pointer), ctypes.c_size_t(array.nbytes))
                allocations.append(pointer)
                self.call(
                    "cuMemcpyHtoD_v2",
                    pointer,
                    ctypes.c_void_p(array.ctypes.data),
                    ctypes.c_size_t(array.nbytes),
                )
            self.call("cuModuleLoad", ctypes.byref(module), str(bundle / kernel["binary"]).encode())
            function = ctypes.c_void_p()
            self.call(
                "cuModuleGetFunction", ctypes.byref(function), module, kernel["entry"].encode()
            )
            packet = ctypes.create_string_buffer(
                struct.pack(
                    "<QQII",
                    allocations[0].value + guard_words * 4,
                    allocations[1].value + guard_words * 4,
                    height,
                    width,
                )
            )
            arguments = (ctypes.c_void_p * 1)(ctypes.addressof(packet))
            self.call(
                "cuLaunchKernel",
                function,
                height * width,
                1,
                1,
                256,
                1,
                1,
                0,
                ctypes.c_void_p(),
                arguments,
                ctypes.c_void_p(),
            )
            self.call("cuCtxSynchronize")
            self.call(
                "cuMemcpyDtoH_v2",
                ctypes.c_void_p(destination.ctypes.data),
                allocations[1],
                ctypes.c_size_t(destination.nbytes),
            )
            self.call(
                "cuMemcpyDtoH_v2",
                ctypes.c_void_p(guarded_source.ctypes.data),
                allocations[0],
                ctypes.c_size_t(guarded_source.nbytes),
            )
            if not (
                np.all(destination[:guard_words] == sentinel)
                and np.all(destination[-guard_words:] == sentinel)
            ):
                raise ValueError("Output guard overwritten")
            if not (
                np.array_equal(guarded_source[guard_words:-guard_words], source)
                and np.all(guarded_source[:guard_words] == 0xA5A5A5A5)
                and np.all(guarded_source[-guard_words:] == 0xA5A5A5A5)
            ):
                raise ValueError("Input buffer modified")
            return destination[guard_words:-guard_words].copy()
        finally:
            if module.value:
                self.call("cuModuleUnload", module)
            for pointer in allocations:
                self.call("cuMemFree_v2", pointer)


def layout_kernels(bundle):
    manifest = json.loads((bundle / "native_manifest.json").read_text())
    result = {}
    for kind in ("image_to_tokens", "tokens_to_image"):
        matches = [
            kernel
            for kernel in manifest["kernels"]
            if kernel["binary"] == f"optimized/layout/{kind}/kernel.cubin"
        ]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous layout kernel: " + kind)
        result[kind] = matches[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, candidate = args.baseline.resolve(), args.candidate.resolve()
    old_kernels, new_kernels = layout_kernels(baseline), layout_kernels(candidate)
    rng = np.random.default_rng(20261011)
    cases = []
    driver = CudaDriver()
    try:
        for height, width in (
            (4, 4),
            (4, 8),
            (8, 4),
            (8, 8),
            (8, 12),
            (12, 8),
            (12, 20),
            (20, 12),
            (20, 32),
            (20, 40),
            (4, 132),
        ):
            count = ((height * width + 31) // 32) * 32 * 256
            for pattern in ("indices", "random_bits"):
                source = (
                    np.arange(1, count + 1, dtype=np.uint32)
                    if pattern == "indices"
                    else rng.integers(0, 2**32, count, dtype=np.uint32)
                )
                for kind in ("image_to_tokens", "tokens_to_image"):
                    old = driver.execute(baseline, old_kernels[kind], height, width, source)
                    new = driver.execute(candidate, new_kernels[kind], height, width, source)
                    if not np.array_equal(old, new):
                        raise ValueError(f"{kind} {height}x{width} {pattern}: output mismatch")
                    cases.append(
                        dict(
                            kind=kind,
                            height=height,
                            width=width,
                            pattern=pattern,
                            exact=True,
                            guards_intact=True,
                            input_unchanged=True,
                        )
                    )
    finally:
        driver.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(passed=True, baseline=str(baseline), candidate=str(candidate), cases=cases),
            indent=2,
        )
        + "\n"
    )
    print(f"Passed {len(cases)} exact GPU layout comparisons, including padding and buffer guards")


if __name__ == "__main__":
    main()
