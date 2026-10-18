"""Build a Phoenix experimental RGB control map from an engine's object-ID plane.

The source IDs must come from the renderer/capture producer. This tool does not
infer object identity from final color. Output values remain linear controls.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib

PROFILE = "phoenix-experimental-rgb/1"


def strengths(value):
    if not isinstance(value, dict) or set(value) - {"intensity", "tone", "structure"}:
        raise ValueError("Controls must contain only intensity, tone and structure")
    values = tuple(value.get(key, 1.0) for key in ("intensity", "tone", "structure"))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Experimental map strengths must be finite values in 0..1")
    return tuple(float(v) for v in values) + (1.0,)


def png(width, height, rgb):
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))
    rows = b"".join(b"\0" + rgb[y * width * 3:(y + 1) * width * 3]
                    for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def build(job_path, destination):
    job_path, destination = Path(job_path).resolve(), Path(destination).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("control_profile") != PROFILE:
        raise ValueError("Explicit experimental control_profile required")
    width, height = job["width"], job["height"]
    if any(type(v) is not int or not 1 <= v <= 8192 for v in (width, height)):
        raise ValueError("Dimensions must be integers in 1..8192")
    source = (job_path.parent / job["object_ids"]).resolve()
    if source.stat().st_size != width * height * 4:
        raise ValueError("Object-ID plane must be full-extent packed little-endian uint32")
    source_bytes = source.read_bytes()
    ids = array("I")
    ids.frombytes(source_bytes)
    if ids.itemsize != 4:
        raise RuntimeError("Host uint32 array representation unavailable")
    if sys.byteorder != "little":
        ids.byteswap()
    default = strengths(job.get("default", {}))
    objects = {}
    for key, values in job.get("objects", {}).items():
        object_id = int(key)
        if str(object_id) != key or not 0 <= object_id <= 0xffffffff:
            raise ValueError("Object IDs must be canonical uint32 decimal keys")
        objects[object_id] = strengths(values)
    controls, preview, counts = array("f"), bytearray(), {}
    for object_id in ids:
        values = objects.get(object_id, default)
        controls.extend(values)
        preview.extend(round(v * 255) for v in values[:3])
        counts[str(object_id)] = counts.get(str(object_id), 0) + 1
    if controls.itemsize != 4:
        raise RuntimeError("Host float32 array representation unavailable")
    if sys.byteorder != "little":
        controls.byteswap()
    payload = controls.tobytes()
    # Fresh directory ensures no previous map/capture can be silently replaced.
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "control.rgba32f").write_bytes(payload)
    (destination / "control-preview.png").write_bytes(png(width, height, bytes(preview)))
    report = {
        "schema": "phoenix-control-mask/1", "control_profile": PROFILE,
        "width": width, "height": height, "format": "RGBA32F_LE",
        "control_mask": "control.rgba32f", "preview": "control-preview.png",
        "channel_meanings": ["display intensity", "local tone", "local structure", "reserved=1"],
        "preview_note": "Direct RGB channel values; preview is not the inference input",
        "source_object_ids_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "control_mask_sha256": hashlib.sha256(payload).hexdigest(),
        "default": default[:3], "objects": {str(k): v[:3] for k, v in objects.items()},
        "object_pixel_counts": counts, "vendor_validated": False,
        "identity_producer": "external full-extent object-ID plane",
    }
    (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("new_output_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.job, args.new_output_directory), indent=2))


if __name__ == "__main__":
    main()
