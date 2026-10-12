"""Generate normalized FP16 color and zero motion for the offline native probe."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(prefix, width, height):
    import numpy as np

    if not (1 <= width <= 16384 and 1 <= height <= 16384):
        raise ValueError("Native dimensions must be between 1 and 16384")
    prefix = prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    files = [Path(str(prefix) + suffix) for suffix in (".proxy.bin", ".motion.bin", ".meta")]
    if any(path.exists() for path in files):
        raise ValueError("Fresh capture prefix required")
    # Simple binary-exact fractions avoid RNG and platform-dependent math.
    rows, columns = np.indices((height, width), dtype=np.uint32)
    color = np.ones((height, width, 4), dtype="<f2")
    color[..., 0] = (columns % 64) / 128 + 0.125
    color[..., 1] = (rows % 64) / 128 + 0.125
    color[..., 2] = ((rows + columns) % 32) / 128 + 0.25
    color.tofile(files[0])
    np.zeros((height, width, 2), dtype="<f2").tofile(files[1])
    metadata = dict(
        width=width,
        height=height,
        color_format=10,
        motion_format=34,
        color_bpp=8,
        motion_bpp=4,
        frame=0,
        blocks=8,
        refinement_passes=1,
        reset=1,
        enabled=1,
        automask=1,
        style=0,
        inverted=0,
        uicorrection=0,
        srgb=0,
        unjitter=1,
        mode=3,
        reference_feed=0,
        jitter_x=0,
        jitter_y=0,
        motion_x=1,
        motion_y=1,
        intensity=1,
        tone=1,
        structure=1,
        skin=-1,
    )
    metadata.update({"block_repeat_" + str(index): 1 for index in range(1, 9)})
    files[2].write_text(
        "".join(f"{key}={value}\n" for key, value in metadata.items()),
        encoding="utf-8",
        newline="\n",
    )
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prefix, args.width, args.height), indent=2))


if __name__ == "__main__":
    main()
