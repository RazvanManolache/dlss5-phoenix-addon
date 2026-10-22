"""Rasterize exported real ImGui draw data on CPU, without a GPU or game.

Input comes from shared_panel_tests with PHOENIX_UI_CAPTURE_DIR set. These are
UI fixture previews, not gameplay screenshots or performance measurements.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def render(source, destination):
    metadata = json.loads((source.parent / 'font.json').read_text())
    texture = np.frombuffer((source.parent / 'font.rgba').read_bytes(), dtype=np.uint8)
    texture = texture.reshape(metadata['height'], metadata['width'], 4).astype(np.float32) / 255
    data = json.loads(source.read_text())
    width, height = map(int, data['size'])
    canvas = np.empty((height, width, 3), dtype=np.float32)
    canvas[:] = (.055, .06, .075)
    for draw in data['lists']:
        vertices = np.asarray(draw['vertices'], dtype=np.float64)
        if not len(vertices):
            continue
        packed = vertices[:, 4].astype(np.uint32)
        colors = np.stack([(packed >> shift) & 255 for shift in (0, 8, 16, 24)], axis=1) / 255
        indices = np.asarray(draw['indices'], dtype=np.int32)
        for command in draw['commands']:
            clip = command['clip']
            for offset in range(command['index'], command['index'] + command['count'], 3):
                selected = indices[offset:offset + 3] + command['vertex']
                points = vertices[selected, :2]
                minimum = np.maximum(np.floor(points.min(axis=0)), [max(0, clip[0]), max(0, clip[1])]).astype(int)
                maximum = np.minimum(np.ceil(points.max(axis=0)), [min(width, clip[2]), min(height, clip[3])]).astype(int)
                x0, y0 = minimum
                x1, y1 = maximum
                if x1 <= x0 or y1 <= y0:
                    continue
                a, b, c = points
                denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
                if abs(denominator) < 1e-8:
                    continue
                ys, xs = np.mgrid[y0:y1, x0:x1] + .5
                wa = ((b[1]-c[1])*(xs-c[0]) + (c[0]-b[0])*(ys-c[1])) / denominator
                wb = ((c[1]-a[1])*(xs-c[0]) + (a[0]-c[0])*(ys-c[1])) / denominator
                wc = 1 - wa - wb
                mask = (wa >= 0) & (wb >= 0) & (wc >= 0)
                if not mask.any():
                    continue
                weights = np.stack((wa[mask], wb[mask], wc[mask]), axis=1)
                uv = weights @ vertices[selected, 2:4]
                tx = np.clip(uv[:, 0] * metadata['width'] - .5, 0, metadata['width'] - 1)
                ty = np.clip(uv[:, 1] * metadata['height'] - .5, 0, metadata['height'] - 1)
                ix, iy = tx.astype(int), ty.astype(int)
                jx, jy = np.minimum(ix + 1, metadata['width'] - 1), np.minimum(iy + 1, metadata['height'] - 1)
                fx, fy = (tx-ix)[:, None], (ty-iy)[:, None]
                texels = (texture[iy, ix]*(1-fx) + texture[iy, jx]*fx)*(1-fy) + (texture[jy, ix]*(1-fx) + texture[jy, jx]*fx)*fy
                rgba = texels * (weights @ colors[selected])
                patch = canvas[y0:y1, x0:x1]
                alpha = rgba[:, 3:4]
                patch[mask] = rgba[:, :3] * alpha + patch[mask] * (1-alpha)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(np.clip(canvas, 0, 1) * 255)).save(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    arguments = parser.parse_args()
    render(arguments.source, arguments.destination)
