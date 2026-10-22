"""One-time, non-executing conversion of the pinned Apache-2.0 tiny checkpoint.

No PyTorch dependency. This build/import helper is not part of the deployed runtime.
Only the exact pinned archive and tensor shapes are accepted; pickle globals cannot execute.
"""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
import pickle
from pathlib import Path
import struct
import zipfile

SOURCE_SHA256 = '5c7ee11d86b685196af2ccb2f2a539e43f17f865448fe900dc61d416465bc79b'


def tensor(storage, offset, shape, stride, *_):
    return storage, offset, tuple(shape), tuple(stride)


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('collections', 'OrderedDict'): OrderedDict,
                   ('torch', 'FloatStorage'): 'float32',
                   ('torch._utils', '_rebuild_tensor_v2'): tensor}
        if (module, name) not in allowed: raise ValueError('Checkpoint global refused')
        return allowed[module, name]

    def persistent_load(self, value):
        if len(value) != 5 or value[:2] != ('storage', 'float32') or value[3] not in ('cpu', 'cuda:0'):
            raise ValueError('Checkpoint storage refused')
        return value[2], value[4]


def layout():
    items = [('skip_scale', (1, 3, 1, 1)), ('skip_bias', (1, 3, 1, 1)), ('res_gain', (1,))]
    layers = [('head', 3, 28)] + [(f'blocks.{b}.conv{c}', 28, 28) for b in range(4) for c in (1, 2)] + [('trunk', 28, 28)]
    for name, inputs, outputs in layers:
        items += [(name + '.rbr.weight', (outputs, inputs, 3, 3)), (name + '.rbr.bias', (outputs,)), (name + '.act.weight', (outputs,))]
    return items + [('up.0.weight', (112, 28, 3, 3)), ('up.0.bias', (112,)), ('up.2.weight', (28,)),
                    ('up.3.weight', (3, 28, 3, 3)), ('up.3.bias', (3,))]


def convert(source):
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256: raise ValueError('Unrecognized tiny checkpoint')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        meta_name, = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        prefix = meta_name[:-len('data.pkl')]
        state = MetadataReader(io.BytesIO(archive.read(meta_name))).load()
        if set(state) != {name for name, _ in layout()}: raise ValueError('Tensor set mismatch')
        result = bytearray(); manifest = []
        for name, expected in layout():
            (key, storage_count), offset, shape, stride = state[name]
            count = 1
            for extent in reversed(shape): count *= extent
            expected_stride = []; contiguous = 1
            for extent in reversed(shape): expected_stride.insert(0, contiguous); contiguous *= extent
            if shape != expected or tuple(expected_stride) != stride or offset < 0 or offset + count > storage_count:
                raise ValueError('Tensor layout mismatch: ' + name)
            raw = archive.read(prefix + 'data/' + key)
            if len(raw) != storage_count * 4: raise ValueError('Storage size mismatch')
            raw = raw[offset * 4:(offset + count) * 4]
            if any(not (-1e6 < value < 1e6) for value, in struct.iter_unpack('<f', raw)):
                raise ValueError('Invalid trained coefficient')
            manifest.append({'name': name, 'offset': len(result) // 4, 'count': count, 'shape': shape,
                             'storage': key, 'storage_offset': offset, 'storage_count': storage_count})
            result.extend(raw)
    return bytes(result), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); weights, tensors = convert(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    # Conversion is deterministic and idempotent. Refuse replacement by different bytes.
    target = args.output / 'weights.bin'
    if target.exists() and target.read_bytes() != weights: raise ValueError('Different weights already exist')
    target.write_bytes(weights)
    description = {'schema': 'phoenix-tiny-2x/1', 'source_sha256': SOURCE_SHA256,
        'weights_sha256': hashlib.sha256(weights).hexdigest(), 'bytes': len(weights), 'tensors': tensors,
        'upstream': 'https://github.com/Benjamin-Wegener/dlss5-distill',
        'revision': '9e3bc2f9e755f0e281b72e439f9452be0e8b7f31', 'license': 'Apache-2.0'}
    (args.output / 'model.json').write_text(json.dumps(description, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'weights_sha256': description['weights_sha256'], 'bytes': len(weights)}))


if __name__ == '__main__': main()
