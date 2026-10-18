"""A graph pin without its bytes must never reach a self-contained add-on."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from embed_runtime import native_graph_files

class EmbeddedGraphTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        manifest = {}
        for filename, pin in (('native_graph.json', 'native_graph_sha256'),
                              ('native_graph_v2.json', 'native_graph_v2_sha256')):
            raw = json.dumps({'graph': filename}).encode()
            (root / filename).write_bytes(raw)
            manifest[pin] = hashlib.sha256(raw).hexdigest()
        return root, manifest

    def test_both_graphs_survive_embedding(self):
        with tempfile.TemporaryDirectory() as folder:
            root, manifest = self.fixture(folder)
            result = native_graph_files(root, manifest)
            self.assertEqual(set(result), {'native/native_graph.json', 'native/native_graph_v2.json'})
            for filename, raw in result.items():
                self.assertEqual(raw, (root / Path(filename).name).read_bytes())

    def test_missing_or_changed_selected_graph_rejects(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as folder:
                root, manifest = self.fixture(folder)
                target = root / 'native_graph_v2.json'
                if missing: target.unlink()
                else: target.write_bytes(b'tampered')
                with self.assertRaises((FileNotFoundError, ValueError)):
                    native_graph_files(root, manifest)

if __name__ == '__main__': unittest.main()
