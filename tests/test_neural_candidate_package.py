"""Candidate promotion must reject stale graph data before copying binaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('neural_package',Path(__file__).parents[1]/'tools/package_neural_candidate.py')
package=importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)

class NativeGraphPromotionTests(unittest.TestCase):
    def fixture(self,root,versioned):
        source=root/'src/native/model/production_native_graph.json'
        source.parent.mkdir(parents=True);source.write_bytes(b'current tile-aligned graph')
        bundle=root/'artifacts/current/native-kernels';bundle.mkdir(parents=True)
        (bundle/'native_graph.json').write_bytes(b'legacy graph' if versioned else source.read_bytes())
        manifest={'native_graph_sha256':package.sha(bundle/'native_graph.json')}
        if versioned:
            (bundle/'native_graph_v2.json').write_bytes(source.read_bytes())
            manifest['native_graph_v2_sha256']=package.sha(source)
        (bundle/'native_manifest.json').write_text(json.dumps(manifest))
        return bundle,source

    def test_current_graph_and_additive_graph_both_promote(self):
        for versioned in (False,True):
            with self.subTest(versioned=versioned),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);bundle,source=self.fixture(root,versioned)
                result=package.validate_native_graph(root)
                self.assertEqual(result['sha256'],package.sha(source))
                self.assertEqual(Path(result['path']).name,'native_graph_v2.json' if versioned else 'native_graph.json')

    def test_stale_or_tampered_graph_cannot_promote(self):
        for mutation in ('stale_legacy','stale_versioned','tampered_versioned','tampered_legacy'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);bundle,source=self.fixture(root,mutation!='stale_legacy')
                if mutation.startswith('stale'):
                    source.write_bytes(b'newer layout than the installed bundle')
                else:
                    (bundle/('native_graph_v2.json' if mutation=='tampered_versioned' else 'native_graph.json')).write_bytes(b'tampered graph')
                with self.assertRaises(ValueError):package.validate_native_graph(root)

if __name__=='__main__':unittest.main()
