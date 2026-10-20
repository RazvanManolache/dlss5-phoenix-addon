"""Model-free regression cases for reproducible layouts and source-input records."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import generate_weight_layouts as layouts
import verify_source_provenance as provenance


class LayoutGenerationTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(layouts.DECLARATIONS.read_text(encoding='utf-8'))

    def test_checked_in_header_and_known_gap(self):
        self.assertEqual(layouts.check(), 23)
        parts = self.document['layouts'][0]['parts']
        self.assertEqual(parts[1][2], 8192)
        self.assertEqual(parts[2][1], 8208)
        self.assertEqual(parts[4], ['qkv', 9312, 12384, 'PackedFp8', 96, 32, 6])

    def test_duplicate_record_rejected(self):
        self.document['layouts'].append(copy.deepcopy(self.document['layouts'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate record'):
            layouts.render(self.document)

    def test_overlapping_parts_rejected(self):
        self.document['layouts'][0]['parts'][1][1] = 0
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            layouts.render(self.document)

    def test_shape_extent_mismatch_rejected(self):
        self.document['layouts'][0]['parts'][0][4] = 127
        with self.assertRaisesRegex(ValueError, 'shape differs'):
            layouts.render(self.document)

    def test_unsupported_encoding_rejected(self):
        self.document['layouts'][0]['parts'][0][3] = 'Unspecified'
        with self.assertRaisesRegex(ValueError, 'encoding'):
            layouts.render(self.document)

    def test_attribution_required(self):
        self.document['attribution'] = ''
        with self.assertRaisesRegex(ValueError, 'attribution'):
            layouts.render(self.document)

    def test_booleans_not_record_extents(self):
        self.document['layouts'][0]['bytes'] = True
        with self.assertRaisesRegex(ValueError, 'extent'):
            layouts.render(self.document)

    def test_nonobject_layout_rejected(self):
        self.document['layouts'][0] = []
        with self.assertRaisesRegex(ValueError, 'record object'):
            layouts.render(self.document)

    def test_stale_header_detected_without_rewriting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / layouts.DECLARATIONS.relative_to(ROOT)
            header = root / layouts.HEADER.relative_to(ROOT)
            data.parent.mkdir(parents=True)
            header.parent.mkdir(parents=True)
            data.write_text(json.dumps(self.document), encoding='utf-8')
            header.write_text('stale\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'header is stale'):
                layouts.check(root)
            self.assertEqual(header.read_text(), 'stale\n')

    def test_small_header_and_independent_schema(self):
        self.assertEqual(layouts.check(model='small_fp8'), 23)
        small = json.loads((ROOT / 'tools/model_layouts/small_fp8.json').read_text())
        self.assertEqual(sum(len(x['parts']) for x in small['layouts']), 157)
        self.assertIn('namespace small_fp8_layouts', layouts.render(small))
        self.assertNotIn('enum TensorEncoding', layouts.render(small))
        # The current models share these packing families; graph topology and
        # selection are separate. Do not invent differences in their layouts.
        self.assertEqual(small['layouts'], self.document['layouts'])

    def test_small_invalid_shape_and_overlap_rejected(self):
        original = json.loads((ROOT / 'tools/model_layouts/small_fp8.json').read_text())
        for index, field, value in ((0, 4, 127), (1, 1, 0)):
            document = copy.deepcopy(original)
            document['layouts'][0]['parts'][index][field] = value
            with self.assertRaises(ValueError):
                layouts.render(document)

    def test_small_stale_header_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, header = layouts.paths(root, 'small_fp8')
            data.parent.mkdir(parents=True)
            header.parent.mkdir(parents=True)
            data.write_bytes((ROOT / 'tools/model_layouts/small_fp8.json').read_bytes())
            header.write_text('stale\n')
            with self.assertRaisesRegex(ValueError, 'header is stale'):
                layouts.check(root, model='small_fp8')
            self.assertEqual(header.read_text(), 'stale\n')


class ImportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'src/host.cpp'
        self.source.parent.mkdir()
        self.source.write_bytes(b'first\nsecond\n')
        self.original_import = 'a' * 64
        self.document = dict(schema='phoenix-source-import/2', files=[dict(
            source='declared/import/host.cpp', path='src/host.cpp',
            import_sha256=self.original_import, **provenance.hashes(self.source.read_bytes()))])
        self.manifest = self.root / provenance.MANIFEST
        self.manifest.parent.mkdir(parents=True)
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.document), encoding='utf-8')

    def test_current_manifest(self):
        self.assertEqual(provenance.verify_manifest(), 5)

    def test_line_ending_only_difference_accepted(self):
        self.source.write_bytes(b'first\r\nsecond\r\n')
        self.assertEqual(provenance.verify_manifest(self.root), 1)

    def test_changed_source_rejected(self):
        self.source.write_bytes(b'changed\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            provenance.verify_manifest(self.root)

    def test_refresh_preserves_import_record(self):
        self.source.write_bytes(b'legitimate maintained adaptation\n')
        provenance.verify_manifest(self.root, refresh=True)
        updated = json.loads(self.manifest.read_text())['files'][0]
        self.assertEqual(updated['import_sha256'], self.original_import)
        self.assertEqual(updated['source'], 'declared/import/host.cpp')
        self.assertEqual(updated['sha256'], provenance.hashes(self.source.read_bytes())['sha256'])
        self.assertEqual(provenance.verify_manifest(self.root), 1)

    def test_escape_path_rejected(self):
        self.document['files'][0]['path'] = '../outside.cpp'
        self.save()
        with self.assertRaisesRegex(ValueError, 'escapes'):
            provenance.verify_manifest(self.root)

    def test_duplicate_destination_rejected(self):
        self.document['files'].append(copy.deepcopy(self.document['files'][0]))
        self.save()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            provenance.verify_manifest(self.root)

    def test_nonobject_source_entry_rejected(self):
        self.document['files'][0] = []
        self.save()
        with self.assertRaisesRegex(ValueError, 'record object'):
            provenance.verify_manifest(self.root)

    def test_refresh_does_not_partially_update_invalid_manifest(self):
        broken = copy.deepcopy(self.document['files'][0])
        broken['path'] = 'src/absent.cpp'
        self.document['files'].append(broken)
        self.save()
        before = self.manifest.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Missing maintained'):
            provenance.verify_manifest(self.root, refresh=True)
        self.assertEqual(self.manifest.read_bytes(), before)

    def make_import_snapshot(self):
        snapshot = self.root / 'separate-import'
        imported = snapshot / 'declared/import/host.cpp'
        imported.parent.mkdir(parents=True)
        imported.write_bytes(b'original, before adaptation\r\n')
        self.document['files'][0]['source'] = 'declared\\import\\host.cpp'
        self.document['files'][0]['import_sha256'] = provenance.hashes(imported.read_bytes())['sha256']
        self.save()
        return snapshot, imported

    def test_declared_import_bytes_verified_separately(self):
        snapshot, imported = self.make_import_snapshot()
        before = self.manifest.read_bytes()
        self.assertNotEqual(imported.read_bytes(), self.source.read_bytes())
        self.assertEqual(provenance.verify_import_snapshot(snapshot, self.root), 1)
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_import_mutation_rejected(self):
        snapshot, imported = self.make_import_snapshot()
        imported.write_bytes(b'tampered\n')
        with self.assertRaisesRegex(ValueError, 'import hash mismatch'):
            provenance.verify_import_snapshot(snapshot, self.root)

    def test_import_escape_rejected(self):
        snapshot, _ = self.make_import_snapshot()
        self.document['files'][0]['source'] = '..\\outside.cpp'
        self.save()
        with self.assertRaisesRegex(ValueError, 'escapes'):
            provenance.verify_import_snapshot(snapshot, self.root)

    def test_missing_import_not_treated_as_authenticated(self):
        snapshot, imported = self.make_import_snapshot()
        imported.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing maintained'):
            provenance.verify_import_snapshot(snapshot, self.root)


if __name__ == '__main__':
    unittest.main()
