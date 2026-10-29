"""Cloned graphics checks must authorize their own fixture directory."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from test_graphics_apis import copy_case_configuration


class GraphicsCaseConfigurationTests(unittest.TestCase):
    def test_relocation_preserves_input_contract_and_external_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'Phoenix.json'
            model = {'kind': 'model_folder', 'path': str(root / 'shared-model'), 'model': 'original'}
            original = {'schema': 'phoenix-live/1', 'armed': True,
                        'sandbox': str(root / 'base'), 'in_place_game_directory': str(root / 'base'),
                        'in_place_approval': 'I approve driving this installed game in place',
                        'input_provider': 'feeder_d3d12', 'weight_source': model}
            source.write_text(json.dumps(original))
            for name in ('motion', 'control', 'resize'):
                target = root / name
                target.mkdir()
                copy_case_configuration(source, target)
                result = json.loads((target / source.name).read_text())
                self.assertEqual(str(target.resolve()), result.pop('sandbox'))
                self.assertEqual(str(target.resolve()), result.pop('in_place_game_directory'))
                self.assertEqual({k: v for k, v in original.items() if k not in ('sandbox', 'in_place_game_directory')}, result)
            self.assertEqual(original, json.loads(source.read_text()))

    def test_non_approval_configuration_is_copied_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'ReShade.ini'
            source.write_bytes(b'[Phoenix]\r\nBackend=3\r\n')
            target = root / 'motion'
            target.mkdir()
            copy_case_configuration(source, target)
            self.assertEqual(source.read_bytes(), (target / source.name).read_bytes())


if __name__ == '__main__':
    unittest.main()
