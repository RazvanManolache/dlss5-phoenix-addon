"""Model-free checks for the native source-maintenance guards."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from audit_kernel_structure import audit, check_contracts


class StructureGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = audit(ROOT)

    def test_current_contracts(self):
        check_contracts(self.inventory, ROOT)
        self.assertEqual(self.inventory['kernel_modules'], 41)

    def test_duplicate_schedule_helper_rejected(self):
        inventory = copy.deepcopy(self.inventory)
        inventory['helper_definitions']['token_tile_work'].append('duplicate.cuh')
        with self.assertRaisesRegex(RuntimeError, 'expected 1'):
            check_contracts(inventory, ROOT)

    def test_duplicate_boundary_helpers_rejected(self):
        for name in ('unsigned_word_quotient', 'boundary_tail_steps',
                     'prepare_unsigned_word_divisor', 'prepared_unsigned_word_quotient',
                     'boundary_item_address', 'clear_boundary_vector', 'clear_boundary_rows'):
            with self.subTest(helper=name):
                inventory = copy.deepcopy(self.inventory)
                inventory['helper_definitions'][name].append('duplicate.cuh')
                with self.assertRaisesRegex(RuntimeError, 'expected 1'):
                    check_contracts(inventory, ROOT)

    def test_scalar_vector_variants_remain_separate(self):
        self.assertEqual(len(self.inventory['helper_definitions']['load_four_words_cached']), 2)
        inventory = copy.deepcopy(self.inventory)
        inventory['helper_definitions']['load_four_words_cached'].pop()
        with self.assertRaisesRegex(RuntimeError, 'expected 2'):
            check_contracts(inventory, ROOT)

    def test_three_dimensional_block_rejected(self):
        source = self.inventory['scheduling_sources'][0]
        catalogue = {'kernels': [{'source': source, 'block': [32, 2, 2]}]}
        inventory = copy.deepcopy(self.inventory)
        inventory['scheduling_sources'] = [source]
        with tempfile.TemporaryDirectory(prefix='structure-guard-') as folder:
            root = Path(folder)
            path = root / 'src/native/model/production_kernel_contracts.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(catalogue))
            with self.assertRaisesRegex(RuntimeError, 'requires blockDim.z=1'):
                check_contracts(inventory, root)

    def test_uncatalogued_schedule_rejected(self):
        inventory = copy.deepcopy(self.inventory)
        inventory['scheduling_sources'] = ['missing.cu']
        with self.assertRaisesRegex(RuntimeError, 'no valid catalogue block'):
            check_contracts(inventory, ROOT)


if __name__ == '__main__':
    unittest.main()
