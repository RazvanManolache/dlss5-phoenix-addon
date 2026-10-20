"""Model-free regression checks for the isolated register experiment tools."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from benchmark_register_limits import set_register_budget, spill_properties
from report_register_limits import summarize


class RegisterExperimentTests(unittest.TestCase):
    def test_spill_properties_keep_entries_separate(self):
        text = ('ptxas info : Function properties for small\n'
                '  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n'
                'ptxas info : Used 160 registers, 4096 bytes smem\n'
                'ptxas info : Function properties for large\n'
                '  8 bytes stack frame, 4 bytes spill stores, 12 bytes spill loads\n'
                'ptxas info : Used 128 registers\n')
        rows = spill_properties(text)
        self.assertEqual(rows['small']['spill_store_bytes'], 0)
        self.assertEqual(rows['large']['spill_load_bytes'], 12)
        self.assertEqual(rows['large']['registers'], 128)

    def test_replace_budget_retains_other_entry_and_instructions(self):
        source = ('.visible .entry a(.param .u32 x)\n.maxntid 32, 2, 1\n.maxnreg 128\n{\nret;\n}\n'
                  '.visible .entry b()\n.maxnreg 192\n{\nret;\n}\n')
        result = set_register_budget(source, ['a'], 160)
        self.assertIn('.maxntid 32, 2, 1', result)
        self.assertIn('.maxnreg 160', result)
        self.assertIn('.entry b()\n.maxnreg 192', result)
        self.assertEqual(result.count('ret;'), 2)
        self.assertEqual(result.count('.maxnreg'), 2)

    def test_add_missing_budget(self):
        self.assertIn('.maxnreg 128\n{', set_register_budget('.visible .entry a()\n{ret;}', ['a'], 128))

    def test_missing_or_duplicate_entry_rejected(self):
        for source in ('', '.visible .entry a(){ret;}\n.visible .entry a(){ret;}'):
            with self.assertRaisesRegex(ValueError, 'Missing/duplicate'):
                set_register_budget(source, ['a'], 128)

    def test_duplicate_directive_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate register'):
            set_register_budget('.visible .entry a()\n.maxnreg 128\n.maxnreg 160\n{ret;}', ['a'], 128)

    def test_out_of_range_budget_rejected(self):
        for budget in (0, 31, 256):
            with self.assertRaisesRegex(ValueError, 'budget'):
                set_register_budget('.visible .entry a(){ret;}', ['a'], budget)

    def report_fixture(self):
        row = dict(passed=True, reset_repeatable=True, debug_errors=0,
                   captures={'reset_a': {'sha256': 'good'}}, kernel_selections=[],
                   gpu={'median_ms': 2})
        profile = dict(passed=True, gpu_ms=2, rounds=[row])
        report = dict(schema='test', completed=True, passed=False, gpu='test',
                      baseline_resources=[], variant_resources={}, compiled_variants=[], cases=[
                          dict(model='original', width=1920, height=1080,
                               profiles=dict(production=profile, failing=dict(passed=False, rounds=[])))])
        return report

    def test_failed_modes_have_no_public_timing_or_hashes(self):
        result = summarize(self.report_fixture(), {}, dict(passed=True))
        profile = result['cases'][0]['profiles']['failing']
        self.assertFalse(profile['passed'])
        self.assertNotIn('gpu_ms', profile)
        self.assertNotIn('output_hashes', profile)

    def test_mismatched_hashes_rejected_even_with_passed_flag(self):
        report = self.report_fixture()
        report['cases'][0]['profiles']['failing'] = dict(passed=True, gpu_ms=1, rounds=[
            dict(passed=True, reset_repeatable=True, debug_errors=0, captures={'wrong': 'hash'})])
        profile = summarize(report, {}, dict(passed=True))['cases'][0]['profiles']['failing']
        self.assertFalse(profile['passed'])
        self.assertNotIn('gpu_ms', profile)

    def test_incomplete_report_or_controls_rejected(self):
        report = self.report_fixture()
        report['completed'] = False
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            summarize(report, {}, dict(passed=True))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            summarize(self.report_fixture(), {}, dict(passed=False))


if __name__ == '__main__':
    unittest.main()
