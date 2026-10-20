"""Model-free checks for benchmark provenance, aggregation and scoped memory."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from benchmark_readme import native_order
from report_readme_benchmarks import aggregate, metric, sha, verified_case, variation_warnings, model_source_comparisons, refresh_comparisons
from report_readme_benchmarks import NATIVE_PERFORMANCE_SUMMARY, performance_analysis, update_readme


class ReadmeBenchmarkTests(unittest.TestCase):
    def test_front_page_regeneration_is_compact_and_preserves_authored_text(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / 'README.md').read_text(encoding='utf-8')
        data = json.loads((root / 'docs/benchmarks/current/measurements.json').read_text())
        def lookup(route, model):
            return next(row for row in data['records']
                        if row['route'] == route and row['model'] == model and
                        row['extent'] == [1920,1080] and row['profile'] == 'default')
        start = '<!-- current-two-model-benchmark -->'
        end = '<!-- /current-two-model-benchmark -->'
        before, rest = source.split(start)
        _, after = rest.split(end)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'README.md'
            path.write_text(source, encoding='utf-8')
            with patch('report_readme_benchmarks.ROOT', Path(folder)):
                update_readme(lookup, {}, '20 October 2026')
                once = path.read_text(encoding='utf-8')
                update_readme(lookup, {}, '20 October 2026')
            self.assertEqual(once, path.read_text(encoding='utf-8'))
        self.assertEqual(once.split(start)[0], before)
        block, actual_after = once.split(start)[1].split(end)
        self.assertEqual(actual_after, after)
        self.assertLess(len(block.splitlines()), 50)
        self.assertIn('## Native RTX: the main event', block)
        for route in ('phoenix', 'nvidia'):
            for model in ('original', 'small-fp8'):
                row = lookup(route, model)
                self.assertIn(f"{row['gpu']['mean_process_median_ms']:.3f} | "
                              f"{row['wall']['mean_process_median_ms']:.3f}", block)
        self.assertIn('including Original at 4K', block)
        for detail in ('Private commit MiB', 'Tensors / packed records', 'GPU-gap follow-up'):
            self.assertNotIn(detail, block)

    def test_performance_summary_is_scoped_and_kept_in_both_readmes(self):
        root = Path(__file__).resolve().parents[2]
        for name in ('README.md', 'docs/benchmarks/current/README.md'):
            self.assertIn(NATIVE_PERFORMANCE_SUMMARY, (root / name).read_text(encoding='utf-8'))
        self.assertIn(NATIVE_PERFORMANCE_SUMMARY,
                      performance_analysis(None, {'native_matched_comparisons': []}))
        for boundary in ('1080p', 'RTX 5080', 'not end-to-end game FPS', 'portable-backend'):
            self.assertIn(boundary, NATIVE_PERFORMANCE_SUMMARY)

    def test_invalid_timing_cannot_be_published_as_success(self):
        for bad in (0,-1,float('nan'),float('inf')):
            with self.subTest(value=bad),self.assertRaises(RuntimeError):
                metric([dict(wall={'median_ms':bad},wall_samples_ms=[bad])],'wall')
        with self.assertRaises(RuntimeError):
            metric([dict(wall={'median_ms':1})],'wall')

    def test_refresh_comparison_never_pools_different_routes_or_profiles(self):
        row=dict(route='cuda',model='original',extent=[1920,1080],profile='default',
                 wall=dict(mean_process_median_ms=2),gpu=None)
        previous=dict(source_sha256='previous',date='2026-10-20',records=[row])
        current=copy.deepcopy(row);current['wall']['mean_process_median_ms']=1
        other=copy.deepcopy(current);other['route']='cpu32'
        feature=copy.deepcopy(current);feature['profile']='shared_export'
        value=refresh_comparisons([current,other,feature],previous)
        self.assertFalse(value['controlled_before_after'])
        self.assertEqual(len(value['records']),1)
        self.assertEqual(value['records'][0]['wall']['change_percent'],-50)
        self.assertNotIn('gpu',value['records'][0])

    def test_shared_export_requires_matched_inputs_and_all_capture_hashes(self):
        row=dict(route='cuda',model='original',extent=[320,180],profile='shared_export',
                 processes=2,across_process_hashes_equal=True,input_hashes=['input']*2,
                 output_hashes=[dict(reset='r',history='h')]*2)
        baseline=copy.deepcopy(row);baseline['profile']='default'
        lookup=lambda *_:baseline
        self.assertTrue(model_source_comparisons([row],lookup)[0]['reset_and_history_exact'])
        for field,value in [('input_hashes',['wrong']*2),
                            ('output_hashes',[dict(reset='r',history='wrong')]*2),
                            ('processes',1),('across_process_hashes_equal',False)]:
            changed=copy.deepcopy(row);changed[field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):
                model_source_comparisons([changed],lookup)
        self.assertEqual(model_source_comparisons([baseline],lookup),[])

    def test_native_order_counterbalances_all_cells(self):
        first,second=native_order(0),native_order(1)
        self.assertEqual(first,list(reversed(second)))
        self.assertEqual(len(set(first)),4)
        self.assertEqual(first,native_order(2))

    def test_equal_process_weight_despite_unequal_frame_counts(self):
        value=metric([dict(wall={'median_ms':1},wall_samples_ms=[1]*100),
                      dict(wall={'median_ms':3},wall_samples_ms=[3]*2)],'wall')
        self.assertEqual(value['mean_process_median_ms'],2)
        self.assertEqual(value['sample_count'],102)
        self.assertEqual(value['process_median_range_ms'],[1,3])
        self.assertEqual(value['process_median_spread_percent'],100)
        self.assertIsNone(metric([{}],'gpu'))

    def test_variation_warning_is_per_model_and_extent(self):
        row=dict(route='cpu32',model='small-fp8',extent=[1920,1080],profile='default',
                 wall=dict(process_median_spread_percent=25,process_median_range_ms=[5,7]),gpu=None)
        self.assertEqual(len(variation_warnings([row])),1)
        other=copy.deepcopy(row);other['extent']=[320,180]
        self.assertEqual(variation_warnings([other])[0]['extent'],[320,180])
        other['extent']=[1920,1080];other['profile']='passes2'
        self.assertEqual(variation_warnings([other]),[])

    def test_changed_or_failed_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'case.json'
            path.write_text(json.dumps(dict(passed=True)))
            entry={'sha256':sha(path)}
            self.assertTrue(verified_case(path,entry)['passed'])
            path.write_text(json.dumps(dict(passed=False)))
            with self.assertRaisesRegex(RuntimeError,'changed'):
                verified_case(path,entry)
            with self.assertRaisesRegex(RuntimeError,'did not pass'):
                verified_case(path,{'sha256':sha(path)})

    def test_memory_ranges_deltas_and_ownership_stay_separate(self):
        case=dict(route='cuda',model='original',extent=[1920,1080],profile='default',
            memory={'device_local_bytes':100,'host_private_bytes':300},
            memory_before={'device_local_bytes':40,'host_private_bytes':250},
            runtime_status={'backend_allocation_bytes':20,'host_frame_buffers':{'noise_capacity_bytes':10}},
            output_hashes={'reset':'r','history':'h'},create_ms=5,first_call_ms=8,
            finite_output=True,reset_repeatable=True,output_mean_abs_change=.1,input_sha256='input',
            wall={'median_ms':1},wall_samples_ms=[1],measured_depths=[8])
        other=copy.deepcopy(case)
        other['memory']['device_local_bytes']=120
        with patch('report_readme_benchmarks.sha',return_value='receipt'):
            rows=aggregate([('cuda-original-1920x1080-default-p0',case),
                            ('cuda-original-1920x1080-default-p1',other)])
        mem=rows[0]['memory']
        self.assertEqual(mem['device_local_bytes'],110)
        self.assertEqual(mem['process_ranges_bytes']['device_local_bytes'],[100,120])
        self.assertEqual(mem['change_from_pre_session_bytes']['device_local_bytes'],70)
        self.assertEqual(mem['requested_backend_bytes'],20)
        self.assertEqual(mem['host_frame_buffers']['noise_capacity_bytes'],10)
        self.assertNotIn('network_owned_allocation_info_bytes',mem)
        self.assertTrue(rows[0]['across_process_hashes_equal'])


if __name__=='__main__':
    unittest.main()
