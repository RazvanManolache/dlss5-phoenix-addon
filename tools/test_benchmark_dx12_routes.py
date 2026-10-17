import math
import unittest
from benchmark_dx12_routes import CASES, execution_setting, stats, summarize


class MeasurementTests(unittest.TestCase):
    def test_nvidia_native_case_really_selects_in_process(self):
        case=next(case for case in CASES if case[0]=='nvidia_native_x64')
        self.assertEqual(case[1:],('dx12-x64','nvidia','in_process'))
        self.assertEqual(execution_setting(case[3]),'in_process')
        self.assertEqual(execution_setting('in_process_portable'),'portable')
        self.assertEqual(execution_setting('helper'),'helper')
        with self.assertRaises(ValueError): execution_setting('automatic')

    def test_native_measurement_requires_same_pid_without_helper(self):
        result=dict(passed=True,execution_mode='in_process',fixture_pid=42,backend='nvidia',neural_fps=100,
            measurement=dict(start_render_seconds=5,end_render_seconds=25,frames_rendered=2000))
        rows=[dict(seconds=i,delivered=i,execution_mode='in_process',inference_pid=42,helper_pid=0,
            gpu_timing=dict(rows=[dict(mode=1,width=1920,height=1080,nvidia=dict(last_ms=4),route=dict(last_ms=5))]))
            for i in (6,7,8)]
        value=summarize(result,rows,'in_process')
        self.assertTrue(value['passed'])
        self.assertEqual(value['gpu_model']['median_ms'],4)
        for key,value in (('helper_pid',99),('inference_pid',99)):
            bad=[dict(row,**{key:value}) for row in rows]
            self.assertFalse(summarize(result,bad,'in_process')['passed'])

    def test_nonfinite_and_zero_are_not_timings(self):
        self.assertIsNone(stats([0, -1, math.nan, math.inf, None]))
        self.assertEqual(stats([1, 2, 3])['median_ms'], 2)

    def test_failure_never_receives_performance_numbers(self):
        value = summarize(dict(passed=False, error='not processing', neural_fps=900), [], 'helper')
        self.assertFalse(value['passed'])
        self.assertNotIn('output_interval_ms', value)

    def test_fallback_cannot_masquerade_as_requested_route(self):
        value = summarize(dict(passed=True, execution_mode='helper'), [], 'in_process_portable')
        self.assertFalse(value['passed'])
        self.assertIn('mechanism differed', value['error'])

    def test_unique_completed_frames_only(self):
        result = dict(passed=True, execution_mode='helper', backend='phoenix', neural_fps=100,
            measurement=dict(start_render_seconds=5, end_render_seconds=25, frames_rendered=2000))
        rows = [dict(seconds=t, delivered=1, execution_mode='helper', helper_pid=3) for t in (6, 7, 8, 9)]
        self.assertFalse(summarize(result, rows, 'helper')['passed'])
        for i, row in enumerate(rows):
            row.update(delivered=i+1, processing_roundtrip_ms=7,
                gpu_timing=dict(rows=[dict(mode=3, width=1920, height=1080, phoenix=dict(last_ms=4), route=dict(last_ms=5))]))
        value = summarize(result, rows, 'helper')
        self.assertTrue(value['passed'])
        self.assertEqual(value['output_interval_ms'], 10)
        self.assertEqual(value['gpu_model']['median_ms'], 4)
        self.assertEqual(value['processing_roundtrip']['median_ms'], 7)

    def test_mechanism_change_invalidates_run(self):
        result = dict(passed=True, execution_mode='helper', measurement=dict(start_render_seconds=5, end_render_seconds=25))
        self.assertFalse(summarize(result, [dict(seconds=8, execution_mode='in_process')], 'helper')['passed'])

    def test_gpu_route_does_not_mix_transport_only_rows(self):
        result = dict(passed=True, execution_mode='helper', backend='nvidia', neural_fps=50,
            measurement=dict(start_render_seconds=5, end_render_seconds=25, frames_rendered=1000))
        rows = [dict(seconds=i, delivered=i, execution_mode='helper', helper_pid=1,
            gpu_timing=dict(rows=[dict(mode=1, width=1920, height=1080, nvidia=dict(last_ms=4), route=dict(last_ms=5)),
                                  dict(mode=1, width=1920, height=1080, nvidia=dict(last_ms=0), route=dict(last_ms=.1))])) for i in (6, 7, 8)]
        self.assertEqual(summarize(result, rows, 'helper')['gpu_route']['median_ms'], 5)


if __name__ == '__main__':
    unittest.main()
