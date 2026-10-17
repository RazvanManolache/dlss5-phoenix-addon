import copy
import unittest

from compare_native_methods import build_comparison


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        profile = dict(width=1920, height=1080, warmup_seconds=5, measured_seconds=20, reset_each_frame=True)
        timing = dict(gpu_processing=dict(mean_ms=4, p95_ms=5, samples=4000),
            synchronized_call=dict(mean_ms=5, p95_ms=6, samples=4000), completed_output_fps=200, reset_each_frame=True)
        case = dict(passed=True, timing=timing)
        native = dict(gpu='test GPU, driver', benchmark_profile=profile,
            benchmark_conclusions={backend:dict(passed=True, cases=dict(native=case, thread=case)) for backend in ('phoenix', 'nvidia')})
        host = dict(gpu=native['gpu'], benchmark_profile=profile,
            x86_host_benchmark_conclusions=dict(passed=True, cases=dict(thread_driver_host=case)))
        helper_case = dict(passed=True, execution_mode='helper', helper_pid=42, inference_pid=42,
            fixture='dx12-x86', completed_output_fps=60, gpu_model=timing['gpu_processing'],
            processing_roundtrip=dict(mean_ms=12, p95_ms=13, samples=41))
        helper = dict(gpu=native['gpu'], **profile,
            runs=dict(helper_x86=helper_case, nvidia_x86=helper_case))
        return native, host, helper

    def test_all_rows_and_unavailable_nvidia_host(self):
        result = build_comparison(*self.fixture())
        self.assertEqual(len(result['rows']), 8)
        missing = [row for row in result['rows'] if not row['available']]
        self.assertEqual([(row['backend'], row['method']) for row in missing], [('nvidia', 3)])
        self.assertFalse(result['game_fps'])
        self.assertIn('scope differs', result['limitations'][0])

    def test_refuse_failed_or_mismatched_profile(self):
        data = self.fixture()
        for index, field, value in [(0, 'width', 640), (1, 'warmup_seconds', 0), (2, 'measured_seconds', 10)]:
            bad = copy.deepcopy(data)
            profile = bad[index]['benchmark_profile'] if index < 2 else bad[index]
            profile[field] = value
            with self.assertRaises(ValueError): build_comparison(*bad)
        for index in range(3):
            bad = copy.deepcopy(data)
            bad[index]['gpu'] = 'another GPU'
            with self.assertRaises(ValueError): build_comparison(*bad)

    def test_refuse_invalid_timing_or_fake_helper(self):
        native, host, helper = self.fixture()
        host['x86_host_benchmark_conclusions']['passed'] = False
        with self.assertRaises(ValueError): build_comparison(native, host, helper)
        native, host, helper = self.fixture()
        helper['runs']['helper_x86']['inference_pid'] = 99
        with self.assertRaises(ValueError): build_comparison(native, host, helper)
        native, host, helper = self.fixture()
        native['benchmark_conclusions']['phoenix']['cases']['thread']['timing']['gpu_processing']['mean_ms'] = float('nan')
        with self.assertRaises(ValueError): build_comparison(native, host, helper)


if __name__ == '__main__':
    unittest.main()
