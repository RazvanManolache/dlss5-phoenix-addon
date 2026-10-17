import copy
import unittest

from probe_wow64_tls import benchmark_conclusions


class BenchmarkTests(unittest.TestCase):
    def fixture(self):
        rows, frames = {}, {}
        for backend in ('phoenix', 'nvidia'):
            for prefix, bits in (('native', 64), ('thread', 32)):
                key = prefix+'_d3d12_'+backend
                rows[key] = dict(exit_code=0, process_pid=bits)
                timing = dict(width=1920, height=1080, scale=1, warmup_seconds=5,
                    requested_seconds=20, measured_seconds=20.004, completed_frames=4000,
                    completed_output_fps=4000/20.004, changed_rgb_samples=100,
                    output_sha256='a'*64,
                    gpu_processing=dict(samples=4000, mean_ms=4, median_ms=4, p95_ms=5),
                    synchronized_call=dict(samples=4000, mean_ms=5, median_ms=5, p95_ms=6))
                frames[key] = dict(passed=True, gpu_copy_verified=True, process_pid=bits,
                    process_bits=bits, pointer_bits=64, helper_pid=0, benchmark=timing)
        return rows, frames

    def test_valid_and_explicit_scope(self):
        rows, frames = self.fixture()
        for row in benchmark_conclusions(rows, frames).values():
            self.assertTrue(row['passed'])
            self.assertFalse(row['game_presentation_benchmarked'])
            self.assertFalse(row['per_frame_x86_handoff_benchmarked'])

    def test_refuse_failed_wrong_process_or_missing_pixels(self):
        rows, frames = self.fixture()
        key = 'thread_d3d12_phoenix'
        for field, value in [('passed', False), ('gpu_copy_verified', False), ('process_pid', 50),
                             ('process_bits', 64), ('pointer_bits', 32), ('helper_pid', 55)]:
            bad = copy.deepcopy(frames)
            bad[key][field] = value
            self.assertFalse(benchmark_conclusions(rows, bad)['phoenix']['passed'])
        rows[key]['exit_code'] = 0xc0000005
        self.assertFalse(benchmark_conclusions(rows, frames)['phoenix']['passed'])

    def test_refuse_wrong_profile_fake_throughput_or_invalid_timing(self):
        rows, frames = self.fixture()
        key = 'thread_d3d12_nvidia'
        for field, value in [('width', 640), ('height', 384), ('scale', 2), ('warmup_seconds', 0),
                             ('measured_seconds', 1), ('measured_seconds', float('nan')),
                             ('completed_frames', 0), ('completed_output_fps', 600),
                             ('changed_rgb_samples', 0), ('output_sha256', 'invalid')]:
            bad = copy.deepcopy(frames)
            bad[key]['benchmark'][field] = value
            self.assertFalse(benchmark_conclusions(rows, bad)['nvidia']['passed'])
        for field in ('gpu_processing', 'synchronized_call'):
            for metric, value in [('samples', 3999), ('mean_ms', float('nan')), ('median_ms', 0), ('p95_ms', -1)]:
                bad = copy.deepcopy(frames)
                bad[key]['benchmark'][field][metric] = value
                self.assertFalse(benchmark_conclusions(rows, bad)['nvidia']['passed'])

    def test_absent_or_timed_out_control_has_no_timing(self):
        rows, frames = self.fixture()
        rows['native_d3d12_phoenix']['timed_out'] = True
        result = benchmark_conclusions(rows, frames)['phoenix']
        self.assertFalse(result['passed'])
        self.assertNotIn('timing', result['cases']['native'])
        self.assertFalse(benchmark_conclusions({}, {})['nvidia']['passed'])


if __name__ == '__main__':
    unittest.main()
