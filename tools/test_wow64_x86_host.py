import copy
import unittest

from probe_wow64_tls import x86_host_conclusions, x86_host_benchmark_conclusions


class X86HostTests(unittest.TestCase):
    def fixture(self):
        rows, frames = {}, {}
        hashes = ['a', 'a', 'a', 'b', 'c', 'a']
        for name, bits, pid in [('native_driver_host', 64, 10), ('thread_driver_host', 32, 11)]:
            rows[name] = dict(exit_code=0, process_pid=pid)
            frames[name] = dict(passed=True, host_bits=bits, driver_bits=64, process_pid=pid, driver_pid=pid,
                helper_pid=0, phoenix64_loaded=False, host_owns_graph=True, host_owns_controls=True,
                controls_and_input_verified=True, authenticated_kernels=41, completed_frames=6,
                dispatched_kernels=924, functions_destroyed=41, modules_destroyed=41, cleanup_failures=0,
                native_compat_restored=True,
                frames=[dict(index=i, changed_rgb_samples=1, gpu_ms=1, output_sha256=key*64) for i, key in enumerate(hashes)])
        rows['native_d3d12_phoenix'] = dict(exit_code=0, process_pid=12)
        native = dict(native_d3d12_phoenix=dict(passed=True, process_pid=12, process_bits=64, pointer_bits=64,
            helper_pid=0, frames=[dict(output_sha256='a'*64) for _ in range(3)]))
        return rows, frames, native

    def test_real_host_control_and_exact_output(self):
        result = x86_host_conclusions(*self.fixture())
        for field in ('same_process_x86_inference', 'bit_exact_host_architectures', 'bit_exact_against_native_api'):
            self.assertTrue(result[field])
        self.assertFalse(result['game_presentation_validated'])
        self.assertFalse(result['production_route_integrated'])

    def test_refuses_helper_wrong_architecture_or_x64_renderer(self):
        rows, frames, native = self.fixture()
        for field, value in [('host_bits', 64), ('driver_bits', 32), ('driver_pid', 12), ('helper_pid', 50),
                             ('phoenix64_loaded', True), ('host_owns_graph', False), ('host_owns_controls', False),
                             ('completed_frames', 5), ('dispatched_kernels', 923), ('modules_destroyed', 40),
                             ('functions_destroyed', 40), ('cleanup_failures', 1), ('native_compat_restored', False)]:
            bad = copy.deepcopy(frames)
            bad['thread_driver_host'][field] = value
            self.assertFalse(x86_host_conclusions(rows, bad, native)['same_process_x86_inference'])

    def test_rejects_stale_pixels_and_claimed_controls_without_change(self):
        rows, frames, native = self.fixture()
        for index in (3, 4):
            bad = copy.deepcopy(frames)
            bad['thread_driver_host']['frames'][index]['output_sha256'] = 'a'*64 if index == 3 else 'b'*64
            self.assertFalse(x86_host_conclusions(rows, bad, native)['same_process_x86_inference'])
        for field, value in [('changed_rgb_samples', 0), ('output_sha256', 'invalid'), ('gpu_ms', float('nan')), ('index', 99)]:
            bad = copy.deepcopy(frames)
            bad['thread_driver_host']['frames'][0][field] = value
            self.assertFalse(x86_host_conclusions(rows, bad, native)['same_process_x86_inference'])

    def test_native_renderer_control_is_required_for_renderer_parity(self):
        rows, frames, native = self.fixture()
        result = x86_host_conclusions(rows, frames, {})
        self.assertTrue(result['same_process_x86_inference'])
        self.assertFalse(result['bit_exact_against_native_api'])
        native['native_d3d12_phoenix']['frames'][0]['output_sha256'] = 'b'*64
        self.assertFalse(x86_host_conclusions(rows, frames, native)['bit_exact_against_native_api'])
        rows['thread_driver_host']['timed_out'] = True
        self.assertFalse(x86_host_conclusions(rows, frames, native)['same_process_x86_inference'])

    def benchmark_fixture(self):
        rows, frames, native = self.fixture()
        for frame in frames.values():
            frame['completed_frames'] = 5007
            frame['dispatched_kernels'] = 5007*154
            frame['benchmark'] = dict(width=1920, height=1080, scale=1, warmup_seconds=5,
                requested_seconds=20, measured_seconds=20.004, completed_frames=4000,
                total_iterations=5000, completed_output_fps=4000/20.004,
                reset_each_frame=True, changed_rgb_samples=10, output_sha256='a'*64,
                gpu_processing=dict(samples=4000, mean_ms=4, median_ms=4, p95_ms=5),
                synchronized_call=dict(samples=4000, mean_ms=5, median_ms=5, p95_ms=6))
        return rows, frames, native

    def test_benchmark_needs_proof_and_all_actual_dispatches(self):
        rows, frames, native = self.benchmark_fixture()
        result = x86_host_benchmark_conclusions(rows, frames, native)
        self.assertTrue(result['passed'])
        self.assertTrue(result['per_frame_x86_handoff_benchmarked'])
        self.assertFalse(result['game_presentation_benchmarked'])
        frames['thread_driver_host']['dispatched_kernels'] -= 154
        self.assertFalse(x86_host_benchmark_conclusions(rows, frames, native)['passed'])

    def test_benchmark_refuses_wrong_window_bad_timing_or_missing_validation(self):
        rows, frames, native = self.benchmark_fixture()
        for field, value in [('width', 640), ('height', 384), ('scale', 2), ('warmup_seconds', 0),
                             ('measured_seconds', 19), ('completed_output_fps', 900), ('total_iterations', 100),
                             ('reset_each_frame', False), ('output_sha256', 'invalid'), ('changed_rgb_samples', 0)]:
            bad = copy.deepcopy(frames)
            bad['thread_driver_host']['benchmark'][field] = value
            self.assertFalse(x86_host_benchmark_conclusions(rows, bad, native)['passed'])
        for field in ('gpu_processing', 'synchronized_call'):
            bad = copy.deepcopy(frames)
            bad['thread_driver_host']['benchmark'][field]['mean_ms'] = float('nan')
            self.assertFalse(x86_host_benchmark_conclusions(rows, bad, native)['passed'])


if __name__ == '__main__':
    unittest.main()
