import unittest
from probe_wow64_native import conclusion, debugger_findings
from probe_x86_vulkan import conclusion as vulkan_conclusion


class Wow64ProbeTests(unittest.TestCase):
    def loaded(self, stage=12, created=1, code=0):
        return dict(exit_code=code, events=[dict(worker_stage=stage, module_created=created)])

    def test_x64_control_and_bare_code_do_not_prove_cross_bitness_graphics(self):
        result = conclusion(dict(native_x64_control=self.loaded(),
                                 cross_bare_worker=self.loaded(stage=100, created=0)))
        self.assertTrue(result['native_x64_control_works'])
        self.assertTrue(result['import_free_x64_dll_works'])
        self.assertFalse(result['cross_bitness_gpu_module_loading_works'])

    def test_module_creation_alone_never_claims_inference(self):
        result = conclusion(dict(native_thread_worker=self.loaded()))
        self.assertTrue(result['cross_bitness_gpu_module_loading_works'])
        self.assertFalse(result['completed_gpu_inference'])

    def test_crash_or_incomplete_teardown_is_not_a_success(self):
        for row in (self.loaded(code=0xc0000005), self.loaded(stage=10), self.loaded(created=0)):
            with self.subTest(row=row):
                self.assertFalse(conclusion(dict(cross_worker=row))['cross_bitness_gpu_module_loading_works'])

    def test_activation_and_gui_paths_require_completed_module_creation(self):
        for name in ('activation_worker', 'activation_thread_worker', 'gui_worker'):
            with self.subTest(name=name):
                self.assertTrue(conclusion({name: self.loaded()})['cross_bitness_gpu_module_loading_works'])
                self.assertFalse(conclusion({name: self.loaded(stage=0, created=0)})['cross_bitness_gpu_module_loading_works'])

    def test_debugger_exit_zero_does_not_hide_fatal_fault(self):
        result = debugger_findings('ExceptionCode: c0000005 (Access violation)\n'
                                   'gdi32full!_dyn_tls_init+0x2d:\n', 0)
        self.assertTrue(result['access_violation_observed'])
        self.assertEqual(result['debugger_exit_code'], 0)
        self.assertTrue(result['lines'])

    def test_exit_zero_without_control_output_does_not_prove_execution(self):
        result = conclusion(dict(mode_switch=dict(exit_code=0, events=[]),
                                 cross_bare_worker=dict(exit_code=0, events=[])))
        self.assertFalse(result['mode_switch_works'])
        self.assertFalse(result['import_free_x64_dll_works'])

    def test_mode_switch_needs_a_matching_control_result(self):
        for matched in (False, None):
            self.assertFalse(conclusion(dict(mode_switch=dict(exit_code=0,
                events=[dict(matched=matched)])))['mode_switch_works'])
        self.assertTrue(conclusion(dict(mode_switch=dict(exit_code=0,
            events=[dict(matched=True)])))['mode_switch_works'])

    def test_vulkan_device_creation_does_not_prove_compute(self):
        result = vulkan_conclusion({'x32': dict(exit_code=0, events=[dict(stage='device_created')])})
        self.assertFalse(result['x86_spirv_compute_works'])

    def test_vulkan_readback_is_not_full_model_inference(self):
        result = vulkan_conclusion({'x32': dict(exit_code=0, events=[
            dict(stage='spirv_result', matched=4096, expected=4096, passed=True)])})
        self.assertTrue(result['x86_spirv_compute_works'])
        self.assertFalse(result['x86_cuda_dispatch_works'])
        self.assertFalse(result['completed_gpu_inference'])

    def test_vulkan_mismatch_or_crash_is_not_compute_success(self):
        for code, matched in ((0, 4095), (0xc0000005, 4096)):
            result = vulkan_conclusion({'x32': dict(exit_code=code, events=[
                dict(stage='spirv_result', matched=matched, expected=4096, passed=True)])})
            self.assertFalse(result['x86_spirv_compute_works'])


if __name__ == '__main__':
    unittest.main()
